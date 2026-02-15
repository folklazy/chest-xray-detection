# src/backbone.py
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models

try:
    import torchxrayvision as xrv
    XRV_AVAILABLE = True
except ImportError:
    XRV_AVAILABLE = False


# =====================================================
# Supported models (only 2, keep it strict)
# =====================================================
MODEL_ALIASES: dict[str, str] = {
    "densenet121": "densenet121",
    "xrv_densenet121_res224_all": "densenet121-res224-all",
}


def _is_head_param(name: str) -> bool:
    """
    Parameters that belong to the task head / classifier.
    - torchvision DenseNet: classifier.*
    - our XRV wrapper:      head.*
    - (generic)             fc.*
    """
    return name.startswith(("classifier.", "head.", "fc."))


@dataclass(frozen=True)
class ModelIO:
    expected_in_channels: int


# =====================================================
# XRV wrapper: features -> relu -> GAP -> head
# =====================================================
class XrvDenseNetFeatures(nn.Module):
    """
    TorchXRayVision DenseNet applies op_norm if you call model(x) directly.
    We avoid that:
      x -> base.features -> relu -> GAP -> head

    Expected input: (B,1,H,W) float32, already xrv-normalized in dataset.

    Note:
    - XRV pretrained weight here is res224; it can still accept other sizes
      (conv + GAP), but distribution differs from training resolution.
    """
    def __init__(self, base: nn.Module, num_classes: int, dropout: float):
        super().__init__()
        self.base = base
        self.features = base.features  # feature extractor only

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Robustly infer feature dim by running a dummy forward through features.
        # This avoids hard-coding (e.g., 1024) which can break if a model variant changes.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 224, 224)  # weights are res224-all
            feats = self.features(dummy)
            feats = torch.relu(feats)
            feats = self.pool(feats)
            in_f = int(feats.flatten(1).shape[1])

        self.head = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(in_f, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"XRV expects input (B,1,H,W) but got {tuple(x.shape)}")

        feats = self.features(x)
        feats = torch.relu(feats)
        feats = self.pool(feats)
        feats = torch.flatten(feats, 1)
        return self.head(feats)


# =====================================================
# Builders
# =====================================================
def _build_torchvision_densenet121(
    num_classes: int,
    pretrained: bool,
    dropout: float
) -> tuple[nn.Module, ModelIO]:
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    net = models.densenet121(weights=weights)

    in_f = int(net.classifier.in_features)
    net.classifier = nn.Sequential(
        nn.Dropout(float(dropout)),
        nn.Linear(in_f, int(num_classes)),
    )
    return net, ModelIO(expected_in_channels=3)


def _build_xrv_densenet121_all(
    num_classes: int,
    pretrained: bool,
    dropout: float
) -> tuple[nn.Module, ModelIO]:
    if not XRV_AVAILABLE:
        raise ValueError("torchxrayvision not installed. pip install torchxrayvision")

    weights_name = MODEL_ALIASES["xrv_densenet121_res224_all"]
    if pretrained:
        try:
            base = xrv.models.get_model(weights_name)
        except TypeError:
            base = xrv.models.DenseNet(weights=weights_name)
    else:
        base = xrv.models.DenseNet(weights=None)

    net = XrvDenseNetFeatures(base, num_classes=int(num_classes), dropout=float(dropout))
    return net, ModelIO(expected_in_channels=1)


# =====================================================
# Main model
# =====================================================
class CheXpertModel(nn.Module):
    """
    Supported model_name:
      - "densenet121"                (torchvision, expects 3ch, any HxW ok)
      - "xrv_densenet121_res224_all" (torchxrayvision, expects 1ch, pretrained on ~224)
    """
    def __init__(
        self,
        model_name: str = "densenet121",
        num_classes: int = 14,
        pretrained: bool = True,
        dropout: float = 0.2,
        img_size: int = 384,  # kept for logging/consistency (model accepts variable size)
    ):
        super().__init__()
        self.model_name = str(model_name)
        self.model_key = MODEL_ALIASES.get(self.model_name, self.model_name)

        self.num_classes = int(num_classes)
        self.img_size = int(img_size)

        if self.model_name == "densenet121":
            self.backbone, io = _build_torchvision_densenet121(self.num_classes, pretrained, dropout)

        elif self.model_name == "xrv_densenet121_res224_all":
            self.backbone, io = _build_xrv_densenet121_all(self.num_classes, pretrained, dropout)

        else:
            raise ValueError(
                f"Unsupported model_name: {self.model_name}. "
                f"Allowed: {list(MODEL_ALIASES.keys())}"
            )

        self.expected_in_channels = int(io.expected_in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.expected_in_channels:
            raise ValueError(
                f"{self.model_name} expects input (B,{self.expected_in_channels},H,W) "
                f"but got {tuple(x.shape)}. Check dataset preprocess mode."
            )
        return self.backbone(x)

    def freeze_features(self) -> None:
        """Freeze everything except task head/classifier."""
        for name, p in self.backbone.named_parameters():
            p.requires_grad = _is_head_param(name)

    def unfreeze_all(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True