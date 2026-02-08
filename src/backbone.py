# =====================================================
# CheXpert Backbone Models
# =====================================================
#
# Supported Models:
# -----------------
# torchvision:
#   - densenet121
#   - efficientnet-b0
#   - convnext_tiny
#
# timm (if installed):
#   - Any timm model name (e.g., convnextv2_tiny, efficientnetv2_s)
#
# =====================================================

import torch
import torch.nn as nn
from torchvision import models

# Check if timm is available
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class CheXpertModel(nn.Module):
    def __init__(
        self,
        model_name="densenet121",
        num_classes=5,
        pretrained=True,
        dropout=0.3
    ):
        super().__init__()

        self.model_name = model_name

        # =====================
        # torchvision models
        # =====================
        if model_name == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            backbone = models.densenet121(weights=weights)

            in_f = backbone.classifier.in_features
            backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_f, num_classes)
            )

        elif model_name == "efficientnet-b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)

            in_f = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_f, num_classes)
            )

        elif model_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            backbone = models.convnext_tiny(weights=weights)

            in_f = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_f, num_classes)
            )

        # =====================
        # timm models (fallback)
        # =====================
        elif TIMM_AVAILABLE:
            # Use timm for any other model name
            backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout,
            )
        else:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Install timm for more options: pip install timm"
            )

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    # Freeze only feature extractor, keep head trainable
    def freeze_features(self):
        for name, p in self.backbone.named_parameters():
            if "classifier" not in name and "head" not in name and "fc" not in name:
                p.requires_grad = False

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True