import torch
import torch.nn as nn
from torchvision import models

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


MODEL_ALIASES = {
    # torchvision
    "densenet121": "densenet121",
    "efficientnet_b0": "efficientnet_b0",
    "convnext_tiny": "convnext_tiny",

    # timm
    "efficientnet_b3": "efficientnet_b3",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "swin_t": "swin_tiny_patch4_window7_224",
}


class CheXpertModel(nn.Module):
    def __init__(
        self,
        model_name="densenet121",
        num_classes=5,
        pretrained=True,
        dropout=0.3,
    ):
        super().__init__()

        model_key = MODEL_ALIASES.get(model_name, model_name)
        self.model_name = model_name
        self.model_key = model_key

        if model_key == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            backbone = models.densenet121(weights=weights)
            in_f = backbone.classifier.in_features
            backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_f, num_classes),
            )

        elif model_key == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            in_f = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_f, num_classes),
            )

        elif model_key == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            backbone = models.convnext_tiny(weights=weights)
            in_f = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_f, num_classes),
            )

        else:
            if not TIMM_AVAILABLE:
                raise ValueError(
                    f"Unsupported model: {model_name} (resolved: {model_key}). "
                    f"Install timm: pip install timm"
                )

            backbone = timm.create_model(
                model_key,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout,
            )

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def freeze_features(self):
        for name, p in self.backbone.named_parameters():
            if ("classifier" not in name) and ("head" not in name) and ("fc" not in name):
                p.requires_grad = False

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True