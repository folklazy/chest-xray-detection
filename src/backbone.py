import torch
import torch.nn as nn
from torchvision import models


class CheXpertModel(nn.Module):
    def __init__(
        self,
        model_name="densenet121",
        num_classes=5,
        pretrained=True,
        dropout=0.3
    ):
        super().__init__()

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
            backbone.classifier[-1] = nn.Linear(in_f, num_classes)

        else:
            raise ValueError("Unsupported model")

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    # 🔥 optional helpers
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
