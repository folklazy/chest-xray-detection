import torch
import torch.nn as nn
from torchvision import models

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


MODEL_ALIASES = {
    # torchvision (รองรับทั้ง - และ _)
    "densenet121": "densenet121",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet-b0": "efficientnet_b0",
    "convnext_tiny": "convnext_tiny",
    "convnext-tiny": "convnext_tiny",

    # timm
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet-b3": "efficientnet_b3",

    "swin_tiny": "swin_tiny_patch4_window7_224",
    "swin_t": "swin_tiny_patch4_window7_224",
}


def _try_set_swin_img_size(backbone, img_size: int):
    """
    timm Swin บางเวอร์ชันจะ assert ว่าขนาด input ต้องตรงกับ patch_embed.img_size
    วิธี robust:
    - ถ้าปรับได้: set patch_embed.img_size = (img_size, img_size)
    - บางโมเดลเก็บเป็น list/tuple/torch.Size ก็รองรับ
    """
    if hasattr(backbone, "patch_embed") and hasattr(backbone.patch_embed, "img_size"):
        backbone.patch_embed.img_size = (img_size, img_size)

    # บางเวอร์ชัน patch_embed มี _img_size
    if hasattr(backbone, "patch_embed") and hasattr(backbone.patch_embed, "_img_size"):
        backbone.patch_embed._img_size = (img_size, img_size)


class CheXpertModel(nn.Module):
    def __init__(
        self,
        model_name: str = "densenet121",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
        img_size: int = 384,
    ):
        super().__init__()

        model_key = MODEL_ALIASES.get(model_name, model_name)
        self.model_name = model_name
        self.model_key = model_key
        self.img_size = img_size

        # =====================
        # torchvision models
        # =====================
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

        # =====================
        # timm models
        # =====================
        else:
            if not TIMM_AVAILABLE:
                raise ValueError(
                    f"Unsupported model: {model_name} (resolved: {model_key}). "
                    f"Install timm: pip install timm"
                )

            kwargs = dict(
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout,
            )

            # ✅ พยายามส่ง img_size ให้ Swin (ถ้า timm รองรับ)
            if "swin" in model_key:
                kwargs["img_size"] = img_size

            try:
                backbone = timm.create_model(model_key, **kwargs)
            except TypeError:
                # timm บางเวอร์ชันไม่รับ img_size ตอนสร้าง
                kwargs.pop("img_size", None)
                backbone = timm.create_model(model_key, **kwargs)

            # ✅ fallback: ปรับ patch_embed.img_size หลังสร้าง กัน assert 224/384
            if "swin" in model_key:
                _try_set_swin_img_size(backbone, img_size)

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