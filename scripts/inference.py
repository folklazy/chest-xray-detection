import os
import cv2
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.model import CheXpertLightning
from src.constants import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD

IMG_SIZE = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = CheXpertLightning.load_from_checkpoint(ckpt_path)
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(img_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = np.stack([img] * 3, axis=-1)

    resize_tf = A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(
            min_height=IMG_SIZE,
            min_width=IMG_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
        ),
    ])
    img = resize_tf(image=img)["image"]

    img_float = np.float32(img) / 255.0

    tensor_tf = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
    tensor = tensor_tf(image=img)["image"].unsqueeze(0).to(DEVICE)

    return img_float, tensor


def predict(model, tensor) -> np.ndarray:
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
    return probs


def predict_tta(model, tensor) -> np.ndarray:
    with torch.no_grad():
        probs1 = torch.sigmoid(model(tensor))
        tensor_flip = torch.flip(tensor, dims=[3])
        probs2 = torch.sigmoid(model(tensor_flip))
        probs = (probs1 + probs2) / 2.0
    return probs.squeeze(0).detach().cpu().numpy()


def print_probs(title: str, probs: np.ndarray):
    print(f"\n📊 {title}")
    for i, name in enumerate(CLASS_NAMES):
        bar = "█" * int(float(probs[i]) * 20)
        print(f"{name:<20}: {float(probs[i]):.4f} {bar}")


# -----------------------------
# Grad-CAM target layer resolver
# -----------------------------
def get_gradcam_target_layer(lightning_model: CheXpertLightning):
    """
    Returns a reasonable last conv-like layer for Grad-CAM for both:
    - torchvision backbones (DenseNet/ConvNeXt/EfficientNet)
    - timm backbones (EfficientNet-B3, Swin, etc.)
    """
    backbone = lightning_model.model.backbone  # CheXpertModel.backbone
    name = getattr(lightning_model.model, "model_name", None)  # might not exist
    # better: check backbone class name
    clsname = backbone.__class__.__name__.lower()

    # torchvision DenseNet
    if "densenet" in clsname and hasattr(backbone, "features"):
        return backbone.features[-1]

    # torchvision ConvNeXt
    if "convnext" in clsname and hasattr(backbone, "features"):
        # [-2] มักคมกว่า [-1]
        return backbone.features[-2]

    # torchvision EfficientNet
    if "efficientnet" in clsname and hasattr(backbone, "features"):
        return backbone.features[-1]

    # timm models:
    # - many CNNs have backbone.conv_head / backbone.blocks / backbone.stages
    # - Swin has backbone.layers
    if hasattr(backbone, "layers"):  # Swin Transformer
        return backbone.layers[-1]

    if hasattr(backbone, "stages"):  # ConvNeXtV2 / some timm convnets
        return backbone.stages[-1]

    if hasattr(backbone, "blocks"):  # EfficientNet timm, etc.
        return backbone.blocks[-1]

    if hasattr(backbone, "features"):  # generic timm fallback
        try:
            return backbone.features[-1]
        except Exception:
            pass

    raise ValueError("Could not automatically find a Grad-CAM target layer for this backbone.")


def run_gradcam(model, tensor, rgb_img_float, class_idx: int):
    target_layer = get_gradcam_target_layer(model)
    targets = [ClassifierOutputTarget(class_idx)]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
    return visualization


def visualize_and_save(cams, probs, output_dir: str, img_name: str):
    os.makedirs(output_dir, exist_ok=True)
    for name, cam_img in cams.items():
        plt.figure(figsize=(6, 6))
        plt.imshow(cam_img)
        plt.title(f"{name} | prob={float(probs[CLASS_NAMES.index(name)]):.3f}")
        plt.axis("off")
        save_path = os.path.join(output_dir, f"{img_name}_cam_{name.replace(' ', '_')}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"✅ Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt")
    parser.add_argument("--img", required=True, help="Path to X-ray image")
    parser.add_argument("--mode", default="topk", choices=["topk", "single"])
    parser.add_argument("--class_name", default="Consolidation")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    model = load_model(args.ckpt)
    rgb_img, tensor = preprocess_image(args.img)

    probs = predict_tta(model, tensor) if args.tta else predict(model, tensor)
    print_probs("Prediction Results" + (" (TTA)" if args.tta else ""), probs)

    cams = {}
    if args.mode == "single":
        if args.class_name not in CLASS_NAMES:
            raise ValueError(f"Unknown class: {args.class_name}")
        idx = CLASS_NAMES.index(args.class_name)
        cams[args.class_name] = run_gradcam(model, tensor, rgb_img, idx)
    else:
        topk = np.argsort(probs)[::-1][:2]
        for idx in topk:
            cams[CLASS_NAMES[idx]] = run_gradcam(model, tensor, rgb_img, idx)

    img_name = os.path.splitext(os.path.basename(args.img))[0]
    visualize_and_save(cams, probs, args.output, img_name)


if __name__ == "__main__":
    main()