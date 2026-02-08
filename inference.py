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

# =====================================================
# CONFIG
# =====================================================

CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# ⚠️ ต้องให้ตรงกับตอน train (คุณเทรนที่ 384 แล้ว)
IMG_SIZE = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# Utils
# =====================================================

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

    # grayscale -> 3ch (เหมือนตอน train)
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

    # สำหรับโชว์/Grad-CAM overlay (0..1)
    img_float = np.float32(img) / 255.0

    # สำหรับเข้า model (normalize เหมือน train)
    tensor_tf = A.Compose([
        A.Normalize(mean=[0.485] * 3, std=[0.229] * 3),
        ToTensorV2()
    ])
    tensor = tensor_tf(image=img)["image"].unsqueeze(0).to(DEVICE)

    return img_float, tensor


def predict(model, tensor) -> np.ndarray:
    """Return probs shape (num_classes,) for a single image tensor [1,C,H,W]."""
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
    return probs


def predict_tta(model, tensor) -> np.ndarray:
    """
    Minimal TTA:
    - original
    - horizontal flip
    Average probs.
    """
    with torch.no_grad():
        logits1 = model(tensor)
        probs1 = torch.sigmoid(logits1)

        tensor_flip = torch.flip(tensor, dims=[3])  # flip W dimension
        logits2 = model(tensor_flip)
        probs2 = torch.sigmoid(logits2)

        probs = (probs1 + probs2) / 2.0
    return probs.squeeze(0).detach().cpu().numpy()


def print_probs(title: str, probs: np.ndarray):
    print(f"\n📊 {title}")
    for i, name in enumerate(CLASS_NAMES):
        bar = "█" * int(float(probs[i]) * 20)
        print(f"{name:<20}: {float(probs[i]):.4f} {bar}")


def run_gradcam(model, tensor, rgb_img_float, class_idx: int):
    """
    Grad-CAM for ConvNeXt-Tiny:
    - Use backbone.features[-2] (often sharper CAM than [-1])
    """
    backbone = model.model.backbone

    if hasattr(backbone, "features") and isinstance(backbone.features, torch.nn.Sequential):
        target_layers = [backbone.features[-2]]  # ✅ แนะนำสำหรับ tiny
    else:
        raise ValueError("Backbone is not ConvNeXt with .features Sequential. Check model.")

    targets = [ClassifierOutputTarget(class_idx)]
    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
    return visualization


def visualize_and_save(rgb_img, cams, probs, output_dir: str, img_name: str):
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

# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt")
    parser.add_argument("--img", required=True, help="Path to X-ray image")
    parser.add_argument(
        "--mode",
        default="topk",
        choices=["topk", "single"],
        help="topk = top-2 classes, single = specific class",
    )
    parser.add_argument(
        "--class_name",
        default="Consolidation",
        help="Used when mode=single",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Folder to save CAM images",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable TTA (orig + horizontal flip avg) for prediction (Grad-CAM still uses original image).",
    )
    args = parser.parse_args()

    model = load_model(args.ckpt)
    rgb_img, tensor = preprocess_image(args.img)

    probs_no_tta = predict(model, tensor)
    if args.tta:
        probs_tta = predict_tta(model, tensor)
        print_probs("Prediction Results (NO TTA)", probs_no_tta)
        print_probs("Prediction Results (TTA = orig + flip avg)", probs_tta)
        probs = probs_tta
    else:
        probs = probs_no_tta
        print_probs("Prediction Results", probs)

    # Grad-CAM: ทำบนภาพ original เท่านั้น (best practice)
    cams = {}
    if args.mode == "single":
        if args.class_name not in CLASS_NAMES:
            raise ValueError(f"Unknown class: {args.class_name}")
        idx = CLASS_NAMES.index(args.class_name)
        cams[args.class_name] = run_gradcam(model, tensor, rgb_img, idx)
    else:
        topk = np.argsort(probs)[::-1][:2]
        for idx in topk:
            name = CLASS_NAMES[idx]
            cams[name] = run_gradcam(model, tensor, rgb_img, idx)

    img_name = os.path.splitext(os.path.basename(args.img))[0]
    visualize_and_save(rgb_img, cams, probs, args.output, img_name)


if __name__ == "__main__":
    main()