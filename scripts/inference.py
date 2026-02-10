import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import csv
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# Utils: IO
# =====================================================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def safe_run_name(ckpt_path: str, override: str | None = None) -> str:
    if override:
        return override
    # default: use ckpt filename without extension
    name = os.path.splitext(os.path.basename(ckpt_path))[0]
    return name.replace("=", "_").replace(".", "_")


# =====================================================
# Load model
# =====================================================

def load_lightning(ckpt_path: str) -> CheXpertLightning:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = CheXpertLightning.load_from_checkpoint(ckpt_path)
    model.to(DEVICE)
    model.eval()
    return model


# =====================================================
# Preprocess
# =====================================================

def preprocess_image(img_path: str, img_size: int):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = np.stack([img] * 3, axis=-1)

    resize_tf = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
        ),
    ])
    img = resize_tf(image=img)["image"]

    img_float = np.float32(img) / 255.0  # for overlay (0..1)

    tensor_tf = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
    tensor = tensor_tf(image=img)["image"].unsqueeze(0).to(DEVICE)

    return img_float, tensor


# =====================================================
# Predict
# =====================================================

@torch.no_grad()
def predict_probs(lightning: CheXpertLightning, tensor: torch.Tensor) -> np.ndarray:
    logits = lightning(tensor)
    probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
    return probs


@torch.no_grad()
def predict_probs_tta(lightning: CheXpertLightning, tensor: torch.Tensor) -> np.ndarray:
    # minimal TTA: orig + horizontal flip
    probs1 = torch.sigmoid(lightning(tensor))
    tensor_flip = torch.flip(tensor, dims=[3])
    probs2 = torch.sigmoid(lightning(tensor_flip))
    probs = (probs1 + probs2) / 2.0
    return probs.squeeze(0).detach().cpu().numpy()


def print_probs(title: str, probs: np.ndarray):
    print(f"\n📊 {title}")
    for i, name in enumerate(CLASS_NAMES):
        bar = "█" * int(float(probs[i]) * 20)
        print(f"{name:<20}: {float(probs[i]):.4f} {bar}")


# =====================================================
# Grad-CAM target layer resolver (robust)
# =====================================================

def get_backbone(lightning: CheXpertLightning):
    # CheXpertLightning.model is CheXpertModel wrapper
    # CheXpertModel.backbone is the actual model
    return lightning.model.backbone


def resolve_target_layer(lightning: CheXpertLightning):
    backbone = get_backbone(lightning)
    clsname = backbone.__class__.__name__.lower()

    # ---- torchvision DenseNet ----
    if "densenet" in clsname and hasattr(backbone, "features"):
        # best practical target for DenseNet CAM
        if hasattr(backbone.features, "denseblock4"):
            return backbone.features.denseblock4
        return backbone.features[-1]

    # ---- torchvision ConvNeXt ----
    if "convnext" in clsname and hasattr(backbone, "features"):
        # [-2] often sharper than [-1]
        if isinstance(backbone.features, torch.nn.Sequential) and len(backbone.features) >= 2:
            return backbone.features[-2]
        return backbone.features[-1]

    # ---- torchvision EfficientNet ----
    if "efficientnet" in clsname and hasattr(backbone, "features"):
        return backbone.features[-1]

    # ---- timm / transformers ----
    # Swin (timm): layers
    if hasattr(backbone, "layers"):
        return backbone.layers[-1]
    # ConvNeXtV2 (timm): stages
    if hasattr(backbone, "stages"):
        return backbone.stages[-1]
    # Many timm CNNs: blocks
    if hasattr(backbone, "blocks"):
        return backbone.blocks[-1]
    # Generic fallback
    if hasattr(backbone, "features"):
        try:
            return backbone.features[-1]
        except Exception:
            pass

    raise ValueError("Cannot find a Grad-CAM target layer for this backbone.")


def run_gradcam(lightning: CheXpertLightning, tensor: torch.Tensor, rgb_img_float: np.ndarray, class_idx: int):
    target_layer = resolve_target_layer(lightning)
    targets = [ClassifierOutputTarget(class_idx)]

    # Pass lightning (works), but hooks target layer on backbone.
    cam = GradCAM(model=lightning, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    overlay = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
    return overlay, grayscale_cam


# =====================================================
# Save outputs
# =====================================================

def save_cam_images(run_dir: str, img_stem: str, cams: dict, probs: np.ndarray):
    out_dir = os.path.join(run_dir, "images", img_stem)
    ensure_dir(out_dir)

    for cls_name, cam_img in cams.items():
        save_path = os.path.join(out_dir, f"cam_{cls_name.replace(' ', '_')}.png")
        plt.figure(figsize=(6, 6))
        plt.imshow(cam_img)
        plt.title(f"{cls_name} | prob={float(probs[CLASS_NAMES.index(cls_name)]):.3f}")
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"✅ Saved: {save_path}")

    return out_dir


def save_overlay_summary(run_dir: str, img_stem: str, rgb_img_float: np.ndarray, cams: dict, probs: np.ndarray):
    out_dir = os.path.join(run_dir, "images", img_stem)
    ensure_dir(out_dir)

    # Make 1 portfolio-ready summary image: original + top2 CAMs
    topk = np.argsort(probs)[::-1][:2]
    top_names = [CLASS_NAMES[i] for i in topk]

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(rgb_img_float)
    ax0.set_title("Original")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(cams[top_names[0]])
    ax1.set_title(f"Grad-CAM: {top_names[0]} ({probs[topk[0]]:.3f})")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(cams[top_names[1]])
    ax2.set_title(f"Grad-CAM: {top_names[1]} ({probs[topk[1]]:.3f})")
    ax2.axis("off")

    save_path = os.path.join(out_dir, f"overlay_top2.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"✅ Saved: {save_path}")


def append_predictions_csv(csv_path: str, row: dict, header: list[str]):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def save_summary_json(run_dir: str, summary: dict):
    path = os.path.join(run_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {path}")


# =====================================================
# MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt")
    parser.add_argument("--img", required=True, help="Path to X-ray image")
    parser.add_argument("--img_size", type=int, default=384, help="Resize/pad size (must match training)")
    parser.add_argument("--mode", default="topk", choices=["topk", "single"])
    parser.add_argument("--class_name", default="Consolidation")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--save_dir", default="runs_infer", help="Root folder for all inference runs")
    parser.add_argument("--run_name", default=None, help="Override run folder name")
    args = parser.parse_args()

    # run folder
    run_name = safe_run_name(args.ckpt, args.run_name)
    run_dir = os.path.join(args.save_dir, run_name)
    ensure_dir(run_dir)

    # load + preprocess
    model = load_lightning(args.ckpt)
    rgb_img, tensor = preprocess_image(args.img, args.img_size)

    # predict
    probs = predict_probs_tta(model, tensor) if args.tta else predict_probs(model, tensor)
    print_probs("Prediction Results" + (" (TTA)" if args.tta else ""), probs)

    # grad-cam selection
    cams = {}
    if args.mode == "single":
        if args.class_name not in CLASS_NAMES:
            raise ValueError(f"Unknown class: {args.class_name}")
        idx = CLASS_NAMES.index(args.class_name)
        cam_img, _ = run_gradcam(model, tensor, rgb_img, idx)
        cams[args.class_name] = cam_img
    else:
        topk = np.argsort(probs)[::-1][:2]
        for idx in topk:
            cls_name = CLASS_NAMES[idx]
            cam_img, _ = run_gradcam(model, tensor, rgb_img, idx)
            cams[cls_name] = cam_img

    img_stem = os.path.splitext(os.path.basename(args.img))[0]

    # save images
    save_cam_images(run_dir, img_stem, cams, probs)
    save_overlay_summary(run_dir, img_stem, rgb_img, cams, probs)

    # write prediction row
    pred_csv = os.path.join(run_dir, "predictions.csv")
    header = ["image_path", "img_size", "tta", "ckpt_path"] + [f"{c}_prob" for c in CLASS_NAMES]
    row = {
        "image_path": args.img,
        "img_size": args.img_size,
        "tta": bool(args.tta),
        "ckpt_path": args.ckpt,
    }
    for i, c in enumerate(CLASS_NAMES):
        row[f"{c}_prob"] = float(probs[i])
    append_predictions_csv(pred_csv, row, header)

    # summary json (metadata)
    summary = {
        "run_name": run_name,
        "ckpt_path": args.ckpt,
        "image_path": args.img,
        "img_size": args.img_size,
        "tta": bool(args.tta),
        "device": DEVICE,
        "classes": CLASS_NAMES,
    }
    save_summary_json(run_dir, summary)

    print(f"\n✅ Done. Outputs saved in: {run_dir}")


if __name__ == "__main__":
    main()