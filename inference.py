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

IMG_SIZE = 320
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# Utils
# =====================================================

def load_model(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = CheXpertLightning.load_from_checkpoint(ckpt_path)
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(img_path):
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
            value=0,
        ),
    ])

    img = resize_tf(image=img)["image"]

    img_float = np.float32(img) / 255.0

    tensor_tf = A.Compose([
        A.Normalize(mean=[0.485]*3, std=[0.229]*3),
        ToTensorV2()
    ])

    tensor = tensor_tf(image=img)["image"].unsqueeze(0).to(DEVICE)

    return img_float, tensor


def predict(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return probs


def run_gradcam(model, tensor, rgb_img_float, class_idx):
    # DenseNet121: ใช้ denseblock4 จะคมและเสถียรสุด
    target_layers = [model.model.backbone.features.denseblock4]
    targets = [ClassifierOutputTarget(class_idx)]

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        use_cuda=(DEVICE == "cuda"),
    )

    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    visualization = show_cam_on_image(
        rgb_img_float, grayscale_cam, use_rgb=True
    )

    return visualization


def visualize_and_save(
    rgb_img,
    cams,
    probs,
    output_dir,
    img_name,
):
    os.makedirs(output_dir, exist_ok=True)

    for name, cam_img in cams.items():
        plt.figure(figsize=(6, 6))
        plt.imshow(cam_img)
        plt.title(f"{name} | prob={probs[CLASS_NAMES.index(name)]:.3f}")
        plt.axis("off")

        save_path = os.path.join(
            output_dir, f"{img_name}_cam_{name.replace(' ', '_')}.png"
        )
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
    args = parser.parse_args()

    model = load_model(args.ckpt)
    rgb_img, tensor = preprocess_image(args.img)
    probs = predict(model, tensor)

    print("\n📊 Prediction Results")
    for i, name in enumerate(CLASS_NAMES):
        bar = "█" * int(probs[i] * 20)
        print(f"{name:<20}: {probs[i]:.4f} {bar}")

    cams = {}

    if args.mode == "single":
        if args.class_name not in CLASS_NAMES:
            raise ValueError(f"Unknown class: {args.class_name}")
        idx = CLASS_NAMES.index(args.class_name)
        cams[args.class_name] = run_gradcam(
            model, tensor, rgb_img, idx
        )

    else:  # topk
        topk = np.argsort(probs)[::-1][:2]
        for idx in topk:
            name = CLASS_NAMES[idx]
            cams[name] = run_gradcam(
                model, tensor, rgb_img, idx
            )

    img_name = os.path.splitext(os.path.basename(args.img))[0]
    visualize_and_save(rgb_img, cams, probs, args.output, img_name)


if __name__ == "__main__":
    main()