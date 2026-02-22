# =====================================================
# CheXpert Dataset & DataModule
# =====================================================
#
# Uncertainty Policy (Custom) — ตาม CheXpert paper + EDA:
# ----------------------------
# | Disease          | Policy   | -1 →  | เหตุผล                          |
# |------------------|----------|-------|---------------------------------|
# | Atelectasis      | U-Ones   |  1    | uncertain ≈ positive (33k each) |
# | Cardiomegaly     | U-Zeros  |  0    | positive มีพอแล้ว (27k)          |
# | Consolidation    | U-Ignore | -1    | uncertain > positive 2x         |
# | Edema            | U-Ones   |  1    | paper แนะนำ                      |
# | Pleural Effusion | U-Ones   |  1    | positive เยอะ แต่ uncertain ≈ 12k|
#
# ⚠️  fillna และ replace -1 ต้องทำแยกต่อ column เพื่อไม่ให้ Consolidation
#     ถูก fillna(0) ก่อนแล้ว -1 หายไป
# =====================================================

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.constants import TARGET_COLS, IMAGENET_MEAN, IMAGENET_STD


# =====================================================
# CLAHE helper (เพิ่ม contrast ให้ X-ray)
# =====================================================
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """รับ grayscale uint8 คืน grayscale uint8 ที่ contrast ดีขึ้น"""
    return _CLAHE.apply(gray)


# =====================================================
# Label helper
# =====================================================

def apply_uncertainty_policy(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    """
    จัดการ -1 (uncertain) และ NaN แยกต่อ column
    ลำดับสำคัญ: ต้อง replace -1 ก่อน fillna เสมอ
    """
    df = df.copy()

    if policy == "custom":
        # --- Atelectasis: U-Ones (-1 → 1), NaN → 0
        df["Atelectasis"] = df["Atelectasis"].replace(-1, 1).fillna(0)

        # --- Cardiomegaly: U-Zeros (-1 → 0), NaN → 0
        df["Cardiomegaly"] = df["Cardiomegaly"].replace(-1, 0).fillna(0)

        # --- Consolidation: U-Ignore (คง -1 ไว้ → masked loss), NaN → 0
        #     ⚠️ ต้องไม่ replace -1 !
        df["Consolidation"] = df["Consolidation"].fillna(0)
        # -1 ยังคงอยู่ → _masked_bce จะ mask ออก

        # --- Edema: U-Ones (-1 → 1), NaN → 0
        df["Edema"] = df["Edema"].replace(-1, 1).fillna(0)

        # --- Pleural Effusion: U-Ones (-1 → 1), NaN → 0
        df["Pleural Effusion"] = df["Pleural Effusion"].replace(-1, 1).fillna(0)

    elif policy == "u-zeros":
        for col in TARGET_COLS:
            df[col] = df[col].replace(-1, 0).fillna(0)

    elif policy == "u-ones":
        for col in TARGET_COLS:
            df[col] = df[col].replace(-1, 1).fillna(0)

    else:
        raise ValueError(f"Unknown policy: {policy!r}. Choose: custom | u-zeros | u-ones")

    return df


# =====================================================
# Dataset
# =====================================================

class CheXpertDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, img_size=384, use_clahe=True):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.use_clahe = use_clahe

        self.paths = self.df["Path"].values
        self.labels = self.df[TARGET_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.paths[idx]
        img_path = os.path.join(self.root_dir, rel_path)

        # --- Load grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # fallback: blank image (กัน crash ถ้าไฟล์หาย)
            image = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        # --- CLAHE (เพิ่ม local contrast ก่อน augment)
        if self.use_clahe:
            image = apply_clahe(image)

        # --- 1ch → 3ch (DenseNet/pretrained ต้องการ 3ch)
        image = np.stack([image] * 3, axis=-1)  # (H, W, 3)

        # --- Augmentation + Normalize
        if self.transform:
            image = self.transform(image=image)["image"]

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


# =====================================================
# DataModule
# =====================================================

class CheXpertDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        csv_path: str,
        img_size: int = 384,
        batch_size: int = 32,
        num_workers: int = 4,
        policy: str = "custom",
        seed: int = 42,
        frontal_only: bool = True,
        use_clahe: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.policy = policy
        self.seed = seed
        self.frontal_only = frontal_only
        self.use_clahe = use_clahe

    def setup(self, stage=None):
        # =====================================================
        # TRAIN: ใช้ train.csv ทั้งหมด (ไม่ split)
        #   → เพิ่มข้อมูล train จาก ~172k → ~191k images
        # VAL:   ใช้ valid.csv (radiologist-annotated ground truth)
        #   → label สะอาด, ตรงกับ leaderboard, น่าเชื่อถือกว่า auto-labeler
        # =====================================================

        # --- Train CSV
        train_df = pd.read_csv(self.csv_path)

        # --- Val CSV (อยู่ folder เดียวกับ train.csv)
        val_csv_path = self.csv_path.replace("train.csv", "valid.csv")
        val_df = pd.read_csv(val_csv_path)

        # --- Frontal only สำหรับ train (Lateral confuse model)
        if self.frontal_only and "Frontal/Lateral" in train_df.columns:
            before = len(train_df)
            train_df = train_df[train_df["Frontal/Lateral"] == "Frontal"].copy()
            print(f"📷 Train Frontal only: {before:,} → {len(train_df):,} images")

        # --- Apply uncertainty policy เฉพาะ train
        # valid.csv ไม่มี -1 (radiologist annotate มาสะอาดแล้ว)
        train_df = apply_uncertainty_policy(train_df, self.policy)

        # --- valid.csv: fillna(0) เฉยๆ ไม่ต้อง policy
        for col in TARGET_COLS:
            val_df[col] = val_df[col].fillna(0)

        print(f"📊 Train: {len(train_df):,} | Val: {len(val_df):,} (radiologist-annotated)")

        # --- Transforms
        train_tf = A.Compose([
            # รักษา aspect ratio (สำคัญสำหรับ X-ray)
            A.LongestMaxSize(max_size=self.img_size),
            A.PadIfNeeded(
                min_height=self.img_size,
                min_width=self.img_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
            ),
            # Geometric augmentation
            A.HorizontalFlip(p=0.5),
            # Albumentations 2.0.x: Affine ใช้ fill แทน cval, ไม่ใช้ mode
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.90, 1.10),
                rotate=(-10, 10),
                fill=0,             # แทน cval (API ใหม่)
                p=0.5,
            ),
            # Intensity augmentation (simulate different exposure)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3,
            ),
            # Noise / blur เล็กน้อย (กัน overfit texture)
            A.GaussianBlur(blur_limit=3, p=0.1),
            A.GaussNoise(p=0.1),
            # CoarseDropout — บังคับโมเดลไม่ให้จำ support devices (สาย/อุปกรณ์)
            A.CoarseDropout(
                num_holes_range=(2, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=0,
                p=0.3,
            ),
            # Normalize ด้วย ImageNet stats (ใช้กับ pretrained model)
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

        val_tf = A.Compose([
            A.LongestMaxSize(max_size=self.img_size),
            A.PadIfNeeded(
                min_height=self.img_size,
                min_width=self.img_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

        self.train_ds = CheXpertDataset(
            train_df, self.data_dir, train_tf,
            img_size=self.img_size, use_clahe=self.use_clahe,
        )
        self.val_ds = CheXpertDataset(
            val_df, self.data_dir, val_tf,
            img_size=self.img_size, use_clahe=self.use_clahe,
        )

        self._print_label_stats(train_df, "Train")
        self._print_label_stats(val_df, "Val (radiologist)")

    def _print_label_stats(self, df: pd.DataFrame, split: str):
        """แสดง label distribution หลัง apply policy"""
        print(f"\n📈 {split} label distribution (after policy):")
        print(f"  {'Disease':<20} {'Pos':>7} {'Neg':>7} {'Ign(-1)':>8} {'Pos%':>7}")
        print("  " + "-" * 55)
        for col in TARGET_COLS:
            pos  = (df[col] == 1).sum()
            neg  = (df[col] == 0).sum()
            ign  = (df[col] == -1).sum()
            total_valid = pos + neg
            pct  = pos / total_valid * 100 if total_valid > 0 else 0
            print(f"  {col:<20} {pos:>7,} {neg:>7,} {ign:>8,} {pct:>6.1f}%")
        print()

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if self.num_workers > 0 else None,
        )