import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

TARGET_COLS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


class CheXpertDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

        self.paths = self.df["Path"].values
        self.labels = self.df[TARGET_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.paths[idx]
        img_path = os.path.join(self.root_dir, rel_path)

        # Read grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # กัน training crash ถ้า path ผิด/ไฟล์หาย
            image = np.zeros((320, 320), dtype=np.uint8)

        # Convert 1ch -> 3ch (สำหรับ ImageNet pretrained)
        image = np.stack([image] * 3, axis=-1)

        if self.transform:
            image = self.transform(image=image)["image"]

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


class CheXpertDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        csv_path,
        img_size=320,
        batch_size=32,
        num_workers=4,
        policy="u-zeros",  # u-zeros | u-ones | custom
        seed=42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.policy = policy
        self.seed = seed

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)

        # Fill NaN with 0
        df[TARGET_COLS] = df[TARGET_COLS].fillna(0)

        # ----------------------------
        # Uncertainty policy (-1)
        # ----------------------------
        if self.policy == "custom":
            # Recommended hybrid:
            # - Atelectasis, Cardiomegaly: U-Zeros (-1 -> 0)
            # - Consolidation: U-Ignore (keep -1 for masking in loss/metric)
            # - Edema, Pleural Effusion: U-Ones (-1 -> 1)
            u_zeros_cols = ["Atelectasis", "Cardiomegaly"]
            u_ones_cols = ["Edema", "Pleural Effusion"]

            for col in u_zeros_cols:
                df[col] = df[col].replace(-1, 0)
            for col in u_ones_cols:
                df[col] = df[col].replace(-1, 1)
            # Consolidation stays -1 (ignored by mask)

        elif self.policy == "u-zeros":
            df[TARGET_COLS] = df[TARGET_COLS].replace(-1, 0)

        elif self.policy == "u-ones":
            df[TARGET_COLS] = df[TARGET_COLS].replace(-1, 1)

        else:
            raise ValueError(f"Unknown policy: {self.policy}")

        # ----------------------------
        # Patient-level split (สำคัญสุด)
        # Path format: CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
        # patient id = patient00001
        # ----------------------------
        df["Patient"] = df["Path"].apply(lambda x: x.split("/")[2])

        patients = df["Patient"].unique()
        train_p, val_p = train_test_split(
            patients, test_size=0.1, random_state=self.seed
        )

        train_df = df[df["Patient"].isin(train_p)].copy()
        val_df = df[df["Patient"].isin(val_p)].copy()

        # ----------------------------
        # Transforms (preserve aspect ratio)
        # ----------------------------
        train_tf = A.Compose(
            [
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(
                    min_height=self.img_size,
                    min_width=self.img_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=5, p=0.3),
                A.Normalize(mean=[0.485] * 3, std=[0.229] * 3),
                ToTensorV2(),
            ]
        )

        val_tf = A.Compose(
            [
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(
                    min_height=self.img_size,
                    min_width=self.img_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.Normalize(mean=[0.485] * 3, std=[0.229] * 3),
                ToTensorV2(),
            ]
        )

        self.train_ds = CheXpertDataset(train_df, self.data_dir, train_tf)
        self.val_ds = CheXpertDataset(val_df, self.data_dir, val_tf)

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