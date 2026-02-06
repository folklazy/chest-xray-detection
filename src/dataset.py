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
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Pleural Effusion'
]


# =========================
# Dataset
# =========================
class CheXpertDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

        self.paths = self.df['Path'].values
        self.labels = self.df[TARGET_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.paths[idx]
        img_path = os.path.join(self.root_dir, rel_path)

        # grayscale (เร็ว + เบา)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = np.stack([image]*3, axis=-1)

        if self.transform:
            image = self.transform(image=image)["image"]

        label = torch.tensor(self.labels[idx])

        return image, label


# =========================
# DataModule
# =========================
class CheXpertDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        csv_path,
        img_size=320,
        batch_size=32,
        num_workers=4,
        policy="u-zeros"
    ):
        super().__init__()
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.policy = policy

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)

        # label policy
        df[TARGET_COLS] = df[TARGET_COLS].fillna(0)
        if self.policy == "u-zeros":
            df[TARGET_COLS] = df[TARGET_COLS].replace(-1, 0)
        else:
            df[TARGET_COLS] = df[TARGET_COLS].replace(-1, 1)

        # 🔥 Extract Patient ID จาก Path
        # Path format: "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"
        df['Patient'] = df['Path'].apply(lambda x: x.split('/')[2])
        
        # 🔥 patient split (สำคัญสุด)
        patients = df['Patient'].unique()

        train_p, val_p = train_test_split(
            patients, test_size=0.1, random_state=42
        )

        train_df = df[df['Patient'].isin(train_p)]
        val_df = df[df['Patient'].isin(val_p)]

        train_tf = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=5, p=0.3),
            A.Normalize(mean=[0.485]*3, std=[0.229]*3),
            ToTensorV2()
        ])

        val_tf = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485]*3, std=[0.229]*3),
            ToTensorV2()
        ])

        self.train_ds = CheXpertDataset(train_df, self.data_dir, train_tf)
        self.val_ds = CheXpertDataset(val_df, self.data_dir, val_tf)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
