# src/dataset.py
from __future__ import annotations

import os
import re
from typing import Optional, Union, Mapping

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.constants import (
    TARGET_COLS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PREPROCESS_IMAGENET,
    PREPROCESS_XRV,
    DEFAULT_SEED,
    DEFAULT_POLICY_NAME,
    POLICIES,
    POLICY_DEFAULT_ACTION,
    PolicyAction,
)

# optional
try:
    import torchxrayvision as xrv
    XRV_AVAILABLE = True
except ImportError:
    XRV_AVAILABLE = False

PolicySpec = Union[str, Mapping[str, Union[str, PolicyAction]], None]


# =====================================================
# Helpers
# =====================================================
def _ensure_targets(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in target_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[target_cols] = df[target_cols].fillna(0)
    return df


def _extract_patient_id(path_value: str) -> str:
    s = str(path_value)
    m = re.search(r"(patient\d+)", s)
    if m:
        return m.group(1)
    return "unknown"


def _normalize_action(x: Union[str, PolicyAction]) -> PolicyAction:
    a = str(x).strip().lower()
    if a in {"ignore"}:
        return "ignore"
    if a in {"zeros", "zero"}:
        return "zeros"
    if a in {"ones", "one"}:
        return "ones"
    raise ValueError(f"Invalid policy action: {x}")


def resolve_policy(policy: PolicySpec) -> dict[str, PolicyAction]:
    if policy is None:
        policy = DEFAULT_POLICY_NAME
    if isinstance(policy, Mapping):
        return {k: _normalize_action(v) for k, v in policy.items()}
    return {k: _normalize_action(v) for k, v in POLICIES[policy].items()}


def apply_uncertainty_policy(df: pd.DataFrame, policy: PolicySpec) -> pd.DataFrame:
    df = df.copy()
    p = resolve_policy(policy)

    for c in TARGET_COLS:
        if c not in df.columns:
            continue

        action = p.get(c, POLICY_DEFAULT_ACTION)
        if action == "ignore":
            continue
        if action == "zeros":
            df[c] = df[c].replace(-1, 0)
        if action == "ones":
            df[c] = df[c].replace(-1, 1)

    return df


# =====================================================
# Dataset
# =====================================================
class CheXpertDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str,
        transform: Optional[A.Compose] = None,
        img_size: int = 384,
        preprocess: str = PREPROCESS_IMAGENET,
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = int(img_size)
        self.preprocess = preprocess

        self.paths = self.df["Path"].astype(str).values
        self.labels = self.df[TARGET_COLS].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.df)

    # ✅ CLEAN PATH FIX (fast + robust)
    def _build_path(self, rel_path: str) -> str:
        p = str(rel_path).strip()

        # normalize slashes
        p = p.replace("\\", "/")

        # strip leading "./"
        if p.startswith("./"):
            p = p[2:]

        # strip official prefixes that appear in CSV
        for prefix in ("CheXpert-v1.0/", "CheXpert-v1.0-small/"):
            if p.startswith(prefix):
                p = p[len(prefix):]
                break

        # join with root_dir (root_dir = ./data/CheXpert_full)
        return os.path.join(self.root_dir, p.replace("/", os.sep))

    def _read_grayscale(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        return img

    def __getitem__(self, idx):
        abs_path = self._build_path(self.paths[idx])
        img = self._read_grayscale(abs_path)

        if self.preprocess == PREPROCESS_IMAGENET:
            img = np.stack([img] * 3, axis=-1)  # (H,W,3)
            img = self.transform(image=img)["image"]

        elif self.preprocess == PREPROCESS_XRV:
            if not XRV_AVAILABLE:
                raise ValueError("preprocess='xrv' requires torchxrayvision")
            img = img.astype(np.float32)
            img = xrv.datasets.normalize(img, 255)
            img = np.expand_dims(img, axis=0)  # (1,H,W)
            img = self.transform(image=img)["image"]

        else:
            raise ValueError(f"Unknown preprocess: {self.preprocess}")

        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, y


# =====================================================
# DataModule
# =====================================================
class CheXpertDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        csv_path: str,
        img_size: int = 384,
        batch_size: int = 16,
        num_workers: int = 4,
        policy: PolicySpec = DEFAULT_POLICY_NAME,
        seed: int = DEFAULT_SEED,
        frontal_only: bool = True,
        preprocess: str = PREPROCESS_IMAGENET,
        # 🚀 speed knobs
        prefetch_factor: int = 8,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.img_size = int(img_size)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.policy = policy
        self.seed = int(seed)
        self.frontal_only = bool(frontal_only)
        self.preprocess = preprocess

        self.prefetch_factor = int(prefetch_factor)
        self.persistent_workers = bool(persistent_workers)

        self.train_ds: Optional[CheXpertDataset] = None
        self.val_ds: Optional[CheXpertDataset] = None

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)

        if self.frontal_only and "Frontal/Lateral" in df.columns:
            df = df[df["Frontal/Lateral"] == "Frontal"].copy()

        df = _ensure_targets(df, TARGET_COLS)
        df = apply_uncertainty_policy(df, self.policy)

        df["Patient"] = df["Path"].apply(_extract_patient_id)
        train_p, val_p = train_test_split(
            df["Patient"].unique(),
            test_size=0.1,
            random_state=self.seed,
            shuffle=True,
        )

        train_df = df[df["Patient"].isin(train_p)].copy()
        val_df = df[df["Patient"].isin(val_p)].copy()

        # 🚀 Faster than LongestMaxSize+Pad (big win)
        train_tf = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=5, p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ToTensorV2(),
        ])

        val_tf = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ToTensorV2(),
        ])

        self.train_ds = CheXpertDataset(train_df, self.data_dir, train_tf, self.img_size, self.preprocess)
        self.val_ds = CheXpertDataset(val_df, self.data_dir, val_tf, self.img_size, self.preprocess)

        print(f"Train={len(self.train_ds)} | Val={len(self.val_ds)}")

    def train_dataloader(self):
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.persistent_workers and self.num_workers > 0),
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.persistent_workers and self.num_workers > 0),
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )