# scripts/train_xrv_all_384.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.dataset import CheXpertDataModule
from src.model import CheXpertLightning
from src.constants import NUM_CLASSES, CLASS_NAMES, DEFAULT_SEED, DEFAULT_POLICY_NAME
from src.utils import compute_pos_weight_from_train_split

# -----------------
# Config
# -----------------
DATA_DIR = "./data"
CSV_PATH = "./data/CheXpert_full/train.csv"

MODEL_NAME = "xrv_densenet121_res224_all"
PREPROCESS = "xrv"

IMG_SIZE = 384
BATCH_SIZE = 20          # XRV 1ch มักเบากว่า เริ่มสูงกว่าได้
ACCUMULATE = 2           # effective batch ~ 40
NUM_WORKERS = 4

LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 16
FREEZE_EPOCHS = 1

POLICY = DEFAULT_POLICY_NAME  # "mixed_v1"


def main():
    pl.seed_everything(DEFAULT_SEED, workers=True)
    torch.set_float32_matmul_precision("medium")

    pos_weight, stats = compute_pos_weight_from_train_split(
        csv_path=CSV_PATH,
        target_cols=CLASS_NAMES,
        seed=DEFAULT_SEED,
        frontal_only=True,
        policy=POLICY,
        test_size=0.1,
        return_stats=True,
    )
    print("✅ POS_WEIGHT:", pos_weight)
    print("✅ POS stats (first 5):", {k: stats[k] for k in list(stats.keys())[:5]})

    dm = CheXpertDataModule(
        data_dir=DATA_DIR,
        csv_path=CSV_PATH,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        policy=POLICY,
        seed=DEFAULT_SEED,
        frontal_only=True,
        preprocess=PREPROCESS,
    )

    model = CheXpertLightning(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        pos_weight=pos_weight,
        scheduler_monitor="val_auc_comp",
        img_size=IMG_SIZE,
        freeze_epochs=FREEZE_EPOCHS,
        dropout=0.2,
    )

    exp_name = f"{MODEL_NAME}_img{IMG_SIZE}_bs{BATCH_SIZE}_acc{ACCUMULATE}_lr{LR}_e{EPOCHS}_fr{FREEZE_EPOCHS}_{POLICY}"
    logger = TensorBoardLogger("logs", name="chexpert_r3", version=exp_name)

    ckpt = ModelCheckpoint(
        monitor="val_auc_comp",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_auc_comp:.4f}",
    )
    es = EarlyStopping(monitor="val_auc_comp", mode="max", patience=3)
    lrmon = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        logger=logger,
        callbacks=[ckpt, es, lrmon],
        log_every_n_steps=25,
        accumulate_grad_batches=ACCUMULATE,
        gradient_clip_val=1.0,
        deterministic=False,
    )

    trainer.fit(model, dm)
    print("\n🏆 Best checkpoint:", ckpt.best_model_path)


if __name__ == "__main__":
    main()
