# ======================================================
# CheXpert Training Template (Production-ready)
# PyTorch Lightning + Weighted BCE + AUC + Scheduler
# ======================================================

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.dataset import CheXpertDataModule
from src.model import CheXpertLightning

# ======================================================
# CONFIG (แก้ตรงนี้ที่เดียวพอ)
# ======================================================

DATA_DIR = "./data"  # โฟลเดอร์ที่มี CheXpert-v1.0-small
CSV_PATH = "./data/CheXpert-v1.0-small/train.csv"

IMG_SIZE = 320
BATCH_SIZE = 16
NUM_WORKERS = 4

MODEL_NAME = "densenet121"  # or efficientnet-b0
LR = 1e-4
EPOCHS = 15

# จากที่คุณคำนวณมา
POS_WEIGHT = [5.68, 7.29, 14.01, 3.28, 1.59]

# ======================================================
# MAIN TRAINING
# ======================================================


def main():
    pl.seed_everything(42, workers=True)

    # -----------------------------
    # DataModule
    # -----------------------------
    dm = CheXpertDataModule(
        data_dir=DATA_DIR,
        csv_path=CSV_PATH,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # -----------------------------
    # Lightning Model
    # -----------------------------
    model = CheXpertLightning(
        model_name=MODEL_NAME,
        num_classes=5,
        lr=LR,
        pos_weight=POS_WEIGHT,  # ⭐ สำคัญสุด
    )

    # -----------------------------
    # Callbacks (ของจำเป็น)
    # -----------------------------

    checkpoint = ModelCheckpoint(
        monitor="val_auc",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_auc:.4f}",
    )

    early_stop = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    logger = TensorBoardLogger("logs", name="chexpert")

    # -----------------------------
    # Trainer
    # -----------------------------

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",  # ⭐ เร็วขึ้น ~1.7x
        callbacks=[checkpoint, early_stop, lr_monitor],
        logger=logger,
        log_every_n_steps=20,
    )

    # -----------------------------
    # Train
    # -----------------------------

    trainer.fit(model, dm)

    print("\nBest checkpoint:", checkpoint.best_model_path)


# ======================================================

if __name__ == "__main__":
    main()
