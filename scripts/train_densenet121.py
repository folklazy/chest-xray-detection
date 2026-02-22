"""
Train DenseNet121 on CheXpert Small

การแก้ไขจาก training run แรก:
  - LR: 1e-4 -> 3e-5  (แก้ train_loss ที่ขึ้นแทนลง)
  - Scheduler: ReduceLROnPlateau -> CosineAnnealingLR + Warmup
  - warmup_epochs=1 (ป้องกัน gradient explosion ช่วงแรก)
  - แก้ albumentations warnings (ShiftScaleRotate, CoarseDropout)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.dataset import CheXpertDataModule
from src.model import CheXpertLightning
from src.constants import NUM_CLASSES

# =====================================================
# Config
# =====================================================
DATA_DIR    = "./data"
CSV_PATH    = "./data/CheXpert-v1.0-small/train.csv"

MODEL_NAME  = "densenet121"
IMG_SIZE    = 384
BATCH_SIZE  = 16
NUM_WORKERS = 8
EPOCHS      = 10
WARMUP      = 1

LR          = 3e-5      # ลดจาก 1e-4 แก้ train_loss ขึ้น

POS_WEIGHT  = [2.21, 7.19, 11.84, 2.1, 1.21]

# =====================================================

def main():
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = True

    dm = CheXpertDataModule(
        data_dir=DATA_DIR,
        csv_path=CSV_PATH,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        policy="custom",
        frontal_only=True,
        use_clahe=True,
    )

    model = CheXpertLightning(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE,
        lr=LR,
        pos_weight=POS_WEIGHT,
        scheduler_monitor="val_auc",
        epochs=EPOCHS,
        warmup_epochs=WARMUP,
    )

    exp_name = f"{MODEL_NAME}_img{IMG_SIZE}_bs{BATCH_SIZE}_lr{LR}_cosine_e{EPOCHS}"
    logger = TensorBoardLogger("logs", name="chexpert", version=exp_name)

    ckpt = ModelCheckpoint(
        monitor="val_auc",
        mode="max",
        save_top_k=3,
        filename="best-{epoch:02d}-{val_auc:.4f}",
    )
    es = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=5,
        min_delta=0.001,
    )
    lrmon = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        logger=logger,
        callbacks=[ckpt, es, lrmon],
        log_every_n_steps=20,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, dm)
    print(f"\nBest checkpoint: {ckpt.best_model_path}")
    print(f"Best val_auc:    {ckpt.best_model_score:.4f}")


if __name__ == "__main__":
    main()