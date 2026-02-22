"""
Train DenseNet121 on CheXpert Small

Steps ก่อน run:
    1. python utils/calc_pos_weight.py   ← ดู POS_WEIGHT ที่ถูกต้อง
    2. python scripts/train_densenet121.py
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
IMG_SIZE    = 384       # เพิ่มจาก 320 → 384 (paper ใช้ 320 แต่ 384 ให้ AUC ดีกว่า)
BATCH_SIZE  = 16
NUM_WORKERS = 8
LR          = 1e-4
EPOCHS      = 10        # เพิ่มจาก 8 → 10 (DenseNet เรียนรู้ช้ากว่า ViT)

# ⚠️  รัน utils/calc_pos_weight.py ก่อนเพื่อให้ได้ค่าจริง
# ค่านี้คำนวณจาก frontal-only + custom policy
POS_WEIGHT  = [2.21, 7.19, 11.84, 2.1, 1.21]

# =====================================================

def main():
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = True   # เร็วขึ้นเมื่อ input size คงที่

    dm = CheXpertDataModule(
        data_dir=DATA_DIR,
        csv_path=CSV_PATH,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        policy="custom",
        frontal_only=True,
        use_clahe=True,     # เพิ่ม CLAHE → +1-2% AUC
    )

    model = CheXpertLightning(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE,
        lr=LR,
        pos_weight=POS_WEIGHT,
        scheduler_monitor="val_auc",
    )

    exp_name = f"{MODEL_NAME}_img{IMG_SIZE}_bs{BATCH_SIZE}_clahe_e{EPOCHS}"
    logger = TensorBoardLogger("logs", name="chexpert", version=exp_name)

    ckpt = ModelCheckpoint(
        monitor="val_auc",
        mode="max",
        save_top_k=3,       # เก็บ top-3 ไว้ ensemble ทีหลัง
        filename="best-{epoch:02d}-{val_auc:.4f}",
    )
    es = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=4,         # เพิ่มจาก 3 → 4 (ให้โอกาส LR decay ทำงาน)
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
        # ไม่ใช้ accumulate_grad_batches สำหรับ DenseNet
        # (bs=16 พอแล้ว)
    )

    trainer.fit(model, dm)
    print(f"\n✅ Best checkpoint: {ckpt.best_model_path}")
    print(f"   Best val_auc:    {ckpt.best_model_score:.4f}")


if __name__ == "__main__":
    main()