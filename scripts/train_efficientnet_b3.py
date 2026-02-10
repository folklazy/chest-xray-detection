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

DATA_DIR = "./data"
CSV_PATH = "./data/CheXpert-v1.0-small/train.csv"

IMG_SIZE = 384
BATCH_SIZE = 16
NUM_WORKERS = 4
MODEL_NAME = "efficientnet_b3"
LR = 2e-4
EPOCHS = 8

POS_WEIGHT = [5.68, 7.29, 14.01, 3.28, 1.59]

def main():
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")

    dm = CheXpertDataModule(
        data_dir=DATA_DIR,
        csv_path=CSV_PATH,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        policy="custom",
        frontal_only=True,
    )

    model = CheXpertLightning(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        lr=LR,
        pos_weight=POS_WEIGHT,
        scheduler_monitor="val_auc",
    )

    exp_name = f"{MODEL_NAME}_img{IMG_SIZE}_bs{BATCH_SIZE}_lr{LR}_e{EPOCHS}"
    logger = TensorBoardLogger("logs", name="chexpert", version=exp_name)

    ckpt = ModelCheckpoint(monitor="val_auc", mode="max", save_top_k=1, filename="best-{epoch:02d}-{val_auc:.4f}")
    es = EarlyStopping(monitor="val_auc", mode="max", patience=3)
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
    print("\nBest checkpoint:", ckpt.best_model_path)

if __name__ == "__main__":
    main()