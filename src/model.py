import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import roc_auc_score

from src.backbone import CheXpertModel

# ต้องตรงกับ TARGET_COLS ใน dataset.py (ชื่อให้เหมือนกันไปเลย)
CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


class CheXpertLightning(pl.LightningModule):
    def __init__(
        self,
        model_name="densenet121",
        num_classes=5,
        lr=1e-4,
        dropout=0.3,
        pos_weight=None,  # list หรือ tensor shape (C,)
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = CheXpertModel(
            model_name=model_name,
            num_classes=num_classes,
            dropout=dropout,
        )
        
        # ✅ แปลง memory format ของโมเดลให้เป็น channels_last (N, H, W, C) เพื่อให้ Tensor Core ทำงานเร็วขึ้น
        self.model = self.model.to(memory_format=torch.channels_last)

        # pos_weight: register_buffer เพื่อย้าย GPU อัตโนมัติ
        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

        self.lr = lr
        self.num_classes = num_classes

        # เก็บ val outputs เพื่อคำนวณ AUC แบบ ignore -1 ได้ถูกต้อง
        self._val_probs = []
        self._val_targets = []

    def forward(self, x):
        # ✅ มั่นใจว่า Input x เป็น channels_last ก่อนส่งเข้าโมเดล
        if x.dim() == 4:
            x = x.to(memory_format=torch.channels_last)
        return self.model(x)

    # -----------------------------
    # Masked BCE (ignore -1)
    # -----------------------------
    def _masked_bce(self, logits, y):
        """
        logits: (B,C)
        y: (B,C) with values in {0,1,-1}
        """
        mask = (y >= 0).float()           # 1 where label valid
        y01 = y.clamp(0, 1)               # -1 -> 0 temporarily

        bce = F.binary_cross_entropy_with_logits(
            logits,
            y01,
            pos_weight=self.pos_weight if self.pos_weight is not None else None,  # shape (C,)
            reduction="none",
        )                                  # (B,C)

        loss = (bce * mask).sum() / (mask.sum() + 1e-6)
        return loss

    # -----------------------------
    # Training
    # -----------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._masked_bce(logits, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    # -----------------------------
    # Validation
    # -----------------------------
    def on_validation_epoch_start(self):
        self._val_probs.clear()
        self._val_targets.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self._masked_bce(logits, y)
        self.log("val_loss", loss, prog_bar=True, logger=True)

        probs = torch.sigmoid(logits)

        # เก็บไว้คำนวณ AUC แบบ ignore -1 จริง
        self._val_probs.append(probs.detach().cpu())
        self._val_targets.append(y.detach().cpu())

    def on_validation_epoch_end(self):
        probs = torch.cat(self._val_probs, dim=0).numpy()      # (N,C)
        targets = torch.cat(self._val_targets, dim=0).numpy()  # (N,C) with -1 possible

        aucs = []
        per_class = {}

        for c in range(self.num_classes):
            t = targets[:, c]
            p = probs[:, c]
            valid = t >= 0

            # ต้องมีทั้ง 0 และ 1 ถึงคำนวณ AUC ได้
            if valid.sum() < 10 or len(np.unique(t[valid])) < 2:
                auc = np.nan
            else:
                auc = roc_auc_score((t[valid] == 1).astype(int), p[valid])

            aucs.append(auc)
            per_class[CLASS_NAMES[c]] = auc

        mean_auc = np.nanmean(aucs)

        # log mean (ใช้ checkpoint / early stop)
        self.log("val_auc", float(mean_auc), prog_bar=True, logger=True)

        # log per-class
        for k, v in per_class.items():
            if np.isnan(v):
                continue
            self.log(f"val_auc_{k}", float(v), logger=True)

        print(f"\n[Epoch {self.current_epoch}] Mean AUC: {mean_auc:.4f}")
        print({k: (None if np.isnan(v) else round(float(v), 4)) for k, v in per_class.items()})

    # -----------------------------
    # Optimizer
    # -----------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=3,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }