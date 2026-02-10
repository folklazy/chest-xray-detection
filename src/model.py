import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import roc_auc_score

from src.backbone import CheXpertModel
from src.constants import CLASS_NAMES, NUM_CLASSES


class CheXpertLightning(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "densenet121",
        num_classes: int = NUM_CLASSES,
        img_size: int = 384,            # ✅ เพิ่ม: ส่งให้ timm models (เช่น Swin)
        lr: float = 1e-4,
        dropout: float = 0.3,
        pos_weight=None,                # list หรือ tensor shape (C,)
        weight_decay: float = 1e-4,
        scheduler_monitor: str = "val_auc",  # "val_auc" หรือ "val_loss"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight"])

        # -----------------------------
        # Backbone
        # -----------------------------
        self.model = CheXpertModel(
            model_name=model_name,
            num_classes=num_classes,
            dropout=dropout,
            img_size=img_size,          # ✅ สำคัญ: กัน Swin assert 224/384
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.scheduler_monitor = scheduler_monitor

        # -----------------------------
        # pos_weight buffer (always shape (C,))
        # ✅ แก้ ckpt mismatch ตอน inference
        # -----------------------------
        if pos_weight is None:
            pos_weight = torch.zeros(num_classes, dtype=torch.float32)
        else:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            if pos_weight.ndim != 1 or pos_weight.numel() != num_classes:
                raise ValueError(
                    f"pos_weight must be shape ({num_classes},) but got {tuple(pos_weight.shape)}"
                )

        self.register_buffer("pos_weight", pos_weight)

        # เก็บ outputs สำหรับ AUC แบบ ignore -1 จริง
        self._val_probs = []
        self._val_targets = []

    def forward(self, x):
        return self.model(x)

    # -----------------------------
    # Masked BCE (ignore -1)
    # -----------------------------
    def _masked_bce(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,C)
        y: (B,C) in {0,1,-1}
        """
        mask = (y >= 0).float()      # 1 where label valid
        y01 = y.clamp(0, 1)          # -1 -> 0 temporarily

        # ใช้ pos_weight ก็ต่อเมื่อมีค่าไม่เป็นศูนย์อย่างน้อย 1 class
        pw = self.pos_weight if torch.any(self.pos_weight != 0) else None

        bce = F.binary_cross_entropy_with_logits(
            logits,
            y01,
            pos_weight=pw,
            reduction="none",
        )  # (B,C)

        denom = mask.sum().clamp_min(1.0)
        loss = (bce * mask).sum() / denom
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

        mean_auc = float(np.nanmean(aucs)) if np.isfinite(np.nanmean(aucs)) else 0.0

        self.log("val_auc", mean_auc, prog_bar=True, logger=True)

        for k, v in per_class.items():
            if np.isnan(v):
                continue
            self.log(f"val_auc_{k}", float(v), logger=True)

        print(f"\n[Epoch {self.current_epoch}] Mean AUC: {mean_auc:.4f}")
        print({k: (None if np.isnan(v) else round(float(v), 4)) for k, v in per_class.items()})

    # -----------------------------
    # Optimizer & Scheduler
    # -----------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.scheduler_monitor == "val_auc":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.1,
                patience=2,
            )
            monitor = "val_auc"
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=3,
            )
            monitor = "val_loss"

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": monitor,
        }