import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.backbone import CheXpertModel
from src.constants import CLASS_NAMES, NUM_CLASSES


class CheXpertLightning(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "densenet121",
        num_classes: int = NUM_CLASSES,
        img_size: int = 384,
        lr: float = 3e-5,               # ลดจาก 1e-4 → 3e-5 แก้ปัญหา train_loss ขึ้น
        dropout: float = 0.3,
        pos_weight=None,
        weight_decay: float = 1e-4,
        scheduler_monitor: str = "val_auc",
        epochs: int = 10,               # ใช้กับ CosineAnnealingLR T_max
        warmup_epochs: int = 1,         # warmup 1 epoch ก่อน cosine
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
            img_size=img_size,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.scheduler_monitor = scheduler_monitor
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs

        # -----------------------------
        # pos_weight buffer
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

        self._val_probs = []
        self._val_targets = []

    def forward(self, x):
        return self.model(x)

    # -----------------------------
    # Masked BCE (ignore -1)
    # -----------------------------
    def _masked_bce(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mask = (y >= 0).float()
        y01  = y.clamp(0, 1)

        pw = self.pos_weight if torch.any(self.pos_weight != 0) else None

        bce = F.binary_cross_entropy_with_logits(
            logits, y01, pos_weight=pw, reduction="none",
        )

        denom = mask.sum().clamp_min(1.0)
        return (bce * mask).sum() / denom

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
        self._val_probs.append(probs.detach().cpu())
        self._val_targets.append(y.detach().cpu())

    def on_validation_epoch_end(self):
        probs   = torch.cat(self._val_probs,   dim=0).numpy()
        targets = torch.cat(self._val_targets, dim=0).numpy()

        aucs      = []
        per_class = {}

        for c in range(self.num_classes):
            t     = targets[:, c]
            p     = probs[:, c]
            valid = t >= 0

            if valid.sum() < 10 or len(np.unique(t[valid])) < 2:
                auc = np.nan
            else:
                auc = roc_auc_score((t[valid] == 1).astype(int), p[valid])

            aucs.append(auc)
            per_class[CLASS_NAMES[c]] = auc

        mean_auc = float(np.nanmean(aucs)) if np.isfinite(np.nanmean(aucs)) else 0.0

        self.log("val_auc", mean_auc, prog_bar=True, logger=True)
        for k, v in per_class.items():
            if not np.isnan(v):
                self.log(f"val_auc_{k}", float(v), logger=True)

        # แสดง LR ปัจจุบันด้วย (ช่วย debug)
        current_lr = self.optimizers().param_groups[0]["lr"]
        print(f"\n[Epoch {self.current_epoch}] Mean AUC: {mean_auc:.4f} | LR: {current_lr:.2e}")
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

        # Cosine Annealing: LR ลดเรียบๆ จาก lr → eta_min ตลอด training
        # ดีกว่า ReduceLROnPlateau ตรงที่ไม่ต้องรอ patience
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=1e-7,
        )

        # Linear Warmup: epoch แรก LR ค่อยๆ เพิ่มจาก 0 → lr
        # ป้องกัน gradient explosion ช่วงแรก
        def warmup_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            return 1.0

        warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

        # SequentialLR: warmup ก่อน แล้วค่อย cosine
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }