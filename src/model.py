# src/model.py
# =====================================================
# LightningModule for CheXpert (multi-label, 14 outputs)
# - Masked BCEWithLogits (ignores labels < 0)
# - Freeze -> unfreeze schedule
# - Metrics:
#     val_auc_mean : mean AUC over ALL classes (14)
#     val_auc_comp : mean AUC over COMPETITION_COLS (5)
#     val_auc_<cls>: per-class AUC (logged when defined)
# =====================================================

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.backbone import CheXpertModel
from src.constants import CLASS_NAMES, NUM_CLASSES, COMPETITION_COLS, COMPETITION_IDXS


def _safe_sigmoid(x: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(x)
    return torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)


def _masked_bce_with_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    pos_weight: torch.Tensor,
    clamp_logits: bool = True,
    clamp_value: float = 20.0,
) -> torch.Tensor:
    """Masked BCE: ignores labels < 0."""
    mask = (y >= 0).float()
    y01 = y.clamp(0, 1)

    if clamp_logits:
        logits = torch.clamp(logits, -clamp_value, clamp_value)

    bce = F.binary_cross_entropy_with_logits(
        logits,
        y01,
        pos_weight=pos_weight,
        reduction="none",
    )
    denom = mask.sum().clamp_min(1.0)
    loss = (bce * mask).sum() / denom

    if not torch.isfinite(loss):
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    return loss


def _compute_auc_per_class(
    probs: np.ndarray,
    targets: np.ndarray,
    class_names: List[str],
    min_valid: int = 10,
) -> Dict[str, float]:
    """Per-class ROC-AUC with mask (targets < 0 ignored)."""
    out: Dict[str, float] = {}
    for i, name in enumerate(class_names):
        t = targets[:, i]
        p = probs[:, i]

        valid = (t >= 0) & np.isfinite(p)
        if int(valid.sum()) < int(min_valid):
            out[name] = np.nan
            continue

        tv = (t[valid] == 1).astype(np.int32)
        if len(np.unique(tv)) < 2:
            out[name] = np.nan
            continue

        try:
            out[name] = float(roc_auc_score(tv, p[valid]))
        except Exception:
            out[name] = np.nan
    return out


def _nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    m = np.nanmean(arr)
    return float(m) if np.isfinite(m) else 0.0


def _sanitize_key(name: str) -> str:
    """Make logging keys stable (avoid spaces/slashes)."""
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


class CheXpertLightning(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "densenet121",
        num_classes: int = NUM_CLASSES,
        lr: float = 1e-4,
        dropout: float = 0.2,
        pos_weight: Optional[Union[torch.Tensor, List[float]]] = None,
        weight_decay: float = 1e-4,
        scheduler_monitor: str = "val_auc_comp",  # ✅ default: competition mean AUC
        img_size: int = 384,
        # stability/debug
        log_nan_debug: bool = True,
        clamp_logits: bool = True,
        logits_clamp_value: float = 20.0,
        # freeze schedule
        freeze_epochs: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = CheXpertModel(
            model_name=model_name,
            num_classes=num_classes,
            dropout=dropout,
            img_size=img_size,
        )

        # pos_weight buffer
        if pos_weight is None:
            pw = torch.ones(num_classes, dtype=torch.float32)
        else:
            pw = pos_weight if isinstance(pos_weight, torch.Tensor) else torch.tensor(pos_weight, dtype=torch.float32)
            pw = pw.float()
            if pw.ndim != 1 or pw.numel() != num_classes:
                raise ValueError(f"pos_weight must be shape ({num_classes},) but got {tuple(pw.shape)}")
        self.register_buffer("pos_weight", pw)

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.num_classes = int(num_classes)

        self.scheduler_monitor = str(scheduler_monitor)
        self.log_nan_debug = bool(log_nan_debug)
        self.clamp_logits = bool(clamp_logits)
        self.logits_clamp_value = float(logits_clamp_value)

        self.freeze_epochs = int(freeze_epochs)
        self._did_unfreeze = False

        self._val_probs: List[torch.Tensor] = []
        self._val_targets: List[torch.Tensor] = []

        # Pre-compute competition indices (from constants)
        self._comp_idxs = list(COMPETITION_IDXS)
        self._comp_names = list(COMPETITION_COLS)

    # -----------------
    # freezing schedule
    # -----------------
    def on_fit_start(self):
        if self.freeze_epochs > 0 and hasattr(self.model, "freeze_features"):
            self.model.freeze_features()
            self._did_unfreeze = False

    def on_train_epoch_start(self):
        if (
            self.freeze_epochs > 0
            and not self._did_unfreeze
            and self.current_epoch >= self.freeze_epochs
            and hasattr(self.model, "unfreeze_all")
        ):
            self.model.unfreeze_all()
            self._did_unfreeze = True

    # -------------
    # forward/loss
    # -------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _masked_bce_with_logits(
            logits=logits,
            y=y,
            pos_weight=self.pos_weight,
            clamp_logits=self.clamp_logits,
            clamp_value=self.logits_clamp_value,
        )

    # -----------------
    # training/validation
    # -----------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._loss(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_probs.clear()
        self._val_targets.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self._loss(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        probs = _safe_sigmoid(logits)
        self._val_probs.append(probs.detach().cpu())
        self._val_targets.append(y.detach().cpu())

    def on_validation_epoch_end(self):
        probs_t = torch.cat(self._val_probs, dim=0)
        targets_t = torch.cat(self._val_targets, dim=0)

        probs = probs_t.numpy()
        targets = targets_t.numpy()

        nan_rate = float((~np.isfinite(probs)).mean()) if probs.size else 0.0
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

        if self.log_nan_debug:
            self.log("val_nan_rate", nan_rate, prog_bar=False, logger=True)

        per_class = _compute_auc_per_class(probs, targets, CLASS_NAMES)

        # mean AUC over all outputs (14)
        mean_auc_all = _nanmean(list(per_class.values()))
        self.log("val_auc_mean", mean_auc_all, prog_bar=False, logger=True)

        # mean AUC over competition tasks only (5)
        comp_vals = [per_class.get(n, np.nan) for n in self._comp_names]
        mean_auc_comp = _nanmean(comp_vals)
        self.log("val_auc_comp", mean_auc_comp, prog_bar=True, logger=True)

        # log per-class AUC (sanitize key)
        for name, auc in per_class.items():
            if not np.isnan(auc):
                self.log(f"val_auc_{_sanitize_key(name)}", float(auc), logger=True)

        # console print (readable)
        comp_str = {k: (None if np.isnan(per_class.get(k, np.nan)) else round(float(per_class[k]), 4)) for k in self._comp_names}
        print(
            f"\n[Epoch {self.current_epoch}] "
            f"val_auc_comp={mean_auc_comp:.4f} | val_auc_mean={mean_auc_all:.4f} | nan_rate={nan_rate:.6f}"
        )
        print("competition per-class:", comp_str)

    # -----------------
    # optimizer/scheduler
    # -----------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        monitor = self.scheduler_monitor
        if monitor not in {"val_auc_comp", "val_auc_mean", "val_loss"}:
            monitor = "val_auc_comp"

        if monitor in {"val_auc_comp", "val_auc_mean"}:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.1,
                patience=2,
            )
            mode_monitor = monitor
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=3,
            )
            mode_monitor = "val_loss"

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": mode_monitor}