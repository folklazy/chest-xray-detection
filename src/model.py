import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from src.backbone import CheXpertModel

# ชื่อโรคสำหรับ log (ต้องตรงกับลำดับใน dataset)
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural_Effusion']


class CheXpertLightning(pl.LightningModule):
    def __init__(
        self,
        model_name="densenet121",
        num_classes=5,
        lr=3e-4,
        dropout=0.3,
        pos_weight=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # -------------------------
        # 1. Model Architecture
        # -------------------------
        self.model = CheXpertModel(
            model_name=model_name,
            num_classes=num_classes,
            dropout=dropout
        )

        # -------------------------
        # 2. Loss Function
        # -------------------------
        # register_buffer ช่วยให้ pos_weight ย้ายลง GPU อัตโนมัติ
        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            self.register_buffer('pos_weight', pos_weight)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.register_buffer('pos_weight', None)
            self.criterion = nn.BCEWithLogitsLoss()

        # -------------------------
        # 3. Metrics
        # -------------------------
        self.val_auc = torchmetrics.AUROC(
            task="multilabel",
            num_labels=num_classes,
            average=None
        )

        self.lr = lr

    # =====================================================
    # Forward
    # =====================================================
    def forward(self, x):
        return self.model(x)

    # =====================================================
    # Training
    # =====================================================
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    # =====================================================
    # Validation
    # =====================================================
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        self.log("val_loss", loss, prog_bar=True, logger=True)

        probs = torch.sigmoid(logits)
        self.val_auc.update(probs, y.int())

    def on_validation_epoch_end(self):
        scores = self.val_auc.compute()
        mean_auc = scores.mean()

        # Log ค่าเฉลี่ยรวม
        self.log("val_auc", mean_auc, prog_bar=True, logger=True)

        # Log แยกรายโรค
        score_dict = {f"val_auc_{name}": sc for name, sc in zip(CLASS_NAMES, scores)}
        self.log_dict(score_dict, logger=True)

        # Print รายงาน
        print(f"\n[Epoch {self.current_epoch}] Mean AUC: {mean_auc:.4f}")
        print(f"Details: {dict(zip(CLASS_NAMES, [round(x.item(), 4) for x in scores]))}")

        self.val_auc.reset()

    # =====================================================
    # Optimizer & Scheduler
    # =====================================================
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=1e-4  # Conservative value
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=3
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
