# src/constants.py
from __future__ import annotations

from typing import Literal

# =====================================================
# CheXpert labels
# - Train: all 14 observations (multi-task)
# - Report: mean AUC on 5 competition tasks
# =====================================================

ALL_TARGET_COLS: list[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

COMPETITION_COLS: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# Backwards-friendly names
TARGET_COLS: list[str] = ALL_TARGET_COLS
CLASS_NAMES: list[str] = list(TARGET_COLS)
NUM_CLASSES: int = len(TARGET_COLS)

# Fast lookup
COL2IDX: dict[str, int] = {c: i for i, c in enumerate(TARGET_COLS)}
COMPETITION_IDXS: list[int] = [COL2IDX[c] for c in COMPETITION_COLS]

# =====================================================
# Normalization (ImageNet)
# =====================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =====================================================
# Preprocess modes
# =====================================================
PREPROCESS_IMAGENET = "imagenet"  # 3ch + ImageNet normalize
PREPROCESS_XRV = "xrv"            # 1ch + torchxrayvision normalize/crop

# =====================================================
# Uncertainty policies (per-class)
# - ignore: keep -1 (masked in loss / metrics)
# - zeros : map -1 -> 0
# - ones  : map -1 -> 1
# =====================================================

PolicyAction = Literal["ignore", "zeros", "ones"]

POLICY_DEFAULT_ACTION: PolicyAction = "zeros"  # safe default for unspecified labels

POLICY_MIXED_V1: dict[str, PolicyAction] = {
    # 5 competition tasks (Kaggle pain points)
    "Atelectasis": "ones",
    "Consolidation": "ignore",
    "Edema": "ones",
    "Cardiomegaly": "zeros",
    "Pleural Effusion": "ones",
    # remaining 9 observations -> zeros
    "No Finding": "zeros",
    "Enlarged Cardiomediastinum": "zeros",
    "Lung Opacity": "zeros",
    "Lung Lesion": "zeros",
    "Pneumonia": "zeros",
    "Pneumothorax": "zeros",
    "Pleural Other": "zeros",
    "Fracture": "zeros",
    "Support Devices": "zeros",
}

POLICIES: dict[str, dict[str, PolicyAction]] = {
    "mixed_v1": POLICY_MIXED_V1,
}

DEFAULT_POLICY_NAME = "mixed_v1"

# Validate policy coverage at import time (fail fast)
_missing = set(TARGET_COLS) - set(POLICY_MIXED_V1.keys())
_extra = set(POLICY_MIXED_V1.keys()) - set(TARGET_COLS)
if _missing:
    raise ValueError(f"POLICY_MIXED_V1 missing labels: {sorted(_missing)}")
if _extra:
    raise ValueError(f"POLICY_MIXED_V1 has unknown labels: {sorted(_extra)}")

# =====================================================
# Defaults
# =====================================================
DEFAULT_SEED = 42