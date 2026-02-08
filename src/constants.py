# src/constants.py

# =========================
# Labels / Classes
# =========================
TARGET_COLS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

CLASS_NAMES = TARGET_COLS  # ใช้ชุดเดียวกันให้ชัวร์
NUM_CLASSES = len(TARGET_COLS)

# =========================
# Normalization (ImageNet)
# =========================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =========================
# Defaults (optional)
# =========================
DEFAULT_SEED = 42
DEFAULT_POLICY = "custom"