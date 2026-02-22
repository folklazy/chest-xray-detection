"""
คำนวณ pos_weight สำหรับ BCEWithLogitsLoss
รันครั้งเดียวก่อน train เพื่อได้ค่าที่ถูกต้อง

Usage:
    python utils/calc_pos_weight.py

⚠️  ต้อง ignore -1 (Consolidation U-Ignore) ตอนนับ
    ไม่งั้น weight จะเพี้ยน
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.constants import TARGET_COLS
from src.dataset import apply_uncertainty_policy


def calc_pos_weight(
    csv_path: str = "./data/CheXpert-v1.0-small/train.csv",
    policy: str = "custom",
    frontal_only: bool = True,
    seed: int = 42,
) -> list:
    """
    คำนวณ pos_weight = neg_valid / pos_valid
    โดย valid = label ที่ไม่ใช่ -1

    Returns:
        list of floats ความยาว len(TARGET_COLS)
    """
    df = pd.read_csv(csv_path)

    # Frontal only
    if frontal_only and "Frontal/Lateral" in df.columns:
        df = df[df["Frontal/Lateral"] == "Frontal"].copy()

    # Apply policy (ทำก่อน split เหมือนใน DataModule)
    df = apply_uncertainty_policy(df, policy)

    # Patient-level split → ใช้แค่ train split
    df["Patient"] = df["Path"].apply(lambda x: x.split("/")[2])
    patients = df["Patient"].unique()
    train_p, _ = train_test_split(patients, test_size=0.1, random_state=seed)
    train_df = df[df["Patient"].isin(train_p)].copy()

    y = train_df[TARGET_COLS].values.astype(np.float32)  # (N, 5)

    print("=" * 65)
    print(f"{'Class Imbalance Analysis':^65}")
    print(f"{'Policy: ' + policy:^65}")
    print("=" * 65)
    print(f"{'Disease':<22} {'Pos':>8} {'Neg':>8} {'Ign(-1)':>8} {'Weight':>8}")
    print("-" * 65)

    pos_weights = []
    for i, col in enumerate(TARGET_COLS):
        col_vals = y[:, i]

        pos_count  = int((col_vals ==  1).sum())
        neg_count  = int((col_vals ==  0).sum())
        ign_count  = int((col_vals == -1).sum())
        valid_total = pos_count + neg_count

        if pos_count == 0:
            weight = 1.0
            print(f"  ⚠️  {col:<20} — no positive samples, weight set to 1.0")
        else:
            weight = neg_count / pos_count

        pos_weights.append(round(weight, 2))
        print(f"  {col:<20} {pos_count:>8,} {neg_count:>8,} {ign_count:>8,} {weight:>8.2f}")

    print("=" * 65)
    print("\n✅ Copy this to your train scripts / configs:\n")
    print(f"POS_WEIGHT = {pos_weights}")
    print()

    # ยังแสดงสัดส่วน positive% ด้วย (มีประโยชน์ตอน debug)
    print(f"\n{'Positive Prevalence (valid labels only)':^65}")
    print("-" * 65)
    for i, col in enumerate(TARGET_COLS):
        col_vals = y[:, i]
        valid    = col_vals >= 0
        pos_pct  = (col_vals[valid] == 1).mean() * 100 if valid.sum() > 0 else 0
        bar      = "█" * int(pos_pct / 2)
        print(f"  {col:<22} {pos_pct:5.1f}% {bar}")
    print()

    return pos_weights


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    default="./data/CheXpert-v1.0-small/train.csv")
    parser.add_argument("--policy", default="custom",
                        choices=["custom", "u-zeros", "u-ones"])
    parser.add_argument("--no_frontal_only", action="store_true")
    args = parser.parse_args()

    calc_pos_weight(
        csv_path=args.csv,
        policy=args.policy,
        frontal_only=not args.no_frontal_only,
    )