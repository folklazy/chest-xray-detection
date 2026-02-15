"""
คำนวณ pos_weight สำหรับ BCEWithLogitsLoss
รันครั้งเดียวเพื่อดูค่า แล้วใส่ใน config/train.py
"""
import sys
sys.path.append('.')

import torch
from src.dataset import CheXpertDataModule, TARGET_COLS


def calc_pos_weight(data_dir: str, csv_path: str) -> list:
    """
    คำนวณ pos_weight สำหรับแต่ละ class
    
    pos_weight = neg_samples / pos_samples
    ใช้กับ BCEWithLogitsLoss เพื่อจัดการ class imbalance
    """
    # สร้าง DataModule
    dm = CheXpertDataModule(
        data_dir=data_dir,
        csv_path=csv_path,
        batch_size=64,
        num_workers=0
    )
    dm.setup()
    
    # ดึง labels จาก training set
    y = torch.tensor(dm.train_ds.labels)  # (N, 5)
    
    # นับ positive และ negative
    pos = y.sum(dim=0)
    neg = len(y) - pos
    
    pos_weight = neg / pos
    
    # แสดงผล
    print("=" * 50)
    print("Class Imbalance Analysis")
    print("=" * 50)
    print(f"Total samples: {len(y)}")
    print()
    
    for i, name in enumerate(TARGET_COLS):
        print(f"{name:20s} | pos: {int(pos[i]):6d} | neg: {int(neg[i]):6d} | weight: {pos_weight[i]:.2f}")
    
    print()
    print("=" * 50)
    print("Copy this to train.py:")
    print("=" * 50)
    print(f"pos_weight = {[round(w.item(), 2) for w in pos_weight]}")
    
    return pos_weight.tolist()


if __name__ == "__main__":
    # === แก้ path ตรงนี้ ===
    DATA_DIR = "./data"
    CSV_PATH = "./data/CheXpert_full/train.csv"
    
    calc_pos_weight(DATA_DIR, CSV_PATH)
