import pandas as pd
import numpy as np
import cv2
import random

train = pd.read_csv('data/CheXpert-v1.0-small/train.csv')

pixel_means, pixel_stds = [], []
sample_paths = random.sample(list(train['Path']), 500)

for path in sample_paths:
    img_path = f"data/{path}"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        pixel_means.append(img.mean() / 255.0)
        pixel_stds.append(img.std() / 255.0)

print(f"Dataset mean: {np.mean(pixel_means):.4f}")
print(f"Dataset std:  {np.mean(pixel_stds):.4f}")
# ใช้ค่านี้ตอน normalize แทน ImageNet stats ก็ได้
