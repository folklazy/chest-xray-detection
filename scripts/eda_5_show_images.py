import pandas as pd
import cv2
import matplotlib.pyplot as plt

train = pd.read_csv('data/CheXpert-v1.0-small/train.csv')

def show_samples(df, n=6):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    samples = df.sample(n)
    for ax, (_, row) in zip(axes.flat, samples.iterrows()):
        img_path = f"data/{row['Path']}"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Age:{row['Age']} {row['Frontal/Lateral']}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

show_samples(train)
