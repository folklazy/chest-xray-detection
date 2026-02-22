import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('data/CheXpert-v1.0-small/train.csv')
valid = pd.read_csv('data/CheXpert-v1.0-small/valid.csv')

print(f"Train size: {len(train)}")
print(f"Valid size: {len(valid)}")
print(train.head())
print(train.dtypes)
