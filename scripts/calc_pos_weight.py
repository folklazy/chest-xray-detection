import pandas as pd

df = pd.read_csv("data/CheXpert_full/train.csv")
corr = df[labels].replace(-1,0).corr()
print(corr["Edema"].sort_values(ascending=False)[:6])