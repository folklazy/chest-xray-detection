import pandas as pd

train = pd.read_csv('data/CheXpert-v1.0-small/train.csv')

print(train['Sex'].value_counts())
print(train['Age'].describe())
print(train['AP/PA'].value_counts())
