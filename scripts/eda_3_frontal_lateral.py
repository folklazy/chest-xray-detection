import pandas as pd

train = pd.read_csv('data/CheXpert-v1.0-small/train.csv')

print(train['Frontal/Lateral'].value_counts())
# Lateral มักไม่ค่อยมีประโยชน์สำหรับ competition
# หลายคน filter เอาแค่ Frontal
