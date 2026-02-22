import pandas as pd

train = pd.read_csv('data/CheXpert-v1.0-small/train.csv')

labels = ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly',
          'Lung Opacity','Lung Lesion','Edema','Consolidation',
          'Pneumonia','Atelectasis','Pneumothorax',
          'Pleural Effusion','Pleural Other','Fracture','Support Devices']

# ดูสัดส่วน 1 / 0 / -1 / NaN ของแต่ละ label
label_counts = {}
for col in labels:
    counts = train[col].value_counts(dropna=False)
    label_counts[col] = counts

print(pd.DataFrame(label_counts).T)
