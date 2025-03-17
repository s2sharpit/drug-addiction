import pandas as pd

#Loading datasets
train_path = "D:/Drug_Prediction_ML/train_drop_records.csv"
test_path = "D:/Drug_Prediction_ML/test_drop_records.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

#Displaying basic information about datasets
train_df.info(), test_df.info(), train_df.head(), test_df.head()
