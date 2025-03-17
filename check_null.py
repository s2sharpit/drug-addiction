import pandas as pd
import numpy as np

df =  pd.read_excel("D:/Drug_Prediction_ML/checkwork.xlsx")

drop_rows = df.dropna()
print("Rows with a null value in any of the columns will be dropped:\n", drop_rows)

print(df)
