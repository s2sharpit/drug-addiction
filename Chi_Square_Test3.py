from sklearn.feature_selection import chi2
import numpy as np
import pandas as pd

#Separating features and target variables
train_df = pd.read_csv("D:/Drug_Prediction_ML/train_drop_records.csv")

X = train_df.drop(columns=["Addiction_Class"])
y = train_df["Addiction_Class"]

#Performing Chi-Square Test
chi_scores, p_values = chi2(X, y)

#Creating a Dataframe with results
chi2_results = pd.DataFrame({"Feature":X.columns, "Chi2 Score":chi_scores, "P-value":p_values})
chi2_results.sort_values(by="Chi2 Score", ascending=False, inplace=True)
print(chi2_results)
print("Maximum_chi_square_value: ", max(chi_scores))

#Write a code to find the three most important features according to the order of their values