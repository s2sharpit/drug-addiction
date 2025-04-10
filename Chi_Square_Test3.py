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
# X.columns- List of all feature names you're testing
# chi_scores- The Chi-Square statistic for each feature with respect to the target
# p_values- The p-values for each Chi-Square test. Lower = more significant

chi2_results.sort_values(by="Chi2 Score", ascending=False, inplace=True)
print(chi2_results)
print("Maximum_chi_square_value: ", max(chi_scores))

#Performing the Chi-Square test to identify statistically significant features for the target variable. ​
#The most important features (highest Chi-Square scores) are:

#1.Physical_Mental_Health_Problems (0.957)
#2.Social_Isolation (0.831)
#3.Withdrawal_Symptoms (0.607)

#However, all p-values are greater than 0.05, indicating that none of these features have a statistically significant relationship with Addiction_Class.
