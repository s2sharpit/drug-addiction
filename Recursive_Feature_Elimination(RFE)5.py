from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd

#Separating features and target variables
train_df = pd.read_csv("D:/Drug_Prediction_ML/train_drop_records.csv")
X = train_df.drop(columns=["Addiction_Class"])
y = train_df["Addiction_Class"]

#Using Logistic Regression as the estimator
log_reg = LogisticRegression(max_iter=1000, random_state=42)

#Applying RFE to select top five(5) features
rfe = RFE(log_reg, n_features_to_select=5)
rfe.fit(X, y)

#Getting Sellected features
selected_features = X.columns[rfe.support_].tolist()

print("Selected Features:",selected_features)

#This method eliminates less important features iteratively.
