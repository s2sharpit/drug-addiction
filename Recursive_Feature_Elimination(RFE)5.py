from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd

#Separating features and target variables
train_df = pd.read_csv("D:/Drug_Prediction_ML/train_drop_records.csv")
X = train_df.drop(columns=["Addiction_Class"])
y = train_df["Addiction_Class"]

#Using Logistic Regression as the estimator
#max_iter- Increases the maximum number of iterations allowed during model convergence. Sometimes LR needs more than default (100) to converge.
#random_state- Ensures reproducibility of results (same output each time you run it).
log_reg = LogisticRegression(max_iter=1000, random_state=42)


#Applying RFE to select top five(5) features
rfe = RFE(log_reg, n_features_to_select=5) #log_reg - The estimator used to evaluate each subset of features
rfe.fit(X, y) #Training

#Getting Selected features
#X.columns[...]	- Indexes the column names using that boolean array.
#rfe.support_ -	A boolean array indicating which features are selected (True) and which are not (False).
#.tolist() - Converts the result to a regular Python list.
selected_features = X.columns[rfe.support_].tolist()

print("Selected Features:",selected_features)
#This method eliminates less important features iteratively
