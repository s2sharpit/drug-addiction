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
#print("Selected Features:",selected_features)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Selecting the best features
X_selected = X[selected_features]

#Splitting dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#Train a logistic regression model using Selected features
log_reg.fit(X_train, y_train)

#Make Predictions
y_pred = log_reg.predict(X_val)

#Calculating Accuracy
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy: ", accuracy)

#The logistic regression model trained with the selected features achieved an accuracy of ~69.68% on the validation set.