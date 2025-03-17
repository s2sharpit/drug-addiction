from sklearn.ensemble import RandomForestClassifier
import pandas as pd

#Separating features and target variables
train_df = pd.read_csv("D:/Drug_Prediction_ML/train_drop_records.csv")
X = train_df.drop(columns=["Addiction_Class"])
y = train_df["Addiction_Class"]

#Training a Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

#Getting Feature Importance
feature_importance =pd.DataFrame({"Feature":X.columns, "Importance":rf_model.feature_importances_})
feature_importance.sort_values(by="Importance", ascending=False, inplace=True)

print("Feature Importance: \n", feature_importance)

#Write a code to find the three most important features according to the order of their values