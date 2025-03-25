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

#Using a Random Forest classifier to rank feature importance using a tree-based model.
#Top 3 Important Features:

#1.Relationship_Strain (10.65%)
#2.Risk_Taking_Behavior (10.58%)
#3.Financial_Issues (10.55%)

#Least Important: Physical_Mental_Health_Problems (8.53%)
#Random Forest considers feature interactions, it provides better insights than correlation or chi-square tests.