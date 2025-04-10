from sklearn.ensemble import RandomForestClassifier
import pandas as pd

#Separating features and target variables
train_df = pd.read_csv("D:/Drug_Prediction_ML/train_drop_records.csv")
X = train_df.drop(columns=["Addiction_Class"])
y = train_df["Addiction_Class"]

#Training a Random Forest Model(Random Forest Classifier- randforclas) (Create the model)
randforclas_model = RandomForestClassifier(n_estimators=100, random_state=42)
randforclas_model.fit(X, y) #train it
#n_estimators=100- Builds 100 trees in the forest. More trees = better performance (to a point)
#random_state=42- Fixes the random seed so you get the same result every time (helps with reproducibility)


#Getting Feature Importance
feature_importance =pd.DataFrame({"Feature":X.columns, "Importance":randforclas_model.feature_importances_})
feature_importance.sort_values(by="Importance", ascending=False, inplace=True)
# X.columns- List of all feature names you're testing
# randforclas_model.feature_importances_- Check accuracy or feature importance

print("Feature Importance: \n", feature_importance)

#Using a Random Forest classifier to rank feature importance using a tree-based model.
#Top 3 Important Features:

#1.Relationship_Strain (10.65%)
#2.Risk_Taking_Behavior (10.58%)
#3.Financial_Issues (10.55%)

#Least Important: Physical_Mental_Health_Problems (8.53%)
#Random Forest considers feature interactions, it provides better insights than correlation or chi-square tests.