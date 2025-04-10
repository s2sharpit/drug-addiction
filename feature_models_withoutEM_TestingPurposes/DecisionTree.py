import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load datasets
train_df = pd.read_csv("D:/practice_models/train_drop_records.csv")
X_train = train_df.drop(columns=["Addiction_Class"])
y_train = train_df["Addiction_Class"]

# Decision Tree Feature Importance
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_importance = dt_model.feature_importances_
dt_results = pd.DataFrame({"Feature": X_train.columns, "Importance": dt_importance}).sort_values(by="Importance", ascending=False)

# Save results
dt_results.to_csv("decision_tree_results.csv", index=False)
print("Decision Tree Feature Selection completed. Results saved to decision_tree_results.csv")
