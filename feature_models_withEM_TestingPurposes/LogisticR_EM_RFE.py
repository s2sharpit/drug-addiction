import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load datasets
train_df = pd.read_csv("D:/practice_models/train_drop_records.csv")
test_df = pd.read_csv("D:/practice_models/test_drop_records.csv")

X_train = train_df.drop(columns=["Addiction_Class"])
y_train = train_df["Addiction_Class"]
X_test = test_df.drop(columns=["Addiction_Class"])
y_test = test_df["Addiction_Class"]

# Recursive Feature Elimination (RFE)
model = LogisticRegression()
rfe_selector = RFE(model, n_features_to_select=5)
X_train_selected = rfe_selector.fit_transform(X_train, y_train)
X_test_selected = rfe_selector.transform(X_test)

# Train and evaluate model
model.fit(X_train_selected, y_train)
y_pred = model.predict(X_test_selected)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Save results
results = pd.DataFrame({"Metric": ["Accuracy", "Precision", "Recall", "F1-score"], "Score": [accuracy, precision, recall, f1]})
results.to_csv("rfe_evaluation_results.csv", index=False)
print("RFE Evaluation completed. Results saved to rfe_evaluation_results.csv")
