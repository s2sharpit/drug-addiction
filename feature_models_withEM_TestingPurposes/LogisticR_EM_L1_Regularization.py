import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load datasets
train_df = pd.read_csv("D:/practice_models/train_drop_records.csv")
test_df = pd.read_csv("D:/practice_models/test_drop_records.csv")

X_train = train_df.drop(columns=["Addiction_Class"])
y_train = train_df["Addiction_Class"]
X_test = test_df.drop(columns=["Addiction_Class"])
y_test = test_df["Addiction_Class"]

# L1 Regularization Feature Selection
l1_model = LogisticRegression(penalty='l1', solver='liblinear')
l1_selector = SelectFromModel(l1_model)
l1_selector.fit(X_train, y_train)
X_train_selected = l1_selector.transform(X_train)
X_test_selected = l1_selector.transform(X_test)

# Train and evaluate model
l1_model.fit(X_train_selected, y_train)
y_pred = l1_model.predict(X_test_selected)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Save results
results = pd.DataFrame({"Metric": ["Accuracy", "Precision", "Recall", "F1-score"],"Score": [accuracy, precision, recall, f1]})
results.to_csv("l1_regularization_evaluation_results.csv", index=False)
print("L1 Regularization Evaluation completed. Results saved to l1_regularization_evaluation_results.csv")
