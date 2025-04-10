import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load datasets
train_df = pd.read_csv("D:/practice_models/train_drop_records.csv")
test_df = pd.read_csv("D:/practice_models/test_drop_records.csv")

X_train = train_df.drop(columns=["Addiction_Class"])
y_train = train_df["Addiction_Class"]
X_test = test_df.drop(columns=["Addiction_Class"])
y_test = test_df["Addiction_Class"]

# Decision Tree Feature Selection
dt_model = DecisionTreeClassifier()
dt_selector = SelectFromModel(dt_model)
dt_selector.fit(X_train, y_train)
X_train_selected = dt_selector.transform(X_train)
X_test_selected = dt_selector.transform(X_test)

# Train and evaluate model using Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train_selected, y_train)
y_pred = log_model.predict(X_test_selected)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Save results
results = pd.DataFrame({"Metric": ["Accuracy", "Precision", "Recall", "F1-score"], "Score": [accuracy, precision, recall, f1]})
results.to_csv("decision_tree_evaluation_results.csv", index=False)
print("Decision Tree Feature Selection with Logistic Regression completed. Results saved to decision_tree_evaluation_results.csv")
