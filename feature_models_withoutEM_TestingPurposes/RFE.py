import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Load datasets
train_df = pd.read_csv("D:/practice_models/train_drop_records.csv")
X_train = train_df.drop(columns=["Addiction_Class"])
y_train = train_df["Addiction_Class"]

# Recursive Feature Elimination (RFE)
log_model = LogisticRegression()
rfe_selector = RFE(log_model, n_features_to_select=5)
rfe_selector.fit(X_train, y_train)
rfe_results = pd.DataFrame({"Feature": X_train.columns, "Selected": rfe_selector.support_})

# Save results
rfe_results.to_csv("rfe_results.csv", index=False)
print("RFE Feature Selection completed. Results saved to rfe_results.csv")
