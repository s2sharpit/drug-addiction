import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load datasets
train_df = pd.read_csv("D:/practice_models/train_drop_records.csv")
X_train = train_df.drop(columns=["Addiction_Class"])
y_train = train_df["Addiction_Class"]

# L1 Regularization (Lasso Regression)
lasso_model = LogisticRegression(penalty='l1', solver='liblinear')
lasso_model.fit(X_train, y_train)
lasso_coefficients = np.abs(lasso_model.coef_)[0]
lasso_results = pd.DataFrame({"Feature": X_train.columns, "L1 Coefficient": lasso_coefficients}).sort_values(by="L1 Coefficient", ascending=False)

# Save results
lasso_results.to_csv("l1_regularization_results.csv", index=False)
print("L1 Regularization Feature Selection completed. Results saved to l1_regularization_results.csv")
