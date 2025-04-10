import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# Load datasets
train_df = pd.read_csv("D:/practice_models/train_drop_records.csv")
X_train = train_df.drop(columns=["Addiction_Class"])
y_train = train_df["Addiction_Class"]

# Mutual Information Feature Selection
mi_scores = mutual_info_classif(X_train, y_train)
mi_results = pd.DataFrame({"Feature": X_train.columns, "Mutual Info Score": mi_scores}).sort_values(by="Mutual Info Score", ascending=False)

# Save results
mi_results.to_csv("mutual_info_results.csv", index=False)
print("Mutual Information Feature Selection completed. Results saved to mutual_info_results.csv")
