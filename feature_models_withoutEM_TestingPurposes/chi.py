import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest

# Load datasets
train_df = pd.read_csv("D:/practice_models/train_drop_records.csv")
X_train = train_df.drop(columns=["Addiction_Class"])
y_train = train_df["Addiction_Class"]

# Chi-Square Feature Selection
chi_selector = SelectKBest(score_func=chi2, k='all')
chi_selector.fit(X_train, y_train)
chi_scores = chi_selector.scores_
chi_results = pd.DataFrame({"Feature": X_train.columns, "Chi2 Score": chi_scores}).sort_values(by="Chi2 Score", ascending=False)

# Save results
chi_results.to_csv("chi_square_results.csv", index=False)
print("Chi-Square Feature Selection completed. Results saved to chi_square_results.csv")

