import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Computing Correlation matrix
train_df = pd.read_csv("D:/Drug_Prediction_ML/train_drop_records.csv")
corr_matrix = train_df.corr()

#Plotting Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidth=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

#Sorting features based on correlation with the target variable
corr_with_target = corr_matrix["Addiction_Class"].drop("Addiction_Class").sort_values(ascending=False)
# corr_matrix["Addiction_Class"]- Pulls the correlation values of all features with Addiction_Class from the correlation matrix
# .drop("Addiction_Class") - Removes the target’s correlation with itself (which is always 1)
# .sort_values(ascending=False)- Sorts features by descending correlation with the target. So the most positively features come first.

print(corr_with_target)
print("Correlation_higest value: \n",max(corr_with_target))
print("Correlation_minimum value: \n",min(corr_with_target))

#The highest positive correlation with Addiction_Class is Physical_Mental_Health_Problems (0.00798).
#The lowest correlation is Withdrawal_Symptoms (-0.00639).
#Overall, correlations are weak, indicating that individual features may not have strong predictive power alone.