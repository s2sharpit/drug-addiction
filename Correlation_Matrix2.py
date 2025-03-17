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
print(corr_with_target)
print("Correlation_higest value: \n",max(corr_with_target))
print("Correlation_minimum value: \n",min(corr_with_target))

#Write a code to find the three most important features according to the order of their values