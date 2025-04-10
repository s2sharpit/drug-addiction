import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set style for the plots
sns.set_style("whitegrid")

train_df = pd.read_csv("D:/Drug_Prediction_ML/train_drop_records.csv")
# Plot the distribution of the target variable
plt.figure(figsize=(6, 4)) #(width, height)
sns.countplot(x=train_df["Addiction_Class"], palette="coolwarm") #creates a bar chart showing the count of each unique value(plots the counts of each class) 
plt.title("Distribution of Addiction_Class")
plt.xlabel("Addiction Class (0: No, 1: Yes)")
plt.ylabel("Count")
plt.show()
