# [30072 rows x 11 columns]
import pandas as pd
import numpy as np

df = pd.read_csv("D:/Drug_Prediction_ML/student_addiction_dataset_train.csv")
print(df)

#print('Shape: ',df.shape)
rows, columns = df.shape
#print('rows & columns: ', rows, columns)
#print('Describe: ', df.describe())
print()

print("Prints all column names:\n ", df.columns)

drop_rows = df.dropna()
print("Rows with a null value in any of the columns will be dropped:\n", drop_rows)

def convert_experimentation(cell):
    if cell == "Yes":
        return "1"
    elif cell == "No":
        return "0"

def convert_academic_performance_decline(cell):
    if cell == "Yes":
        return "1"
    elif cell =="No":
        return "0"
    
def convert_social_isolation(cell):
    if cell == "Yes":
        return "1"
    elif cell =="No":
        return "0"
    
def convert_financial_issues(cell):
    if cell == "Yes":
        return "1"
    elif cell =="No":
        return "0"
    
def convert_physical_mental_health_problems(cell):
    if cell == "Yes":
        return "1"
    elif cell =="No":
        return "0"
    
def convert_legal_consequences(cell):
    if cell == "Yes":
        return "1"
    elif cell =="No":
        return "0"
    
def convert_relationship_strain(cell):
    if cell == "Yes":
        return "1"
    elif cell =="No":
        return "0"
    
def convert_risk_taking_behavior(cell):
    if cell == "Yes":
        return "1"
    elif cell =="No":
        return "0"
    
def convert_withdrawal_symptoms(cell):
    if cell == "Yes":
        return "1"
    elif cell =="No":
        return "0"
    
def convert_denial_and_resistance_to_treatment(cell):
    if cell == "Yes":
        return "1"
    elif cell =="No":
        return "0"
    
def convert_addiction_class(cell):
    if cell == "Yes":
        return "1"
    elif cell =="No":
        return "0"
    
Encod_Categorize_rows = pd.read_csv("D:/Drug_Prediction_ML/student_addiction_dataset_train.csv", converters= {
    'Experimentation': convert_experimentation,
    'Academic_Performance_Decline' : convert_academic_performance_decline,
    'Social_Isolation' : convert_social_isolation,
    'Financial_Issues' : convert_financial_issues,
    'Physical_Mental_Health_Problems' : convert_physical_mental_health_problems,
    'Legal_Consequences' : convert_legal_consequences,
    'Relationship_Strain' : convert_relationship_strain,
    'Risk_Taking_Behavior' : convert_risk_taking_behavior,
    'Withdrawal_Symptoms' : convert_withdrawal_symptoms,
    'Denial_and_Resistance_to_Treatment' : convert_denial_and_resistance_to_treatment,
    'Addiction_Class' : convert_addiction_class
})

print(Encod_Categorize_rows) #Encoding categorizing variables
Train_drop_rows = Encod_Categorize_rows.dropna() #Rows with null values will be dropped
print(Train_drop_rows)

#Train_drop_rows.to_csv("D:/Drug_Prediction_ML/train_drop_records.csv", index= False)
