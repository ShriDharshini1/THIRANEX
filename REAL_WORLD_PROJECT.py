'''REAL WORLD PROJECT'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("loan_dataset.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

print("\nStatistical Summary")
print(df.describe())

plt.figure(figsize=(8,5))
sns.histplot(df["Income"], bins=5, kde=True)
plt.title("Income Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["Credit_Score"], bins=5, kde=True)
plt.title("Credit Score Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Loan_Status", data=df)
plt.title("Loan Approval Status")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(
    x="Income",
    y="Loan_Amount",
    hue="Loan_Status",
    data=df
)
plt.title("Income vs Loan Amount")
plt.show()

approved_income = df.groupby("Loan_Status")["Income"].mean()

plt.figure(figsize=(6,4))
approved_income.plot(kind="bar")
plt.title("Average Income by Loan Status")
plt.ylabel("Income")
plt.show()

correlation = df[["Age", "Income", "Loan_Amount", "Credit_Score"]].corr()

print("\nCorrelation Matrix")
print(correlation)

plt.figure(figsize=(8,5))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\nAverage Loan Amount")
print(df["Loan_Amount"].mean())

print("\nAverage Credit Score")
print(df["Credit_Score"].mean())

print("\nLoan Status Count")
print(df["Loan_Status"].value_counts())

print("\nAverage Income by Loan Status")
print(df.groupby("Loan_Status")["Income"].mean())

print("\nAverage Credit Score by Loan Status")
print(df.groupby("Loan_Status")["Credit_Score"].mean())

print("\nProject Completed Successfully")



'''

SAMPLE DATASET --> loan_dataset.csv

Customer_ID,Age,Income,Loan_Amount,Credit_Score,Loan_Status
C001,25,30000,100000,650,Approved
C002,40,60000,200000,750,Approved
C003,22,25000,150000,580,Rejected
C004,35,50000,180000,720,Approved
C005,28,28000,120000,600,Rejected
C006,45,70000,250000,800,Approved
C007,30,35000,130000,620,Rejected
C008,50,80000,300000,820,Approved
C009,27,32000,110000,610,Rejected
C010,38,55000,190000,740,Approved
C011,24,27000,115000,590,Rejected
C012,42,65000,220000,780,Approved
C013,29,34000,125000,630,Rejected
C014,47,72000,260000,810,Approved
C015,26,29000,118000,605,Rejected
C016,39,58000,195000,750,Approved
C017,31,36000,135000,640,Rejected
C018,44,68000,240000,790,Approved
C019,23,26000,105000,570,Rejected
C020,41,62000,210000,770,Approved



'''







