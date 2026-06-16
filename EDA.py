import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("hospital_management.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nPatients Per Department:")
print(df["Department"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x="Department", data=df)
plt.title("Patients by Department")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df["Age"], bins=5, kde=True)
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df["Bill_Amount"], bins=5, kde=True)
plt.title("Bill Amount Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=df["Days_Admitted"])
plt.title("Days Admitted")
plt.show()

department_revenue = df.groupby("Department")["Bill_Amount"].sum()

print("\nRevenue By Department:")
print(department_revenue)

department_revenue.plot(kind="bar")
plt.title("Revenue by Department")
plt.ylabel("Bill Amount")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(
    x="Days_Admitted",
    y="Bill_Amount",
    data=df
)
plt.title("Days Admitted vs Bill Amount")
plt.show()

correlation = df[["Age", "Days_Admitted", "Bill_Amount"]].corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(6,4))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\nAverage Bill Amount:")
print(df["Bill_Amount"].mean())

print("\nMaximum Bill Amount:")
print(df["Bill_Amount"].max())

print("\nDepartment With Highest Revenue:")
print(department_revenue.idxmax())

print("\nEDA Completed Successfully!")


'''

SAMPLE DATASET --> hospital_management.csv

Patient_ID,Age,Department,Days_Admitted,Bill_Amount
P001,25,General,2,5000
P002,40,Cardiology,5,15000
P003,30,Orthopedics,3,8000
P004,55,Cardiology,7,22000
P005,22,General,1,3000
P006,45,Neurology,6,18000
P007,35,Orthopedics,4,10000
P008,60,Cardiology,8,25000
P009,28,General,2,4500
P010,50,Neurology,5,17000
P011,32,Orthopedics,3,9000
P012,65,Cardiology,9,28000
P013,24,General,1,3500
P014,48,Neurology,6,19000
P015,38,Orthopedics,4,11000'''
