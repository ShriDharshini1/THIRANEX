'''
sales_data.csv:
Order_ID,Date,Region,Category,Sales,Profit
1001,2025-01-05,North,Technology,1200,300
1002,2025-01-08,South,Furniture,850,120
1003,2025-01-10,East,Office Supplies,400,80
1004,2025-01-12,West,Technology,1500,450
1005,2025-01-15,North,Furniture,700,100
1006,2025-01-18,South,Office Supplies,350,70
1007,2025-01-20,East,Technology,1800,500
1008,2025-01-22,West,Furniture,950,150
1009,2025-01-25,North,Office Supplies,300,60
1010,2025-01-28,South,Technology,2200,650
1011,2025-02-01,East,Furniture,780,110
1012,2025-02-04,West,Office Supplies,450,90
1013,2025-02-07,North,Technology,1600,400
1014,2025-02-10,South,Furniture,920,140
1015,2025-02-12,East,Office Supplies,500,100
1016,2025-02-15,West,Technology,2100,600
1017,2025-02-18,North,Furniture,680,95
1018,2025-02-20,South,Office Supplies,420,85
1019,2025-02-23,East,Technology,1900,550
1020,2025-02-26,West,Furniture,1000,170'''





# CODE 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("sales_data.csv")

print(df.head())
print(df.info())
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# Sales by Category
sns.barplot(x='Category', y='Sales', data=df)
plt.title("Sales by Category")
plt.show()

# Profit by Region
df.groupby('Region')['Profit'].sum().plot(kind='bar')
plt.title("Profit by Region")
plt.show()

# Sales Distribution
sns.histplot(df['Sales'], kde=True)
plt.title("Sales Distribution")
plt.show()

# Correlation Heatmap
sns.heatmap(df[['Sales','Profit']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()
