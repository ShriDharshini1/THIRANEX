import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

# Load dataset
data = pd.read_csv("student_performance.csv")

# Features and Target
X = data[["Hours_Studied", "Attendance", "Previous_Score"]]
y = data["Passed"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

best_model = None
best_accuracy = 0

print("Model Results")
print("-" * 30)

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"{name}: {accuracy:.2f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Confusion Matrix
y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# ROC Curve
y_prob = best_model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.show()

print("\nBest Accuracy:", best_accuracy)



'''
SAMPLE DATASET --> student_performance.csv

Hours_Studied,Attendance,Previous_Score,Passed
2,60,45,0
3,65,50,0
4,70,55,0
5,75,60,1
6,80,65,1
7,85,70,1
8,90,75,1
1,55,40,0
2,58,42,0
3,62,48,0
4,68,52,0
5,72,58,1
6,78,64,1
7,82,68,1
8,88,74,1
9,92,80,1
2,61,46,0
3,66,51,0
4,71,56,1
5,76,61,1
6,81,66,1
7,86,71,1
8,91,76,1
1,54,39,0
2,59,44,0
3,64,49,0
4,69,54,1
5,74,59,1
6,79,63,1
7,84,69,1
8,89,73,1
9,94,82,1
2,63,47,0
3,67,53,1
4,72,57,1
5,77,62,1
6,83,67,1
7,87,72,1
8,93,78,1
1,53,38,0'''
