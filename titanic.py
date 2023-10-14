import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree  


data = pd.read_csv("train.csv")
data.dropna(subset=["Age"], inplace=True)
data["Fare"].fillna(data["Fare"].mean(), inplace=True)
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
data = pd.get_dummies(data, columns=["Sex", "Pclass", "Embarked"], drop_first=True)
data.drop(["Name", "Cabin", "Ticket"], axis=1, inplace=True)


scaler = StandardScaler()
data[["Age", "Fare", "FamilySize"]] = scaler.fit_transform(data[["Age", "Fare", "FamilySize"]])

X = data.drop("Survived", axis=1)
y = data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Decision Tree Model
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt.fit(X_train, y_train)
plt.figure(figsize=(15, 10))
plot_tree(clf_dt, filled=True, feature_names=X.columns, class_names=["Not Survived", "Survived"])
plt.show(block=False)


input("Press Enter to continue...")


y_pred_dt = clf_dt.predict(X_test)

# Calculate accuracy
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Model:")
print(f"Accuracy: {accuracy_dt:.2f}")

# Apply 5-fold cross-validation for the Decision Tree Model
cross_val_scores_dt = cross_val_score(clf_dt, X, y, cv=5, scoring='accuracy')

# Calculate the average classification accuracy for the Decision Tree Model
average_accuracy_dt = np.mean(cross_val_scores_dt)
print(f"Average Classification Accuracy (5-Fold Cross-Validation - Decision Tree): {average_accuracy_dt:.2f}")

# Random Forest Model
# Train a Random Forest classifier
clf_rf = RandomForestClassifier(random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

# Calculate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("\nRandom Forest Model:")
print(f"Accuracy: {accuracy_rf:.2f}")

# Apply 5-fold cross-validation for the Random Forest Model
cross_val_scores_rf = cross_val_score(clf_rf, X, y, cv=5, scoring='accuracy')

# Calculate the average classification accuracy for the Random Forest Model
average_accuracy_rf = np.mean(cross_val_scores_rf)
print(f"Average Classification Accuracy (5-Fold Cross-Validation - Random Forest): {average_accuracy_rf:.2f}")
