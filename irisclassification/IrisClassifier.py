# iris_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("✅ Script started...")


# Load dataset
data = pd.read_csv('Iris.csv')

# Drop 'Id' column if exists
data = data.drop(columns=['Id'])

# Encode labels
le = LabelEncoder()
data['Species'] = le.fit_transform(data['Species'])

# Split features and labels
X = data.drop('Species', axis=1)
y = data['Species']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
print("\n📊 Close the plot window to continue with manual prediction...\n")
plt.show()

# Manual input for custom prediction
print("\n🌸 Predict the species of a new Iris flower:")
try:
    sepal_length = float(input("Enter Sepal Length (cm): "))
    sepal_width = float(input("Enter Sepal Width (cm): "))
    petal_length = float(input("Enter Petal Length (cm): "))
    petal_width = float(input("Enter Petal Width (cm): "))

    # Make prediction
    custom_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    custom_pred = model.predict(custom_data)
    predicted_species = le.inverse_transform(custom_pred)

    print(f"✅ Predicted Species: {predicted_species[0]}")

except ValueError:
    print("❌ Invalid input. Please enter numeric values only.")
