import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

print("=" * 60)
print("CREDIT CARD FRAUD DETECTION - MODEL TRAINING")
print("=" * 60)

# Make required folders
os.makedirs('models', exist_ok=True)

# ✅ Correct path check only (don't recreate data folder here)
data_path = os.path.join("data", "creditcard.csv")

print("\n[1/6] Loading dataset...")

if not os.path.exists(data_path):
    print("✗ ERROR: creditcard.csv not found inside /data folder!")
    print("Download dataset from Kaggle:")
    print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("Place it here: data/creditcard.csv")
    exit(1)

df = pd.read_csv(data_path)
print(f"✓ Dataset loaded successfully!")
print(f"  Total transactions: {len(df)}")
print(f"  Fraudulent: {df['Class'].sum()} ({(df['Class'].sum()/len(df)*100):.4f}%)")
print(f"  Legitimate: {len(df) - df['Class'].sum()} ({((len(df)-df['Class'].sum())/len(df)*100):.4f}%)")

# Prepare features and labels
X = df.drop('Class', axis=1)
y = df['Class']

print("\n[2/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("✓ Data split completed")
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

print("\n[3/6] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Feature scaling completed")

# Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(probability=True, kernel='rbf')
}

results = {}
best_model = None
best_f1 = 0

print("\n[4/6] Training models...")

for name, model in models.items():
    print(f"\n→ Training {name}...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)

    results[name] = {
        'accuracy': round(accuracy*100, 2),
        'precision': round(precision*100, 2),
        'recall': round(recall*100, 2),
        'f1_score': round(f1*100, 2)
    }

    print(f"  ✓ {name} Completed: F1 Score = {results[name]['f1_score']}%")

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_name = name

print("\n[5/6] Saving best model...")
print(f"Best Model: {best_name} (F1 Score = {best_f1*100:.2f}%)")

joblib.dump(best_model, "models/model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(results, "models/metrics.joblib")

print("✓ Model saved successfully!")

print("\n[6/6] Confusion Matrix:")

y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)

print(f"  TN: {cm[0][0]}")
print(f"  FP: {cm[0][1]}")
print(f"  FN: {cm[1][0]}")
print(f"  TP: {cm[1][1]}")

print("\nALL MODEL SCORES:")
for name, metrics in results.items():
    print(f"\n{name}")
    for key, value in metrics.items():
        print(f"  {key.capitalize()}: {value}%")

print("\n✓ Training Complete — Run your app now:")
print("python app.py")
