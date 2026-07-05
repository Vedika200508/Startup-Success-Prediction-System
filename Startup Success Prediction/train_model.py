# =============================================================
#   Startup Success Prediction — Model Training Script
#   Trains a Random Forest classifier on dataset.csv
#   and saves the model as startup_model.pkl
# =============================================================

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ─────────────────────────────────────────────
#  STEP 1 — Load Dataset
# ─────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH   = os.path.join(BASE_DIR, "startup_model.pkl")

print("=" * 55)
print("  Startup Success Prediction — Model Training")
print("=" * 55)

print("\n[1/5] Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"      Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

# ─────────────────────────────────────────────
#  STEP 2 — Drop Irrelevant Columns
# ─────────────────────────────────────────────

print("\n[2/5] Preprocessing data...")

drop_columns = [
    "Unnamed: 0", "state_code", "latitude", "longitude", "zip_code",
    "id", "city", "Unnamed: 6", "name", "labels",
    "founded_at", "closed_at", "first_funding_at", "last_funding_at",
    "state_code.1", "object_id", "category_code"
]

for col in drop_columns:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# ─────────────────────────────────────────────
#  STEP 3 — Encode Target Column
#  acquired / operating / ipo  → 1 (Success)
#  closed                      → 0 (Failure)
# ─────────────────────────────────────────────

df["status"] = df["status"].map({
    "closed":    0,
    "acquired":  1,
    "operating": 1,
    "ipo":       1
})

# Drop rows where status is still NaN after mapping
df = df.dropna(subset=["status"])
df["status"] = df["status"].astype(int)

# Fill remaining NaN values with 0
df = df.fillna(0)

print(f"      Success (1): {df['status'].sum()}  |  Failure (0): {(df['status'] == 0).sum()}")
print(f"      Features used: {df.shape[1] - 1}")

# ─────────────────────────────────────────────
#  STEP 4 — Split into Train & Test Sets
# ─────────────────────────────────────────────

print("\n[3/5] Splitting dataset (80% train / 20% test)...")

X = df.drop("status", axis=1)
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"      Training samples : {len(X_train)}")
print(f"      Testing  samples : {len(X_test)}")

# ─────────────────────────────────────────────
#  STEP 5 — Train Random Forest Model
# ─────────────────────────────────────────────

print("\n[4/5] Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,    # 100 decision trees
    max_depth=10,        # max depth per tree
    random_state=42
)

model.fit(X_train, y_train)
print("      Training complete.")

# ─────────────────────────────────────────────
#  STEP 6 — Evaluate Model
# ─────────────────────────────────────────────

print("\n[5/5] Evaluating model on test data...")

y_pred = model.predict(X_test)

accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

print(f"\n      Accuracy : {accuracy}%")
print("\n      Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"      TN={cm[0][0]}  FP={cm[0][1]}")
print(f"      FN={cm[1][0]}  TP={cm[1][1]}")

print("\n      Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Failure", "Success"]))

# ─────────────────────────────────────────────
#  STEP 7 — Save Model
# ─────────────────────────────────────────────

joblib.dump(model, MODEL_PATH)
print(f"\n  Model saved → {MODEL_PATH}")

# ─────────────────────────────────────────────
#  STEP 8 — Save Model Report
# ─────────────────────────────────────────────

REPORT_PATH = os.path.join(BASE_DIR, "model_report.txt")

report_lines = [
    "=" * 55,
    "  Startup Success Prediction — Model Report",
    "=" * 55,
    "",
    "DATASET",
    f"  Total Records     : {df.shape[0]}",
    f"  Features Used     : {df.shape[1] - 1}",
    f"  Success (1)       : {df['status'].sum()}",
    f"  Failure (0)       : {(df['status'] == 0).sum()}",
    "",
    "TRAIN / TEST SPLIT",
    f"  Training Samples  : {len(X_train)}  (80%)",
    f"  Testing  Samples  : {len(X_test)}  (20%)",
    "",
    "MODEL CONFIGURATION",
    "  Algorithm         : Random Forest Classifier",
    "  Number of Trees   : 100",
    "  Max Depth         : 10",
    "  Random State      : 42",
    "",
    "EVALUATION RESULTS",
    f"  Accuracy          : {accuracy}%",
    "",
    "CONFUSION MATRIX",
    "                  Predicted",
    "                  Failure   Success",
    f"  Actual Failure :   {cm[0][0]}        {cm[0][1]}",
    f"  Actual Success :   {cm[1][0]}        {cm[1][1]}",
    "",
    "  TN (True Negative)  = " + str(cm[0][0]) + "  → Correctly predicted Failure",
    "  FP (False Positive) = " + str(cm[0][1]) + "  → Predicted Success, actually Failure",
    "  FN (False Negative) = " + str(cm[1][0]) + "   → Predicted Failure, actually Success",
    "  TP (True Positive)  = " + str(cm[1][1]) + "  → Correctly predicted Success",
    "",
    "CLASSIFICATION REPORT",
    classification_report(y_test, y_pred, target_names=["Failure", "Success"]),
    "TOP 10 IMPORTANT FEATURES",
]

feature_names = X.columns.tolist()
importances = model.feature_importances_
feat_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
for i, (feat, imp) in enumerate(feat_pairs, 1):
    report_lines.append(f"  {i:2}. {feat:<35} {round(imp * 100, 2)}%")

report_lines += [
    "",
    "=" * 55,
    "  Model saved as  : startup_model.pkl",
    "  Report saved as : model_report.txt",
    "=" * 55,
]

with open(REPORT_PATH, "w") as f:
    f.write("\n".join(report_lines))

print(f"  Report saved → {REPORT_PATH}")
print("=" * 55)
print("  Done! Run app.py to launch the Streamlit app.")
print("=" * 55)
