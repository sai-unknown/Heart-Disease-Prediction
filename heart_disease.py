# ⚙️ Install Requirements
# Run in terminal:
# pip install -r requirements.txt

from itertools import count
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    RocCurveDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# 📂 Load Dataset
__path__ = "E:/Major Project/Dataset/"
csv_file = "heart_data.csv"

df = pd.read_csv(__path__ + csv_file)

print("Dataset shape:", df.shape)
print(df.head())


# 🔍 EDA
print("\nMissing values (count):")
print(df.isna().sum())

print("\nMissing values (%)")
print((df.isna().mean() * 100).round(2).sort_values(ascending=False))

plt.figure(figsize=(10,4))
missing_pct = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
sns.barplot(x=missing_pct.index, y=missing_pct.values)
plt.title("Missing Values (%) by Columns")
plt.tight_layout()
plt.show()

print("\nTarget balance:")
print(df["target"].value_counts())

plt.figure(figsize=(5,4))
sns.countplot(x="target", data=df)
plt.title("Class Balance")
plt.show()

if "age" in df.columns:
    sns.histplot(df["age"], kde=True)
    plt.title("Age Distribution")
    plt.show()


# ⚙️ Preprocessing
target_col = "target"
X_raw = df.drop(columns=[target_col]).copy()
y = df[target_col].astype(int)

# Missing values
for c in X_raw.columns:
    if X_raw[c].isna().sum() > 0:
        if pd.api.types.is_numeric_dtype(X_raw[c]):
            X_raw[c] = X_raw[c].fillna(X_raw[c].median())
        else:
            X_raw[c] = X_raw[c].fillna(X_raw[c].mode().iloc[0])

# Feature types
categorical_cols = []
numeric_cols = []

for c in X_raw.columns:
    if pd.api.types.is_numeric_dtype(X_raw[c]):
        if X_raw[c].nunique() <= 10:
            categorical_cols.append(c)
        else:
            numeric_cols.append(c)
    else:
        categorical_cols.append(c)

# Encoding
X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 🤖 Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}


# 📊 Evaluation
def evaluate_model(name, model, scaled=False):
    if scaled:
        model.fit(X_train_scaled, y_train)
        prob = model.predict_proba(X_test_scaled)[:,1]
    else:
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:,1]

    pred = (prob >= 0.5).astype(int)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC_AUC": roc_auc_score(y_test, prob)
    }


results = []
results.append(evaluate_model("Logistic Regression", models["Logistic Regression"], True))
results.append(evaluate_model("SVM", models["SVM"], True))
results.append(evaluate_model("Random Forest", models["Random Forest"]))
results.append(evaluate_model("XGBoost", models["XGBoost"]))

results_df = pd.DataFrame(results)
print(results_df)


# 🏆 Best model
best_name = results_df.sort_values("ROC_AUC", ascending=False).iloc[0]["Model"]
best_model = models[best_name]

print("Best model:", best_name)


# 📉 ROC Curve
if best_name in ["Logistic Regression", "SVM"]:
    best_model.fit(X_train_scaled, y_train)
    RocCurveDisplay.from_estimator(best_model, X_test_scaled, y_test)
else:
    best_model.fit(X_train, y_train)
    RocCurveDisplay.from_estimator(best_model, X_test, y_test)

plt.show()


# 📋 Report
if best_name in ["Logistic Regression", "SVM"]:
    print(classification_report(y_test, best_model.predict(X_test_scaled)))
else:
    print(classification_report(y_test, best_model.predict(X_test)))


# 🧠 Prediction Function
def predict_patient_risk(patient_dict):
    row = pd.DataFrame([patient_dict])
    row_enc = pd.get_dummies(row)

    for col in X.columns:
        if col not in row_enc.columns:
            row_enc[col] = 0

    row_enc = row_enc[X.columns]

    if best_name in ["Logistic Regression", "SVM"]:
        row_enc = scaler.transform(row_enc)

    prob = best_model.predict_proba(row_enc)[0,1]
    pred = int(prob >= 0.5)

    return pred, prob


# Example
sample = {
    "age": 55,
    "gender": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 250
}

print(predict_patient_risk(sample))