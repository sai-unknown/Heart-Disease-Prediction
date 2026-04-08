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

# --------------------------------------------------------------------------
# Load Data
# --------------------------------------------------------------------------
__path__ = "E:/Major Project/Dataset/"
csv_file = "heart_data.csv"
df = pd.read_csv(__path__ + csv_file)


print("Dataset shape:", df.shape)
print(df.head())

# --------------------------------------------------------------------------
# EDA
# Missing Values analysis
# --------------------------------------------------------------------------

print("\n 1) Missing values (count):")
print(df.isna().sum())

print("\n Missing values (%)")
print((df.isna().mean() * 100).round(2).sort_values(ascending=False))

plt.figure(figsize=(10,4))
missing_pct = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
sns.barplot(x = missing_pct.index, y = missing_pct.values)
plt.title("Missing Values (%) by Columns")
plt.ylabel("Missing %")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------
# Class balance (target distributoion)

print("\n2) Target balance (0=no diseace, 1 = disease):")
print(df["target"].value_counts())
print(df["target"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

plt.figure(figsize=(5,4))
sns.countplot(x="target", data=df)
plt.title("Class Balance: Heart Disease Target")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------
#Age and gender distributoion
if "age" in df.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df["age"], bins=30, kde=True)
    plt.title("Age Distributoion")
    plt.tight_layout()
    plt.show()

if "gender" in df.columns:
    plt.figure(figsize=(5,4))
    sns.countplot(x = "gender", data=df)
    plt.title("Gender Distributoion (0/1)")
    plt.tight_layout()
    plt.show()

# Cholestrol and resting blood pressure distributoions
for col in ["chol", "trestbps"]:
    if col in df.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f"Distributoion: {col}")
        plt.tight_layout()
        plt.show()

# Chest pain type vs disease presence
if "cp" in df.columns:
    cp_rate = df.groupby("cp")["target"].mean().sort_values(ascending=False)
    plt.figure(figsize=(6,4))
    sns.barplot(x=cp_rate.index.astype(str), y=cp_rate.values)
    plt.title("Heart Disease Rate by Chest Pain Type (cp)")
    plt.xlabel("cp")
    plt.ylabel("Disease Rate")
    plt.tight_layout()
    plt.show()

# Maximum heart rate vs disease outcome
if "thalach" in df.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="target", y="thalach", data=df)
    plt.title("Max Heart Rate (thalach) vs Target")
    plt.tight_layout()
    plt.show()

# Correlation heatmap for numeric features
num_cols = df.select_dtypes(include=[np.number]).columns.to_list()
plt.figure(figsize=(10,7))
sns.heatmap(df[num_cols].corr(), cmap="coolwarm", annot=False,
            linewidths=0.2)
plt.title("Correlation heatmap (Numeric Features)")
plt.tight_layout()
plt.show()

# Outliers in cholesterol and resting blood  prerssure
for col in ["chol", "trestbps"]:
    if col in df.columns:
        plt.figure(figsize=(6,3))

        sns.boxplot(x=df[col])

        plt.title(f"Outliers in {col}")

        plt.tight_layout()
        plt.show()

# Exercise-induced angina and ST depression effects
if "exang" in df.columns:
    exang_rate = df.groupby("exang")["target"].mean().sort_values(ascending=False)
    plt.figure(figsize=(5,4))
    sns.barplot(x=exang_rate.index.astype(str), y=exang_rate.values)
    plt.title("Heart Disease Rate by Exercise Induced Angina (exang)")
    plt.xlabel("Exercise Induced Angina")
    plt.ylabel("Heart Disease Rate")
    plt.tight_layout()
    plt.show()

if "oldpeak" in df.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="target", y="oldpeak", data=df)
    plt.title("ST Depression (oldpeak) vs Target")
    plt.tight_layout()
    plt.show()

# Feature importance using tree-based model

# --------------------------------------------------------------------------
# Preprocessing + Feature Engineering
# --------------------------------------------------------------------------

target_col = "target"

# Separate X (features) and y (target)
X_raw = df.drop(columns=[target_col]).copy()
y = df[target_col].astype(int)

# Impute missing values (robust even if dataset has non-numeric columns)
for c in X_raw.columns:
    if X_raw[c].isna().sum() > 0:
        if pd.api.types.is_numeric_dtype(X_raw[c]):
            X_raw[c] = X_raw[c].fillna(X_raw[c].median())
        else:
            X_raw[c] = X_raw[c].fillna(X_raw[c].mode().iloc[0])

# Identify low-cardinality categoricals (<= 10 unique values)
categorical_cols = []
numeric_cols = []

for c in X_raw.columns:
    if pd.api.types.is_numeric_dtype(X_raw[c]):
        if X_raw[c].nunique() <= 10 and c not in ["age", "chol", "trestbps", "thalach", "oldpeak"]:
            categorical_cols.append(c)
        else:
            numeric_cols.append(c)
    else:
        # Object categorical columns
        categorical_cols.append(c)

# One-hot encode categorical columns (low-cardinality)
X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)

print("\nAfter encoding, X shape:", X.shape)

# --------------------------------------------------------------------------
# Train-test split
# --------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric features for logistic regression and SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------------------------------
# Training Models
# --------------------------------------------------------------------------
models = {}

# Logistic Regression (scaled)
models["Logistic Regression"] = LogisticRegression(max_iter=2000,class_weight="balanced", random_state=42)

# Support Vector Machine (scaled)
models["SVM (RBF)"] = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)

# Random Forest (unscaled)
models["Random Forest"] = RandomForestClassifier(
    n_estimators=400, 
    class_weight="balanced_subsample", 
    random_state=42
    )

# XGBoost Classifier (unscaled)
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = neg / pos if pos > 0 else 1.0
models["XGBoost"] = XGBClassifier(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight
)

# ----------------------------------------------------------------------------------
# Evaluate + Compare
# ----------------------------------------------------------------------------------
def evaluate_model(name, model, scaled=False):
    # Train
    if scaled:
        model.fit(X_train_scaled, y_train)
        prob = model.predict_proba(X_test_scaled) [:, 1]
    else:
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test) [:, 1]

    pred = (prob >= 0.5).astype(int)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, prob)
    }
results = []
results.append(evaluate_model("Logistic Regression", models["Logistic Regression"], scaled=True))
results.append(evaluate_model("SVM (RBF)", models["SVM (RBF)"], scaled=True))
results.append(evaluate_model("Random Forest", models["Random Forest"], scaled=False))
results.append(evaluate_model("XGBoost", models["XGBoost"], scaled=False))
results_df = pd.DataFrame(results).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
print("\n--- Model Comparison (sorted by ROC_AUC)---")
print(results_df)

plt.figure(figsize=(8,4))
plt.bar(results_df["Model"], results_df["ROC AUC"])
plt.title("Model Comparison by ROC-AUC")
plt.ylabel("ROC AUC Score")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

best_name = results_df.iloc[0]["Model"]
best_model = models[best_name]
print(f"\nBest model: {best_name} with ROC AUC: {results_df.iloc[0]['ROC AUC']:.4f}")

# -----------------------------------------------------------------------------------
# ROC Curve for best model
# -----------------------------------------------------------------------------------
if best_name in ["Logistic Regression", "SVM (RBF)"]:
    best_model.fit(X_train_scaled, y_train)
    RocCurveDisplay.from_estimator(best_model, X_test_scaled, y_test)
else:
    best_model.fit(X_train, y_train)
    RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.title(f"ROC Curve - Best Model: {best_name}")
plt.tight_layout()
plt.show()

# Print classification report for best model
print(f"\nClassification Report for Best Model: {best_name}")
if best_name in ["Logistic Regression", "SVM (RBF)"]:
    print(classification_report(y_test, best_model.predict(X_test_scaled)))
else:
    print(classification_report(y_test, best_model.predict(X_test)))

# -----------------------------------------------------------------------------------
# Feature importance for tree-based models
# -----------------------------------------------------------------------------------
# If best model is not tree-based, we still train a random forest for explainability
explain_model = None
if best_name in ["Random Forest", "XGBoost"]:
    explain_model = best_model
else:
    explain_model = models["Random Forest"]
    explain_model.fit(X_train, y_train)

if hasattr(explain_model, "feature_importances_"):
    importances = pd.Series(explain_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 15 Feature Importances:")
    print(importances.head(15))

    plt.figure(figsize=(9,5))
    importances.head(15).sort_values().plot(kind="barh")
    plt.title(f"Top Feature Importances (Tree-based)")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------------
# Risk Prediction / Propensity Scoring + Single Patitent Prediction
# -----------------------------------------------------------------------------------
def predict_patient_risk(patient_dict, threshold=0.5):
    """
    Input: patient_dict (dictionary of features:value using ORIGINAL column names)
   
    Output: predicted_class (0/1), probability (0..1), risk_label
    """
    row = pd.DataFrame([patient_dict])

    #Ensure all original columns are exist
    for c in X_raw.columns:
        if c not in row.columns:
            row[c] = np.nan
    
    # Impute missing in the input row
    for c in row.columns:
        if row[c].isna().sum() > 0:
            if c in X_raw.columns and pd.api.types.is_numeric_dtype(X_raw[c]):
                row[c] = row[c].fillna(X_raw[c].median())
            else:
                # mode for categorical
                row[c] = row[c].fillna(X_raw[c].mode().iloc[0]) if c in X_raw.columns else 0

    # One-hot encode the row to match training columns
    row_enc = pd.get_dummies(row, columns=categorical_cols, drop_first=True)

    # Add missing dummy columns
    for col in X.columns:
        if col not in row_enc.columns:
            row_enc[col] = 0
    row_enc = row_enc[X.columns]  # Ensure same column order

    # Predict
    if best_name in ["Logistic Regression", "SVM (RBF)"]:
        row_scaled = scaler.transform(row_enc)
        prob = best_model.predict_proba(row_scaled) [0,1]
    else:
        prob = best_model.predict_proba(row_enc) [0,1]

    pred_class = int(prob >= threshold)

    # Simple risk label
    if prob >= 0.8:
        risk_label = "High Risk"
    elif prob >= 0.5:
        risk_label = "Moderate Risk"
    else:
        risk_label = "Low Risk"

    return pred_class, float(prob), risk_label

print("\n--- Example: Single Patient Risk Prediction ---")
example_patient = {
    "age": 55,
    "gender": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.4,
    "slope": 1,
    "ca": 0,
    "thal": 2
}

pred, prob, label = predict_patient_risk(example_patient, threshold=0.5)
print("Predicted Target (0=No Disease, 1=Disease):", pred)
print("Predicted Probability of Heart Disease:", round(prob, 4))
print("Risk Label:", label)
