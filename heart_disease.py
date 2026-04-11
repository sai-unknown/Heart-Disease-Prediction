# ==============================
# 📦 IMPORTS & SETUP
# ==============================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    RocCurveDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ==============================
# 📂 LOAD DATA
# ==============================
path = "./Dataset/"   # change if needed
csv_file = "heart_data.csv" 

df = pd.read_csv(path + csv_file)

print("Dataset shape:", df.shape)
print(df.head())

# ==============================
# 🔍 EDA
# ==============================
print("\nMissing values:")
print(df.isna().sum())

plt.figure(figsize=(10,4))
missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
sns.barplot(x=missing_pct.index, y=missing_pct.values)
plt.xticks(rotation=60)
plt.title("Missing Values (%)")
plt.show()

print("\nTarget Distribution:")
print(df["target"].value_counts())

sns.countplot(x="target", data=df)
plt.title("Target Distribution")
plt.show()

# ==============================
# ⚙️ PREPROCESSING
# ==============================
target_col = "target"

X_raw = df.drop(columns=[target_col]).copy()
y = df[target_col].astype(int)

# Handle missing values
for c in X_raw.columns:
    if X_raw[c].isna().sum() > 0:
        if pd.api.types.is_numeric_dtype(X_raw[c]):
            X_raw[c] = X_raw[c].fillna(X_raw[c].median())
        else:
            X_raw[c] = X_raw[c].fillna(X_raw[c].mode().iloc[0])

# Identify columns
categorical_cols = []
numeric_cols = []

for c in X_raw.columns:
    if pd.api.types.is_numeric_dtype(X_raw[c]):
        if X_raw[c].nunique() <= 10 and c not in ["age","trestbps","chol","thalach","oldpeak"]:
            categorical_cols.append(c)
        else:
            numeric_cols.append(c)
    else:
        categorical_cols.append(c)

# Encoding
X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)

print("Encoded shape:", X.shape)

# ==============================
# 🧠 TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 🤖 MODELS
# ==============================
models = {}

models["Logistic Regression"] = LogisticRegression(max_iter=2000, class_weight="balanced")
models["SVM"] = SVC(kernel="rbf", probability=True, class_weight="balanced")
models["Random Forest"] = RandomForestClassifier(n_estimators=400)

pos = (y_train == 1).sum()
neg = (y_train == 0).sum()

models["XGBoost"] = XGBClassifier(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.05,
    eval_metric="logloss",
    scale_pos_weight=(neg/pos)
)

# ==============================
# 📊 EVALUATION
# ==============================
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
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC_AUC": roc_auc_score(y_test, prob)
    }

results = []
results.append(evaluate_model("Logistic Regression", models["Logistic Regression"], True))
results.append(evaluate_model("SVM", models["SVM"], True))
results.append(evaluate_model("Random Forest", models["Random Forest"]))
results.append(evaluate_model("XGBoost", models["XGBoost"]))

results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)

print("\nModel Comparison:\n", results_df)

# Plot
plt.bar(results_df["Model"], results_df["ROC_AUC"])
plt.title("ROC-AUC Comparison")
plt.xticks(rotation=20)
plt.show()

# Best model
best_name = results_df.iloc[0]["Model"]
best_model = models[best_name]

print("\nBest Model:", best_name)

# ==============================
# 📉 ROC CURVE
# ==============================
if best_name in ["Logistic Regression","SVM"]:
    best_model.fit(X_train_scaled, y_train)
    RocCurveDisplay.from_estimator(best_model, X_test_scaled, y_test)
else:
    best_model.fit(X_train, y_train)
    RocCurveDisplay.from_estimator(best_model, X_test, y_test)

plt.show()

# ==============================
# 📋 CLASSIFICATION REPORT
# ==============================
if best_name in ["Logistic Regression","SVM"]:
    print(classification_report(y_test, best_model.predict(X_test_scaled)))
else:
    print(classification_report(y_test, best_model.predict(X_test)))

# ==============================
# 🧠 PREDICTION FUNCTION
# ==============================
def predict_patient_risk(patient_dict, threshold=0.5):

    row = pd.DataFrame([patient_dict])

    for c in X_raw.columns:
        if c not in row.columns:
            row[c] = np.nan

    for c in row.columns:
        if row[c].isna().sum() > 0:
            if pd.api.types.is_numeric_dtype(X_raw[c]):
                row[c] = row[c].fillna(X_raw[c].median())
            else:
                row[c] = row[c].fillna(X_raw[c].mode().iloc[0])

    row_enc = pd.get_dummies(row, columns=categorical_cols, drop_first=True)

    for col in X.columns:
        if col not in row_enc.columns:
            row_enc[col] = 0

    row_enc = row_enc[X.columns]

    if best_name in ["Logistic Regression","SVM"]:
        row_scaled = scaler.transform(row_enc)
        prob = best_model.predict_proba(row_scaled)[0,1]
    else:
        prob = best_model.predict_proba(row_enc)[0,1]

    pred = int(prob >= threshold)

    if prob >= 0.8:
        risk_label = "🔴 High Risk"
    elif prob >= 0.5:
        risk_label = "🟡 Moderate Risk"
    else:
        risk_label = "🟢 Low Risk"

    return pred, prob, risk_label

# ==============================
# 👤 USER INPUT SYSTEM
# ==============================
def get_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except:
            print("Invalid input!")

def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except:
            print("Invalid input!")

def user_input():
    return {
        "name": input("Name of the Patient:"),
        "age": get_int("Age: "),
        "gender": get_int("Gender (0=F,1=M): "),
        "cp": get_int("Chest Pain (1-4): "),
        "trestbps": get_int("BP: "),
        "chol": get_int("Cholesterol: "),
        "fbs": get_int("FBS (0/1): "),
        "restecg": get_int("RestECG (0-2): "),
        "thalach": get_int("Max HR: "),
        "exang": get_int("Exang (0/1): "),
        "oldpeak": get_float("Oldpeak: "),
        "slope": get_int("Slope (1-3): "),
        "ca": get_int("CA (0-3): "),
        "thal": get_int("Thal (1-3): ")
    }

# ==============================
# 🧪 RUN PREDICTION
# ==============================
print("\nEnter Patient Details:")
patient = user_input()

pred, prob, label = predict_patient_risk(patient)

output = [(patient["name"], pred, prob, label)]
output_df = pd.DataFrame(output, columns=["Name", "Prediction", "Probability", "Risk Level"])
print("\n 📄 DataFrame Report:")
print(output_df)