# 🫀 Heart Disease Prediction — Machine Learning Project

A supervised machine learning pipeline that predicts the likelihood of heart disease in patients using clinical features. The project covers the full ML workflow: exploratory data analysis, preprocessing, model training, evaluation, and individual patient risk scoring.

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── heart_disease_.ipynb       # Main Jupyter Notebook (full pipeline)
├── Dataset/
│   └── heart_data.csv         # Input dataset (place here before running)
└── README.md                  # Project documentation
```

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Project Pipeline](#project-pipeline)
6. [Models Used](#models-used)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Patient Risk Prediction](#patient-risk-prediction)
9. [Feature Descriptions](#feature-descriptions)
10. [Requirements](#requirements)

---

## 🔍 Overview

Heart disease is one of the leading causes of death globally. This project builds a binary classification system that predicts whether a patient has heart disease (`target = 1`) or not (`target = 0`) based on clinical measurements. Four machine learning algorithms are trained, compared, and the best-performing model is used for final predictions.

---

## 📊 Dataset

- **File:** `heart_data.csv`
- **Target column:** `target` (0 = No Disease, 1 = Disease)
- **Format:** CSV with clinical feature columns

The dataset is expected to follow the structure of the classic UCI Heart Disease dataset. Place the file at:

```
E:/Major Project/Dataset/heart_data.csv
```

> **Note:** Update the `__path__` variable in the notebook's *Load Dataset* cell if your file is stored in a different location.

### Feature Descriptions

| Feature     | Description                                               | Type        |
|-------------|-----------------------------------------------------------|-------------|
| `age`       | Age of the patient (years)                                | Numeric     |
| `gender`    | Sex (0 = Female, 1 = Male)                                | Categorical |
| `cp`        | Chest pain type (0–3)                                     | Categorical |
| `trestbps`  | Resting blood pressure (mm Hg)                            | Numeric     |
| `chol`      | Serum cholesterol (mg/dl)                                 | Numeric     |
| `fbs`       | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)     | Categorical |
| `restecg`   | Resting ECG results (0–2)                                 | Categorical |
| `thalach`   | Maximum heart rate achieved                               | Numeric     |
| `exang`     | Exercise-induced angina (1 = Yes, 0 = No)                 | Categorical |
| `oldpeak`   | ST depression induced by exercise relative to rest        | Numeric     |
| `slope`     | Slope of peak exercise ST segment (0–2)                   | Categorical |
| `ca`        | Number of major vessels colored by fluoroscopy (0–4)      | Categorical |
| `thal`      | Thalassemia type (0–3)                                    | Categorical |
| `target`    | Heart disease diagnosis (0 = No, 1 = Yes)                 | Target      |

---

## ⚙️ Installation

### 1. Clone or download the project

```bash
git clone https://github.com/sai-unknown/Heart-Disease-Prediction
cd heart-disease-prediction
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

Or install everything at once if a `requirements.txt` is provided:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

1. Place `heart_data.csv` in the correct directory (update the path in the notebook if needed).
2. Open the notebook:

```bash
jupyter notebook heart_disease_.ipynb
```

3. Run all cells from top to bottom (`Kernel → Restart & Run All`).

---

## 🔄 Project Pipeline

The notebook is organized into clearly labeled sections:

### 1. 📦 Import Libraries
Loads all necessary Python libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, and `xgboost`.

### 2. 📂 Load Dataset
Reads `heart_data.csv` into a Pandas DataFrame and displays its shape and first few rows.

### 3. 🔍 Exploratory Data Analysis (EDA)

Visualizations and analysis covering:
- **Missing Value Analysis** — Counts and percentage bar chart of missing data per column
- **Target Class Distribution** — Count plot showing disease vs. no-disease balance
- **Age Distribution** — Histogram with KDE curve
- **Gender Distribution** — Count plot (Male vs. Female)
- **Cholesterol & Blood Pressure Distributions** — Histograms for `chol` and `trestbps`
- **Chest Pain Type vs Heart Disease** — Bar chart of disease rate by chest pain category
- **Max Heart Rate vs Target** — Box plot comparing `thalach` between disease groups
- **Correlation Heatmap** — Heatmap of all numeric feature correlations

### 4. ⚙️ Data Preprocessing & Feature Engineering

- **Separate Features & Target** — Splits `X` (features) and `y` (target)
- **Handle Missing Values** — Numeric columns filled with median; categorical columns filled with mode
- **Identify Feature Types** — Automatically separates numeric and categorical columns based on unique value counts
- **One-Hot Encoding** — Applies `pd.get_dummies()` to categorical columns with `drop_first=True` to avoid multicollinearity

### 5. 🧠 Train-Test Split
Splits data into 80% training and 20% testing using stratified sampling to maintain class balance.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 6. 📏 Feature Scaling
Applies `StandardScaler` to normalize numeric features. Scaler is fit only on training data and applied to both train and test sets (for models that require scaling: Logistic Regression, SVM).

### 7. 🤖 Model Training
Four classifiers are configured and trained (see [Models Used](#models-used) below).

### 8. 📊 Model Evaluation
All models are evaluated using a unified `evaluate_model()` function. Results are compiled into a comparison DataFrame.

### 9. 📈 Visualization & Best Model Selection

- Bar chart comparing ROC AUC scores across all models
- ROC Curve for the best-performing model
- Classification Report (precision, recall, F1 per class)
- Feature Importance chart (for tree-based models)

### 10. 🩺 Patient Risk Prediction
A `predict_patient_risk()` function accepts a dictionary of patient values and returns:
- Predicted class (0 or 1)
- Probability of disease
- Risk label: **Low Risk**, **Moderate Risk**, or **High Risk**

---

## 🤖 Models Used

| Model                   | Scaling Required | Key Hyperparameters                                                    |
|-------------------------|------------------|------------------------------------------------------------------------|
| **Logistic Regression** | ✅ Yes           | `max_iter=2000`, `class_weight='balanced'`                             |
| **SVM (RBF Kernel)**    | ✅ Yes           | `kernel='rbf'`, `probability=True`, `class_weight='balanced'`          |
| **Random Forest**       | ❌ No            | `n_estimators=400`, `class_weight='balanced_subsample'`                |
| **XGBoost**             | ❌ No            | `n_estimators=600`, `max_depth=4`, `learning_rate=0.05`, auto-balanced |

All models use `class_weight` balancing or `scale_pos_weight` (XGBoost) to handle any class imbalance in the dataset.

---

## 📏 Evaluation Metrics

Each model is evaluated on the held-out test set using:

| Metric        | Description                                              |
|---------------|----------------------------------------------------------|
| **Accuracy**  | Proportion of correct predictions                        |
| **Precision** | Of predicted positives, how many are truly positive      |
| **Recall**    | Of actual positives, how many were correctly identified  |
| **F1 Score**  | Harmonic mean of Precision and Recall                    |
| **ROC AUC**   | Area under the ROC curve; robust to class imbalance      |

---

## 🩺 Patient Risk Prediction

The `predict_patient_risk()` function enables real-time inference for a new patient:

```python
example_patient = {
    "age": 55,
    "gender": 1,        # 1 = Male
    "cp": 2,            # Chest pain type
    "trestbps": 140,    # Resting blood pressure
    "chol": 250,        # Cholesterol
    "fbs": 0,           # Fasting blood sugar
    "restecg": 1,       # Resting ECG
    "thalach": 150,     # Max heart rate
    "exang": 0,         # Exercise angina
    "oldpeak": 1.4,     # ST depression
    "slope": 1,
    "ca": 0,
    "thal": 2
}

pred, prob, label = predict_patient_risk(example_patient)
print("Predicted Target:", pred)
print("Probability:", round(prob, 4))
print("Risk Level:", label)
```

### Risk Level Thresholds

| Probability Range | Risk Label       |
|-------------------|-----------------|
| ≥ 0.80            | 🔴 High Risk     |
| 0.50 – 0.79       | 🟡 Moderate Risk |
| < 0.50            | 🟢 Low Risk      |

---

## 📦 Requirements

```
Python >= 3.8
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

Install all at once:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**. The model outputs should **not** be used as a substitute for professional medical diagnosis. Always consult a qualified healthcare provider for medical decisions.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
