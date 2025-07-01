# 🏦 Bank Loan Classification

An end-to-end machine learning project to predict whether a customer will accept a personal loan based on demographic, financial, and behavioral features. The pipeline includes EDA, preprocessing, model training, hyperparameter tuning, and MLflow-based model tracking.

---

## 📌 Project Highlights

- 🔍 Exploratory Data Analysis (EDA)
- 🧼 Data Preprocessing (PowerTransformer, Scalers)
- 🧠 Feature Selection with RFE
- ⚖️ Class Imbalance Handling using SMOTE
- 🤖 Model Training with GridSearchCV on 6 algorithms
- 📈 MLflow Tracking with model registry and promotion
- 🌐 Flask and Streamlit Web Interfaces

---

## 🧾 Features

**Input Features:**
- Demographic: `Age`, `Experience`, `Education`, `Family`
- Financial: `Income`, `Mortgage`, `CCAvg`
- Behavioral: `Online`, `CreditCard`

**Target Variable:**  
- `Personal Loan` (0 = No, 1 = Yes)

---

## ⚙️ ML Workflow

### 1. EDA
- Visuals: Histograms, boxplots, pairplots, correlation heatmap
- Categorical analysis vs target

### 2. Preprocessing
- Skewness correction: `PowerTransformer`
- Scaling: `RobustScaler`, `StandardScaler`
- Saved as `.joblib` files

### 3. Feature Selection
- RFE using Logistic Regression

### 4. Imbalance Handling
- SMOTE for oversampling

### 5. Model Training (with GridSearchCV)
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- Support Vector Machine

### 6. MLflow Tracking
- Metrics: Accuracy, Precision, Recall, F1
- Best model:
  - Registered as: `BankLoanBestModel`
  - Promoted to: **Production**

---

## 🚀 How to Run

### 📦 1. Install Requirements
```bash
pip install -r requirements.txt
