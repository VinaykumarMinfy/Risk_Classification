# 💼 Bank Loan Classification

This project implements an end-to-end machine learning pipeline to predict whether a customer will opt for a personal loan based on demographic and financial attributes. It includes detailed EDA, feature engineering, preprocessing, model training, hyperparameter tuning, and MLflow-based model tracking.

---

## 📊 Features Used

| Type          | Features                                      |
|---------------|-----------------------------------------------|
| Demographic   | `Age`, `Experience`, `Education`, `Family`    |
| Financial     | `Income`, `Mortgage`, `CCAvg`                 |
| Behavioral    | `Online`, `CreditCard`                        |
| Target        | `Personal Loan (0/1)`                         |

---

## 🧪 Workflow Summary

### 1. Exploratory Data Analysis (EDA)
- Visualize target distribution
- Histograms and boxplots for numeric features
- Pairplots and correlation heatmap
- Categorical features vs. target analysis (`Education`, `Family`, `Online`, `CreditCard`)

### 2. Preprocessing
- Skewness correction using `PowerTransformer`
- Feature scaling using `RobustScaler` and `StandardScaler`
- Transformers saved using `joblib`

### 3. Feature Selection
- Recursive Feature Elimination (RFE) using Logistic Regression

### 4. Handling Class Imbalance
- Applied SMOTE to oversample the minority class (loan = 1)

### 5. Model Training & Hyperparameter Tuning
Used `GridSearchCV` on:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

### 6. MLflow Tracking
- Metrics logged: Accuracy, Precision, Recall, F1-score
- Best model:
  - Registered as `BankLoanBestModel`
  - Promoted to `Production` stage

---

## 🚀 How to Run the Project

### ✅ Step 1: Install Dependencies

pip install -r requirements.txt
✅ Step 2: Run the Jupyter Notebook

jupyter notebook Assignment.ipynb
✅ Step 3: Execute Application Scripts

python server.py
python Client.py
🖥️ Web Interface Options
🔷 Flask-based UI:
python app.py
🟢 Streamlit-based UI:
streamlit run streamlit_app.py
