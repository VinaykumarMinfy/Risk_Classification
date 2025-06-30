import streamlit as st
import pandas as pd
import joblib
import warnings
import io
from sklearn.exceptions import InconsistentVersionWarning

# Suppress common sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Page settings
st.set_page_config(page_title="Bank Loan Predictor", layout="wide")
st.title("🏦 Bank Customer Loan Classifier")
st.markdown("Upload a bank customer dataset to classify Personal Loan eligibility using multiple ML models.")

# Load all models
@st.cache_resource
def load_models():
    model_paths = {
        'DecisionTree': 'DecisionTree_model.pkl',
        'RandomForest': 'RandomForest_model.pkl',
        'GradientBoosting': 'GradientBoosting_model.pkl',
        'KNN': 'KNN_model.pkl',
        'SVM': 'SVM_model.pkl',
        'LogisticRegression': 'LogisticRegression_model.pkl',
    }

    loaded_models = {}
    for name, path in model_paths.items():
        try:
            loaded_models[name] = joblib.load(path)
        except Exception as e:
            st.warning(f"⚠️ Failed to load {name}: {e}")
    return loaded_models

models = load_models()

# Define the expected features (must match training)
feature_cols = [
    'Age',
    'Experience',
    'Income',
    'Family',
    'CCAvg',
    'Education',
    'Mortgage',
    'Securities Account'
]

# Upload file
uploaded_file = st.file_uploader("📁 Upload a CSV file", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded and read successfully.")

        # Check for required columns
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            X = df[feature_cols]
            results_df = df.copy()

            # Apply each model
            for name, model in models.items():
                try:
                    results_df[f'{name}_Prediction'] = model.predict(X)
                except Exception as e:
                    results_df[f'{name}_Prediction'] = f'Error: {e}'

            # Display results
            st.subheader("📊 Prediction Results")
            st.dataframe(results_df, use_container_width=True)

            # CSV download
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="loan_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f" Could not process the file: {e}")
