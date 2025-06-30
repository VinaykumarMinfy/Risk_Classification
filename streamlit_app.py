import streamlit as st
import pandas as pd
import joblib
import io

# Page config
st.set_page_config(page_title="Bank Loan Classifier", layout="centered")

st.title(" Bank Customer Loan Classifier")
st.markdown("Upload your CSV file and get predictions from multiple models.")

# Load models
@st.cache_resource
def load_models():
    model_files = {
        'DecisionTree': 'DecisionTree_model.pkl',
        'RandomForest': 'RandomForest_model.pkl',
        'GradientBoosting': 'GradientBoosting_model.pkl',
        'KNN': 'KNN_model.pkl',
        'SVM': 'SVM_model.pkl',
        'LogisticRegression': 'LogisticRegression_model.pkl',
    }
    models = {}
    for name, path in model_files.items():
        try:
            models[name] = joblib.load(path)
        except Exception as e:
            st.error(f"Error loading {name} model: {e}")
    return models

models = load_models()

# Define the expected features used in training
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

# File uploader
uploaded_file = st.file_uploader(" Upload a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("CSV loaded successfully!")

        # Check feature columns
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            X = df[feature_cols]
            results_df = df.copy()

            for name, model in models.items():
                try:
                    results_df[f'{name}_Prediction'] = model.predict(X)
                except Exception as e:
                    results_df[f'{name}_Prediction'] = f'Prediction error: {e}'

            st.subheader(" Prediction Results")
            st.dataframe(results_df, use_container_width=True)

            # Download link
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label=" Download Results as CSV",
                data=csv_buffer.getvalue(),
                file_name='predicted_results.csv',
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"Could not process the file: {e}")
