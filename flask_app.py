from flask import Flask, render_template, request
import pandas as pd
import joblib
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress sklearn-related warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)

# Model file paths
model_files = {
    'DecisionTree': 'DecisionTree_model.pkl',
    'RandomForest': 'RandomForest_model.pkl',
    'GradientBoosting': 'GradientBoosting_model.pkl',
    'KNN': 'KNN_model.pkl',
    'SVM': 'SVM_model.pkl',
    'LogisticRegression': 'LogisticRegression_model.pkl',
}

# Load all models
models = {}
for name, path in model_files.items():
    try:
        models[name] = joblib.load(path)
        print(f"[INFO] Loaded model: {name}")
    except Exception as e:
        print(f"[ERROR] Failed to load model {name}: {e}")

# Feature columns used during model training
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods=['POST'])
def success():
    file = request.files.get('file')
    if not file:
        return "No file uploaded.", 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return f"Failed to read CSV file: {e}", 400

    try:
        X = df[feature_cols]
    except KeyError as e:
        return f"Missing expected columns: {e}", 400

    results_df = df.copy()

    for name, model in models.items():
        try:
            results_df[f'{name}_Prediction'] = model.predict(X)
        except Exception as e:
            results_df[f'{name}_Prediction'] = f'Prediction error: {e}'

    # Convert results to HTML table
    table_html = results_df.to_html(classes='table', index=False)

    return render_template('data.html', Y=table_html)

if __name__ == '__main__':
    app.run(debug=True)
