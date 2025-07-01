import requests
import json
import joblib
import pandas as pd
import os

MLFLOW_URL = "http://127.0.0.1:5001/invocations"
SELECTOR_PATH = "saved_models/selector.pkl"
CSV_FILE = "New_Customer_Bank_Personal_Loan.csv" 

if not os.path.exists(SELECTOR_PATH):
    raise FileNotFoundError(f"❌ Selector not found at {SELECTOR_PATH}")
selector = joblib.load(SELECTOR_PATH)

all_columns = [
    "Age", "Experience", "Income", "Family", "CCAvg",
    "Education", "Mortgage", "Securities Account", "CD Account",
    "Online", "CreditCard"
]

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"❌ CSV file not found at {CSV_FILE}")

df_input = pd.read_csv(CSV_FILE)

missing_cols = set(all_columns) - set(df_input.columns)
if missing_cols:
    raise ValueError(f"❌ Missing columns in input CSV: {missing_cols}")

df_input = df_input[all_columns]  

X_selected = selector.transform(df_input)

if hasattr(selector, "get_support"):
    selected_indices = selector.get_support(indices=True)
    selected_columns = [all_columns[i] for i in selected_indices]
else:
    selected_columns = [f"feature_{i}" for i in range(X_selected.shape[1])]

payload = {
    "dataframe_split": {
        "columns": selected_columns,
        "data": X_selected.tolist()
    }
}

headers = {"Content-Type": "application/json"}
try:
    response = requests.post(MLFLOW_URL, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    predictions = response.json()
    print("✅ Predictions:")
    print(predictions)

    df_input["Prediction"] = predictions
    df_input.to_csv("predicted_output.csv", index=False)
    print("📁 Saved predictions to predicted_output.csv")

except requests.exceptions.RequestException as e:
    print("❌ Request failed:", e)
    if response.content:
        print("🔍 Response content:", response.content.decode())
