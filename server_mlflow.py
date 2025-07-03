import subprocess

def serve_model(model_uri="models:/BankLoanBestModel/Production", host="127.0.0.1", port=5001):
    try:
        subprocess.run([
            "mlflow", "models", "serve",
            "-m", model_uri,
            "--host", host,
            "--port", str(port),
            "--env-manager=local"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(" Failed to start MLflow server:", e)

serve_model()
