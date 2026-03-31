from fastapi import FastAPI
from app.schema import LoanRequest
import mlflow.pyfunc
import numpy as np
from mlflow.tracking import MlflowClient
import os

app = FastAPI(title="Loan Prediction API (Best Model)")

MODEL_NAME = "LoanBestModel"

# Optionally set MLflow tracking URI from env or default to local
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Flag to detect if a server is running (for models:/) or use local path
USE_SERVER = MLFLOW_TRACKING_URI.startswith("http")

def get_best_model():
    """
    Return the best model by accuracy.
    Works with MLflow server or local mlruns folder.
    """
    try:
        if USE_SERVER:
            # Load from model registry (models:/)
            versions = client.get_latest_versions(MODEL_NAME)
            if not versions:
                return None, None, None, None

            best_version = None
            best_acc = -1
            for v in versions:
                run = client.get_run(v.run_id)
                acc = run.data.metrics.get("accuracy", 0)
                if acc > best_acc:
                    best_acc = acc
                    best_version = v

            model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{best_version.version}")
            model_name = client.get_run(best_version.run_id).data.params.get("model_name", "Unknown")
            return model, best_version.version, best_version.run_id, model_name

        else:
            # Load locally from mlruns folder
            experiments = client.list_experiments()
            if not experiments:
                return None, None, None, None

            best_acc = -1
            best_model = None
            best_run_id = None
            best_model_name = None

            for exp in experiments:
                runs = client.search_runs(exp.experiment_id)
                for run in runs:
                    acc = run.data.metrics.get("accuracy", 0)
                    if acc > best_acc:
                        best_acc = acc
                        best_run_id = run.info.run_id
                        best_model_name = run.data.params.get("model_name", "Unknown")

            if best_run_id is None:
                return None, None, None, None

            # Local artifact path
            model_path = f"./mlruns/0/{best_run_id}/artifacts/model"
            model = mlflow.pyfunc.load_model(model_path)
            return model, "local", best_run_id, best_model_name

    except Exception as e:
        print("Error loading model:", e)
        return None, None, None, None


@app.get("/")
def home():
    model, version, run_id, model_name = get_best_model()
    return {
        "message": "Loan Prediction API",
        "best_model_name": model_name,
        "best_model_version": version,
        "mlflow_run_id": run_id
    }


@app.post("/predict")
def predict(request: LoanRequest):
    model, version, run_id, model_name = get_best_model()
    if model is None:
        return {"error": "No model available"}

    data = np.array([[request.income, request.loan_amount, request.credit_score]])
    prediction = model.predict(data)

    return {
        "loan_approved": int(prediction[0]),
        "best_model_name": model_name,
        "best_model_version": version,
        "mlflow_run_id": run_id
    }

