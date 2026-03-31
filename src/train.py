import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("loan_prediction")
MODEL_NAME = "LoanBestModel"

# Load dataset
df = pd.read_csv("data/loan_data.csv")
X = df[["income", "loan_amount", "credit_score"]]
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Models to try
models = {
    "LogisticRegression": LogisticRegression(),
    "DecisionTree": DecisionTreeClassifier(max_depth=5),
    "RandomForest": RandomForestClassifier(n_estimators=100)
}

best_acc = 0
best_model_name = None
best_run_id = None

# Train and log models
for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print(f"{name} Accuracy:", acc)

        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_run_id = run.info.run_id

# Register the best model automatically
from mlflow.tracking import MlflowClient
client = MlflowClient()

model_uri = f"runs:/{best_run_id}/model"

try:
    client.create_registered_model(MODEL_NAME)
except:
    pass  # Already exists

model_version = mlflow.register_model(model_uri, MODEL_NAME)
print(f"Best model: {best_model_name}, Version: {model_version.version}")
