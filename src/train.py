"""Train and evaluate a Random Forest Iris classifier; log with MLflow."""
from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from src.data_preprocessing import load_dataset, FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "iris_model.joblib"
METRICS_PATH = ROOT / "metrics.json"


def train_model(random_state: int = 42):
    df = load_dataset()
    X = df[FEATURES]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    mlflow.set_experiment("iris-flower-classification")

    with mlflow.start_run():
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        mlflow.log_artifact(str(METRICS_PATH))

    print(classification_report(
        y_test,
        predictions,
        target_names=["setosa", "versicolor", "virginica"]
    ))
    print(f"Saved model to {MODEL_PATH}\nMetrics: {metrics}")
    return model, metrics


if __name__ == "__main__":
    train_model()
