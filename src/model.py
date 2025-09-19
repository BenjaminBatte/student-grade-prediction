import os
import joblib
import json
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_model(X, y, pipeline, test_size=0.2, random_state=42):
    """Train pipeline with train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test

def evaluate_model(model, X_test, y_test, metrics_path=None, dataset_name="dataset"):
    """Evaluate model with R², RMSE, MAE."""
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "r2_score": r2_score(y_test, y_pred),
        "rmse": sqrt(mse),
        "mae": mean_absolute_error(y_test, y_pred)
    }

    print(f"Model Performance on {dataset_name}: {metrics}")

    if metrics_path:
        os.makedirs(metrics_path, exist_ok=True)
        out_file = os.path.join(metrics_path, f"{dataset_name}_metrics.json")
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=4)

    return metrics

def save_model(model, path, filename="model.pkl"):
    """Save full pipeline to disk."""
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, filename)
    joblib.dump(model, file_path)
    print(f"Model saved at: {file_path}")
