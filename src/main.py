import argparse
import os
import pandas as pd
from data_loader import load_data
from utils import get_logger
from preprocessing import build_preprocessor
from model import train_model, evaluate_model, save_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

logger = get_logger(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def run_pipeline(dataset: str, model_name: str):
    mat, por = load_data()
    df = mat if dataset == "math" else por

    if df is None or "G3" not in df.columns:
        logger.error("Dataset not loaded or target 'G3' missing.")
        return

    y = df["G3"]
    X = df.drop("G3", axis=1)

    # Identify column types
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Build preprocessing
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # Choose model
    if model_name == "random_forest":
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "linear_regression":
        regressor = LinearRegression()
    else:
        logger.error(f"Unsupported model: {model_name}")
        return

    # Build full pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", regressor)
    ])

    # Train
    logger.info(f"Training {model_name} on {dataset} dataset...")
    pipeline, X_test, y_test = train_model(X, y, pipeline)

    # Evaluate
    metrics_path = os.path.join(RESULTS_DIR, "metrics")
    evaluate_model(pipeline, X_test, y_test, metrics_path=metrics_path, dataset_name=dataset)

    # Save model
    models_path = os.path.join(RESULTS_DIR, "models")
    save_model(pipeline, models_path, filename=f"{model_name}_{dataset}.pkl")

    logger.info(f"Pipeline completed for {dataset} with {model_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student Grade Prediction Pipeline")
    parser.add_argument("--dataset", choices=["math", "portuguese"], required=True)
    parser.add_argument("--model", choices=["random_forest", "linear_regression"], required=True)
    args = parser.parse_args()
    run_pipeline(args.dataset, args.model)
