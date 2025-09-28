import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import load_data
from utils import get_logger
from preprocessing import build_preprocessor
from model import train_model, evaluate_model, save_model
from eda import plot_distributions, plot_correlation_heatmap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import time

logger = get_logger(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def run_pipeline(dataset: str, model_name: str):
    """Run the full ML pipeline for a given dataset and model."""
    start_time = time.time()

    try:
        # Load dataset
        mat, por = load_data()
        df = mat if dataset == "math" else por

        if df is None or df.empty:
            logger.error(f"Failed to load {dataset} dataset")
            return None
        
        if "G3" not in df.columns:
            logger.error("Target column 'G3' missing from dataset")
            return None

        y = df["G3"]
        X = df.drop("G3", axis=1)

        # Identify column types
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

        # Build preprocessing pipeline
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)

        # Choose model
        if model_name == "random_forest":
            regressor = RandomForestRegressor(
                n_estimators=100, 
                random_state=42, 
                n_jobs=1
            )
        elif model_name == "linear_regression":
            regressor = LinearRegression()
        else:
            logger.error(f"Unsupported model: {model_name}")
            return None

        # Build full pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", regressor)
        ])

        # Train and evaluate
        logger.info(f"Training {model_name} on {dataset} dataset...")
        pipeline, X_test, y_test = train_model(X, y, pipeline)

        metrics_path = os.path.join(RESULTS_DIR, "metrics")
        metrics = evaluate_model(
            pipeline, X_test, y_test,
            metrics_path=metrics_path,
            dataset_name=f"{dataset}_{model_name}"
        )

        # Save model
        models_path = os.path.join(RESULTS_DIR, "models")
        save_model(
            pipeline,
            models_path,
            filename=f"{model_name}_{dataset}.pkl"
        )

        elapsed = round(time.time() - start_time, 2)
        logger.info(
            f"✅ Pipeline completed in {elapsed}s. "
            f"MAE: {metrics['mae']}, R²: {metrics['r2']}"
        )

        return metrics   # ✅ return metrics for printing later

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise
