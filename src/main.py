import argparse
import os
import matplotlib
# Set the backend to Agg (non-interactive) before importing pyplot
matplotlib.use('Agg')  # This must be done before importing pyplot
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

# ------------------------------------------------------------
# Initialize logger and project directories
# ------------------------------------------------------------
logger = get_logger(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def run_pipeline(dataset: str, model_name: str):
    """
    Run the full ML pipeline for a given dataset and model.

    Steps:
        1. Load dataset (math or portuguese).
        2. Split into features (X) and target (y = G3).
        3. Identify numeric vs categorical features.
        4. Build preprocessing transformer.
        5. Select and initialize the model.
        6. Train pipeline with train/test split.
        7. Evaluate model and save metrics.
        8. Save trained model (versioned + latest).

    Args:
        dataset (str): "math" or "portuguese".
        model_name (str): "random_forest" or "linear_regression".
    """
    start_time = time.time()

    # --- Load dataset ---
    mat, por = load_data()
    df = mat if dataset == "math" else por

    if df is None or "G3" not in df.columns:
        logger.error("Dataset not loaded or target 'G3' missing.")
        return

    y = df["G3"]          # Target variable (final grade)
    X = df.drop("G3", axis=1)  # Features

    # --- Identify column types ---
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # --- Build preprocessing ---
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    # --- Choose model ---
    if model_name == "random_forest":
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "linear_regression":
        regressor = LinearRegression()
    else:
        logger.error(f"Unsupported model: {model_name}")
        return

    # --- Build full pipeline (preprocessor + model) ---
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", regressor)
    ])

    # --- Train ---
    logger.info(f"Training {model_name} on {dataset} dataset...")
    pipeline, X_test, y_test = train_model(X, y, pipeline)

    # --- Evaluate ---
    metrics_path = os.path.join(RESULTS_DIR, "metrics")
    metrics = evaluate_model(
        pipeline, X_test, y_test,
        metrics_path=metrics_path,
        dataset_name=f"{dataset}_{model_name}"
    )

    # --- Save trained model ---
    models_path = os.path.join(RESULTS_DIR, "models")
    save_model(
        pipeline,
        models_path,
        filename=f"{model_name}_{dataset}.pkl"
    )

    # --- Log completion ---
    elapsed = round(time.time() - start_time, 2)
    logger.info(
        f"✅ Pipeline completed for {dataset} with {model_name} "
        f"in {elapsed} seconds. Metrics: {metrics}"
    )


# ------------------------------------------------------------
# CLI Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    # Simple automation - just run everything
    logger.info("🚀 Starting automated ML pipeline...")
    
    # Run EDA
    mat, por = load_data()
    if mat is not None:
        logger.info("📊 Generating EDA plots for Math dataset...")
        try:
            plot_distributions(mat, dataset_name="math")
            plot_correlation_heatmap(mat, dataset_name="math")
            logger.info("✅ Math EDA completed")
        except Exception as e:
            logger.error(f"❌ Math EDA failed: {e}")
    if por is not None:  
        logger.info("📊 Generating EDA plots for Portuguese dataset...")
        try:
            plot_distributions(por, dataset_name="portuguese")
            plot_correlation_heatmap(por, dataset_name="portuguese")
            logger.info("✅ Portuguese EDA completed")
        except Exception as e:
            logger.error(f"❌ Portuguese EDA failed: {e}")
    
    # Train all combinations
    configurations = [
        ("math", "random_forest"),
        ("math", "linear_regression"), 
        ("portuguese", "random_forest"),
        ("portuguese", "linear_regression")
    ]
    
    logger.info("🤖 Training all model-dataset combinations...")
    for i, (dataset, model) in enumerate(configurations, 1):
        logger.info(f"🔧 Training {i}/{len(configurations)}: {model} on {dataset}")
        run_pipeline(dataset, model)
    
    logger.info("🎉 All done! Check results/ folder for outputs.")
    logger.info("📁 Models saved in: results/models/")
    logger.info("📊 Metrics saved in: results/metrics/")
    logger.info("📈 Plots saved in: results/figures/")