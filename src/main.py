import argparse
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (prevents GUI popups when saving plots)
import matplotlib.pyplot as plt

# --- Project imports ---
from data_loader import load_data                     # Load datasets (Math & Portuguese)
from utils import get_logger                          # Custom logger (console + file)
from preprocessing import build_preprocessor          # ColumnTransformer (scaling + encoding)
from model import train_model, evaluate_model, save_model  # Training, evaluation, persistence
from eda import plot_distributions, plot_correlation_heatmap
from sklearn.ensemble import RandomForestRegressor    # Tree-based ensemble model
from sklearn.linear_model import LinearRegression     # Simple baseline model
from sklearn.pipeline import Pipeline                 # Combine preprocessing + model
import time

# --- Global variables ---
logger = get_logger(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def run_pipeline(dataset: str, model_name: str):
    """
    Run the complete machine learning pipeline for one dataset-model combination.
    
    Args:
        dataset (str): Which dataset to use ("math" or "portuguese").
        model_name (str): Which model to train ("random_forest" or "linear_regression").

    Returns:
        dict: Evaluation metrics (MAE, RMSE, R², etc.) for the trained model.
    """
    start_time = time.time()

    try:
        # -----------------------------------------------------------------
        # STEP 1: Load the dataset (Math or Portuguese)
        # -----------------------------------------------------------------
        mat, por = load_data()
        df = mat if dataset == "math" else por

        if df is None or df.empty:
            logger.error(f"Failed to load {dataset} dataset")
            return None
        
        if "G3" not in df.columns:
            # G3 = final grade (target variable) must exist
            logger.error("Target column 'G3' missing from dataset")
            return None

        # Split into features (X) and target (y)
        y = df["G3"]                  # target = final grade
        X = df.drop("G3", axis=1)     # predictors = all other columns

        # -----------------------------------------------------------------
        # STEP 2: Identify numeric vs categorical features
        # Needed for preprocessing with ColumnTransformer
        # -----------------------------------------------------------------
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

        # Build preprocessing pipeline
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)

        # -----------------------------------------------------------------
        # STEP 3: Choose the ML model
        # - Random Forest: robust for non-linear, categorical-heavy data
        # - Linear Regression: baseline model (interpretable but weaker)
        # -----------------------------------------------------------------
        if model_name == "random_forest":
            regressor = RandomForestRegressor(
                n_estimators=100,   # number of decision trees
                random_state=42,    # reproducibility
                n_jobs=1            # single-thread (keeps grading machines stable)
            )
        elif model_name == "linear_regression":
            regressor = LinearRegression()
        else:
            logger.error(f"Unsupported model: {model_name}")
            return None

        # Combine preprocessing + model into a single pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", regressor)
        ])

        # -----------------------------------------------------------------
        # STEP 4: Train the pipeline and evaluate performance
        # -----------------------------------------------------------------
        logger.info(f"Training {model_name} on {dataset} dataset...")
        pipeline, X_test, y_test = train_model(X, y, pipeline)

        # Evaluate the trained pipeline
        metrics_path = os.path.join(RESULTS_DIR, "metrics")
        metrics = evaluate_model(
            pipeline, X_test, y_test,
            metrics_path=metrics_path,
            dataset_name=f"{dataset}_{model_name}"
        )

        # -----------------------------------------------------------------
        # STEP 5: Save the trained model for reuse
        # -----------------------------------------------------------------
        models_path = os.path.join(RESULTS_DIR, "models")
        save_model(
            pipeline,
            models_path,
            filename=f"{model_name}_{dataset}.pkl"  # versioned by dataset+model
        )

        # -----------------------------------------------------------------
        # STEP 6: Log runtime and key results
        # -----------------------------------------------------------------
        elapsed = round(time.time() - start_time, 2)
        logger.info(
            f"✅ Pipeline completed in {elapsed}s. "
            f"MAE: {metrics['mae']}, R²: {metrics['r2']}"
        )

        return metrics   # return metrics to caller (e.g., run.py)

    except Exception as e:
        # Catch-all for unexpected errors (logged for debugging)
        logger.error(f"❌ Pipeline failed: {e}")
        raise
