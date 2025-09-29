import os
import sys

# ---------------------------------------------------------------------
# Ensure that Python can find the `src/` directory where all modules live.
# This allows you to run `run.py` directly from the project root without
# needing to install the package or adjust PYTHONPATH manually.
# ---------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import key project modules
from main import run_pipeline                # Main ML pipeline (preprocess + train + evaluate)
from utils import get_logger                 # Logger for consistent log output
from data_loader import load_data            # Loads Math and Portuguese datasets
from eda import (                            # EDA utilities: plots + summaries
    plot_distributions,
    plot_correlation_heatmap,
    summarize_dataset
)

# Initialize a logger instance for this script
logger = get_logger(__name__)


def main():
    """
    Main entry point for running the full student grade prediction pipeline.
    
    Responsibilities:
      1. Load both datasets (Math & Portuguese).
      2. Run exploratory data analysis (EDA) → saves plots + prints summaries.
      3. Train + evaluate multiple models (Linear Regression, Random Forest).
      4. Collect and print evaluation metrics for comparison.
    """
    print("🎓 Student Performance Prediction Pipeline")
    print("=" * 50)

    # Define the datasets and models to loop over
    datasets = ["math", "portuguese"]
    models = ["random_forest", "linear_regression"]

    # Dictionary to hold results for each dataset-model combination
    results = {}

    for dataset in datasets:
        # -----------------------------------------------------------------
        # STEP 1: Load the appropriate dataset (math or portuguese)
        # -----------------------------------------------------------------
        mat, por = load_data()
        df = mat if dataset == "math" else por

        if df is not None:
            # Print a dataset summary (shape, column types, missing values)
            print(f"\n=== {dataset.capitalize()} Dataset Summary ===")
            summarize_dataset(df)

            # Generate and save EDA plots
            plot_distributions(df, dataset_name=dataset)
            plot_correlation_heatmap(df, dataset_name=dataset)

        # -----------------------------------------------------------------
        # STEP 2: Train and evaluate models
        # For each dataset, run both Random Forest and Linear Regression
        # -----------------------------------------------------------------
        for model in models:
            key = f"{dataset}_{model}"
            results[key] = run_pipeline(dataset, model)

    # ---------------------------------------------------------------------
    # STEP 3: Print a consolidated summary of all model results
    # Metrics include: MAE, RMSE, R², etc. (depending on evaluation)
    # ---------------------------------------------------------------------
    print("\n=== Model Results ===")
    for key, metrics in results.items():
        print(f"\n{key}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")  # format floats nicely
            else:
                print(f"  {metric}: {value}")

    # Log a completion message
    logger.info("🎉 All done! Check results/ folder for outputs.")  


# -------------------------------------------------------------------------
# Script entry point:
# Running `python run.py` will trigger the pipeline for all datasets/models.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
