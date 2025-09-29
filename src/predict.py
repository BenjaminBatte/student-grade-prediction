import argparse
import os
import joblib
import pandas as pd
from datetime import datetime


def run_prediction(model_path: str, data_path: str, output_dir: str = "results/predictions"):
    """
    Run predictions using a trained model pipeline.

    Args:
        model_path (str): Path to a trained model (.pkl file saved by save_model).
        data_path (str): Path to a CSV file with new data (no target column 'G3').
        output_dir (str): Directory to save prediction results (default: results/predictions).

    Behavior:
        - Loads the trained pipeline and input dataset.
        - Drops target column 'G3' if present (since we only want features for prediction).
        - Generates predictions and creates a results DataFrame with both raw 
          and rounded values (for easier interpretation).
        - Prints a summary of predictions to the console.
        - Saves the results into a timestamped CSV file in output_dir.
    """
    # -----------------------------------------------------------------
    # STEP 1: Validate input paths
    # Ensure both the model file and the new data file exist
    # -----------------------------------------------------------------
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # -----------------------------------------------------------------
    # STEP 2: Load the trained model pipeline and input data
    # - joblib is used because it efficiently handles sklearn models
    # - Input data uses ";" as separator (UCI dataset format)
    # -----------------------------------------------------------------
    pipeline = joblib.load(model_path)
    df = pd.read_csv(data_path, sep=";")

    # -----------------------------------------------------------------
    # STEP 3: Drop target column if accidentally present
    # Normally, new unseen data should NOT contain 'G3'
    # -----------------------------------------------------------------
    if "G3" in df.columns:
        df = df.drop("G3", axis=1)
        print("⚠️  Target column 'G3' removed from input data")

    # -----------------------------------------------------------------
    # STEP 4: Generate predictions
    # - pipeline handles preprocessing + model inference automatically
    # - predictions are continuous (regression), so we also provide rounded values
    # -----------------------------------------------------------------
    predictions = pipeline.predict(df)

    results_df = pd.DataFrame({
        "prediction": predictions,                         # raw regression outputs
        "prediction_rounded": predictions.round().astype(int)  # easier to interpret as grades
    })

    # -----------------------------------------------------------------
    # STEP 5: Print a human-readable summary of predictions
    # Includes basic statistics and preview of first 10 predictions
    # -----------------------------------------------------------------
    print("\n" + "=" * 50)
    print("PREDICTION SUMMARY")
    print("=" * 50)
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Input data: {os.path.basename(data_path)}")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Prediction range: {predictions.min():.1f} - {predictions.max():.1f}")
    print(f"Average prediction: {predictions.mean():.2f}")
    print("\nFirst 10 predictions:")
    for i, pred in enumerate(predictions[:10]):
        print(f"  Student {i+1}: {pred:.2f} (rounded: {round(pred)})")

    # -----------------------------------------------------------------
    # STEP 6: Save results to CSV
    # - Uses timestamp to avoid overwriting past predictions
    # - Files are stored in results/predictions/
    # -----------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"predictions_{timestamp}.csv")

    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Predictions saved to: {output_file}")


# -------------------------------------------------------------------------
# Script entry point:
# Allows running predictions from the command line:
# Example:
#   $ python -m src.predict --model results/models/random_forest_math.pkl --data data/new_data_math.csv
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions using a trained model")
    parser.add_argument("--model", required=True, help="Path to trained model (.pkl)")
    parser.add_argument("--data", required=True, help="Path to CSV file with new data")
    parser.add_argument("--out", default="results/predictions", help="Directory to save predictions")

    args = parser.parse_args()
    run_prediction(args.model, args.data, args.out)
