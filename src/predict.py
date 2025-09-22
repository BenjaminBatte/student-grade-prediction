import argparse
import os
import joblib
import pandas as pd


def run_prediction(model_path: str, data_path: str, output_dir: str = "results/predictions"):
    """
    Run predictions using a trained model pipeline.

    Args:
        model_path (str): Path to the trained model file (.pkl).
        data_path (str): Path to the input dataset (CSV file).
        output_dir (str, optional): Directory where predictions will be saved 
            (default: "results/predictions").

    Behavior:
        - Loads a trained model pipeline (joblib).
        - Loads new input data (CSV).
        - Drops target column 'G3' if present (since we want predictions).
        - Generates predictions.
        - Prints first 10 predictions for quick inspection.
        - Saves all predictions to a CSV file.

    Raises:
        FileNotFoundError: If the model file or data file does not exist.

    Example:
        >>> run_prediction("results/models/latest_model.pkl", "data/student-mat.csv")
    """
    # --- Load trained model ---
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    pipeline = joblib.load(model_path)

    # --- Load input dataset ---
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path, sep=";")

    # Drop target column if included in dataset
    if "G3" in df.columns:
        df = df.drop("G3", axis=1)

    # --- Run predictions ---
    preds = pipeline.predict(df)
    print("First 10 predictions:", preds[:10])

    # --- Save predictions ---
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "predictions.csv")
    pd.DataFrame({"prediction": preds}).to_csv(out_file, index=False)
    print(f"Predictions saved to: {out_file}")


if __name__ == "__main__":
    # CLI interface for running predictions
    parser = argparse.ArgumentParser(description="Run predictions using a trained model")
    parser.add_argument("--model", required=True, help="Path to trained model (.pkl)")
    parser.add_argument("--data", required=True, help="Path to CSV file with new data")
    parser.add_argument(
        "--out", default="results/predictions", help="Directory to save predictions"
    )

    args = parser.parse_args()
    run_prediction(args.model, args.data, args.out)
