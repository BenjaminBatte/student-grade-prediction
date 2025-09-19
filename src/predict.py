import argparse
import os
import joblib
import pandas as pd


def run_prediction(model_path: str, data_path: str, output_dir: str = "results/predictions"):
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    pipeline = joblib.load(model_path)

    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path, sep=";")

    # Drop target column if present
    if "G3" in df.columns:
        df = df.drop("G3", axis=1)

    # Predict
    preds = pipeline.predict(df)
    print("First 10 predictions:", preds[:10])

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "predictions.csv")
    pd.DataFrame({"prediction": preds}).to_csv(out_file, index=False)
    print(f"Predictions saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions using a trained model")
    parser.add_argument("--model", required=True, help="Path to trained model (.pkl)")
    parser.add_argument("--data", required=True, help="Path to CSV file with new data")
    parser.add_argument("--out", default="results/predictions", help="Directory to save predictions")

    args = parser.parse_args()
    run_prediction(args.model, args.data, args.out)
