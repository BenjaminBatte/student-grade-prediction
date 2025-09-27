import argparse
import os
import joblib
import pandas as pd
from datetime import datetime

def run_prediction(model_path: str, data_path: str, output_dir: str = "results/predictions"):
    """Run predictions using a trained model pipeline."""
    
    # Validate inputs
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load model and data
    pipeline = joblib.load(model_path)
    df = pd.read_csv(data_path, sep=";")

    # Drop target column if present
    if "G3" in df.columns:
        df = df.drop("G3", axis=1)
        print("⚠️  Target column 'G3' removed from input data")

    # Generate predictions
    predictions = pipeline.predict(df)
    
    # Create detailed output
    results_df = pd.DataFrame({
        'prediction': predictions,
        'prediction_rounded': predictions.round().astype(int)
    })
    
    # Print summary
    print("\n" + "="*50)
    print("PREDICTION SUMMARY")
    print("="*50)
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Input data: {os.path.basename(data_path)}")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Prediction range: {predictions.min():.1f} - {predictions.max():.1f}")
    print(f"Average prediction: {predictions.mean():.2f}")
    print("\nFirst 10 predictions:")
    for i, pred in enumerate(predictions[:10]):
        print(f"  Student {i+1}: {pred:.2f} (rounded: {round(pred)})")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"predictions_{timestamp}.csv")
    
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Predictions saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions using a trained model")
    parser.add_argument("--model", required=True, help="Path to trained model (.pkl)")
    parser.add_argument("--data", required=True, help="Path to CSV file with new data")
    parser.add_argument("--out", default="results/predictions", help="Directory to save predictions")

    args = parser.parse_args()
    run_prediction(args.model, args.data, args.out)