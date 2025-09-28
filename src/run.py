import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import run_pipeline
from utils import get_logger

logger = get_logger(__name__)

def main():
    """Main entry point with user-friendly interface."""
    print("🎓 Student Performance Prediction Pipeline")
    print("=" * 50)
    
    # Run all dataset-model combinations
    datasets = ["math", "portuguese"]
    models = ["random_forest", "linear_regression"]

    results = {}
    for dataset in datasets:
        for model in models:
            key = f"{dataset}_{model}"
            results[key] = run_pipeline(dataset, model)

    # Print summary of results
    print("\n=== Model Results ===")
    for key, metrics in results.items():
        print(f"\n{key}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    logger.info("🎉 All done! Check results/ folder for outputs.")  


if __name__ == "__main__":
    main()
