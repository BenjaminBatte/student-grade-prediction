import pandas as pd
import os

# Define paths for project root and data directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def generate_samples():
    """
    Generate small sample datasets (without target column) 
    for testing predictions.

    Behavior:
        - Reads original Math and Portuguese datasets.
        - Drops the target column 'G3' (final grade).
        - Selects the first 5 rows for quick testing.
        - Saves them as new CSV files in the data/ directory.

    Outputs:
        - data/new_data_math.csv
        - data/new_data.csv

    Example:
        >>> python generate_sample_data.py
        ✅ Created: data/new_data_math.csv
        ✅ Created: data/new_data.csv
    """
    # --- Generate Math dataset sample ---
    mat = pd.read_csv(os.path.join(DATA_DIR, "student-mat.csv"), sep=";")
    mat_out = os.path.join(DATA_DIR, "new_data_math.csv")
    mat.drop(columns=["G3"]).head(5).to_csv(mat_out, sep=";", index=False)

    # --- Generate Portuguese dataset sample ---
    por = pd.read_csv(os.path.join(DATA_DIR, "student-por.csv"), sep=";")
    por_out = os.path.join(DATA_DIR, "new_data.csv")
    por.drop(columns=["G3"]).head(5).to_csv(por_out, sep=";", index=False)

    print(f"✅ Created: {mat_out}")
    print(f"✅ Created: {por_out}")


if __name__ == "__main__":
    generate_samples()
