import pandas as pd
import os

# ---------------------------------------------------------------------
# Define paths for project root and data directory.
# This ensures the script works no matter where it is run from.
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def generate_samples():
    """
    Generate small sample datasets (without target column) 
    for testing predictions.

    Behavior:
        - Reads the original Math and Portuguese datasets from /data.
        - Removes the target column 'G3' (final grade), since predictions
          should be made on unseen data without the answer.
        - Selects the first 5 rows only, creating a very small dataset
          for quick testing and demonstration.
        - Saves the samples back into /data with new filenames.

    Outputs:
        - data/new_data_math.csv   → sample from student-mat.csv
        - data/new_data.csv        → sample from student-por.csv

    Example:
        $ python generate_sample_data.py
        ✅ Created: data/new_data_math.csv
        ✅ Created: data/new_data.csv
    """

    # --- Generate Math dataset sample ---
    mat_path = os.path.join(DATA_DIR, "student-mat.csv")
    mat = pd.read_csv(mat_path, sep=";")   # UCI datasets use ";" separator
    mat_out = os.path.join(DATA_DIR, "new_data_math.csv")

    # Drop target column 'G3' and keep only first 5 rows
    mat.drop(columns=["G3"]).head(5).to_csv(mat_out, sep=";", index=False)

    # --- Generate Portuguese dataset sample ---
    por_path = os.path.join(DATA_DIR, "student-por.csv")
    por = pd.read_csv(por_path, sep=";")
    por_out = os.path.join(DATA_DIR, "new_data.csv")

    por.drop(columns=["G3"]).head(5).to_csv(por_out, sep=";", index=False)

    # Confirmation messages for user
    print(f"✅ Created: {mat_out}")
    print(f"✅ Created: {por_out}")


# -------------------------------------------------------------------------
# Script entry point:
# Running `python generate_sample_data.py` will create both sample files.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    generate_samples()
