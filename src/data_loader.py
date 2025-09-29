import os
import pandas as pd
from utils import get_logger

# ---------------------------------------------------------------------
# Initialize a logger specifically for this module.
# All messages (info/errors) will go both to console and results/logs/project.log
# ---------------------------------------------------------------------
logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Define the absolute paths for the project root and the data directory.
# This ensures the script works regardless of where it is run from.
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")


def load_data():
    """
    Load the student performance datasets (Math and Portuguese) from CSV files.

    Returns:
        tuple: (mat_df, por_df)
            mat_df: pandas.DataFrame or None
                Dataset for student performance in Mathematics (student-mat.csv).
            por_df: pandas.DataFrame or None
                Dataset for student performance in Portuguese (student-por.csv).

    Behavior:
        - Logs dataset shapes if loaded successfully.
        - Raises/logs errors if files are missing or unreadable.
        - Returns (None, None) if loading fails.
    """
    try:
        # -----------------------------------------------------------------
        # File paths for Math and Portuguese datasets
        # Both must exist in the /data directory for the pipeline to work
        # -----------------------------------------------------------------
        mat_path = os.path.join(DATA_PATH, "student-mat.csv")
        por_path = os.path.join(DATA_PATH, "student-por.csv")

        # Validate that both dataset files are present
        if not os.path.exists(mat_path) or not os.path.exists(por_path):
            raise FileNotFoundError(
                f"One or both dataset files are missing in: {DATA_PATH}"
            )

        # -----------------------------------------------------------------
        # Load CSV files
        # - The UCI Student Performance dataset uses a semicolon (;) delimiter,
        #   not the usual comma, so we explicitly set sep=";".
        # -----------------------------------------------------------------
        mat = pd.read_csv(mat_path, sep=";")
        por = pd.read_csv(por_path, sep=";")

        # Log successful loads with dataset dimensions (rows, columns)
        logger.info("Math dataset loaded successfully with shape %s", mat.shape)
        logger.info("Portuguese dataset loaded successfully with shape %s", por.shape)

        return mat, por

    except Exception as e:
        # Catch any error (e.g., missing file, read failure), log it,
        # and return (None, None) so downstream code can handle gracefully.
        logger.error("Failed to load data: %s", e)
        return None, None
