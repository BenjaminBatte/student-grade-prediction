import os
import pandas as pd
from utils import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

# Define project root and data path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")


def load_data():
    """
    Load the student performance datasets (Math and Portuguese).

    Returns:
        tuple: (mat_df, por_df)
            mat_df: pandas.DataFrame or None
                Dataset for student performance in Mathematics.
            por_df: pandas.DataFrame or None
                Dataset for student performance in Portuguese.
    
    The function logs dataset shapes if loaded successfully, or logs an error
    if files are missing/unreadable. If loading fails, returns (None, None).
    """
    try:
        # File paths for Math and Portuguese datasets
        mat_path = os.path.join(DATA_PATH, "student-mat.csv")
        por_path = os.path.join(DATA_PATH, "student-por.csv")

        # Check if both files exist before attempting to load
        if not os.path.exists(mat_path) or not os.path.exists(por_path):
            raise FileNotFoundError(
                f"One or both dataset files are missing in: {DATA_PATH}"
            )

        # Load CSV files with semicolon separator (specific to this dataset)
        mat = pd.read_csv(mat_path, sep=";")
        por = pd.read_csv(por_path, sep=";")

        # Log successful loads with dataset shapes
        logger.info("Math dataset loaded successfully with shape %s", mat.shape)
        logger.info("Portuguese dataset loaded successfully with shape %s", por.shape)

        return mat, por

    except Exception as e:
        # Log failure and return None for both datasets
        logger.error("Failed to load data: %s", e)
        return None, None
