import os
import pandas as pd
from utils import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")

def load_data():
    try:
        mat_path = os.path.join(DATA_PATH, "student-mat.csv")
        por_path = os.path.join(DATA_PATH, "student-por.csv")

        if not os.path.exists(mat_path) or not os.path.exists(por_path):
            raise FileNotFoundError(f"One or both dataset files are missing in: {DATA_PATH}")

        mat = pd.read_csv(mat_path, sep=";")
        por = pd.read_csv(por_path, sep=";")

        logger.info("Math dataset loaded successfully with shape %s", mat.shape)
        logger.info("Portuguese dataset loaded successfully with shape %s", por.shape)

        return mat, por
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        return None, None
