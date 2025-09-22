from datetime import datetime
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from utils import get_logger
import numpy as np

# Initialize module-level logger
logger = get_logger(__name__)


def train_model(X, y, pipeline, test_size=0.2, random_state=42):
    """
    Train the model pipeline and return the trained pipeline along with test data.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        pipeline (sklearn.pipeline.Pipeline): Preprocessing and model pipeline
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (trained_pipeline, X_test, y_test)
    """
    try:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train the pipeline
        pipeline.fit(X_train, y_train)
        
        logger.info(f"✅ Model training completed. Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        return pipeline, X_test, y_test
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        raise


def evaluate_model(pipeline, X_test, y_test, metrics_path, dataset_name):
    """
    Evaluate the trained model and save metrics to file.
    
    Args:
        pipeline (sklearn.pipeline.Pipeline): Trained pipeline
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): True target values
        metrics_path (str): Directory to save metrics
        dataset_name (str): Identifier for the dataset/model combination
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    try:
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # Calculate RMSE manually for compatibility with older scikit-learn versions
        rmse = np.sqrt(mse)
        
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            "mae": round(mae, 3),
            "mse": round(mse, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 3),
            "dataset": dataset_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save metrics to file
        os.makedirs(metrics_path, exist_ok=True)
        metrics_file = os.path.join(metrics_path, f"metrics_{dataset_name}.csv")
        
        # Create DataFrame and save
        metrics_df = pd.DataFrame([metrics])
        if os.path.exists(metrics_file):
            # Append to existing file
            existing_df = pd.read_csv(metrics_file)
            metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
        
        metrics_df.to_csv(metrics_file, index=False)
        
        logger.info(f"📊 Model evaluation completed. Metrics saved to: {metrics_file}")
        logger.info(f"   MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Model evaluation failed: {e}")
        raise


def save_model(model, path, filename=None, save_latest=True):
    """
    Save a trained model (or pipeline) to disk.

    Behavior:
        - Always saves a versioned model file with a timestamp (or a custom filename if provided).
        - Optionally overwrites/creates 'latest_model.pkl' for production use.

    Args:
        model (sklearn.pipeline.Pipeline or estimator): 
            The trained model or pipeline to save.
        path (str): 
            Directory where the model will be stored.
        filename (str, optional): 
            Custom filename. If None, a timestamped filename is generated.
        save_latest (bool, optional): 
            Whether to also overwrite 'latest_model.pkl' (default=True).

    Returns:
        dict: Dictionary with keys:
            - "versioned": Path of the versioned model file
            - "latest": Path of the latest model file (if saved)

    Example:
        >>> save_model(model, "results/models")
        {
            "versioned": "results/models/model_20250922_104500.pkl",
            "latest": "results/models/latest_model.pkl"
        }
    """
    os.makedirs(path, exist_ok=True)  # Ensure the directory exists

    try:
        # Generate a timestamped filename if none provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_{timestamp}.pkl"

        # --- Save versioned model ---
        versioned_path = os.path.join(path, filename)
        joblib.dump(model, versioned_path)
        logger.info(f"[SAVE] Versioned model saved at: {versioned_path}")

        saved_paths = {"versioned": versioned_path}

        # --- Save/overwrite latest model for production ---
        if save_latest:
            latest_path = os.path.join(path, "latest_model.pkl")
            joblib.dump(model, latest_path)
            logger.info(f"[SAVE] Latest model updated at: {latest_path}")
            saved_paths["latest"] = latest_path

        return saved_paths

    except Exception as e:
        logger.error(f"[ERROR] Failed to save model to {path}: {e}")
        return {}