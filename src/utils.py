import logging
import sys
import os
from datetime import datetime


def get_logger(name: str = __name__, level: int = logging.INFO):
    """
    Configure and return a logger instance for consistent logging
    across the project.

    Features:
        - Writes logs to both console (stdout) and a file.
        - Creates results/logs/project.log if it does not exist.
        - Each new run writes a clear header separator with timestamp + run ID.

    Args:
        name (str, optional): Name of the logger (default: __name__).
            → Helps distinguish logs from different modules.
        level (int, optional): Logging level (default: INFO).
            → Can be set to DEBUG, WARNING, ERROR, etc.

    Returns:
        logging.Logger: Configured logger object.

    Example:
        >>> from utils import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Logger initialized")
    """
    # Create or retrieve a logger with the specified name
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate logs from root logger

    # -----------------------------------------------------------------
    # Add handlers only once (avoid duplication if function called again)
    # -----------------------------------------------------------------
    if not logger.handlers:
        # Format: 2025-09-29 12:30:15 [INFO] Message text
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        # -----------------------------
        # Console handler (real-time logs)
        # -----------------------------
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # -----------------------------
        # File handler (persistent logs)
        # Saves logs to results/logs/project.log
        # -----------------------------
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        log_dir = os.path.join(project_root, "results", "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "project.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # -----------------------------
        # Add a visual separator for each new run
        # This makes the log file easier to navigate
        # -----------------------------
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        separator = (
            "\n" + "=" * 80 +
            f"\nNEW RUN STARTED: {datetime.now()} | Run ID: {run_id}\n" +
            "=" * 80 + "\n"
        )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(separator)

    return logger
