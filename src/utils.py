import logging
import sys
import os
from datetime import datetime


def get_logger(name: str = __name__, level: int = logging.INFO):
    """
    Configure and return a logger instance.

    Logs are written both to the console and to a file:
    - Console: Real-time output of logs
    - File: Stored at results/logs/project.log (UTF-8 encoding)

    Each new run is marked with a timestamped header in the log file.

    Args:
        name (str, optional): Logger name (default: __name__).
        level (int, optional): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger object.

    Example:
        >>> from utils import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Logger initialized")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicate logs from the root logger

    if not logger.handlers:  # Prevent adding multiple handlers on repeated calls
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        # Console handler → outputs logs to stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler → logs saved to results/logs/project.log
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        log_dir = os.path.join(project_root, "results", "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "project.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Write a visual separator in the log file for each new run
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        separator = (
            "\n" + "=" * 80 +
            f"\nNEW RUN STARTED: {datetime.now()} | Run ID: {run_id}\n" +
            "=" * 80 + "\n"
        )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(separator)

    return logger
