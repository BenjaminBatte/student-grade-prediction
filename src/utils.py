import logging
import sys
import os
from datetime import datetime


def get_logger(name: str = __name__, level: int = logging.INFO):
    """
    Configure and return a logger.
    Logs go to both console and results/logs/project.log (UTF-8).
    Each run is separated by a header line.
    
    Args:
        name (str): Logger name (usually __name__).
        level (int): Logging level (default: logging.INFO).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # prevent duplicate logs from root logger

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        log_dir = os.path.join(project_root, "results", "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "project.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Write a separator at the start of each run
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        separator = (
            "\n" + "=" * 80 +
            f"\nNEW RUN STARTED: {datetime.now()} | Run ID: {run_id}\n" +
            "=" * 80 + "\n"
        )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(separator)

    return logger
