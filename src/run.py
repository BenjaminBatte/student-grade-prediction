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
    
    # You can add interactive options here
    run_pipeline("math", "random_forest")
    run_pipeline("portuguese", "random_forest")

if __name__ == "__main__":
    main()