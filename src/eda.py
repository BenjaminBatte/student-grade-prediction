import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# ---------------------------------------------------------------------
# --- Configuration ---
# ---------------------------------------------------------------------
# Use a non-interactive backend (Agg) by default.
# This ensures plots can be saved as image files even on servers/VMs
# without a graphical user interface (e.g., grading environments).
matplotlib.use('Agg')

# Define directory for saving generated figures.
# Creates folder: results/figures if it does not exist.
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Toggle for showing plots interactively.
# By default, False (plots are only saved, not displayed).
SHOW_PLOTS = False


def plot_distributions(df: pd.DataFrame, columns=None, dataset_name="dataset"):
    """
    Plot and save histograms for numeric columns in the dataset.

    Args:
        df (pd.DataFrame): Input dataset.
        columns (list, optional): Subset of numeric columns to plot. 
                                  If None, all numeric columns are used.
        dataset_name (str): Prefix for saved plot filenames (e.g., "math").
    """
    if columns is None:
        # Automatically select numeric columns if not specified
        columns = df.select_dtypes(include=["int64", "float64"]).columns

    for col in columns:
        # Create histogram for each numeric feature
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=20, color="skyblue")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()

        # Save figure with descriptive filename
        out_path = os.path.join(FIGURES_DIR, f"{dataset_name}_{col}_distribution.png")
        plt.savefig(out_path)

        # Optionally show the plot in interactive mode
        if SHOW_PLOTS:
            plt.show()

        plt.close()  # Free up memory between plots
        print(f"Saved distribution plot: {out_path}")


def plot_correlation_heatmap(df: pd.DataFrame, dataset_name="dataset"):
    """
    Generate and save a correlation heatmap for numeric features.

    Args:
        df (pd.DataFrame): Input dataset.
        dataset_name (str): Prefix for saved heatmap filename.
    """
    plt.figure(figsize=(12, 8))

    # Compute pairwise correlation for numeric columns only
    corr = df.corr(numeric_only=True)

    # Draw heatmap (blue = negative, red = positive correlation)
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title(f"Correlation Heatmap - {dataset_name}")
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, f"{dataset_name}_correlation_heatmap.png")
    plt.savefig(out_path)

    if SHOW_PLOTS:
        plt.show()

    plt.close()
    print(f"Saved correlation heatmap: {out_path}")


def summarize_dataset(df: pd.DataFrame):
    """
    Print a dataset summary including:
      - Shape (#rows, #columns)
      - Data types
      - Missing values per column
      - First 5 rows (sample preview)

    Args:
        df (pd.DataFrame): Dataset to summarize.
    """
    print("=== Dataset Summary ===")
    print(f"Shape: {df.shape}")
    print("\nColumn Types:\n", df.dtypes)
    print("\nMissing Values per Column:\n", df.isnull().sum())
    print("\nFirst 5 rows:\n", df.head())


# -------------------------------------------------------------------------
# Standalone execution mode:
# Allows running `python eda.py` directly for quick testing of plots.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA utility")
    parser.add_argument(
        "--show", 
        action="store_true", 
        help="Show plots interactively as well as saving them"
    )
    args = parser.parse_args()

    # If --show is provided, enable interactive mode
    if args.show:
        SHOW_PLOTS = True
        matplotlib.use("TkAgg")  # Switch backend to interactive
        import matplotlib.pyplot as plt  # Reload pyplot with new backend

    from data_loader import load_data

    print("Testing EDA functions...")

    # Load datasets for testing
    mat, por = load_data()

    if mat is not None:
        print("\n=== Math Dataset ===")
        summarize_dataset(mat)
        plot_distributions(mat, dataset_name="math")
        plot_correlation_heatmap(mat, dataset_name="math")

    if por is not None:
        print("\n=== Portuguese Dataset ===")
        summarize_dataset(por)
        plot_distributions(por, dataset_name="portuguese")
        plot_correlation_heatmap(por, dataset_name="portuguese")

    print("EDA testing completed!")
