import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# --- Configuration ---
# Default: headless mode (Agg backend)
matplotlib.use('Agg')

# Directory where generated figures will be saved
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Toggle interactive display
SHOW_PLOTS = False


def plot_distributions(df: pd.DataFrame, columns=None, dataset_name="dataset"):
    """
    Plot and save histograms for numeric columns in the dataset.
    """
    if columns is None:
        # Select numeric columns if not explicitly provided
        columns = df.select_dtypes(include=["int64", "float64"]).columns

    for col in columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=20, color="skyblue")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()

        out_path = os.path.join(FIGURES_DIR, f"{dataset_name}_{col}_distribution.png")
        plt.savefig(out_path)
        if SHOW_PLOTS:
            plt.show()
        plt.close()
        print(f"Saved distribution plot: {out_path}")


def plot_correlation_heatmap(df: pd.DataFrame, dataset_name="dataset"):
    """
    Generate and save a correlation heatmap for numeric features.
    """
    plt.figure(figsize=(12, 8))
    corr = df.corr(numeric_only=True)
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
    Print a dataset summary including shape, data types, missing values, and head.
    """
    print("=== Dataset Summary ===")
    print(f"Shape: {df.shape}")
    print("\nColumn Types:\n", df.dtypes)
    print("\nMissing Values per Column:\n", df.isnull().sum())
    print("\nFirst 5 rows:\n", df.head())


# Function to test EDA independently
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA utility")
    parser.add_argument("--show", action="store_true", help="Show plots interactively as well as saving them")
    args = parser.parse_args()

    # Enable showing plots if requested
    if args.show:
        SHOW_PLOTS = True
        matplotlib.use("TkAgg")  # or another interactive backend
        import matplotlib.pyplot as plt  # reload pyplot with new backend

    from data_loader import load_data

    print("Testing EDA functions...")
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
