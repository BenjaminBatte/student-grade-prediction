import pandas as pd
import matplotlib
# Set non-interactive backend to avoid Qt issues
matplotlib.use('Agg')  # This must come before pyplot import
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Directory where generated figures will be saved
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_distributions(df: pd.DataFrame, columns=None, dataset_name="dataset"):
    """
    Plot and save histograms for numeric columns in the dataset.

    Args:
        df (pd.DataFrame): Input dataset.
        columns (list, optional): List of columns to plot. If None, all numeric
            columns will be used.
        dataset_name (str, optional): Prefix for saved plot filenames.

    Saves:
        PNG files for each numeric column distribution in results/figures/.

    Example:
        >>> plot_distributions(df, ["age", "G3"], dataset_name="math")
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
        plt.close()
        print(f"Saved distribution plot: {out_path}")  # Add feedback


def plot_correlation_heatmap(df: pd.DataFrame, dataset_name="dataset"):
    """
    Generate and save a correlation heatmap for numeric features.

    Args:
        df (pd.DataFrame): Input dataset.
        dataset_name (str, optional): Prefix for saved plot filename.

    Saves:
        A PNG file containing the correlation heatmap.
    
    Example:
        >>> plot_correlation_heatmap(df, dataset_name="portuguese")
    """
    plt.figure(figsize=(12, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title(f"Correlation Heatmap - {dataset_name}")
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, f"{dataset_name}_correlation_heatmap.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved correlation heatmap: {out_path}")  # Add feedback


def summarize_dataset(df: pd.DataFrame):
    """
    Print a dataset summary including shape, data types, missing values, and head.

    Args:
        df (pd.DataFrame): Input dataset.

    Prints:
        - Dataset shape (rows, columns)
        - Column data types
        - Missing values per column
        - First 5 rows of data
    
    Example:
        >>> summarize_dataset(df)
    """
    print("=== Dataset Summary ===")
    print(f"Shape: {df.shape}")
    print("\nColumn Types:\n", df.dtypes)
    print("\nMissing Values per Column:\n", df.isnull().sum())
    print("\nFirst 5 rows:\n", df.head())


# Function to test EDA independently
if __name__ == "__main__":
    # Test the EDA functions
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