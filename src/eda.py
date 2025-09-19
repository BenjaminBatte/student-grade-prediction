import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure results/figures exists
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def plot_distributions(df: pd.DataFrame, columns=None, dataset_name="dataset"):
    """
    Plot distributions of numeric columns.
    Saves histogram plots for each numeric feature.
    """
    if columns is None:
        columns = df.select_dtypes(include=["int64", "float64"]).columns

    for col in columns:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True, bins=20, color="skyblue")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()

        out_path = os.path.join(FIGURES_DIR, f"{dataset_name}_{col}_distribution.png")
        plt.savefig(out_path)
        plt.close()

def plot_correlation_heatmap(df: pd.DataFrame, dataset_name="dataset"):
    """
    Plot and save a correlation heatmap for numeric features.
    """
    plt.figure(figsize=(12, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, f"{dataset_name}_correlation_heatmap.png")
    plt.savefig(out_path)
    plt.close()

def summarize_dataset(df: pd.DataFrame):
    """
    Print dataset summary: shape, dtypes, null counts, and head.
    """
    print("=== Dataset Summary ===")
    print(f"Shape: {df.shape}")
    print("\nColumn Types:\n", df.dtypes)
    print("\nMissing Values per Column:\n", df.isnull().sum())
    print("\nFirst 5 rows:\n", df.head())
