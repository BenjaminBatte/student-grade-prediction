import pandas as pd
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def generate_samples():
    # Math dataset
    mat = pd.read_csv(os.path.join(DATA_DIR, "student-mat.csv"), sep=";")
    mat_out = os.path.join(DATA_DIR, "new_data_math.csv")
    mat.drop(columns=["G3"]).head(5).to_csv(mat_out, sep=";", index=False)

    # Portuguese dataset
    por = pd.read_csv(os.path.join(DATA_DIR, "student-por.csv"), sep=";")
    por_out = os.path.join(DATA_DIR, "new_data.csv")
    por.drop(columns=["G3"]).head(5).to_csv(por_out, sep=";", index=False)

    print(f"✅ Created: {mat_out}")
    print(f"✅ Created: {por_out}")

if __name__ == "__main__":
    generate_samples()
