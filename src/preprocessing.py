from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_preprocessor(numeric_cols, categorical_cols):
    """
    Build a ColumnTransformer for preprocessing:
    - Standardize numeric columns
    - One-hot encode categorical columns
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ]
    )
