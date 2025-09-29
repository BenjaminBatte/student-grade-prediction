from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def build_preprocessor(numeric_cols, categorical_cols):
    """
    Construct a preprocessing pipeline using ColumnTransformer.

    Purpose:
        - Ensures consistent preprocessing of numeric and categorical data 
          before feeding into machine learning models.
        - Keeps the transformation steps reusable and reproducible.

    Args:
        numeric_cols (list of str): Names of numeric columns to standardize.
            → StandardScaler scales values so they have mean=0 and variance=1,
              which prevents large-scale features (e.g., absences) from dominating.
        categorical_cols (list of str): Names of categorical columns to encode.
            → OneHotEncoder converts categories into binary vectors for ML models.

    Returns:
        sklearn.compose.ColumnTransformer:
            A transformer that applies:
              - StandardScaler to numeric features
              - OneHotEncoder to categorical features
                (drop="first" avoids multicollinearity / dummy variable trap,
                 handle_unknown="ignore" prevents crashes on unseen categories)

    Example:
        >>> preprocessor = build_preprocessor(
        ...     numeric_cols=["age", "studytime"],
        ...     categorical_cols=["sex", "address"]
        ... )
        >>> X_transformed = preprocessor.fit_transform(df)
    """

    # -----------------------------------------------------------------
    # ColumnTransformer lets us define multiple preprocessing steps
    # and apply them selectively to different column types:
    # - "num": applies StandardScaler to numeric columns
    # - "cat": applies OneHotEncoder to categorical columns
    # -----------------------------------------------------------------
    return ColumnTransformer(
        transformers=[
            # Apply scaling to numeric columns
            ("num", StandardScaler(), numeric_cols),
            
            # Apply encoding to categorical columns
            # - drop="first": prevents redundant categories
            # - handle_unknown="ignore": safe for unseen categories at inference
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ]
    )
