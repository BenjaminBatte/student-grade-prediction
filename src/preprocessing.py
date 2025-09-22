from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def build_preprocessor(numeric_cols, categorical_cols):
    """
    Construct a preprocessing pipeline using ColumnTransformer.

    Args:
        numeric_cols (list of str): List of column names containing numeric features
            to be standardized (mean=0, variance=1).
        categorical_cols (list of str): List of column names containing categorical 
            features to be one-hot encoded.

    Returns:
        sklearn.compose.ColumnTransformer:
            A transformer that applies:
              - StandardScaler to numeric features
              - OneHotEncoder to categorical features (dropping first category to avoid multicollinearity,
                and ignoring unknown categories at prediction time)
    
    Example:
        >>> preprocessor = build_preprocessor(
        ...     numeric_cols=["age", "studytime"],
        ...     categorical_cols=["sex", "address"]
        ... )
        >>> X_transformed = preprocessor.fit_transform(df)
    """
    return ColumnTransformer(
        transformers=[
            # Apply standard scaling to numeric columns
            ("num", StandardScaler(), numeric_cols),
            
            # Apply one-hot encoding to categorical columns
            # - drop="first": avoids dummy variable trap (multicollinearity)
            # - handle_unknown="ignore": prevents errors on unseen categories during prediction
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ]
    )
