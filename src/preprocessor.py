"""
Preprocessing pipelines and ColumnTransformers for numerical scaling and categorical encoding.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.feature_engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def get_feature_preprocessor() -> ColumnTransformer:
    """
    Returns ColumnTransformer for full engineered features:
    StandardScaler for numeric features, OneHotEncoder for categorical features.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )


def get_basic_preprocessor() -> ColumnTransformer:
    """
    Returns ColumnTransformer for basic match state (used in initial/DL exploration).
    """
    num_cols = ["ball", "batter_runs", "batter_balls", "team_runs", "over"]
    cat_cols = ["bat_pos", "bowler"]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )
