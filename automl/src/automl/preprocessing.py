import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def build_preprocessor(X_train: pd.DataFrame):
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(
        include=["string", "object", "category", "bool"]
    ).columns.tolist()
    handled = set(numeric_cols) | set(categorical_cols)
    unsupported_cols = [c for c in X_train.columns if c not in handled]

    if unsupported_cols:
        logger.warning("Unsupported dtypes, dropped from model: %s", unsupported_cols)

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        [("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)],
        remainder="drop",
    )
    info = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "unsupported_cols": unsupported_cols,
    }
    return preprocessor, info