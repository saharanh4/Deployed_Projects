import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

def build_preprocessor(
        X_train: pd.DataFrame,
        num_strategy: str = "median",
        add_indicator: bool = True,
        use_scaler: bool = True,
        cat_strategy: str = "most_frequent",
):
    numeric_cols = X_train.select_dtypes(include='number').columns.to_list()
    categorical_cols = X_train.select_dtypes(include=['string', 'object', 'category', 'bool']).columns.to_list()
    handled = set(numeric_cols) | set(categorical_cols)
    handled = [c for c in X_train.columns if c not in handled]

    unsupported_cols = [c for c in X_train.columns if c not in handled]
    if unsupported_cols:
        logger.warning("WARNING: unsupported dtypes, dropped from model: %s", unsupported_cols)

    nums_steps = [("imputer", SimpleImputer(strategy=num_strategy, add_indicator=add_indicator))]
    if use_scaler:
        nums_steps.append(('scaler', StandardScaler()))

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy=cat_strategy)),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ('num', Pipeline(nums_steps), numeric_cols),
        ('cat', cat_pipe, categorical_cols),
    ],
    remainder='drop',
    )

    return preprocessor, numeric_cols, categorical_cols, unsupported_cols