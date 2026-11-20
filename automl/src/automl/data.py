from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from automl.config import Config


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return pd.read_csv(p, sep=None, engine="python")


def split_data(df: pd.DataFrame, config: Config):
    """Returns X_train, X_test, y_train, y_test, class_names (None for regression)."""
    target = config.target
    X = df.drop(columns=target)
    class_names = None

    if config.task == "classification":
        counts = df[target].value_counts()
        if counts.min() < 2:
            raise ValueError(f"Classes with fewer than 2 samples: {counts[counts < 2].to_dict()}")
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(df[target]), index=df.index, name=target)
        class_names = le.classes_.tolist()
        stratify = y
    else:
        y = df[target]
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=stratify
    )

    if config.task == "classification":
        train_counts = y_train.value_counts()
        rare = train_counts[train_counts < config.n_splits]
        if not rare.empty:
            raise ValueError(
                f"Classes with fewer than {config.n_splits} train samples "
                f"(CV needs >= {config.n_splits}): "
                f"{ {class_names[i]: int(n) for i, n in rare.items()} }"
            )

    return X_train, X_test, y_train, y_test, class_names