from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FILE_PATH = "../../data/insurance/insurance.csv"

def load_data(path:str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return pd.read_csv(p, sep= None, engine= "python")

def split_data(df: pd.DataFrame, target: str, task: str, test_size: float, random_state: int, n_splits: int):
    """Returns X_train, X_test, y_train, y_test, class_names (None for regression)."""
    X = df.drop(columns = target)
    class_names = None

    if task == "classification":
        counts = df[target].value_counts()
        if counts.min() < 2:
            raise ValueError(f"Classes with fewer than 2 samples: {counts[counts < 2].to_dict()}")
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(df[target]), index = df.index, name = target)
        class_names = le.classes_.tolist()
        stratify = y

    else:
        y = df[target]
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    if task == "classification":
        train_counts = y_train.value_counts()
        rare = train_counts[train_counts < n_splits]
        if not rare.empty:
            raise ValueError(
                f"Classes with fewer than {n_splits} train samples "
                f"(CV needs >= {n_splits}): "
                f"{ {class_names[i]: int(n) for i, n in rare.items()} }"
            )

    return X_train, X_test, y_train, y_test, class_names
