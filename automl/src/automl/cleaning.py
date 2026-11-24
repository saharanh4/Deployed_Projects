import pandas as pd


def clean_rows(df: pd.DataFrame, target: str):
    """Row-level cleaning. Safe before the split (uses no column statistics)."""
    report = {"initial_shape": df.shape}

    n = len(df)
    df = df.dropna(subset=[target])
    report["rows_dropped_missing_target"] = n - len(df)

    n = len(df)
    df = df.drop_duplicates()
    report["duplicate_rows_dropped"] = n - len(df)

    report["final_shape"] = df.shape
    return df, report


def find_drop_columns(X_train: pd.DataFrame, missing_thresh: float, id_ratio: float) -> dict:
    """Decide which feature columns to drop. Call on TRAIN only."""
    feats = list(X_train.columns)

    constant = [c for c in feats if X_train[c].nunique(dropna=False) <= 1]
    feats = [c for c in feats if c not in constant]

    high_missing = [c for c in feats if X_train[c].isna().mean() > missing_thresh]
    feats = [c for c in feats if c not in high_missing]

    id_like = [
        c for c in feats
        if X_train[c].dtype.kind != "f" and X_train[c].nunique() / len(X_train) >= id_ratio
    ]

    return {"constant": constant, "high_missing": high_missing, "id_like": id_like}


def apply_drop_columns(X: pd.DataFrame, drops: dict) -> pd.DataFrame:
    """Apply a drop-list from find_drop_columns to train or test."""
    all_cols = [c for cols in drops.values() for c in cols]
    return X.drop(columns=all_cols, errors="ignore")