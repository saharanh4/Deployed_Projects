import pandas as pd

def inspection(df: pd.DataFrame) -> dict:
    num_col = df.select_dtypes(include='number').columns.tolist()
    string_col = df.select_dtypes(include=['object', 'string']).columns.tolist()
    null_pct = (df.isna().mean() * 100).round(3)

    return{
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numeric_cols": num_col,
        "string_cols": string_col,
        "null_pct_col": null_pct[null_pct > 0].to_dict(),
        "unique_val": df[string_col].nunique().to_dict(),
        "duplicated_rows": int(df.duplicated().sum()),
        "constant_cols": [c for c in df.columns if df[c].nunique(dropna=False) == 1],
    }

def clean_rows(df: pd.DataFrame, target:str):
    report = {"initial_shape": df.shape}

    n = len(df)
    df = df.dropna(subset= [target])
    report['rows_dropped_missing_target'] = n - len(df)

    n = len(df)
    df = df.drop_duplicates()
    report['duplicated_rows_dropped'] = n - len(df)

    report["final_shape"] = df.shape
    return df, report

def find_drop_col(X_train: pd.DataFrame, missing_thresh: float, id_ratio: float) -> dict:
    feats = list(X_train.columns)

    constant = [c for c in feats if X_train[c].nunique(dropna=False) <= 1]
    feats = [c for c in feats if c not in constant]

    high_missing = [c for c in feats if X_train[c].isna().mean() > missing_thresh]
    feats = [c for c in feats if c not in high_missing]

    id_like = [
        c for c in feats
        if X_train[c].dtype.kind != "f" and X_train[c].nunique() / len(X_train) >= id_ratio
    ]
    feats = [c for c in feats if c not in id_like]

    dup_mask = X_train[feats].T.duplicated()
    duplicated = dup_mask[dup_mask].index.tolist()

    return {
        "constant": constant,
        "high_missing": high_missing,
        "id_like": id_like,
        "duplicated": duplicated,
    }


def apply_drop_columns(X: pd.DataFrame, drops: dict) -> pd.DataFrame:
    all_cols = [c for cols in drops.values() for c in cols]
    return X.drop(columns=all_cols, errors="ignore")