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

    