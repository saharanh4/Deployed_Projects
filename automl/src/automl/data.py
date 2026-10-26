from pathlib import Path

import pandas as pd

def load_data(path:str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return pd.read_csv(p, sep= None, engine= "python")