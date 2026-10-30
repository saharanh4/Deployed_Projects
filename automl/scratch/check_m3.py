from pathlib import Path

from sklearn.model_selection import train_test_split

from automl.cleaning import apply_drop_columns, clean_rows, find_drop_columns
from automl.data import load_data
from automl.preprocessing import build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "wine+quality" / "winequality-red.csv"

df, _ = clean_rows(load_data(str(CSV)), "quality")
X, y = df.drop(columns="quality"), df["quality"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

drops = find_drop_columns(X_tr, 0.5, 0.95)
X_tr, X_te = apply_drop_columns(X_tr, drops), apply_drop_columns(X_te, drops)

pre, num, cat, bad = build_preprocessor(X_tr)
Xt = pre.fit_transform(X_tr)
print(len(num), len(cat), bad, Xt.shape, pre.transform(X_te).shape)