import logging
from pathlib import Path

from automl.cleaning import apply_drop_columns, clean_rows, find_drop_columns
from automl.config import config, build_task_config
from automl.data import load_data, split_data
from automl.models import build_model_pipelines
from automl.preprocessing import build_preprocessor
from automl.train import benchmark_models, build_scoreboard, tune_top_k

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "wine+quality" / "winequality-red.csv"

# n_iter=5 is for this check only; the default is 30
config = config(path=str(CSV), target="quality", task="classification", n_iter=5)

df, _ = clean_rows(load_data(config.path), config.target)

X_tr, X_te, y_tr, y_te, names = split_data(
    df, config.target, config.task, config.test_size, config.random_state, config.n_splits
)
print(X_tr.shape, X_te.shape, names, y_tr.value_counts().sort_index().tolist())

drops = find_drop_columns(X_tr, config.missing_thresh, config.id_ratio)
X_tr, X_te = apply_drop_columns(X_tr, drops), apply_drop_columns(X_te, drops)

pre, num, cat, bad = build_preprocessor(X_tr)
Xt = pre.fit_transform(X_tr)
print(len(num), len(cat), bad, Xt.shape, pre.transform(X_te).shape)

task_cfg = build_task_config(config, n_classes=len(names))
pipes = build_model_pipelines(task_cfg, pre)
print(list(pipes), task_cfg["metrics"]["ROC_AUC"])

bench, errs = benchmark_models(task_cfg, pipes, X_tr, y_tr, config.n_jobs)
sb = build_scoreboard(task_cfg, bench, errs)
print(sb.round(4).to_string())
print(errs)

tuning, tuned_models, terrs = tune_top_k(task_cfg, sb, pipes, X_tr, y_tr, config)
print(tuning.drop(columns="Best_params").round(4).to_string())
print(tuning[["Model", "Selected", "Best_params"]].to_string())
print(terrs)