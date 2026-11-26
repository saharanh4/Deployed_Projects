import logging
from pathlib import Path

from automl.cleaning import apply_drop_columns, clean_rows, find_drop_columns
from automl.config import Config
from automl.data import load_data, split_data
from automl.evaluate import build_metadata, final_evaluation, save_artifact
from automl.models import build_model_pipelines, build_task_config
from automl.preprocessing import build_preprocessor
from automl.train import benchmark_models, build_scoreboard, tune_top_k

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "insurance" / "insurance.csv"

# n_iter=5 is for this check only; the default is 30
config = Config(path=str(CSV), target="charges", task="regression",
                n_iter=5, out_dir=str(ROOT / "runs"))

df, _ = clean_rows(load_data(config.path), config.target)

X_tr, X_te, y_tr, y_te, names = split_data(df, config)
print(X_tr.shape, X_te.shape, names)

drops = find_drop_columns(X_tr, config.missing_thresh, config.id_ratio)
X_tr, X_te = apply_drop_columns(X_tr, drops), apply_drop_columns(X_te, drops)

pre, info = build_preprocessor(X_tr)
print(len(info["numeric_cols"]), len(info["categorical_cols"]), info["unsupported_cols"])

task_cfg = build_task_config(config, n_classes=len(names) if names else None)
pipes = build_model_pipelines(task_cfg, pre)
print(list(pipes), task_cfg["metrics"])

bench, errs = benchmark_models(task_cfg, pipes, X_tr, y_tr, config.n_jobs)
sb = build_scoreboard(task_cfg, bench)
print(sb.round(4).to_string())
print(errs)

tuning, tuned_models, terr = tune_top_k(task_cfg, sb, pipes, X_tr, y_tr, config)
print(tuning.drop(columns="Best_params").round(4).to_string())
print(terr)

best_name, best_model, best_scores, base_scores, final_df = final_evaluation(
    task_cfg, tuning, tuned_models, pipes, X_tr, y_tr, X_te, y_te
)
print(final_df.round(4).to_string())

# serving check: extra columns must not break prediction
print(best_model.predict(X_te.assign(junk=1))[:3])

data_info = {
    "class_names": names,
    "drops": drops,
    "numeric_cols": info["numeric_cols"],
    "categorical_cols": info["categorical_cols"],
    "unsupported_cols": info["unsupported_cols"],
    "features": X_tr.columns.tolist(),
}
metadata = build_metadata(config, task_cfg, best_name, best_scores, base_scores, tuning, data_info)
print(save_artifact(config, best_model, metadata))