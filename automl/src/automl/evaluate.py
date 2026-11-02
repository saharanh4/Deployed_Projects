import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.metrics import get_scorer

logger = logging.getLogger(__name__)


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _test_scores(task_cfg: dict, est, X_test, y_test) -> dict:
    out = {}
    for metric, scorer in task_cfg["metrics"].items():
        val = get_scorer(scorer)(est, X_test, y_test)
        out[metric] = -val if scorer.startswith("neg_") else val
    return out


def final_evaluation(task_cfg: dict, tuning_df: pd.DataFrame, tuned_models: dict,
                     pipelines: dict, X_train, y_train, X_test, y_test):
    best_name = tuning_df.loc[0, "Model"]
    best_model = tuned_models[best_name]

    baseline = clone(pipelines["Baseline"]).fit(X_train, y_train)

    best_scores = _test_scores(task_cfg, best_model, X_test, y_test)
    base_scores = _test_scores(task_cfg, baseline, X_test, y_test)
    result = pd.DataFrame({"Baseline": base_scores, best_name: best_scores}).T
    return best_name, best_model, best_scores, base_scores, result


def save_artifact(config, task_cfg: dict, best_name: str, best_model,
                  best_scores: dict, base_scores: dict, tuning_df: pd.DataFrame,
                  X_train, class_names=None, drops=None, unsupported_cols=None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.out_dir) / f"{stamp}_{Path(config.path).stem}"
    run_dir.mkdir(parents=True, exist_ok=False)

    joblib.dump(best_model, run_dir / "model.joblib")

    pm = task_cfg["primary_metric"]
    row = tuning_df.loc[tuning_df["Model"] == best_name].iloc[0]
    drops = drops or {}
    dropped = [c for cols in drops.values() for c in cols]

    metadata = {
        "task": config.task,
        "target": config.target,
        "model": best_name,
        "selected": row["Selected"],
        "best_params": row["Best_params"],
        "primary_metric": pm,
        "cv_score": row[f"Final_{pm}"],
        "holdout_scores": best_scores,
        "baseline_holdout_scores": base_scores,
        "class_names": class_names,
        "drops": drops,
        "unsupported_cols_dropped": unsupported_cols or [],
        "features_after_drop": X_train.columns.tolist(),
        "input_columns": X_train.columns.tolist() + dropped,
        "config": asdict(config),
        "versions": {
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=_jsonable)

    logger.info("Artifact saved to %s", run_dir)
    return run_dir