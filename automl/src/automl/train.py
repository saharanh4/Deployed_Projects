import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, cross_validate

logger = logging.getLogger(__name__)


def benchmark_models(task_cfg: dict, pipelines: dict, X_train, y_train, n_jobs: int = -1):
    cv, metrics, pm = task_cfg["cv"], task_cfg["metrics"], task_cfg["primary_metric"]
    rows, errors = [], {}

    for name, pipe in pipelines.items():
        logger.info("Benchmarking %s", name)
        try:
            res = cross_validate(
                pipe, X_train, y_train,
                cv=cv, scoring=metrics, error_score=np.nan, n_jobs=n_jobs,
            )
            row = {"Model": name}
            for metric, scorer in metrics.items():
                scores = res[f"test_{metric}"]
                if scorer.startswith("neg_"):
                    scores = -scores
                row[f"{metric}_mean"] = scores.mean()
                row[f"{metric}_std"] = scores.std()
            row["Fit_time_s"] = res["fit_time"].mean()

            if np.isnan(row[f"{pm}_mean"]):
                raise RuntimeError(f"primary metric {pm} is NaN (a fold failed)")
            failed = [m for m in metrics if np.isnan(row[f"{m}_mean"])]
            if failed:
                logger.warning("Metric failed for %s: %s", name, failed)
            rows.append(row)
        except Exception as e:
            errors[name] = str(e)
            logger.warning("Model failed: %s -> %s", name, e)

    return pd.DataFrame(rows), errors


def build_scoreboard(task_cfg: dict, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    pm, asc = task_cfg["primary_metric"], task_cfg["ascending"]
    mean_col, std_col = f"{pm}_mean", f"{pm}_std"

    if benchmark_df.empty:
        raise RuntimeError("No model completed benchmarking")
    if "Baseline" not in benchmark_df["Model"].values:
        raise ValueError("Baseline row missing, cannot compute improvement")

    sb = benchmark_df.sort_values(mean_col, ascending=asc).reset_index(drop=True)
    sb.insert(0, "Rank", sb.index + 1)

    base = sb.loc[sb["Model"] == "Baseline", mean_col].iloc[0]
    if asc:
        sb["Improvement_vs_baseline_%"] = (base - sb[mean_col]) / base * 100
    else:
        sb["Improvement_vs_baseline_%"] = (sb[mean_col] - base) / base * 100

    best_mean, best_std = sb.loc[0, mean_col], sb.loc[0, std_col]
    sb["Tied_with_best"] = (sb[mean_col] - best_mean).abs() <= best_std
    return sb


def tune_top_k(task_cfg: dict, scoreboard: pd.DataFrame, pipelines: dict,
               X_train, y_train, cfg):
    pm, asc = task_cfg["primary_metric"], task_cfg["ascending"]
    scoring = task_cfg["metrics"][pm]
    negated = scoring.startswith("neg_")

    candidates = (
        scoreboard.loc[scoreboard["Model"] != "Baseline", "Model"].head(cfg.k).tolist()
    )
    rows, best_estimators, errors = [], {}, {}

    for name in candidates:
        logger.info("Tuning %s", name)
        untuned = scoreboard.loc[scoreboard["Model"] == name, f"{pm}_mean"].iloc[0]
        try:
            space = task_cfg["search_spaces"].get(name)
            tuned, params, est = np.nan, {}, None

            if space:
                search = RandomizedSearchCV(
                    pipelines[name],
                    {f"model__{p}": d for p, d in space.items()},
                    n_iter=cfg.n_iter, scoring=scoring, cv=task_cfg["cv"],
                    random_state=cfg.random_state, n_jobs=cfg.n_jobs,
                    refit=True, error_score="raise",
                )
                search.fit(X_train, y_train)
                tuned = -search.best_score_ if negated else search.best_score_
                params = {
                    p.removeprefix("model__"): v for p, v in search.best_params_.items()
                }
                est = search.best_estimator_

            use_tuned = est is not None and ((tuned < untuned) if asc else (tuned > untuned))
            if not use_tuned:
                est, params = clone(pipelines[name]).fit(X_train, y_train), {}

            rows.append({
                "Model": name,
                f"Untuned_{pm}": untuned,
                f"Tuned_{pm}": tuned,
                f"Final_{pm}": tuned if use_tuned else untuned,
                "Selected": "tuned" if use_tuned else "untuned",
                "Best_params": params,
            })
            best_estimators[name] = est
        except Exception as e:
            errors[name] = str(e)
            logger.warning("Tuning failed: %s -> %s", name, e)

    if not rows:
        raise RuntimeError(f"All tuning candidates failed: {errors}")

    results = (
        pd.DataFrame(rows)
        .sort_values(f"Final_{pm}", ascending=asc)
        .reset_index(drop=True)
    )
    return results, best_estimators, errors