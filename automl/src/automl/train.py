from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, cross_validate


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