from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from automl.config import Config


def _tree_spaces():
    return {
        "RandomForest": {
            "n_estimators": randint(100, 500),
            "max_depth": randint(3, 20),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": uniform(0.3, 0.7),
        },
        "GradientBoosting": {
            "n_estimators": randint(100, 500),
            "learning_rate": loguniform(0.01, 0.3),
            "max_depth": randint(2, 6),
            "subsample": uniform(0.6, 0.4),
            "min_samples_leaf": randint(1, 10),
        },
        "XGBoost": {
            "n_estimators": randint(100, 500),
            "learning_rate": loguniform(0.01, 0.3),
            "max_depth": randint(2, 8),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_lambda": loguniform(0.01, 10),
        },
    }


def build_task_config(config: Config, n_classes: int | None = None) -> dict:
    """Task settings for config.task. n_classes is required for classification."""
    rs = config.random_state

    if config.task == "regression":
        return {
            "baseline": DummyRegressor(strategy="mean"),
            "primary_metric": "RMSE",
            "ascending": True,
            "models": {
                "Linear": LinearRegression(),
                "Ridge": Ridge(random_state=rs),
                "Lasso": Lasso(random_state=rs),
                "RandomForest": RandomForestRegressor(random_state=rs),
                "GradientBoosting": GradientBoostingRegressor(random_state=rs),
                "XGBoost": XGBRegressor(random_state=rs, n_jobs=1),
            },
            "cv": KFold(n_splits=config.n_splits, shuffle=True, random_state=rs),
            "metrics": {
                "RMSE": "neg_root_mean_squared_error",
                "MAE": "neg_mean_absolute_error",
                "R2": "r2",
            },
            "search_spaces": {
                "Ridge": {"alpha": loguniform(1e-3, 1e2)},
                "Lasso": {"alpha": loguniform(1e-4, 1e1)},
                **_tree_spaces(),
            },
        }

    if n_classes is None or n_classes < 2:
        raise ValueError("n_classes (>= 2) is required for classification")

    return {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "primary_metric": "F1",
        "ascending": False,
        "models": {
            "Logistic": LogisticRegression(max_iter=1000, random_state=rs),
            "RandomForest": RandomForestClassifier(random_state=rs),
            "GradientBoosting": GradientBoostingClassifier(random_state=rs),
            "XGBoost": XGBClassifier(random_state=rs, n_jobs=1),
        },
        "cv": StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=rs),
        "metrics": {
            "Accuracy": "accuracy",
            "F1": "f1_weighted",
            "ROC_AUC": "roc_auc" if n_classes == 2 else "roc_auc_ovr",
        },
        "search_spaces": {
            "Logistic": {"C": loguniform(1e-3, 1e2)},
            **_tree_spaces(),
        },
    }


def build_model_pipelines(task_cfg: dict, preprocessor) -> dict:
    """Baseline plus every model, each as Pipeline([preprocessor, model])."""
    pipelines = {
        "Baseline": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", clone(task_cfg["baseline"])),
        ])
    }
    for name, model in task_cfg["models"].items():
        pipelines[name] = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", clone(model)),
        ])
    return pipelines