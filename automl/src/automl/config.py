from dataclasses import dataclass

from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

TASKS = ('regression', 'classification')

@dataclass(frozen= True)
class config:
    path: str
    target: str
    task: str
    test_size: float = 0.2
    random_state: int = 42
    n_splits: int = 5
    k: int = 3
    n_iter: int = 30
    missing_thresh: float = 0.5
    id_ratio: float = 0.95
    out_dir: str = "runs"

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got '{self.task}'")
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be in (0, 1), got {self.test_size}")
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {self.n_splits}")
        if self.k < 1 or self.n_iter < 1:
            raise ValueError("k and n_iter must be >= 1")

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

def build_task_config(cfg: Config, n_classes: int | None = None) -> dict:
    """Task settings for cfg.task. n_classes is required for classification."""
    rs = cfg.random_state

    if cfg.task == "regression":
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
            "cv": KFold(n_splits=cfg.n_splits, shuffle=True, random_state=rs),
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
        "cv": StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=rs),
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