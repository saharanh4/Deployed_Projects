from sklearn.base import clone
from sklearn.pipeline import Pipeline


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