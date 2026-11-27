from dataclasses import dataclass
from pathlib import Path

import yaml

TASKS = ("regression", "classification")
Required = {"path", "target", "task"}

@dataclass(frozen=True)
class Config:
    path: str
    target: str
    task: str
    test_size: float = 0.2
    random_state: int = 42
    n_splits: int = 5
    k: int = 3
    n_iter: int = 30
    n_jobs: int = -1
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

def load_config(config_path) -> Config:
    """Build a Config from YAML. Relative paths resolve against the project root
    (the folder that contains configs/)."""
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    missing = REQUIRED - set(raw)
    if missing:
        raise ValueError(f"Missing required config keys: {sorted(missing)}")

    root = config_path.parent.parent
    for key in ("path", "out_dir"):
        if key in raw and not Path(raw[key]).is_absolute():
            raw[key] = str(root / raw[key])
    raw.setdefault("out_dir", str(root / "runs"))
    return Config(**raw)