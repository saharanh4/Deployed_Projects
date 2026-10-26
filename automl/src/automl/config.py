from dataclasses import dataclass

TASKS = ('regression', 'classification')

@dataclass(frozen= True)
class config:
    path: str
    target: str
    task: str
    missing_thresh: float = 0.5
    id_ratio: float = 0.95

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got '{self.task}'")
        