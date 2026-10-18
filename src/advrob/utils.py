import os
import random
import subprocess
from pathlib import Path


def set_seed(seed: int) -> None:
    """Fix all RNGs so a run can be repeated (needed for mean +/- std over 3 seeds)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path, overrides=None) -> dict:
    """Load a YAML config; `overrides` is a list of 'a.b=value' strings."""
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def git_hash() -> str:
    """Short commit hash, stored with every run so results map to code."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip()
    except Exception:
        return "unknown"


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.n += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.n, 1)
