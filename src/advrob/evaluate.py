"""Clean accuracy of a saved checkpoint:  python -m advrob.evaluate checkpoints/<run>/final.pt"""
import sys

import torch

from .data import get_cifar10
from .models import build_model
from .train import evaluate

if __name__ == "__main__":
    ck = torch.load(sys.argv[1], map_location="cpu")
    cfg = ck["cfg"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg["model"], **cfg.get("model_args", {})).to(device)
    model.load_state_dict(ck["model"])
    _, test = get_cifar10(cfg.get("data_root", "data"), cfg["batch_size"])
    print(f"clean accuracy: {evaluate(model, test, device):.4f}")
