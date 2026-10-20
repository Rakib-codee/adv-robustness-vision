"""Load a trained model from a W&B artifact or a local checkpoint."""
from pathlib import Path

import torch

from .models import build_model


def load_model(artifact=None, ckpt=None, device="cpu"):
    assert artifact or ckpt, "give an artifact or a ckpt path"
    if ckpt:
        ck = torch.load(ckpt, map_location="cpu")
    else:
        import wandb
        folder = Path(wandb.Api().artifact(artifact).download())
        ck = torch.load(next(folder.glob("*.pt")), map_location="cpu")
    cfg = ck["cfg"]
    margs = dict(cfg.get("model_args", {}))
    if "pretrained" in margs:
        margs["pretrained"] = False
    model = build_model(cfg["model"], **margs).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), cfg["name"]
