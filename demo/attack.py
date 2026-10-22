"""Shared logic for the adversarial robustness demo: model loading, attacks,
Grad-CAM. Used by both the Gradio (app.py) and Streamlit (streamlit_app.py) UIs.

Checkpoints live in checkpoints/<name>.pt (gitignored). For hosted deployments
(Streamlit Community Cloud) they are fetched on first use from the public HF
Hub model repo nexgendev/adv-robustness-vision-ckpts.
"""
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models import build_model

torch.set_num_threads(1)  # keep RAM low; free hosting tiers are memory-tight

HERE = Path(__file__).parent
CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MODEL_FILES = [("Standard", "resnet18_std"), ("FGSM-AT", "resnet18_fgsm_at"),
               ("PGD-AT", "resnet18_pgd_at"), ("TRADES", "resnet18_trades_b6")]

HF_CKPT_BASE = "https://huggingface.co/nexgendev/adv-robustness-vision-ckpts/resolve/main"


def ensure_checkpoints():
    """Download any missing checkpoint from the HF Hub model repo."""
    import urllib.request
    ckpt_dir = HERE / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    for _, key in MODEL_FILES:
        path = ckpt_dir / f"{key}.pt"
        if not path.exists():
            print(f"downloading checkpoint {key}.pt ...")
            urllib.request.urlretrieve(f"{HF_CKPT_BASE}/{key}.pt", path)


def checkpoint_exists(label):
    return (HERE / "checkpoints" / f"{dict(MODEL_FILES)[label]}.pt").exists()


def available_labels():
    return [label for label, _ in MODEL_FILES if checkpoint_exists(label)]


_model_cache = {}


def get_model(label):
    """Lazy single-slot model cache: only one model is resident in RAM at a
    time, so the demo fits 512 MB-1 GB hosting tiers."""
    m = _model_cache.get(label)
    if m is not None:
        return m
    key = dict(MODEL_FILES)[label]
    path = HERE / "checkpoints" / f"{key}.pt"
    if not path.exists():
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = build_model(cfg["model"], **dict(cfg.get("model_args", {})))
    m.load_state_dict(ck["model"])
    _model_cache.clear()          # evict the previous model before loading
    gc.collect()
    _model_cache[label] = m.eval()
    return m


def to_tensor(img):
    """Centre-crop to a square, resize to 32x32, scale to [0, 1]. Models normalise internally."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((32, 32), Image.BICUBIC)
    return torch.from_numpy(np.asarray(img).copy()).float().div(255).permute(2, 0, 1).unsqueeze(0)


def fgsm(model, x, y, eps):
    x = x.clone().requires_grad_(True)
    grad, = torch.autograd.grad(F.cross_entropy(model(x), y), x)
    return (x + eps * grad.sign()).clamp(0, 1).detach()


def pgd(model, x, y, eps, steps=10):
    alpha = 2.5 * eps / steps
    xa = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        grad, = torch.autograd.grad(F.cross_entropy(model(xa), y), xa)
        xa = xa.detach() + alpha * grad.sign()
        xa = torch.min(torch.max(xa, x - eps), x + eps).clamp(0, 1)
    return xa.detach()


@torch.no_grad()
def predict(model, x):
    p = F.softmax(model(x), 1)[0]
    c = int(p.argmax())
    return c, float(p[c])


def gradcam(model, x, cls):
    """Grad-CAM on ResNet layer4 for class `cls`; returns a (32, 32) map in [0, 1]."""
    store = {}
    hook = model.net.layer4.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    with torch.enable_grad():
        logits = model(x)
        act = store["a"]
        grad, = torch.autograd.grad(logits[0, cls], act)
    hook.remove()
    w = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(32, 32), mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().numpy()
