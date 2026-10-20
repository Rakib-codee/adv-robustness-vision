"""One backward pass of every defense loss on random data (GPU recommended).
Run: PYTHONPATH=src python scripts/smoke_adv.py"""
import time
import torch
from advrob.adv import defense_loss, quick_robust_acc
from advrob.models import build_model

dev = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.rand(32, 3, 32, 32, device=dev)
y = torch.randint(0, 10, (32,), device=dev)
for d in ["standard", "fgsm_at", "pgd_at", "trades"]:
    m = build_model("resnet18").to(dev).train()
    cfg = {"defense": d, "eps_255": 8, "alpha_255": 10 if d == "fgsm_at" else 2, "steps": 3, "beta": 6.0}
    t0 = time.time()
    loss = defense_loss(m, x, y, cfg)
    loss.backward()
    print(f"{d:9s} loss {loss.item():.3f} ({time.time() - t0:.2f}s)")
print("pgd10 acc on random batch:", round(quick_robust_acc(m, x, y), 3))
print("smoke_adv passed")
