"""Quick end-to-end check on a GPU: build both models, one forward/backward pass,
and verify inputs in [0,1] give valid logits. Run: PYTHONPATH=src python scripts/smoke_test.py"""
import torch
from advrob.models import build_model

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "(no GPU!)")
x = torch.rand(8, 3, 32, 32, device=dev)
y = torch.randint(0, 10, (8,), device=dev)
for name, kw in [("resnet18", {}), ("deit_tiny", {"pretrained": False})]:
    m = build_model(name, **kw).to(dev)
    out = m(x)
    assert out.shape == (8, 10), out.shape
    torch.nn.functional.cross_entropy(out, y).backward()
    print(f"{name}: OK, params {sum(p.numel() for p in m.parameters())/1e6:.1f}M")
print("smoke test passed")
