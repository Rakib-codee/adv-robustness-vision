"""Accuracy under FGSM / PGD at several epsilons (L-inf, pixel space [0,1]).

Example (Kaggle, from repo root, PYTHONPATH=src):
  python -m advrob.attack_eval \
      --artifact rakibislam4913-zhengzhou-university/adv-robustness-vision/resnet18_std_seed0:latest \
      --n 1000 --out results/attack_resnet18_std_seed0.csv

Use the SAME --n for every model so numbers are comparable (first n test images, fixed order).
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import torchattacks

from .data import get_cifar10
from .models import build_model
from .utils import git_hash, set_seed


def load_checkpoint(args):
    if args.ckpt:
        return torch.load(args.ckpt, map_location="cpu")
    import wandb
    art = wandb.Api().artifact(args.artifact)
    folder = Path(art.download())
    return torch.load(next(folder.glob("*.pt")), map_location="cpu")


def first_n_test(n, batch_size):
    _, test_loader = get_cifar10("data", batch_size=128, num_workers=2)
    xs, ys, got = [], [], 0
    for x, y in test_loader:
        xs.append(x); ys.append(y); got += len(y)
        if got >= n:
            break
    x, y = torch.cat(xs)[:n], torch.cat(ys)[:n]
    return [(x[i:i + batch_size], y[i:i + batch_size]) for i in range(0, n, batch_size)]


@torch.no_grad()
def accuracy(model, batches, device):
    correct = total = 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += len(y)
    return correct / total


def robust_accuracy(model, attack, batches, device):
    correct = total = 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        x_adv = attack(x, y)
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
        total += len(y)
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", help="W&B artifact path, entity/project/name:version")
    p.add_argument("--ckpt", help="or a local .pt checkpoint")
    p.add_argument("--eps", default="2,4,8,16", help="comma list, in /255 units")
    p.add_argument("--attacks", default="fgsm,pgd")
    p.add_argument("--pgd-steps", type=int, default=20)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    assert args.ckpt or args.artifact, "give --artifact or --ckpt"

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = load_checkpoint(args)
    cfg = ck["cfg"]
    margs = dict(cfg.get("model_args", {}))
    if "pretrained" in margs:
        margs["pretrained"] = False  # weights come from the checkpoint, no download
    model = build_model(cfg["model"], **margs).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    batches = first_n_test(args.n, args.batch)
    clean = accuracy(model, batches, device)
    print(f"{cfg['name']}: clean accuracy on first {args.n} test images = {clean:.4f}")

    rows = [dict(model=cfg["name"], attack="clean", eps_255=0, n=args.n, acc=round(clean, 4),
                 seed=args.seed, git=git_hash())]
    for name in args.attacks.split(","):
        for e in [int(v) for v in args.eps.split(",")]:
            eps = e / 255
            if name == "fgsm":
                atk = torchattacks.FGSM(model, eps=eps)
            elif name == "pgd":
                atk = torchattacks.PGD(model, eps=eps, alpha=2.5 * eps / args.pgd_steps,
                                       steps=args.pgd_steps, random_start=True)
            else:
                raise ValueError(name)
            t0 = time.time()
            acc = robust_accuracy(model, atk, batches, device)
            print(f"  {name} eps={e}/255: robust acc {acc:.4f} ({time.time() - t0:.0f}s)")
            rows.append(dict(model=cfg["name"], attack=name, eps_255=e, n=args.n,
                             acc=round(acc, 4), seed=args.seed, git=git_hash()))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("saved", out)


if __name__ == "__main__":
    main()
