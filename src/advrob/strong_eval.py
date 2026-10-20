"""AutoAttack (L-inf) and C&W (L2) evaluation.

  python -m advrob.strong_eval --artifact <entity/project/name:latest> --attack autoattack --n 1000 \
      --out results/aa_<name>.csv
  python -m advrob.strong_eval --artifact <...> --attack cw --n 500 --out results/cw_<name>.csv

AutoAttack = the final, trustworthy robustness number (standard version: APGD-CE, APGD-T, FAB-T, Square).
C&W = minimal L2 perturbation. Reported: share of images fooled, median L2 of successful
perturbations, and accuracy when an attacker is limited to L2 <= 0.5.
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import torchattacks

from .attack_eval import accuracy, first_n_test
from .loading import load_model
from .utils import git_hash, set_seed


def run_autoattack(model, batches, device, eps):
    atk = torchattacks.AutoAttack(model, norm="Linf", eps=eps, version="standard", n_classes=10, seed=0)
    correct = total = 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        x_adv = atk(x, y)
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
        total += len(y)
    return correct / total


def run_cw(model, batches, device, cs=(0.1, 1.0, 10.0), steps=100, budget=0.5):
    """For each image keep the smallest successful L2 perturbation over a small grid of c."""
    best_l2, fooled_any, clean_ok = [], [], []
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            ok = model(x).argmax(1) == y
        best = torch.full((len(y),), float("inf"), device=device)
        for c in cs:
            atk = torchattacks.CW(model, c=c, kappa=0, steps=steps, lr=0.01)
            x_adv = atk(x, y)
            with torch.no_grad():
                fooled = model(x_adv).argmax(1) != y
            l2 = (x_adv - x).flatten(1).norm(dim=1)
            best = torch.where(fooled & (l2 < best), l2, best)
        best_l2.append(best.cpu()); clean_ok.append(ok.cpu())
    best = torch.cat(best_l2); ok = torch.cat(clean_ok)
    succ = ok & torch.isfinite(best)
    # misclassified-on-clean images count as already broken (L2 = 0)
    l2_all = torch.where(ok, best, torch.zeros_like(best))
    acc_budget = (l2_all > budget).float().mean().item()
    return dict(fooled_rate=succ.float().sum().item() / max(ok.sum().item(), 1),
                median_l2=best[succ].median().item() if succ.any() else float("nan"),
                acc_at_l2_0_5=acc_budget)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact"); p.add_argument("--ckpt")
    p.add_argument("--attack", choices=["autoattack", "cw"], required=True)
    p.add_argument("--eps", type=int, default=8, help="/255, for autoattack")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, name = load_model(args.artifact, args.ckpt, device)
    batches = first_n_test(args.n, args.batch)
    clean = accuracy(model, batches, device)
    print(f"{name}: clean acc on first {args.n} = {clean:.4f}")
    t0 = time.time()
    if args.attack == "autoattack":
        acc = run_autoattack(model, batches, device, args.eps / 255)
        rows = [dict(model=name, attack="autoattack", eps_255=args.eps, n=args.n, acc=round(acc, 4),
                     git=git_hash())]
        print(f"AutoAttack eps={args.eps}/255: robust acc {acc:.4f} ({time.time() - t0:.0f}s)")
    else:
        r = run_cw(model, batches, device)
        rows = [dict(model=name, attack="cw_l2", n=args.n, clean=round(clean, 4),
                     fooled_rate=round(r["fooled_rate"], 4), median_l2=round(r["median_l2"], 4),
                     acc_at_l2_0_5=round(r["acc_at_l2_0_5"], 4), git=git_hash())]
        print(rows[0], f"({time.time() - t0:.0f}s)")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("saved", out)


if __name__ == "__main__":
    main()
