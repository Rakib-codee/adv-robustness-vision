"""Transfer attack: craft PGD examples on a source model, test them on other models.

  python -m advrob.transfer --source <artifact> --targets <a1>,<a2>,... --n 1000 --out results/transfer.csv

Transfer success rate = share of images the target classified correctly (clean) that the
source-crafted adversarial image fools. The source's own row is the white-box rate.
"""
import argparse
import csv
from pathlib import Path

import torch
import torchattacks

from .attack_eval import first_n_test
from .loading import load_model
from .utils import git_hash, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--targets", required=True, help="comma-separated W&B artifacts")
    p.add_argument("--eps", type=int, default=8)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    src, src_name = load_model(args.source, device=device)
    tgts = [load_model(a, device=device) for a in [args.source] + args.targets.split(",")]
    eps = args.eps / 255
    atk = torchattacks.PGD(src, eps=eps, alpha=2.5 * eps / args.steps, steps=args.steps, random_start=True)

    batches = first_n_test(args.n, args.batch)
    fooled = {name: 0 for _, name in tgts}
    clean_ok = {name: 0 for _, name in tgts}
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        x_adv = atk(x, y)
        with torch.no_grad():
            for m, name in tgts:
                ok = m(x).argmax(1) == y
                bad = m(x_adv).argmax(1) != y
                clean_ok[name] += ok.sum().item()
                fooled[name] += (ok & bad).sum().item()
    rows = []
    for _, name in tgts:
        rate = fooled[name] / max(clean_ok[name], 1)
        print(f"{src_name} -> {name}: attack success {rate:.4f}")
        rows.append(dict(source=src_name, target=name, eps_255=args.eps, n=args.n,
                         success_rate=round(rate, 4), git=git_hash()))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("saved", out)


if __name__ == "__main__":
    main()
