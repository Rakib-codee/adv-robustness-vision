"""Merge all result CSVs into one comparison table (results/summary.csv + results/summary.md).
Run from repo root:  python scripts/make_summary.py
"""
import csv
from pathlib import Path

R = Path("results")
MODELS = [("resnet18_std", "ResNet-18 standard"), ("deit_tiny_std", "DeiT-Tiny standard"),
          ("resnet18_fgsm_at", "ResNet-18 FGSM-AT"), ("resnet18_pgd_at", "ResNet-18 PGD-AT"),
          ("resnet18_trades_b6", "ResNet-18 TRADES (beta=6)")]


def rows(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


def pct(x):
    return "-" if x is None else f"{100 * float(x):.1f}"


out = []
for key, label in MODELS:
    att = {(r["attack"], int(r["eps_255"])): r["acc"] for r in rows(R / f"attack_{key}_seed0.csv")}
    aa = rows(R / f"aa_{key}_seed0.csv")
    cw = rows(R / f"cw_{key}_seed0.csv")
    out.append({
        "model": label,
        "clean_acc_n1000": pct(att.get(("clean", 0))),
        "fgsm_eps8": pct(att.get(("fgsm", 8))),
        "pgd20_eps8": pct(att.get(("pgd", 8))),
        "pgd20_eps16": pct(att.get(("pgd", 16))),
        "autoattack_eps8_n500": pct(aa[0]["acc"]) if aa else "-",
        "cw_median_l2_n500": f"{float(cw[0]['median_l2']):.2f}" if cw else "-",
        "acc_at_l2_0.5_n500": pct(cw[0]["acc_at_l2_0_5"]) if cw else "-",
    })

with open(R / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

heads = ["Model", "Clean", "FGSM 8/255", "PGD-20 8/255", "PGD-20 16/255", "AutoAttack 8/255", "C&W median L2", "Acc @ L2<=0.5"]
lines = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
for r in out:
    lines.append("| " + " | ".join(r.values()) + " |")
lines += ["", "Accuracies in %. Clean, FGSM and PGD-20: first 1000 CIFAR-10 test images; AutoAttack and C&W: first 500. Seed 0, single run."]
(R / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
