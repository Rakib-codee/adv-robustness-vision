"""Accuracy-vs-epsilon figure from results/attack_*.csv.
Run from repo root:  python scripts/plot_attack_curves.py
Writes results/fig_accuracy_vs_eps.png (and .pdf)."""
import csv
import glob
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["resnet18_std", "deit_tiny_std", "resnet18_fgsm_at", "resnet18_pgd_at", "resnet18_trades_b6"]
COLORS = {"resnet18_std": "#2a78d6", "deit_tiny_std": "#eb6834",        # categorical slots 1-4
          "resnet18_fgsm_at": "#1baf7a", "resnet18_pgd_at": "#eda100",
          "resnet18_trades_b6": "#e87ba4"}
MARKERS = {"resnet18_std": "o", "deit_tiny_std": "s", "resnet18_fgsm_at": "^", "resnet18_pgd_at": "D", "resnet18_trades_b6": "v"}
LABELS = {"resnet18_std": "ResNet-18 standard", "deit_tiny_std": "DeiT-Tiny standard",
          "resnet18_fgsm_at": "ResNet-18 FGSM-AT", "resnet18_pgd_at": "ResNet-18 PGD-AT",
          "resnet18_trades_b6": "ResNet-18 TRADES (β=6)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

data = defaultdict(dict)  # data[model][(attack, eps)] = acc
for path in sorted(glob.glob("results/attack_*.csv")):
    for r in csv.DictReader(open(path)):
        data[r["model"]][(r["attack"], int(r["eps_255"]))] = float(r["acc"]) * 100
        n = r["n"]

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
for ax, attack, title in zip(axes, ["fgsm", "pgd"], ["FGSM (1 step)", "PGD-20"]):
    ends = []
    for model in [m for m in ORDER if m in data]:
        d = data[model]
        eps = [0] + sorted(e for (a, e) in d if a == attack)
        acc = [d[("clean", 0)]] + [d[(attack, e)] for e in eps[1:]]
        ax.plot(eps, acc, color=COLORS[model], lw=2, marker=MARKERS[model], ms=6.5,
                mfc=COLORS[model], mec="white", mew=1.5, label=LABELS[model], zorder=3)
        ends.append((model, eps[-1], acc[-1]))
    last = None                                          # label end values, skip ones within 4 points
    for _, x, a in sorted(ends, key=lambda t: -t[2]):
        if last is None or last - a >= 4:
            ax.annotate(f"{a:.1f}%", (x, a), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
            last = a
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xticks([0, 2, 4, 8, 16])
    ax.set_xticklabels(["0\n(clean)", "2", "4", "8", "16"])
    ax.set_xlim(-0.6, 18.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("epsilon (x/255, L-inf)", fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.axvline(8, color=GRID, lw=1.2, ls="--", zorder=1)   # main eps
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
axes[0].set_ylabel("Accuracy (%)", fontsize=9, color=MUTED)
axes[1].legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)
fig.suptitle("Adversarial training lifts PGD-20 accuracy at eps 8/255 from 0% to 45-52%",
             x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
fig.text(0.01, 0.01, f"CIFAR-10, first {n} test images, seed 0. Dashed line = main epsilon 8/255.",
         fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.96))
for ext in ("png", "pdf"):
    fig.savefig(f"results/fig_accuracy_vs_eps.{ext}", dpi=200, facecolor="white")
print("saved results/fig_accuracy_vs_eps.png")
