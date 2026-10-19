"""Accuracy-vs-epsilon figure from results/attack_*.csv.
Run from repo root:  python scripts/plot_attack_curves.py
Writes results/fig_accuracy_vs_eps.png (and .pdf)."""
import csv
import glob
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"resnet18_std": "#2a78d6", "deit_tiny_std": "#eb6834"}   # categorical slots 1, 2
LABELS = {"resnet18_std": "ResNet-18", "deit_tiny_std": "DeiT-Tiny"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

data = defaultdict(dict)  # data[model][(attack, eps)] = acc
for path in sorted(glob.glob("results/attack_*.csv")):
    for r in csv.DictReader(open(path)):
        data[r["model"]][(r["attack"], int(r["eps_255"]))] = float(r["acc"]) * 100
        n = r["n"]

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
for ax, attack, title in zip(axes, ["fgsm", "pgd"], ["FGSM (1 step)", "PGD-20"]):
    ends = []
    for model, d in data.items():
        eps = [0] + sorted(e for (a, e) in d if a == attack)
        acc = [d[("clean", 0)]] + [d[(attack, e)] for e in eps[1:]]
        ax.plot(eps, acc, color=COLORS[model], lw=2, marker="o", ms=6,
                mfc=COLORS[model], mec="white", mew=1.5, label=LABELS[model], zorder=3)
        ends.append((model, eps[-1], acc[-1]))
    if len({round(a, 1) for _, _, a in ends}) == 1:      # identical end values: label once
        ax.annotate(f"{ends[0][2]:.1f}% (both)", (ends[0][1], ends[0][2]), xytext=(-4, 12),
                    textcoords="offset points", ha="right", fontsize=9, color=INK)
    else:
        for _, x, a in ends:
            ax.annotate(f"{a:.1f}%", (x, a), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
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
axes[1].legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
fig.suptitle("Standard models collapse under PGD: 0% at epsilon = 8/255 for both",
             x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
fig.text(0.01, 0.01, f"CIFAR-10, first {n} test images, seed 0. Dashed line = main epsilon 8/255.",
         fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.96))
for ext in ("png", "pdf"):
    fig.savefig(f"results/fig_accuracy_vs_eps.{ext}", dpi=200, facecolor="white")
print("saved results/fig_accuracy_vs_eps.png")
