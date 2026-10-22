"""Grad-CAM before/after PGD for several models: figure + similarity CSV.

  python scripts/gradcam_analysis.py --models resnet18_std=<artifact>,deit_tiny_std=<artifact>,resnet18_pgd_at=<artifact> \
      --n 500 --eps 8
Outputs results/gradcam_grid.png and results/gradcam_similarity.csv (run from repo root, PYTHONPATH=src).
"""
import argparse
import csv
import sys
sys.path.insert(0, "src")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchattacks

from advrob.attack_eval import first_n_test
from advrob.gradcam import cam_similarity, gradcam
from advrob.loading import load_model

CLASSES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

p = argparse.ArgumentParser()
p.add_argument("--models", required=True, help="label=artifact,label=artifact,...")
p.add_argument("--n", type=int, default=500)
p.add_argument("--eps", type=int, default=8)
p.add_argument("--show", default="0,1,2,3", help="indices of images to plot")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
batches = first_n_test(args.n, 100)
x_all = torch.cat([b[0] for b in batches]); y_all = torch.cat([b[1] for b in batches])
show = [int(i) for i in args.show.split(",")]
eps = args.eps / 255
specs = [s.split("=", 1) for s in args.models.split(",")]

rows, grid = [], {}
for label, art in specs:
    model, _ = load_model(art, device=device)
    atk = torchattacks.PGD(model, eps=eps, alpha=2.5 * eps / 20, steps=20, random_start=True)
    corrs, ious, corrs_p, ious_p, vis = [], [], [], [], {}
    for bi, (x, y) in enumerate(batches):
        x, y = x.to(device), y.to(device)
        x_adv = atk(x, y)
        c_clean, p_clean = gradcam(model, x)
        c_adv, p_adv = gradcam(model, x_adv)                       # CAM of the class predicted on x_adv
        c_adv_fix, _ = gradcam(model, x_adv, cls=p_clean)          # CAM of the SAME class as on clean x
        corr, iou = cam_similarity(c_clean, c_adv_fix)
        corr_p, iou_p = cam_similarity(c_clean, c_adv)
        corrs.append(corr.cpu()); ious.append(iou.cpu())
        corrs_p.append(corr_p.cpu()); ious_p.append(iou_p.cpu())
        for i in show:
            if bi * 100 <= i < (bi + 1) * 100:
                j = i - bi * 100
                vis[i] = (x[j].cpu(), c_clean[j].cpu(), x_adv[j].cpu(), c_adv[j].cpu(),
                          p_clean[j].item(), p_adv[j].item())
    corr, iou = torch.cat(corrs), torch.cat(ious)
    corr_p, iou_p = torch.cat(corrs_p), torch.cat(ious_p)
    rows.append(dict(model=label, eps_255=args.eps, n=args.n,
                     cam_corr=round(corr.mean().item(), 4), cam_top20_iou=round(iou.mean().item(), 4),
                     cam_corr_predclass=round(corr_p.mean().item(), 4),
                     cam_top20_iou_predclass=round(iou_p.mean().item(), 4)))
    print(rows[-1])
    grid[label] = vis

with open("results/gradcam_similarity.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# figure: one block per model; columns = clean, CAM, attacked, CAM; rows = images
nm, ni = len(specs), len(show)
fig, axes = plt.subplots(ni, 4 * nm, figsize=(2.0 * 4 * nm, 2.1 * ni))
for mi, (label, _) in enumerate(specs):
    for r, i in enumerate(show):
        x, c1, xa, c2, p1, p2 = grid[label][i]
        for k, (img, cam) in enumerate([(x, None), (x, c1), (xa, None), (xa, c2)]):
            ax = axes[r, 4 * mi + k]
            ax.imshow(img.permute(1, 2, 0).numpy())
            if cam is not None:
                ax.imshow(cam.numpy(), cmap="jet", alpha=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(["clean", "CAM", f"PGD {args.eps}/255", "CAM"][k], fontsize=8)
            if k in (0, 2):
                ax.set_xlabel(CLASSES[p1 if k == 0 else p2], fontsize=8)
    axes[0, 4 * mi + 1].text(1.1, 1.35, label, transform=axes[0, 4 * mi + 1].transAxes,
                             ha="center", fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig("results/gradcam_grid.png", dpi=200)
print("saved results/gradcam_grid.png and results/gradcam_similarity.csv")
