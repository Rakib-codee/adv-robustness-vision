"""Standard training with per-epoch checkpoints, resume, and optional W&B logging.

    python -m advrob.train --config configs/resnet18_cifar10.yaml --seed 0
Run from the repo root with PYTHONPATH=src (or `pip install -e .`).
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from .data import get_cifar10
from .models import build_model
from .utils import AverageMeter, git_hash, load_config, set_seed


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[], help="overrides, e.g. epochs=1")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = f"{cfg['name']}_seed{args.seed}"
    ckpt_dir = Path(args.out) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last = ckpt_dir / "last.pt"

    train_loader, test_loader = get_cifar10(cfg.get("data_root", "data"),
                                            cfg["batch_size"], cfg.get("num_workers", 2))
    model = build_model(cfg["model"], **cfg.get("model_args", {})).to(device)

    if cfg["optimizer"] == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9,
                              weight_decay=cfg.get("weight_decay", 5e-4), nesterov=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg.get("weight_decay", 0.05))
    epochs = cfg["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss()

    start_epoch = 0
    if last.exists():  # resume after a Kaggle/Colab timeout
        state = torch.load(last, map_location=device)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"]); start_epoch = state["epoch"] + 1
        print(f"resumed from epoch {start_epoch}")

    wb = None
    if not args.no_wandb:
        try:
            import wandb
            wb = wandb.init(project=cfg.get("wandb_project", "adv-robustness-vision"),
                            name=run_name, config={**cfg, "seed": args.seed, "git": git_hash()},
                            id=run_name, resume="allow")
        except Exception as e:  # never let logging kill a training run
            print("wandb disabled:", e)

    for epoch in range(start_epoch, epochs):
        model.train()
        meter, t0 = AverageMeter(), time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            meter.update(loss.item(), y.numel())
        acc = evaluate(model, test_loader, device)
        sec = time.time() - t0
        print(f"epoch {epoch+1}/{epochs} loss {meter.avg:.4f} clean_acc {acc:.4f} {sec:.0f}s")
        if wb:
            wb.log({"epoch": epoch + 1, "train_loss": meter.avg, "clean_acc": acc,
                    "epoch_seconds": sec})
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch, "cfg": cfg,
                    "seed": args.seed, "git": git_hash()}, last)

    torch.save({"model": model.state_dict(), "cfg": cfg, "seed": args.seed,
                "git": git_hash(), "clean_acc": acc}, ckpt_dir / "final.pt")
    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
