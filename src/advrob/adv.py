"""Attacks used INSIDE training (L-inf, pixel space [0,1]) and the three defense losses.

- fgsm_rs : FGSM with random start (Wong et al. 2020, "Fast is better than free")
- pgd     : multi-step PGD with random start (Madry et al. 2018)
- trades_loss : CE(clean) + beta * KL(clean || adversarial)  (Zhang et al. 2019)
"""
import torch
import torch.nn.functional as F


def fgsm_rs(model, x, y, eps, alpha):
    delta = (torch.rand_like(x) * 2 - 1) * eps
    delta = ((x + delta).clamp(0, 1) - x).requires_grad_(True)
    loss = F.cross_entropy(model(x + delta), y)
    grad, = torch.autograd.grad(loss, delta)
    delta = (delta.detach() + alpha * grad.sign()).clamp(-eps, eps)
    return (x + delta).clamp(0, 1).detach()


def pgd(model, x, y, eps, alpha, steps):
    delta = (torch.rand_like(x) * 2 - 1) * eps
    delta = ((x + delta).clamp(0, 1) - x)
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(x + delta), y)
        grad, = torch.autograd.grad(loss, delta)
        delta = (delta.detach() + alpha * grad.sign()).clamp(-eps, eps)
        delta = (x + delta).clamp(0, 1) - x
    return (x + delta).detach()


def trades_loss(model, x, y, eps, alpha, steps, beta):
    """Official TRADES recipe: attack generated with the model in eval mode."""
    model.eval()
    with torch.no_grad():
        p_nat = F.softmax(model(x), dim=1)
    x_adv = (x + 0.001 * torch.randn_like(x)).clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        kl = F.kl_div(F.log_softmax(model(x_adv), dim=1), p_nat, reduction="sum")
        grad, = torch.autograd.grad(kl, x_adv)
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)
    model.train()
    logits = model(x)
    loss_nat = F.cross_entropy(logits, y)
    loss_rob = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                        F.softmax(logits, dim=1), reduction="batchmean")
    return loss_nat + beta * loss_rob


def defense_loss(model, x, y, cfg):
    """Training loss for cfg['defense'] in {standard, fgsm_at, pgd_at, trades}."""
    d = cfg.get("defense", "standard")
    eps = cfg.get("eps_255", 8) / 255
    alpha = cfg.get("alpha_255", 2) / 255
    if d == "standard":
        return F.cross_entropy(model(x), y)
    if d == "fgsm_at":
        return F.cross_entropy(model(fgsm_rs(model, x, y, eps, alpha)), y)
    if d == "pgd_at":
        return F.cross_entropy(model(pgd(model, x, y, eps, alpha, cfg.get("steps", 7))), y)
    if d == "trades":
        return trades_loss(model, x, y, eps, alpha, cfg.get("steps", 10), cfg.get("beta", 6.0))
    raise ValueError(f"unknown defense: {d}")


def quick_robust_acc(model, x, y, steps=10, eps=8 / 255, alpha=2 / 255):
    """PGD-10 accuracy on a small fixed batch; used each epoch to spot catastrophic overfitting."""
    model.eval()
    x_adv = pgd(model, x, y, eps, alpha, steps)
    with torch.no_grad():
        return (model(x_adv).argmax(1) == y).float().mean().item()
