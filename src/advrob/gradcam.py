"""Grad-CAM for ResNet-18 (layer4) and DeiT-Tiny (last block tokens), plus map-similarity metrics."""
import torch
import torch.nn.functional as F


def target_layer(model):
    net = model.net
    if hasattr(net, "layer4"):
        return net.layer4, "cnn"
    return net.blocks[-1].norm1, "vit"          # DeiT-Tiny: tokens before the last attention block


def gradcam(model, x, cls=None):
    """x: (B,3,32,32) in [0,1]. Returns (cam in [0,1] of shape (B,32,32), predicted classes)."""
    layer, kind = target_layer(model)
    store = {}
    h = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    x = x.clone().requires_grad_(False)
    with torch.enable_grad():
        logits = model(x)
        cls = logits.argmax(1) if cls is None else cls
        score = logits.gather(1, cls[:, None]).sum()
        act = store["a"]
        grad = torch.autograd.grad(score, act)[0]
    h.remove()
    if kind == "cnn":                               # (B,C,H,W)
        w = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * act).sum(1, keepdim=True))
    else:                                           # (B,1+N,C): drop cls token, reshape to grid
        a, g = act[:, 1:], grad[:, 1:]
        w = g.mean(dim=1, keepdim=True)
        cam = F.relu((w * a).sum(-1))
        s = int(cam.shape[1] ** 0.5)
        cam = cam.reshape(-1, 1, s, s)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    mn = cam.flatten(1).min(1)[0][:, None, None]
    mx = cam.flatten(1).max(1)[0][:, None, None]
    return ((cam - mn) / (mx - mn + 1e-8)).detach(), cls.detach()


def cam_similarity(c1, c2, top=0.2):
    """Per-image Pearson correlation and IoU of the top-`top` fraction of pixels, clean vs adversarial CAM."""
    a, b = c1.flatten(1), c2.flatten(1)
    a0, b0 = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    corr = (a0 * b0).sum(1) / (a0.norm(dim=1) * b0.norm(dim=1) + 1e-8)
    k = int(a.shape[1] * top)
    ta = torch.zeros_like(a).scatter_(1, a.topk(k, dim=1)[1], 1.0)
    tb = torch.zeros_like(b).scatter_(1, b.topk(k, dim=1)[1], 1.0)
    iou = (ta * tb).sum(1) / ((ta + tb).clamp(max=1).sum(1) + 1e-8)
    return corr, iou
