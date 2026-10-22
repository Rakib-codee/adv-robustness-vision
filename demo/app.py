"""Gradio demo: attack 4 CIFAR-10 ResNet-18 models (standard / FGSM-AT /
PGD-AT / TRADES) with FGSM or PGD and watch their predictions and Grad-CAM
maps change. The hosted version of this demo is the Streamlit app
(streamlit_app.py); this Gradio variant is kept for local/HF-PRO use.

Run locally:  python demo/app.py   (needs torch, torchvision, gradio, matplotlib, pillow)
Checkpoints are expected in demo/checkpoints/<name>.pt with
names: resnet18_std, resnet18_fgsm_at, resnet18_pgd_at, resnet18_trades_b6.
"""
import io
import os
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from attack import (CLASSES, MODEL_FILES, available_labels, ensure_checkpoints,
                    fgsm, get_model, gradcam, pgd, predict, to_tensor)

HERE = Path(__file__).parent


def run(image, attack, eps255):
    if image is None:
        raise gr.Error("Upload an image or click an example first.")
    labels = available_labels()
    if not labels:
        raise gr.Error("No model checkpoints found in demo/checkpoints/.")
    x = to_tensor(image)
    eps = float(eps255) / 255
    n = len(MODEL_FILES)
    fig, axes = plt.subplots(n, 4, figsize=(9, 2.4 * n), squeeze=False)
    heads = ["clean", "Grad-CAM (clean)", f"{attack} eps={int(eps255)}/255", "Grad-CAM (attacked)"]
    table = ["| Model | Clean prediction | After attack | Result |", "|---|---|---|---|"]
    for r, (label, _) in enumerate(MODEL_FILES):
        model = get_model(label)
        if model is None:
            continue
        c0, p0 = predict(model, x)
        y = torch.tensor([c0])                      # untargeted: push away from the clean prediction
        if eps > 0:
            xa = fgsm(model, x, y, eps) if attack == "FGSM" else pgd(model, x, y, eps)
        else:
            xa = x
        c1, p1 = predict(model, xa)
        cam0, cam1 = gradcam(model, x, c0), gradcam(model, xa, c0)
        panels = [(x, None, f"{CLASSES[c0]} ({p0:.0%})"), (x, cam0, ""),
                  (xa, None, f"{CLASSES[c1]} ({p1:.0%})"), (xa, cam1, "")]
        for k, (t, cam, caption) in enumerate(panels):
            ax = axes[r, k]
            ax.imshow(t[0].permute(1, 2, 0).numpy(), interpolation="nearest")
            if cam is not None:
                ax.imshow(cam, cmap="jet", alpha=0.5, interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            if caption:
                ax.set_xlabel(caption, fontsize=9)
            if r == 0:
                ax.set_title(heads[k], fontsize=9)
            if k == 0:
                ax.set_ylabel(label, fontsize=10, fontweight="bold")
        result = "fooled" if c1 != c0 else "held"
        table.append(f"| {label} | {CLASSES[c0]} ({p0:.0%}) | {CLASSES[c1]} ({p1:.0%}) | **{result}** |")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy(), "\n".join(table)


INTRO = """# How robust are vision models?
Four ResNet-18 models trained on CIFAR-10: **Standard**, **FGSM-AT**, **PGD-AT** and **TRADES**.
Pick an image, choose an attack and a perturbation size (epsilon, out of 255), and see whether each model's prediction survives.

*Attacks are white-box: they are computed separately against each model. The attack pushes the image away from
the model's own clean prediction. Images are cropped to a square and resized to 32x32.*"""

EXAMPLES = sorted((HERE / "examples").glob("*.png"))
summary_path = HERE / "summary.md"
SUMMARY = summary_path.read_text() if summary_path.exists() else ""

with gr.Blocks(title="Adversarial Robustness Demo") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("Attack demo"):
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="pil", label="Input image")
                attack = gr.Radio(["FGSM", "PGD-10"], value="PGD-10", label="Attack")
                eps = gr.Slider(0, 16, value=8, step=1, label="Epsilon (x/255)")
                btn = gr.Button("Run attack", variant="primary")
                if EXAMPLES:
                    gr.Examples(examples=[[str(p)] for p in EXAMPLES], inputs=[img], label="CIFAR-10 test images")
            with gr.Column(scale=2):
                out_img = gr.Image(label="Predictions and Grad-CAM", type="pil")
                out_md = gr.Markdown()
        btn.click(run, [img, attack, eps], [out_img, out_md])
    with gr.Tab("Results"):
        gr.Markdown("## Evaluation summary (CIFAR-10, seed 0)\n" + SUMMARY)
        fig_path = HERE / "results_fig.png"
        if fig_path.exists():
            gr.Image(str(fig_path), label="Accuracy vs epsilon", interactive=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
