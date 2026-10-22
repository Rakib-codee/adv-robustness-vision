"""Streamlit demo: attack 4 CIFAR-10 ResNet-18 models (standard / FGSM-AT /
PGD-AT / TRADES) with FGSM or PGD and watch their predictions and Grad-CAM
maps change. This is the version deployed on Streamlit Community Cloud.

Run locally:  streamlit run demo/streamlit_app.py
"""
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

from attack import (CLASSES, MODEL_FILES, available_labels, ensure_checkpoints,
                    fgsm, get_model, gradcam, pgd, predict, to_tensor)

HERE = Path(__file__).parent

st.set_page_config(page_title="Adversarial Robustness Demo", layout="wide")

INTRO = """Four ResNet-18 models trained on CIFAR-10: **Standard**, **FGSM-AT**,
**PGD-AT** and **TRADES**. Pick an image, choose an attack and a perturbation
size (epsilon, out of 255), and see whether each model's prediction survives.

*Attacks are white-box: they are computed separately against each model. The
attack pushes the image away from the model's own clean prediction. Images are
cropped to a square and resized to 32x32.*"""


def run_attack(image, attack, eps255):
    x = to_tensor(image)
    eps = float(eps255) / 255
    n = len(MODEL_FILES)
    fig, axes = plt.subplots(n, 4, figsize=(9, 2.4 * n), squeeze=False)
    heads = ["clean", "Grad-CAM (clean)", f"{attack} eps={int(eps255)}/255", "Grad-CAM (attacked)"]
    rows = []
    for r, (label, _) in enumerate(MODEL_FILES):
        model = get_model(label)
        if model is None:
            continue
        c0, p0 = predict(model, x)
        y = torch.tensor([c0])              # untargeted: push away from the clean prediction
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
        rows.append({"Model": label,
                     "Clean prediction": f"{CLASSES[c0]} ({p0:.0%})",
                     "After attack": f"{CLASSES[c1]} ({p1:.0%})",
                     "Result": "fooled" if c1 != c0 else "held"})
    fig.tight_layout()
    return fig, rows


def main():
    st.title("How robust are vision models?")
    st.markdown(INTRO)

    with st.spinner("Loading model checkpoints (first run downloads ~171 MB from Hugging Face)..."):
        ensure_checkpoints()
    labels = available_labels()
    if not labels:
        st.error("No model checkpoints found in demo/checkpoints/.")
        st.stop()

    tab_demo, tab_results = st.tabs(["Attack demo", "Results"])

    with tab_demo:
        col_ctrl, col_out = st.columns([1, 2])
        with col_ctrl:
            image = st.file_uploader("Input image", type=["png", "jpg", "jpeg"])
            attack = st.selectbox("Attack", ["FGSM", "PGD-10"], index=1)
            eps255 = st.slider("Epsilon (x/255)", 0, 16, 8)
            run = st.button("Run attack", type="primary")
        with col_out:
            if run:
                if image is None:
                    st.warning("Upload an image first.")
                else:
                    img = Image.open(image)
                    st.caption(f"Input: {img.size[0]}x{img.size[1]} (cropped to square, resized to 32x32)")
                    with st.spinner("Running attacks on all 4 models (PGD takes ~30s on free hosting)..."):
                        fig, rows = run_attack(img, attack, eps255)
                    st.dataframe(rows, width="stretch", hide_index=True)
                    st.pyplot(fig)
                    plt.close(fig)

    with tab_results:
        st.markdown("## Evaluation summary (CIFAR-10, seed 0)")
        summary_path = HERE / "summary.md"
        if summary_path.exists():
            st.markdown(summary_path.read_text())
        fig_path = HERE / "results_fig.png"
        if fig_path.exists():
            st.image(str(fig_path), caption="Accuracy vs epsilon")


if __name__ == "__main__":
    main()
