"""Streamlit demo: attack 4 CIFAR-10 ResNet-18 models (standard / FGSM-AT /
PGD-AT / TRADES) with FGSM or PGD and watch their predictions and Grad-CAM
maps change. This is the version deployed on Streamlit Community Cloud.

Run locally:  streamlit run demo/streamlit_app.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import torch
from PIL import Image

from attack import (CLASSES, MODEL_FILES, available_labels, ensure_checkpoints,
                    fgsm, get_model, gradcam, pgd, predict, to_tensor)

HERE = Path(__file__).parent

st.set_page_config(page_title="Adversarial Robustness Demo",
                   page_icon="🛡️", layout="wide")

HEADER = """
<div style="background:linear-gradient(100deg,#16213e 0%,#1f3b73 55%,#3a6fc4 100%);
border-radius:14px;padding:26px 30px;margin-bottom:6px;">
  <div style="color:#ffffff;font-size:27px;font-weight:800;line-height:1.25;">
    How robust are vision models?</div>
  <div style="color:#c9d9f5;font-size:14px;margin-top:6px;">
    Four ResNet-18 models trained on CIFAR-10 — <b>Standard</b>, <b>FGSM-AT</b>,
    <b>PGD-AT</b> and <b>TRADES</b>. Pick an image, choose an attack and a
    perturbation size (&epsilon;, out of 255), and see whether each model's
    prediction survives.</div>
</div>
"""

FOOTER = """
<div style="text-align:center;color:#9aa5b1;font-size:12px;margin-top:36px;padding-top:12px;
border-top:1px solid #e6e9ed;">
Attacks are white-box and untargeted, computed per model. Images are centre-cropped to a
square and resized to 32&times;32.<br>
<a href="https://github.com/Rakib-codee/adv-robustness-vision">GitHub — code, training logs &amp; full evaluation</a>
</div>
"""


def card_html(row):
    fooled = row["Result"] == "fooled"
    badge_bg, badge_fg, badge = ("#fdecea", "#c0392b", "FOOLED") if fooled \
        else ("#e9f9f0", "#1e8449", "HELD")
    return f"""
    <div style="border:1px solid #e3e7ec;border-radius:12px;padding:14px 8px;
    text-align:center;background:#ffffff;box-shadow:0 1px 4px rgba(20,40,80,.07);">
      <div style="font-weight:700;font-size:15px;color:#16213e;">{row['Model']}</div>
      <div style="color:#5c6670;font-size:13px;margin-top:8px;">{row['Clean prediction']}</div>
      <div style="font-size:16px;color:#aeb8c2;line-height:1.1;">↓</div>
      <div style="color:#5c6670;font-size:13px;">{row['After attack']}</div>
      <div style="margin-top:9px;">
        <span style="background:{badge_bg};color:{badge_fg};padding:3px 12px;
        border-radius:11px;font-size:11.5px;font-weight:800;letter-spacing:.4px;">{badge}</span>
      </div>
    </div>"""


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
    st.markdown(HEADER, unsafe_allow_html=True)

    with st.spinner("Loading model checkpoints (first run downloads ~171 MB from Hugging Face)..."):
        ensure_checkpoints()
    if not available_labels():
        st.error("No model checkpoints found in demo/checkpoints/.")
        st.stop()

    tab_demo, tab_results = st.tabs(["🎯 Attack demo", "📊 Results"])

    with tab_demo:
        col_ctrl, col_out = st.columns([1, 2.2])
        with col_ctrl:
            st.markdown("**1. Upload an image**")
            image = st.file_uploader("Input image", type=["png", "jpg", "jpeg"],
                                     label_visibility="collapsed")
            st.markdown("**2. Choose the attack**")
            attack = st.selectbox("Attack", ["FGSM", "PGD-10"], index=1,
                                  label_visibility="collapsed")
            st.markdown("**3. Perturbation budget** — ε (x/255)")
            eps255 = st.slider("Epsilon (x/255)", 0, 16, 8, label_visibility="collapsed")
            run = st.button("⚔️ Run attack", type="primary", width="stretch")
        with col_out:
            if run:
                if image is None:
                    st.warning("Upload an image first.")
                else:
                    img = Image.open(image)
                    st.caption(f"Input: {img.size[0]}×{img.size[1]} "
                               "(cropped to square, resized to 32×32)")
                    with st.spinner("Running white-box attacks on all 4 models "
                                    "(PGD takes ~30s on free hosting)..."):
                        fig, rows = run_attack(img, attack, eps255)
                    fooled = sum(r["Result"] == "fooled" for r in rows)
                    st.markdown(f"### {fooled}/{len(rows)} models fooled "
                                f"— {attack}, ε={int(eps255)}/255")
                    cols = st.columns(len(rows))
                    for col, row in zip(cols, rows):
                        col.markdown(card_html(row), unsafe_allow_html=True)
                    with st.expander("View as table"):
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

    st.markdown(FOOTER, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
