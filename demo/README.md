# Adversarial Robustness Demo

Interactive demo: attack four ResNet-18 models trained on CIFAR-10
(standard, FGSM-AT, PGD-AT, TRADES) with FGSM or PGD and watch their
predictions and Grad-CAM maps change.

Code, training logs and full evaluation (AutoAttack, C&W, transfer attacks):
https://github.com/Rakib-codee/adv-robustness-vision

## Two UIs, one shared backend

- `streamlit_app.py` — the **hosted** version (Streamlit Community Cloud, free).
- `app.py` — a Gradio variant, kept for local use or a HF PRO Space.
- `attack.py` — shared logic (model loading, FGSM/PGD, Grad-CAM) used by both.

## Run locally

```bash
pip install -r requirements.txt   # CPU torch wheels via the PyTorch extra index
streamlit run streamlit_app.py    # http://localhost:8501
# or the Gradio variant:
pip install gradio && python app.py
```

Checkpoints are expected in `checkpoints/<name>.pt`
(`resnet18_std`, `resnet18_fgsm_at`, `resnet18_pgd_at`, `resnet18_trades_b6`).
They are gitignored; if missing, the apps download them automatically on first
run from https://huggingface.co/nexgendev/adv-robustness-vision-ckpts (model
repos are free on HF; only Gradio Spaces require a paid plan, which is why the
demo is hosted on Streamlit Cloud rather than Hugging Face).

## Deploy on Streamlit Community Cloud

1. https://share.streamlit.io → sign in with GitHub (free, no card).
2. **New app** → repo `Rakib-codee/adv-robustness-vision`, main file path
   `demo/streamlit_app.py`, branch `main`.
3. The platform installs `demo/requirements.txt` automatically; the app
   downloads the checkpoints from HF on first run.
