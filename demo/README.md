# Adversarial Robustness Demo

Interactive Gradio demo: attack four ResNet-18 models trained on CIFAR-10
(standard, FGSM-AT, PGD-AT, TRADES) with FGSM or PGD and watch their
predictions and Grad-CAM maps change.

Code, training logs and full evaluation (AutoAttack, C&W, transfer attacks):
https://github.com/Rakib-codee/adv-robustness-vision

## Run locally

```bash
pip install -r requirements.txt   # CPU torch wheels via the PyTorch extra index
python app.py                     # http://localhost:7860
```

Checkpoints are expected in `checkpoints/<name>.pt` with names
`resnet18_std`, `resnet18_fgsm_at`, `resnet18_pgd_at`, `resnet18_trades_b6`.
They are gitignored; the Docker image downloads them at build time from
https://huggingface.co/nexgendev/adv-robustness-vision-ckpts (model repos are
free; only Gradio Spaces on HF require a paid plan, which is why this demo is
hosted on Render instead).

## Run with Docker

```bash
docker build -t adv-robustness-demo .
docker run -p 7860:7860 adv-robustness-demo
```

## Deploy on Render

The repo root contains `render.yaml` (Render Blueprint). In the Render
dashboard: **New → Blueprint** → connect the GitHub repo → the blueprint
creates the free web service from this directory's Dockerfile.
