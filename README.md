# How Robust Are Vision Models?
A comparative study of adversarial attacks and defenses on CNNs (ResNet-18) and Vision Transformers (DeiT-Tiny) on CIFAR-10.

**Research questions**
1. RQ1: Which is more robust to adversarial attacks, ResNet or ViT?
2. RQ2: How much robustness does adversarial training add, and what does it cost in clean accuracy?
3. RQ3: How do Grad-CAM explanations change under attack, and are robust models' explanations more object-focused?

**Status:** Week 1 of 10 (started 2026-10-19). See `docs/DECISIONS.md` and `docs/WEEK01.md`.

## Quick start (Kaggle / Colab)
```bash
git clone <YOUR_REPO_URL> && cd adv-robustness-vision
pip install -r requirements.txt
export PYTHONPATH=src
python scripts/smoke_test.py                       # ~1 min, checks GPU + code path
python -m advrob.train --config configs/resnet18_cifar10.yaml --seed 0
```

## Layout
- `src/advrob/` library code (data, models, training, evaluation)
- `configs/` one YAML per experiment; results must be reproducible from config + seed + commit
- `scripts/` one-off helpers; `notebooks/` thin Kaggle/Colab runners only
- `results/` small tables and figures that go in the paper (tracked); checkpoints are NOT tracked
- `docs/` decisions, weekly guides, paper tracker (`papers.csv`)

## Results
_Filled in as experiments finish (Week 3 onward)._

## Links
- W&B project: TBD · Demo: https://adv-robustness-vision.streamlit.app · Report: TBD
