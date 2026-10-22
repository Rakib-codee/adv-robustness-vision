# How Robust Are Vision Models?

A comparative study of **adversarial attacks and defenses** on CNNs (ResNet-18) and
Vision Transformers (DeiT-Tiny) trained on CIFAR-10.

[![Live demo](https://img.shields.io/badge/demo-streamlit-ff4b4b)](https://adv-robustness-vision.streamlit.app/)
[![Checkpoints](https://img.shields.io/badge/checkpoints-hugging%20face-ffd21e)](https://huggingface.co/nexgendev/adv-robustness-vision-ckpts)

**Research questions**

1. **RQ1** — Which is more robust to adversarial attacks, ResNet or ViT?
2. **RQ2** — How much robustness does adversarial training add, and what does it cost in clean accuracy?
3. **RQ3** — How do Grad-CAM explanations change under attack, and are robust models' explanations more object-focused?

---

## TL;DR

| Model | Clean | FGSM 8/255 | PGD-20 8/255 | PGD-20 16/255 | AutoAttack 8/255 | C&W median L₂ | Acc @ L₂ ≤ 0.5 |
|---|---|---|---|---|---|---|---|
| ResNet-18 standard | 94.3 | 13.1 | 0.0 | 0.0 | 0.0 | 0.20 | 0.2 |
| DeiT-Tiny standard | 96.0 | 17.4 | 0.0 | 0.0 | – | – | – |
| ResNet-18 FGSM-AT | 84.2 | 56.6 | 44.7 | 11.4 | 40.4 | 0.70 | 54.0 |
| ResNet-18 PGD-AT | 81.0 | 54.9 | 50.6 | 17.9 | 44.0 | 0.84 | 58.8 |
| ResNet-18 TRADES (β=6) | 80.3 | 55.5 | 51.7 | 22.8 | 45.8 | 0.82 | 57.0 |

Accuracies in %. Clean, FGSM and PGD-20: first 1000 CIFAR-10 test images;
AutoAttack and C&W: first 500. Seed 0, single run. See [`results/`](results) for
per-attack CSVs and [`demo/summary.md`](demo/summary.md) for the live table.

**Headline findings**

- **Standard models have no adversarial robustness.** PGD-20 at ε = 8/255 drops
  both ResNet-18 and DeiT-Tiny from >94% clean accuracy to **0%**; AutoAttack
  confirms this is not gradient masking (AutoAttack = 0.0).
- **ViT is *slightly* more robust than ResNet on the weakest attack** (FGSM:
  17.4 vs 13.1), but the margin vanishes under any stronger attack — so for
  **RQ1**, architecture alone buys almost nothing.
- **Adversarial training works, at a real clean-accuracy cost (RQ2).**
  PGD-AT survives AutoAttack at 44.0% (vs 0.0) while clean accuracy falls
  94.3 → 81.0 (−13.3). TRADES (β=6) is the most robust at PGD-20 ε=16/255
  (22.8) and nearly matches PGD-AT under AutoAttack (45.8), while C&W finds
  PGD-AT slightly more robust in L₂ (median 0.84 vs 0.82).
- **Robust models keep object-focused explanations under attack (RQ3).**
  Grad-CAM correlation between clean and attacked maps at ε = 8/255:
  PGD-AT **0.947** and TRADES **0.938**, vs ResNet-18 standard **0.006** and
  DeiT-Tiny **0.129** (n = 500; [`results/gradcam_similarity.csv`](results/gradcam_similarity.csv)).
  Robust models not only resist the attack — their attention barely moves.

---

## Live demo

**[adv-robustness-vision.streamlit.app](https://adv-robustness-vision.streamlit.app/)**
— upload any image, choose FGSM or PGD-10 and ε, and watch the four ResNet-18
models' predictions and Grad-CAM maps change under the attack. Free hosting on
Streamlit Community Cloud; checkpoints stream from a Hugging Face model repo on
first load, and models are lazy-loaded one at a time to fit the 1 GB tier.

The same backend powers an interactive **Gradio** variant ([`demo/app.py`](demo/app.py))
for local use.

## Repository layout

```
src/advrob/          library code
  train.py           standard + adversarial training (FGSM-AT, PGD-AT, TRADES)
  attack_eval.py     accuracy under FGSM / PGD sweeps
  strong_eval.py     AutoAttack (L∞) and C&W (L₂)
  transfer.py        black-box transfer attacks
  gradcam.py         Grad-CAM + clean-vs-attacked map similarity metrics
  models.py, data.py, adv.py, evaluate.py, loading.py, utils.py
configs/             one YAML per experiment — results reproducible from config + seed + commit
demo/                interactive demo (Streamlit + Gradio UIs, shared attack.py backend)
results/             result tables (CSV) and figures that go in the report — tracked in git
scripts/             one-off helpers (smoke tests, summary tables, Grad-CAM analysis, plots)
notebooks/           thin Kaggle/Colab runners only
docs/                locked decisions, weekly guides, paper tracker, timings
tests/               unit tests (pytest)
```

Design rules: every reported number comes from a config + seed + git commit and
is logged to W&B; checkpoints are never committed — released ones live in a HF
[model repo](https://huggingface.co/nexgendev/adv-robustness-vision-ckpts).

## Quick start

```bash
git clone https://github.com/Rakib-codee/adv-robustness-vision.git
cd adv-robustness-vision
pip install -r requirements.txt
export PYTHONPATH=src

python scripts/smoke_test.py                                   # ~1 min: GPU + code path check
python -m advrob.train --config configs/resnet18_cifar10.yaml --seed 0
```

### Training

```bash
python -m advrob.train --config configs/resnet18_fgsm_at.yaml  --seed 0   # FGSM adversarial training
python -m advrob.train --config configs/resnet18_pgd_at.yaml   --seed 0   # PGD-10 adversarial training
python -m advrob.train --config configs/resnet18_trades.yaml   --seed 0   # TRADES, beta = 6
python -m advrob.train --config configs/deit_tiny_cifar10.yaml --seed 0   # DeiT-Tiny baseline (timm)
```

### Evaluation

```bash
python -m advrob.attack_eval  --run <run_name> --eps 0 2 4 8 12 16          # FGSM/PGD curves
python -m advrob.strong_eval  --artifact <entity/project/run:latest> \
       --attack autoattack --n 1000 --out results/aa_<run>.csv              # AutoAttack
python -m advrob.transfer     --source <artifact> --targets <a1>,<a2>       # black-box transfer
python scripts/gradcam_analysis.py                                          # Grad-CAM similarity
```

## Methodology

- **Models.** ResNet-18 and DeiT-Tiny on CIFAR-10; inputs in [0,1] with
  normalization inside the model (decision D6), so ε is measured in pixel space.
- **Threat model.** L∞, main ε = 8/255; white-box untargeted attacks computed
  per model. Strong attacks: PGD-20, AutoAttack (avoids gradient-masking
  artefacts — decision D7), plus C&W in L₂ for a different norm.
- **Defenses.** FGSM-AT, PGD-AT (10-step), TRADES (β = 6) on ResNet-18.
- **Explanations.** Grad-CAM (ResNet `layer4`, DeiT last-block tokens);
  robustness of explanations measured by clean-vs-attacked map correlation
  and top-20% IoU.

## Reproducibility & docs

- Every number traces to a W&B run with config and git hash (decision D8;
  final paper tables will use 3 seeds, mean ± std).
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — locked design decisions, each
  changed only in its own commit with the reason.
- [`docs/GIT_WORKFLOW.md`](docs/GIT_WORKFLOW.md) — branch/PR/commit conventions.
- [`docs/papers.csv`](docs/papers.csv) — paper tracker with key takeaways.
- [`docs/TIMINGS.md`](docs/TIMINGS.md) — GPU budget timings.
- Tests: `pytest tests/`.

## Links

- **Live demo:** https://adv-robustness-vision.streamlit.app
- **Checkpoints:** https://huggingface.co/nexgendev/adv-robustness-vision-ckpts
- W&B project: TBD · Report: TBD

---

*10-week research project (started 2026-10-19). Status: adversarial-training
and evaluation complete for seed 0; multi-seed runs and the final report are in
progress. See [`docs/WEEK01.md`](docs/WEEK01.md) for the week-by-week plan.*
