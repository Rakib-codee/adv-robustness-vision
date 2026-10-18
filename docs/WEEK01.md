# Week 1 guide (Oct 19-25): Reading + Setup
Gate: `python scripts/smoke_test.py` passes on a Kaggle GPU and one epoch logs to W&B.

## Steps, in order
1. **GitHub repo (10 min).** Create an empty repo `adv-robustness-vision` on GitHub (no README). Then locally: `git remote add origin <url> && git push -u origin main`.
2. **Accounts (15 min).** W&B (wandb.ai) -> copy API key. Kaggle -> verify phone (needed for GPU) -> Settings -> enable Internet. Hugging Face account (needed in Week 9).
3. **Secrets (5 min).** Kaggle notebook -> Add-ons -> Secrets: `WANDB_API_KEY`. Never paste keys in cells.
4. **Smoke test (20 min).** Follow `notebooks/kaggle_runner.md`, Cells 1-3 with `smoke_test.py` only. Send me the output.
5. **One real epoch (30 min).** `python -m advrob.train --config configs/resnet18_cifar10.yaml --set epochs=1 --seed 0`. Write the epoch time in `docs/TIMINGS.md`. This starts our GPU budget.
6. **Download CIFAR-10-C** (Zenodo record 2535967, `CIFAR-10-C.tar`) into a Kaggle dataset so later weeks don't re-download.
7. **Reading (~8 h total).** Order: FGSM -> PGD -> TRADES -> C&W -> AutoAttack. For each paper: 3 lines in `docs/papers.csv` Key Takeaway (what is the attack/defense, what is eps, what is the main claim). Commit: `docs: takeaways for <paper>`.
8. **Tag:** `git tag week01-done && git push --tags`.

## Branch for this week
`git switch -c week01/setup` then merge to main with a PR.
