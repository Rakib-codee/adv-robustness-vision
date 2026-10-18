# Locked decisions (2026-10-19)
Change one only by editing this file in its own commit, with the reason.

| # | Decision | Reason |
|---|---|---|
| D1 | Week 1 starts Mon 2026-10-19; numbers freeze 2026-12-06; report due 2026-12-27 | 10-week plan |
| D2 | Adversarial training on ResNet-18 only (FGSM-AT, PGD-AT, TRADES, 3 seeds). DeiT-Tiny gets FGSM-AT x1 seed only if Week 6 has spare GPU hours | GPU budget |
| D3 | Deliverable: report + public repo + Hugging Face demo. arXiv/workshop is optional, decided in Week 9 | Scope control |
| D4 | Kaggle is the main GPU, Colab is the backup | Free quotas |
| D5 | Language: English for code, README, report | Admissions |
| D6 | All model inputs in [0,1]; normalization inside the model; eps measured in pixel space, main eps = 8/255, L-inf | Correct attacks |
| D7 | Final robustness numbers come from AutoAttack only | Avoid gradient masking |
| D8 | Every reported number: 3 seeds, mean +/- std, logged in W&B with config and git hash | Reproducibility |
