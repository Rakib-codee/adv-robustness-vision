# GPU timings (Kaggle T4)
| Date | Run | Epochs | Seconds/epoch | Clean acc |
|---|---|---|---|---|
| 2026-10-19 | resnet18 standard, seed 0 | 1 | 40 | 63.9% |
| 2026-10-19 | resnet18 standard, seed 0 | 30 | 46 | 93.92% |
| 2026-10-19 | deit_tiny test (pretrained, upsample 224), seed 0 | 1 | 184 | 94.48% |
| 2026-10-19 | deit_tiny standard, seed 0 | 5 | 202 | 96.44% |
| 2026-10-20 | resnet18 FGSM-AT, seed 0 | 30 | 77 | 83.92% clean, 42.6% PGD-10 (500 img) |
| 2026-10-20 | resnet18 PGD-AT attack_eval (n=1000) | - | 12 s per eps | 50.6% PGD-20 @ 8/255 |