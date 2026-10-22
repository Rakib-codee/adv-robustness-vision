| Model | Clean | FGSM 8/255 | PGD-20 8/255 | PGD-20 16/255 | AutoAttack 8/255 | C&W median L2 | Acc @ L2<=0.5 |
|---|---|---|---|---|---|---|---|
| ResNet-18 standard | 94.3 | 13.1 | 0.0 | 0.0 | 0.0 | 0.20 | 0.2 |
| DeiT-Tiny standard | 96.0 | 17.4 | 0.0 | 0.0 | - | - | - |
| ResNet-18 FGSM-AT | 84.2 | 56.6 | 44.7 | 11.4 | 40.4 | 0.70 | 54.0 |
| ResNet-18 PGD-AT | 81.0 | 54.9 | 50.6 | 17.9 | 44.0 | 0.84 | 58.8 |
| ResNet-18 TRADES (beta=6) | 80.3 | 55.5 | 51.7 | 22.8 | 45.8 | 0.82 | 57.0 |

Accuracies in %. Clean, FGSM and PGD-20: first 1000 CIFAR-10 test images; AutoAttack and C&W: first 500. Seed 0, single run.
