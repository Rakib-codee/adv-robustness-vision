---
title: Adversarial Robustness Demo
emoji: 🛡️
colorFrom: blue
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
sdk_version: 6.30.0
---

# How robust are vision models? Interactive demo

Attack four ResNet-18 models trained on CIFAR-10 (standard, FGSM-AT, PGD-AT, TRADES) with FGSM or PGD
and watch their predictions and Grad-CAM maps change.

Code, training logs and full evaluation (AutoAttack, C&W, transfer attacks): https://github.com/Rakib-codee/adv-robustness-vision
