# Kaggle runner (paste into notebook cells; GPU on, Internet on)
```python
# Cell 1: code
!git clone https://github.com/<YOU>/adv-robustness-vision.git
%cd adv-robustness-vision
!pip install -q timm torchattacks grad-cam wandb pyyaml
```
```python
# Cell 2: W&B login (key in Kaggle Secrets as WANDB_API_KEY)
from kaggle_secrets import UserSecretsClient
import os; os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
os.environ["PYTHONPATH"] = "src"
```
```python
# Cell 3: check, then train
!python scripts/smoke_test.py
!python -m advrob.train --config configs/resnet18_cifar10.yaml --seed 0
```
Checkpoints save to `checkpoints/` every epoch; if the session dies, re-run Cell 3 and it resumes. Download `final.pt` via the Output tab (not tracked in git).
