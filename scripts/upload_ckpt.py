"""Upload checkpoints/<run_name>/final.pt to W&B as a model artifact.

Usage (from repo root):  python scripts/upload_ckpt.py resnet18_trades_b6_seed0
"""
import sys
from pathlib import Path

import wandb

run_name = sys.argv[1]
ckpt = Path("checkpoints") / run_name / "final.pt"
assert ckpt.exists(), f"missing {ckpt}"
run = wandb.init(project="adv-robustness-vision", name=f"upload-{run_name}", job_type="upload")
art = wandb.Artifact(run_name, type="model")
art.add_file(str(ckpt))
run.log_artifact(art)
run.finish()
print("uploaded artifact", run_name)
