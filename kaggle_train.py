"""
QSense Kaggle GPU training notebook entry point.

Upload this directory to a Kaggle notebook with GPU (T4 or P100).
Run: !python kaggle_train.py

Outputs:
  artifacts/nn_weights.npz
  artifacts/nn_meta.json
"""
# ─── Setup ───────────────────────────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch"], check=False)

import sys, os
# Add repo root to path (this file lives in the repo root on Kaggle)
sys.path.insert(0, os.path.dirname(__file__) or ".")

from pathlib import Path
from train.train_nn import train

# ─── Train ───────────────────────────────────────────────────────────────────
model, best_val_nll = train(
    steps      = 20_000,
    batch      = 2048,
    lr         = 1e-3,
    lr_min     = 1e-5,
    val_every  = 500,
    out        = Path("artifacts/nn_weights.npz"),
    seed       = 20261004,
    device_str = "auto",   # auto-selects CUDA if available
)
print(f"\nDone. Best validation NLL: {best_val_nll:.4f}")
print("Download artifacts/nn_weights.npz and artifacts/nn_meta.json")
