"""
QSense NN weight export helper.

Usage:
    python -m train.export --weights artifacts/nn_weights.npz

Validates that the numpy forward pass matches the torch forward pass
within 1e-5 absolute tolerance on 1000 random inputs.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train.train_nn import QSenseNet, N_BINS
from est.nn_infer import QSenseNetNumpy


def verify_equivalence(weights_path: Path, n_checks: int = 1000,
                        atol: float = 1e-5) -> None:
    """Compare torch and numpy forward passes on random inputs."""
    rng = np.random.default_rng(42)
    x_np = rng.uniform(-1, 1, (n_checks, 19)).astype(np.float32)
    x_th = torch.from_numpy(x_np)

    # Torch model
    model = QSenseNet()
    state = torch.load(weights_path, map_location="cpu") if str(weights_path).endswith(".pt") \
        else {k: torch.from_numpy(v) for k, v in np.load(str(weights_path)).items()}
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        torch_probs = F.softmax(model(x_th), dim=-1).numpy()

    # Numpy model
    np_model = QSenseNetNumpy(weights_path)
    np_probs = np_model.forward(x_np.astype(np.float64))

    max_diff = np.abs(torch_probs - np_probs).max()
    mean_diff = np.abs(torch_probs - np_probs).mean()
    print(f"Max |torch - numpy| = {max_diff:.2e}  (mean {mean_diff:.2e})")

    if max_diff > atol:
        raise AssertionError(
            f"Equivalence test FAILED: max difference {max_diff:.2e} > {atol:.2e}"
        )
    print("✓ NumPy forward pass matches torch within tolerance.")


def main():
    parser = argparse.ArgumentParser(description="Verify NN numpy export")
    parser.add_argument("--weights", type=Path, default=Path("artifacts/nn_weights.npz"))
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    verify_equivalence(args.weights, n_checks=args.n, atol=args.atol)


if __name__ == "__main__":
    main()
