"""
QSense NN weight export and NumPy-only inference.

After training, this module:
  1. Loads the .npz weights produced by train_nn.py
  2. Provides a pure-NumPy forward pass (3 matmuls + ReLU + softmax)
  3. Provides the same feature-extraction logic used during training

No PyTorch dependency is required at inference time.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

# ──────────────────────────────────────────────
# Constants (must match train_nn.py exactly)
# ──────────────────────────────────────────────
GAMMA = 0.1759
OMEGA_MAX = 1.759
TAUS_US = np.array([0.5, 1., 2., 4., 8., 16.], dtype=np.float64)
SHOTS_PER_TAU = 10
N_BINS = 1024
N_TAU = 6

_bin_centers = np.linspace(0.0, OMEGA_MAX, N_BINS, dtype=np.float64)


# ──────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────
def extract_features(counts: np.ndarray, n: np.ndarray,
                     repeat_counts: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Build the 19-dim feature vector from pooled counts and optional repeat counts.

    Parameters
    ----------
    counts : (J,) int   pooled outcome-0 counts per tau
    n      : (J,) int   total shots per tau
    repeat_counts : (k, J) int   per-repeat counts, or None

    Returns
    -------
    features : (19,) float32
    """
    counts = np.asarray(counts, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    J = len(counts)
    k = int(round(float(n[0]) / SHOTS_PER_TAU))

    f_pooled = counts / np.maximum(n, 1.0)  # (J,) pooled frequency

    # Budget normalised to max k=16
    n_norm = n / (16.0 * SHOTS_PER_TAU)     # (J,)

    # Over-dispersion
    if repeat_counts is not None and k > 1:
        rc = np.asarray(repeat_counts, dtype=np.float64)  # (k, J)
        rep_freq = rc / SHOTS_PER_TAU                     # (k, J)
        var_rep = rep_freq.var(axis=0)                    # (J,)
    else:
        var_rep = np.zeros(J, dtype=np.float64)

    binom_var = f_pooled * (1.0 - f_pooled) / max(k, 1) + 1e-6
    od = np.clip(var_rep / binom_var, 0.0, 10.0)          # (J,)

    k_mask = np.array([1.0 if k == 1 else 0.0])           # (1,)

    features = np.concatenate([f_pooled, n_norm, od, k_mask]).astype(np.float32)
    return features  # (19,)


# ──────────────────────────────────────────────
# NumPy forward pass
# ──────────────────────────────────────────────
class QSenseNetNumpy:
    """Numpy-only equivalent of the trained QSenseNet MLP."""

    def __init__(self, weights_path: str | Path):
        w = np.load(str(weights_path))
        # Layer 0: net.0.weight (hidden, n_in), net.0.bias (hidden,)
        self.W0 = w["net.0.weight"].astype(np.float64)   # (256, 19)
        self.b0 = w["net.0.bias"].astype(np.float64)     # (256,)
        # Layer 2: net.2.weight (hidden, hidden), net.2.bias (hidden,)
        self.W1 = w["net.2.weight"].astype(np.float64)   # (256, 256)
        self.b1 = w["net.2.bias"].astype(np.float64)     # (256,)
        # Layer 4: net.4.weight (n_bins, hidden), net.4.bias (n_bins,)
        self.W2 = w["net.4.weight"].astype(np.float64)   # (1024, 256)
        self.b2 = w["net.4.bias"].astype(np.float64)     # (1024,)

    def logits(self, x: np.ndarray) -> np.ndarray:
        """x : (..., 19) → logits (..., 1024)"""
        h = np.maximum(x @ self.W0.T + self.b0, 0.0)
        h = np.maximum(h @ self.W1.T + self.b1, 0.0)
        return h @ self.W2.T + self.b2

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x : (..., 19) → softmax probabilities (..., 1024)"""
        lg = self.logits(x)
        lg = lg - lg.max(axis=-1, keepdims=True)
        p = np.exp(lg)
        return p / p.sum(axis=-1, keepdims=True)


# ──────────────────────────────────────────────
# Public inference API
# ──────────────────────────────────────────────
_DEFAULT_WEIGHTS = Path(__file__).parent.parent / "artifacts" / "nn_weights.npz"
_model_cache: dict[str, QSenseNetNumpy] = {}


def _get_model(weights_path: str | Path | None = None) -> QSenseNetNumpy:
    path = str(weights_path or _DEFAULT_WEIGHTS)
    if path not in _model_cache:
        _model_cache[path] = QSenseNetNumpy(path)
    return _model_cache[path]


def posterior(counts, n, repeat_counts=None,
              weights_path=None) -> np.ndarray:
    """
    Return 1024-bin posterior probability mass over the omega grid.

    Parameters
    ----------
    counts : array-like (6,)   pooled outcome-0 counts
    n      : array-like (6,)   total shots per tau
    repeat_counts : (k, 6) int or None
    weights_path  : path to nn_weights.npz, or None for default

    Returns
    -------
    mass : (1024,) numpy array, sums to 1
    """
    model = _get_model(weights_path)
    feat = extract_features(counts, n, repeat_counts)
    return model.forward(feat)  # (1024,)


def summary(counts, n, repeat_counts=None, weights_path=None) -> dict:
    """
    Returns the NN point estimate and credible intervals.

    Output schema matches bayes.summary().
    """
    mass = posterior(counts, n, repeat_counts, weights_path)
    grid = _bin_centers
    cdf = np.cumsum(mass)
    q = lambda level: float(np.interp(level, cdf, grid))
    omega_est = float(np.dot(grid, mass))
    return {
        "grid": grid,
        "posterior": mass,
        "omega_rad_per_us": omega_est,
        "ci68": (q(0.16), q(0.84)),
        "ci95": (q(0.025), q(0.975)),
    }
