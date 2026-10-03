"""Local, no-drift classical Fisher-information reference."""
from __future__ import annotations
import numpy as np
from .physics import TAUS_US, probability_zero, coherence
from .sim import Noise


def local_sigma(omega: float, noise: Noise, k: int,
                taus_us=TAUS_US, shots_per_tau: int = 10) -> float:
    taus = np.asarray(taus_us, dtype=float)
    p = np.clip(probability_zero(omega, taus, noise.t2_us, noise.shape,
                                 noise.visibility), 1e-12, 1 - 1e-12)
    derivative = -0.5 * noise.visibility * coherence(taus, noise.t2_us, noise.shape) * taus * np.sin(omega * taus)
    fisher = k * shots_per_tau * np.sum(derivative ** 2 / (p * (1 - p)))
    return float(1 / np.sqrt(fisher)) if fisher > 0 else float("inf")
