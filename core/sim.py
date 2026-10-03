"""Time-ordered Ramsey data simulation, including correlated frequency drift."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .physics import TAUS_US, probability_zero


@dataclass(frozen=True)
class Noise:
    t2_us: float = 10.0
    shape: float = 1.0
    visibility: float = 0.9
    sigma_phi_rad: float = 0.0
    correlation_shots: float = 50.0
    overhead_us: float = 1.0


def _drift(n_shots: int, noise: Noise, rng: np.random.Generator) -> np.ndarray:
    """Stationary AR(1) offset in rad/us, specified by phase std at tau=16 us."""
    if noise.sigma_phi_rad <= 0:
        return np.zeros(n_shots)
    sigma_omega = noise.sigma_phi_rad / 16.0
    rho = np.exp(-1.0 / max(noise.correlation_shots, 1e-9))
    out = np.empty(n_shots)
    out[0] = rng.normal(0, sigma_omega)
    innovation = sigma_omega * np.sqrt(1.0 - rho * rho)
    for i in range(1, n_shots):
        out[i] = rho * out[i - 1] + rng.normal(0, innovation)
    return out


def simulate(omega: float, noise: Noise, k: int, rng: np.random.Generator,
             taus_us: np.ndarray = TAUS_US, shots_per_tau: int = 10,
             return_repeat_counts: bool = False):
    """Return pooled counts/trials, optionally with time-ordered repeat counts."""
    taus = np.asarray(taus_us, dtype=float)
    n = np.full(taus.size, k * shots_per_tau, dtype=int)
    counts = np.zeros(taus.size, dtype=int)
    repeat_counts = np.zeros((k, taus.size), dtype=int)
    drift = _drift(int(n.sum()), noise, rng)
    ix = 0
    for _repeat in range(k):
        for j, tau in enumerate(taus):
            offsets = drift[ix:ix + shots_per_tau]
            ix += shots_per_tau
            p0 = probability_zero(omega + offsets, tau, noise.t2_us,
                                  noise.shape, noise.visibility)
            repeat_counts[_repeat, j] = rng.binomial(1, p0).sum()
            counts[j] += repeat_counts[_repeat, j]
    if return_repeat_counts:
        return counts, n, repeat_counts
    return counts, n


def total_time_us(k: int, taus_us: np.ndarray = TAUS_US, shots_per_tau: int = 10,
                  overhead_us: float = 1.0) -> float:
    return k * shots_per_tau * float(np.sum(np.asarray(taus_us) + overhead_us))
