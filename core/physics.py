"""Ramsey sensor physics in rad / microsecond units."""
from __future__ import annotations

import numpy as np

GAMMA_RAD_PER_US_PER_UT = 0.1759
TAUS_US = np.asarray([0.5, 1., 2., 4., 8., 16.])


def coherence(tau_us, t2_us: float, shape: float = 1.0):
    """Ramsey coherence exp(-(tau/T2*)**p)."""
    tau = np.asarray(tau_us, dtype=float)
    return np.exp(-np.power(tau / t2_us, shape))


def probability_zero(omega_rad_per_us, tau_us, t2_us: float = 10.0,
                     shape: float = 1.0, visibility: float = 0.9):
    """Probability of outcome 0 after a Ramsey sequence."""
    return 0.5 * (1.0 + visibility * coherence(tau_us, t2_us, shape)
                  * np.cos(np.asarray(omega_rad_per_us) * np.asarray(tau_us)))


def b_from_omega(omega):
    return np.asarray(omega) / GAMMA_RAD_PER_US_PER_UT


def omega_from_b(field_ut):
    return np.asarray(field_ut) * GAMMA_RAD_PER_US_PER_UT
