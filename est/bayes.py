"""Grid Bayesian estimators: calibrated B1 and nuisance-marginalized B2."""
from __future__ import annotations
import numpy as np
from functools import lru_cache
from core.physics import TAUS_US, probability_zero

OMEGA_MAX = 1.759
DEFAULT_GRID = np.linspace(0.0, OMEGA_MAX, 2048)
T2_PRIOR_US = np.geomspace(2.5, 40.0, 9)
V_PRIOR = np.asarray([0.6, 0.75, 0.9])
SHAPE_PRIOR = np.asarray([1.0, 1.5, 2.0, 3.0])


def _loglike(grid, counts, n, t2, visibility, shape=1.0, taus_us=TAUS_US):
    p = np.clip(probability_zero(grid[:, None], np.asarray(taus_us)[None, :],
                                 t2, shape, visibility), 1e-6, 1 - 1e-6)
    return (np.asarray(counts) * np.log(p) + (np.asarray(n) - np.asarray(counts)) * np.log1p(-p)).sum(axis=1)


@lru_cache(maxsize=8)
def _tables(grid_values: tuple, tau_values: tuple):
    """Precompute model probabilities; data change between trials, not the models."""
    grid = np.asarray(grid_values)
    taus = np.asarray(tau_values)
    def table(t2, visibility):
        p = np.clip(probability_zero(grid[:, None], taus[None, :], t2, 1.0,
                                     visibility), 1e-6, 1 - 1e-6)
        return np.log(p), np.log1p(-p)
    assumed = table(10.0, .9)
    nuisance_terms = tuple(table(t2, v) for t2 in T2_PRIOR_US for v in V_PRIOR)
    nuisance_shape_terms = tuple(
        (np.log(np.clip(probability_zero(grid[:, None], taus[None, :], t2, shape, v), 1e-6, 1 - 1e-6)),
         np.log1p(-np.clip(probability_zero(grid[:, None], taus[None, :], t2, shape, v), 1e-6, 1 - 1e-6)))
        for t2 in T2_PRIOR_US for v in V_PRIOR for shape in SHAPE_PRIOR
    )
    # Stacked tensors avoid hundreds of tiny matrix products per posterior.
    nuisance = tuple(np.stack([term[i] for term in nuisance_terms]) for i in (0, 1))
    nuisance_shape = tuple(np.stack([term[i] for term in nuisance_shape_terms]) for i in (0, 1))
    return assumed, nuisance, nuisance_shape


def _loglike_from_table(log_p, log_q, counts, n):
    return log_p @ np.asarray(counts) + log_q @ (np.asarray(n) - np.asarray(counts))


def _marginalized_loglike(log_p, log_q, counts, n):
    """Log-mean-exp across nuisance models; tensors have shape model/grid/tau."""
    counts = np.asarray(counts)
    ll = (np.einsum("mgj,j->mg", log_p, counts)
          + np.einsum("mgj,j->mg", log_q, np.asarray(n) - counts))
    maximum = ll.max(axis=0)
    return maximum + np.log(np.exp(ll - maximum).mean(axis=0))


def posterior(counts, n, model: str = "assumed", grid=DEFAULT_GRID, taus_us=TAUS_US):
    """Return normalized posterior density masses over the omega grid."""
    grid = np.asarray(grid, dtype=float)
    assumed, nuisance, nuisance_shape = _tables(tuple(grid), tuple(taus_us))
    if model == "assumed":
        ll = _loglike_from_table(*assumed, counts, n)
    elif model in ("nuisance", "prior"):
        ll = _marginalized_loglike(*nuisance, counts, n)
    elif model in ("nuisance_shape", "shape_prior"):
        ll = _marginalized_loglike(*nuisance_shape, counts, n)
    else:
        raise ValueError("model must be 'assumed', 'nuisance', or 'nuisance_shape'")
    ll -= ll.max()
    mass = np.exp(ll)
    return mass / mass.sum()


def summary(counts, n, model="assumed", grid=DEFAULT_GRID, taus_us=TAUS_US):
    p = posterior(counts, n, model, grid, taus_us)
    cdf = np.cumsum(p)
    q = lambda level: float(np.interp(level, cdf, grid))
    return {"grid": grid, "posterior": p, "omega_rad_per_us": float(np.dot(grid, p)),
            "ci68": (q(.16), q(.84)), "ci95": (q(.025), q(.975))}
