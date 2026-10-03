"""Posterior interval and paired-trial metrics."""
from __future__ import annotations
import numpy as np
from core.physics import b_from_omega


def posterior_summary(grid, mass, omega_true):
    cdf = np.cumsum(mass)
    q = lambda x: float(np.interp(x, cdf, grid))
    mean = float(np.dot(grid, mass))
    return {"omega": mean, "error_uT": float(b_from_omega(mean - omega_true)),
            "ci68": (q(.16), q(.84)), "ci95": (q(.025), q(.975))}


def aggregate(rows):
    errors = np.asarray([r["error_uT"] for r in rows])
    true = np.asarray([r["omega_true"] for r in rows])
    cover = lambda key: np.mean([(r[key][0] <= w <= r[key][1]) for r, w in zip(rows, true)])
    # This is intentionally a *relative* catastrophic threshold.  A fixed .5 uT
    # cut is only about 2 sigma in some budgets and measures ordinary tails.
    failures = [abs(row["error_uT"]) > 4 * row["crb_sigma_uT"] for row in rows]
    return {"rmse_uT": float(np.sqrt(np.mean(errors ** 2))), "mae_uT": float(np.mean(abs(errors))),
            "bias_uT": float(np.mean(errors)), "cov68": float(cover("ci68")),
            "cov95": float(cover("ci95")), "width68_uT": float(np.mean([(b_from_omega(r["ci68"][1])-b_from_omega(r["ci68"][0])) for r in rows])),
            "fail_rate": float(np.mean(failures))}
