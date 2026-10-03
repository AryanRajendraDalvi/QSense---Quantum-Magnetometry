"""Small, reproducible paired benchmark runner for the B1/B2/B3/NN benchmark.

Example:
    python -m bench.run_benchmark --trials 2000 --k 4 16 --suite harsh --include-b3 --include-nn
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from bench.metrics import aggregate, posterior_summary
from core.crb import local_sigma
from core.physics import b_from_omega
from core.sim import Noise, simulate, total_time_us
from est.bayes import posterior, DEFAULT_GRID


def evaluate(scenario: str, axis: str, value: float, noise: Noise, k: int,
             trials: int, rng: np.random.Generator, include_b3: bool = False,
             include_nn: bool = False, nn_weights_path: "Path | None" = None):
    """Run paired trials for all requested methods; return list of benchmark records."""
    methods = {"bayes_assumed": "assumed", "bayes_nuisance": "nuisance"}
    if include_b3:
        methods["bayes_nuisance_shape"] = "nuisance_shape"
    rows = {name: [] for name in methods}
    nn_rows = []
    crbs = []

    # Lazy NN import so benchmark works even without weights
    nn_mod = None
    nn_wp = None
    if include_nn:
        try:
            import est.nn_infer as _nn
            wp = nn_weights_path or _nn._DEFAULT_WEIGHTS
            if Path(wp).exists():
                nn_mod = _nn
                nn_wp = wp
                # Warm up the model cache
                _nn._get_model(nn_wp)
            else:
                print(f"  [NN] weights not found at {wp}, skipping NN")
        except Exception as exc:
            print(f"  [NN] import/load error: {exc}, skipping NN")

    for _ in range(trials):
        omega = rng.uniform(0.0, 1.759)
        counts, n, repeat_counts = simulate(omega, noise, k, rng,
                                            return_repeat_counts=True)
        crbs.append(float(b_from_omega(local_sigma(omega, noise, k))))

        for name, model in methods.items():
            p = posterior(counts, n, model)
            row = posterior_summary(DEFAULT_GRID, p, omega)
            row["omega_true"] = omega
            row["crb_sigma_uT"] = crbs[-1]
            rows[name].append(row)

        if nn_mod is not None:
            try:
                p_nn = nn_mod.posterior(counts, n,
                                        repeat_counts=repeat_counts,
                                        weights_path=nn_wp)
                row = posterior_summary(nn_mod._bin_centers, p_nn, omega)
                row["omega_true"] = omega
                row["crb_sigma_uT"] = crbs[-1]
                nn_rows.append(row)
            except Exception as exc:
                print(f"  [NN] inference error: {exc}")

    common = {
        "scenario": scenario, "axis": axis, "value": value, "k": k,
        "t_total_ms": total_time_us(k) / 1000, "n_trials": trials,
        "crb_median_uT": float(np.median(crbs)),
    }
    result = [
        {**common, "method": method, **aggregate(method_rows)}
        for method, method_rows in rows.items()
    ]
    if nn_rows:
        result.append({**common, "method": "nn", **aggregate(nn_rows)})
    return result


# ── Condition catalogues ────────────────────────────────────────────────────
BASELINE_CONDITIONS = [
    ("matched",          "t2_ratio",  1.0, Noise()),
    ("fast_dephasing",   "t2_ratio",  0.25, Noise(t2_us=2.5)),
    ("gaussian_decay",   "shape",     2.0, Noise(shape=2.0)),
    ("correlated_drift", "drift",     0.5, Noise(sigma_phi_rad=0.5)),
]

HARSH_CONDITIONS = [
    ("matched",              "t2_ratio",  1.0, Noise()),
    ("shape_p2",             "shape",     2.0, Noise(shape=2.0)),
    ("shape_p3",             "shape",     3.0, Noise(shape=3.0)),
    ("shape_p2_t2_half",     "combined",  0.5, Noise(t2_us=5.0,  shape=2.0)),
    ("shape_p2_t2_double",   "combined",  2.0, Noise(t2_us=20.0, shape=2.0)),
    ("shape_p3_t2_half",     "combined",  0.5, Noise(t2_us=5.0,  shape=3.0)),
    ("shape_p3_t2_double",   "combined",  2.0, Noise(t2_us=20.0, shape=3.0)),
    ("drift_1rad_fast",      "drift",     1.0, Noise(sigma_phi_rad=1.0, correlation_shots=10)),
    ("drift_2rad_fast",      "drift",     2.0, Noise(sigma_phi_rad=2.0, correlation_shots=10)),
    ("drift_3rad_fast",      "drift",     3.0, Noise(sigma_phi_rad=3.0, correlation_shots=10)),
    ("visibility_half",      "visibility", 0.5, Noise(visibility=0.5)),
]

OOD_CONDITIONS = [
    ("ood_visibility_low",   "visibility", 0.5,  Noise(visibility=0.5)),
    ("ood_drift_4rad",       "drift",      4.0,  Noise(sigma_phi_rad=4.0, correlation_shots=10)),
    ("ood_t2_very_short",    "t2_ratio",   0.125, Noise(t2_us=1.25)),
    ("ood_drift3_t2_5",      "combined",   0.5,  Noise(sigma_phi_rad=3.0, correlation_shots=10, t2_us=5.0)),
]


def main():
    parser = argparse.ArgumentParser(
        description="QSense paired benchmark: B1/B2/B3/NN across noise scenarios.")
    parser.add_argument("--trials",      type=int,  default=300)
    parser.add_argument("--seed",        type=int,  default=20261003)
    parser.add_argument("--output",      type=Path, default=Path("results/spike.json"))
    parser.add_argument("--k",           type=int,  nargs="+", default=[1, 2, 4, 8, 16],
                        help="Repeat budgets to evaluate.")
    parser.add_argument("--suite",       choices=["baseline", "harsh", "ood"],
                        default="baseline")
    parser.add_argument("--only",        nargs="+",
                        help="Restrict to named scenarios.")
    parser.add_argument("--include-b3",  action="store_true",
                        help="Include B3 (shape-marginalized Bayesian).")
    parser.add_argument("--include-nn",  action="store_true",
                        help="Include NN (requires artifacts/nn_weights.npz).")
    parser.add_argument("--nn-weights",  type=Path, default=None,
                        help="Path to nn_weights.npz.")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    suite_map = {"baseline": BASELINE_CONDITIONS,
                 "harsh": HARSH_CONDITIONS,
                 "ood": OOD_CONDITIONS}
    conditions = suite_map[args.suite]

    if args.only:
        conditions = [c for c in conditions if c[0] in args.only]
        if not conditions:
            parser.error("--only did not match any scenario in the selected suite")

    output = []
    for name, axis, value, noise in conditions:
        for k in args.k:
            print(f"  scenario={name:<28} k={k}")
            output.extend(evaluate(
                name, axis, value, noise, k, args.trials, rng,
                include_b3=args.include_b3,
                include_nn=args.include_nn,
                nn_weights_path=args.nn_weights,
            ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {len(output)} records to {args.output}")


if __name__ == "__main__":
    main()
