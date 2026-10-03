"""
QSense figure generation from precomputed benchmark JSON files.

Generates:
  figures/fringe_plot.png       - Ramsey fringe with decay envelope
  figures/coverage_vs_mismatch.png  - coverage vs mismatch factor for B1/B2/B3/NN
  figures/error_vs_time.png     - RMSE vs total sensing time
  figures/posteriors_overlay.png - posterior curves for one example run
  figures/ood_bars.png          - coverage bars for OOD scenarios

Usage:
    python -m bench.make_figures [--results results/benchmark.json] [--out figures/]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from core.physics import probability_zero, TAUS_US, GAMMA_RAD_PER_US_PER_UT, b_from_omega
from core.sim import Noise, simulate
from est.bayes import posterior as bayes_posterior, summary as bayes_summary, DEFAULT_GRID

# ── Style ────────────────────────────────────────────────────────────────────
METHOD_COLORS = {
    "bayes_assumed": "#e74c3c",
    "bayes_nuisance": "#3498db",
    "bayes_nuisance_shape": "#2ecc71",
    "nn": "#9b59b6",
}
METHOD_LABELS = {
    "bayes_assumed": "B1 (calibrated)",
    "bayes_nuisance": "B2 (T2*/V margin.)",
    "bayes_nuisance_shape": "B3 (+shape margin.)",
    "nn": "NN (amortized)",
}

NOMINAL_68 = 0.68
NOMINAL_95 = 0.95


def fringe_plot(out_dir: Path):
    """P(0) vs tau: matched vs mismatched noise, with sampled points."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    tau_fine = np.linspace(0.01, 16, 500)

    cases = [
        ("Matched noise (T2*=10µs, p=1, V=0.9)", Noise()),
        ("Mismatch (T2*=2.5µs, p=2, V=0.7)", Noise(t2_us=2.5, shape=2.0, visibility=0.7)),
    ]
    rng = np.random.default_rng(42)
    omega = 0.7  # rad/µs → B ≈ 3.98 µT

    for ax, (title, noise) in zip(axes, cases):
        # Decay envelope
        ax.fill_between(
            tau_fine,
            0.5 * (1 - noise.visibility * np.exp(-np.power(tau_fine / noise.t2_us, noise.shape))),
            0.5 * (1 + noise.visibility * np.exp(-np.power(tau_fine / noise.t2_us, noise.shape))),
            alpha=0.15, color="steelblue", label="Decay envelope",
        )
        # Analytic fringe
        p_fine = probability_zero(omega, tau_fine, noise.t2_us, noise.shape, noise.visibility)
        ax.plot(tau_fine, p_fine, color="steelblue", lw=1.5, label="P(0|ω,τ)")

        # Sampled points
        counts, n = simulate(omega, noise, k=4, rng=rng)
        f_obs = counts / n
        ax.scatter(TAUS_US, f_obs, color="crimson", zorder=5, s=40, label="Observed (k=4)")

        ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.set_xlabel("Wait time τ (µs)", fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("P(outcome = 0)", fontsize=11)
    fig.suptitle("Ramsey Fringe: True Signal vs Assumed Model", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "fringe_plot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def coverage_vs_mismatch(data: list[dict], out_dir: Path, axis: str = "t2_ratio",
                         ci: str = "cov95", k: int = 16):
    """Coverage vs mismatch factor for a given axis."""
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = list(METHOD_COLORS.keys())
    for method in methods:
        rows = [r for r in data
                if r["axis"] == axis and r["method"] == method and r["k"] == k]
        if not rows:
            continue
        rows.sort(key=lambda r: r["value"])
        xs = [r["value"] for r in rows]
        ys = [r[ci] for r in rows]
        ax.plot(xs, ys, "o-", color=METHOD_COLORS[method],
                label=METHOD_LABELS.get(method, method), lw=1.8, ms=6)

    nom = NOMINAL_68 if ci == "cov68" else NOMINAL_95
    ax.axhline(nom, color="black", ls="--", lw=1.2, label=f"Nominal ({nom:.0%})")
    ax.axhline(0.90, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax.set_xlabel(axis.replace("_", " ").title(), fontsize=11)
    ax.set_ylabel(f"{ci.replace('cov', '').upper()}% Coverage", fontsize=11)
    ax.set_title(f"{ci[3:]}% Coverage vs {axis.replace('_',' ').title()}  (k={k})", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"coverage_vs_{axis}_{ci}_k{k}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def error_vs_time(data: list[dict], out_dir: Path):
    """RMSE vs total sensing time for matched scenario."""
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = list(METHOD_COLORS.keys())
    for method in methods:
        rows = [r for r in data
                if r["scenario"] == "matched" and r["method"] == method]
        if not rows:
            continue
        rows.sort(key=lambda r: r["t_total_ms"])
        xs = [r["t_total_ms"] for r in rows]
        ys = [r["rmse_uT"] for r in rows]
        ax.plot(xs, ys, "o-", color=METHOD_COLORS[method],
                label=METHOD_LABELS.get(method, method), lw=1.8, ms=6)

    # CRB reference line from matched scenario
    crb_rows = [r for r in data if r["scenario"] == "matched" and r["method"] == "bayes_assumed"]
    if crb_rows:
        crb_rows.sort(key=lambda r: r["t_total_ms"])
        xs_crb = [r["t_total_ms"] for r in crb_rows]
        ys_crb = [r["crb_median_uT"] for r in crb_rows]
        ax.plot(xs_crb, ys_crb, "k--", lw=1.2, label="Median local CRB ref.", alpha=0.7)

    ax.set_xlabel("Total sensing time (ms)", fontsize=11)
    ax.set_ylabel("RMSE (µT)", fontsize=11)
    ax.set_title("Estimation Error vs Sensing Time (Matched Noise)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = out_dir / "error_vs_time.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def posteriors_overlay(out_dir: Path):
    """Show posterior curves for B1, B2, B3 on a single example trial."""
    rng = np.random.default_rng(99)
    noise_true = Noise(t2_us=3.0, shape=2.0, visibility=0.8)
    omega_true = 0.88  # rad/µs → B ≈ 5 µT
    k = 8
    counts, n = simulate(omega_true, noise_true, k, rng)

    fig, ax = plt.subplots(figsize=(8, 4))
    grid = DEFAULT_GRID
    b_grid = grid / GAMMA_RAD_PER_US_PER_UT

    for model, label, color in [
        ("assumed", "B1 (calibrated)", METHOD_COLORS["bayes_assumed"]),
        ("nuisance", "B2 (T2*/V margin.)", METHOD_COLORS["bayes_nuisance"]),
        ("nuisance_shape", "B3 (+shape margin.)", METHOD_COLORS["bayes_nuisance_shape"]),
    ]:
        post = bayes_posterior(counts, n, model)
        ax.plot(b_grid, post / post.max(), color=color, lw=1.8, label=label)

    ax.axvline(b_from_omega(omega_true), color="black", lw=1.5, ls="--",
               label=f"True B = {b_from_omega(omega_true):.2f} µT")
    ax.set_xlabel("Magnetic field B (µT)", fontsize=11)
    ax.set_ylabel("Posterior (normalized to peak)", fontsize=11)
    ax.set_title("Posterior Comparison: T2*=3µs, p=2, V=0.8 (mismatch from p=1 assumed)",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "posteriors_overlay.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def ood_bars(ood_data: list[dict], out_dir: Path, k: int = 16):
    """Coverage bars for OOD scenarios."""
    scenarios = list(dict.fromkeys(r["scenario"] for r in ood_data))
    methods = [m for m in METHOD_COLORS if any(r["method"] == m for r in ood_data)]
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, method in enumerate(methods):
        rows_by_sc = {r["scenario"]: r for r in ood_data
                      if r["method"] == method and r["k"] == k}
        heights = [rows_by_sc.get(sc, {}).get("cov95", 0) for sc in scenarios]
        bars = ax.bar(x + i * width, heights, width * 0.9,
                      color=METHOD_COLORS[method], label=METHOD_LABELS.get(method, method),
                      alpha=0.85)

    ax.axhline(0.95, color="black", ls="--", lw=1.2, label="Nominal 95%")
    ax.axhline(0.90, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([s.replace("ood_", "").replace("_", " ") for s in scenarios],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("95% Coverage", fontsize=11)
    ax.set_title(f"OOD Stress Tests — 95% Coverage (k={k})", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"ood_bars_k{k}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate QSense figures")
    parser.add_argument("--results",     type=Path, default=None,
                        help="Main benchmark JSON (e.g. results/benchmark.json)")
    parser.add_argument("--ood-results", type=Path, default=None,
                        help="OOD benchmark JSON (e.g. results/ood.json)")
    parser.add_argument("--out",         type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Generating fringe plot…")
    fringe_plot(args.out)

    print("Generating posteriors overlay…")
    posteriors_overlay(args.out)

    if args.results and args.results.exists():
        data = json.loads(args.results.read_text(encoding="utf-8"))
        print(f"Loaded {len(data)} records from {args.results}")

        for axis in ("t2_ratio", "shape", "drift", "visibility"):
            for ci in ("cov68", "cov95"):
                for k in (4, 16):
                    coverage_vs_mismatch(data, args.out, axis=axis, ci=ci, k=k)

        error_vs_time(data, args.out)
    else:
        print("No results file provided — skipping benchmark plots.")

    if args.ood_results and args.ood_results.exists():
        ood_data = json.loads(args.ood_results.read_text(encoding="utf-8"))
        print(f"Loaded {len(ood_data)} OOD records from {args.ood_results}")
        for k in (4, 16):
            ood_bars(ood_data, args.out, k=k)
    else:
        print("No OOD results file provided — skipping OOD bars.")

    print(f"\nAll figures saved to {args.out}/")


if __name__ == "__main__":
    main()
