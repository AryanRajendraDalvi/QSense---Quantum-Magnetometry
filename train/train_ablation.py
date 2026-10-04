"""
QSense neural network estimator training script.

Architecture: 19-input → 256 → 256 → 1024-logit classifier over omega bins.
Inputs (19):
  - 6 pooled count frequencies  f_j = k_0j / n_j  for each tau
  - 6 budgets n_j / (k * shots_per_tau)  (normalized total shots)
  - 6 over-dispersion ratios  var(f_j across repeats) / (f_j*(1-f_j)/k + eps)  per tau
  - 1 k=1 mask (1.0 if k==1, else 0.0)

Loss: cross-entropy against Gaussian-smoothed 1024-bin soft labels.
Optimizer: AdamW with cosine decay.
Training: on-the-fly simulation, no fixed dataset.

Usage (Kaggle or local GPU):
    python -m train.train_nn --steps 20000 --batch 2048 --out artifacts/nn_weights.npz

After training, run:
    python -m train.export --weights artifacts/nn_weights.npz  (creates nn_meta.json)
    python -m pytest tests/test_nn_export.py
"""
from __future__ import annotations

import argparse
import math
import json
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────
# Try to import torch; give a clear error if missing
# ──────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for training. Install with: pip install torch"
    ) from exc

# ──────────────────────────────────────────────
# Physics constants (duplicated to keep training self-contained)
# ──────────────────────────────────────────────
GAMMA = 0.1759            # rad/µs/µT
OMEGA_MAX = 1.759         # rad/µs  (B = 10 µT)
TAUS_US = np.array([0.5, 1., 2., 4., 8., 16.], dtype=np.float32)
SHOTS_PER_TAU = 10
N_BINS = 1024
N_TAU = 6
LABEL_SIGMA_BINS = 1.5    # Gaussian smoothing width for soft labels

# ──────────────────────────────────────────────
# Training distribution (revised after B3 result)
# ──────────────────────────────────────────────
TRAIN_DIST = {
    "omega":    (0.0, OMEGA_MAX),         # uniform
    "t2_us":    (2.5, 40.0),              # log-uniform
    "shape":    (1.0, 3.0),              # uniform
    "vis":      (0.6, 0.95),             # uniform
    "drift_prob": 0.5,                   # 50% chance of nonzero drift
    "sigma_phi": (0.0, 3.0),             # uniform when active
    "t_corr":   (10.0, 200.0),           # log-uniform shots
    "k_vals":   [1, 2, 4, 8, 16],        # uniform choice
}


# ──────────────────────────────────────────────
# Numpy simulation (kept here so training is self-contained)
# ──────────────────────────────────────────────
def _prob_zero_np(omega, tau, t2, shape, vis):
    coh = np.exp(-np.power(np.abs(tau) / t2, shape))
    return 0.5 * (1.0 + vis * coh * np.cos(omega * tau))


def _simulate_batch(batch_size: int, device: torch.device):
    """
    Returns (features, omega_true) where features has shape (batch, 19).
    Fully vectorized PyTorch implementation for GPU speed.
    """
    d = TRAIN_DIST
    # Random parameters for the batch
    omega = (d["omega"][1] - d["omega"][0]) * torch.rand(batch_size, device=device) + d["omega"][0]
    t2 = torch.exp((math.log(d["t2_us"][1]) - math.log(d["t2_us"][0])) * torch.rand(batch_size, device=device) + math.log(d["t2_us"][0]))
    shape = (d["shape"][1] - d["shape"][0]) * torch.rand(batch_size, device=device) + d["shape"][0]
    vis = (d["vis"][1] - d["vis"][0]) * torch.rand(batch_size, device=device) + d["vis"][0]
    
    k_vals = torch.tensor(d["k_vals"], device=device, dtype=torch.int32)
    k_idx = torch.randint(0, len(d["k_vals"]), (batch_size,), device=device)
    k_arr = k_vals[k_idx]
    
    has_drift = torch.rand(batch_size, device=device) < d["drift_prob"]
    sigma_phi = torch.where(
        has_drift,
        (d["sigma_phi"][1] - d["sigma_phi"][0]) * torch.rand(batch_size, device=device) + d["sigma_phi"][0],
        torch.zeros(batch_size, device=device)
    )
    t_corr = torch.exp((math.log(d["t_corr"][1]) - math.log(d["t_corr"][0])) * torch.rand(batch_size, device=device) + math.log(d["t_corr"][0]))

    max_k = 16
    total_shots = max_k * N_TAU * SHOTS_PER_TAU
    
    sigma_omega = (sigma_phi / 16.0).to(torch.float64)
    rho = torch.exp(-1.0 / torch.clamp(t_corr, min=1e-9)).to(torch.float64)
    innovation_std = sigma_omega * torch.sqrt(torch.clamp(1.0 - rho * rho, min=0.0))
    
    # Generate AR(1) drift with a plain loop (fast on GPU, avoids float32 cumsum overflow)
    noise = torch.randn((batch_size, total_shots), device=device, dtype=torch.float64) * innovation_std.unsqueeze(1)
    noise[:, 0] = torch.randn(batch_size, device=device, dtype=torch.float64) * sigma_omega
    
    drift = torch.empty((batch_size, total_shots), device=device, dtype=torch.float64)
    drift[:, 0] = noise[:, 0]
    for s in range(1, total_shots):
        drift[:, s] = rho * drift[:, s-1] + noise[:, s]
    
    # tau_rep shape: (max_k * N_TAU)
    taus = torch.tensor(TAUS_US, device=device, dtype=torch.float32)
    tau_rep = taus.repeat(max_k)
    
    # Reshape drift to (batch, max_k * N_TAU, SHOTS_PER_TAU)
    drift_3d = drift.view(batch_size, max_k * N_TAU, SHOTS_PER_TAU).to(torch.float32)
    
    omegas = omega.view(-1, 1, 1) + drift_3d
    tau_col = tau_rep.view(1, -1, 1)
    
    coh = torch.exp(-torch.pow(tau_col / t2.view(-1, 1, 1), shape.view(-1, 1, 1)))
    p0 = 0.5 * (1.0 + vis.view(-1, 1, 1) * coh * torch.cos(omegas * tau_col))
    p0 = torch.clamp(p0, 0.0, 1.0)
    
    # Binomial sampling
    outcomes = torch.rand(p0.shape, device=device) < p0
    shots_per_tau_rep = outcomes.sum(dim=2, dtype=torch.float32)  # (batch, max_k * N_TAU)
    repeat_counts = shots_per_tau_rep.view(batch_size, max_k, N_TAU)
    
    # Mask out unused repeats based on k_arr
    arange_k = torch.arange(max_k, device=device).view(1, max_k, 1)
    valid_mask = (arange_k < k_arr.view(-1, 1, 1)).to(torch.float32)
    
    repeat_counts = repeat_counts * valid_mask
    pooled_counts = repeat_counts.sum(dim=1)  # (batch, N_TAU)
    total_n = (k_arr.view(-1, 1) * SHOTS_PER_TAU).to(torch.float32).expand(-1, N_TAU)
    
    # Over-dispersion
    rep_freq = repeat_counts / SHOTS_PER_TAU
    # Compute variance only over valid k for each sample
    # E[X^2] - E[X]^2
    sum_sq = (rep_freq ** 2 * valid_mask).sum(dim=1)
    mean_val = rep_freq.sum(dim=1) / k_arr.view(-1, 1).to(torch.float32)
    var_rep = (sum_sq / k_arr.view(-1, 1).to(torch.float32)) - mean_val ** 2
    # Fix sample variance bias if k > 1 (optional, using n/(n-1))
    k_f = k_arr.view(-1, 1).to(torch.float32)
    var_rep = var_rep * (k_f / torch.clamp(k_f - 1.0, min=1.0))
    
    var_rep = torch.where(k_arr.view(-1, 1) > 1, var_rep, torch.zeros_like(var_rep))
    
    f_bar = pooled_counts / total_n
    binom_var = f_bar * (1.0 - f_bar) / k_f + 1e-6
    od = torch.clamp(var_rep / binom_var, 0.0, 10.0)
    
    f_pooled = pooled_counts / torch.clamp(total_n, min=1.0)
    n_norm = total_n / (max_k * SHOTS_PER_TAU)
    k_col = (k_arr == 1).to(torch.float32).view(-1, 1)
    
    features = torch.cat([f_pooled, n_norm], dim=1)
    return features, omega


# ──────────────────────────────────────────────
# Soft label construction
# ──────────────────────────────────────────────
_bin_centers = torch.linspace(0.0, float(OMEGA_MAX), N_BINS, dtype=torch.float32)
_bin_sigma = float(LABEL_SIGMA_BINS * (OMEGA_MAX / N_BINS))


def _make_labels(omega_true: torch.Tensor) -> torch.Tensor:
    """Gaussian-smoothed 1024-bin soft labels. Shape: (B, N_BINS)."""
    device = omega_true.device
    centers = _bin_centers.to(device)
    diff = (centers.unsqueeze(0) - omega_true.unsqueeze(1)) / _bin_sigma
    log_p = -0.5 * diff * diff
    log_p = log_p - log_p.max(dim=1, keepdim=True).values
    p = torch.exp(log_p)
    return p / p.sum(dim=1, keepdim=True)


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
class QSenseNet(nn.Module):
    """Two-hidden-layer MLP → 1024 logits over omega bins."""
    def __init__(self, n_in: int = 12, hidden: int = 256, n_bins: int = N_BINS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_bins),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────
def _make_val_set(n: int, device: torch.device, seed: int):
    torch.manual_seed(seed)
    feat, omega = _simulate_batch(n, device)
    labels = _make_labels(omega)
    return feat, labels, omega


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def train(steps: int = 20_000, batch: int = 2048, lr: float = 1e-3,
          lr_min: float = 1e-5, val_every: int = 500,
          val_size: int = 20_000,
          out: Path = Path("artifacts/nn_weights.npz"),
          seed: int = 20261004, device_str: str = "auto"):

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) if device_str == "auto" \
              else torch.device(device_str)
    
    torch.manual_seed(seed)
    print(f"Training on {device}  |  Requested steps={steps}  batch={batch}  val_size={val_size}")

    import time
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(5):
            f_test, o_test = _simulate_batch(batch, device)
            _ = _make_labels(o_test)
        if device.type == "cuda":
            torch.cuda.synchronize()
    t1 = time.perf_counter()
    samples_per_sec = (5 * batch) / (t1 - t0)
    print(f"Measured throughput: {samples_per_sec:,.0f} samples/sec")

    if samples_per_sec < 5000:
        print("Throughput < 5k/sec. Scaling down run to 8k steps, batch 1024.")
        steps = 8_000
        batch = 1024
    else:
        print("Throughput OK. Keeping requested steps and batch.")

    model = QSenseNet().to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=steps, eta_min=lr_min)

    # Fixed validation set
    val_feat, val_labels, val_omega = _make_val_set(val_size, device, seed + 1)

    best_val_nll = float("inf")
    best_state = None
    log = []

    model.train()
    for step in range(1, steps + 1):
        feat, omega = _simulate_batch(batch, device)
        labels = _make_labels(omega)

        logits = model(feat)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(labels * log_probs).sum(dim=-1).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % val_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_feat)
                val_log_p = F.log_softmax(val_logits, dim=-1)
                val_nll = -(val_labels * val_log_p).sum(dim=-1).mean().item()

                probs = F.softmax(val_logits, dim=-1)
                bins = _bin_centers.to(device)
                omega_est = (probs * bins).sum(dim=-1)
                rmse_rad = float(((omega_est - val_omega) ** 2).mean().sqrt())
                rmse_uT = rmse_rad / GAMMA

            model.train()
            entry = {"step": step, "train_loss": loss.item(), "val_nll": val_nll,
                     "val_rmse_uT": rmse_uT, "lr": sched.get_last_lr()[0]}
            log.append(entry)
            print(f"step {step:5d} | train {loss.item():.4f} | val_nll {val_nll:.4f} "
                  f"| rmse {rmse_uT:.4f} µT | lr {sched.get_last_lr()[0]:.2e}")

            if val_nll < best_val_nll:
                best_val_nll = val_nll
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
                # Checkpoint to /kaggle/working if it exists (for session crashes)
                kaggle_work = Path("/kaggle/working")
                if kaggle_work.exists():
                    tmp_out = kaggle_work / "nn_weights_best.npz"
                    np.savez_compressed(str(tmp_out), **{k: v.numpy() for k, v in best_state.items()})
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best checkpoint
    model.load_state_dict(best_state)
    model.eval()

    # Export weights to npz
    out.parent.mkdir(parents=True, exist_ok=True)
    weights = {k: v.numpy() for k, v in best_state.items()}
    np.savez_compressed(str(out), **weights)
    print(f"Saved weights -> {out}  (val_nll={best_val_nll:.4f})")

    # Save metadata
    meta = {
        "n_bins": N_BINS,
        "omega_max": float(OMEGA_MAX),
        "taus_us": TAUS_US.tolist(),
        "n_in": 19,
        "hidden": 256,
        "label_sigma_bins": LABEL_SIGMA_BINS,
        "steps": steps,
        "best_val_nll": best_val_nll,
        "seed": seed,
        "train_dist": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in TRAIN_DIST.items()
        },
        "training_log": log,
    }
    meta_path = out.parent / "nn_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved metadata -> {meta_path}")
    return model, best_val_nll


def main():
    parser = argparse.ArgumentParser(description="Train QSense NN estimator")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-size", type=int, default=20_000,
                        help="Validation set size (use 1000 for local CPU tests).")
    parser.add_argument("--out", type=Path, default=Path("artifacts/nn_weights.npz"))
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    train(steps=args.steps, batch=args.batch, lr=args.lr, lr_min=args.lr_min,
          val_every=args.val_every, val_size=args.val_size, out=args.out,
          seed=args.seed, device_str=args.device)


if __name__ == "__main__":
    main()
