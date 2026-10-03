"""
Simulator parity test: Compare NumPy (core.sim + nn_infer) vs PyTorch (train_nn).
Draws 100k experiments from both simulators for three settings and checks feature moments.
"""
import numpy as np
import torch
import math
from core.sim import Noise, simulate
from est.nn_infer import extract_features
from train.train_nn import _simulate_batch, TRAIN_DIST

def check_parity(name: str, omega: float, t2: float, shape: float, vis: float, 
                 sigma_phi: float, t_corr: float, k: int, n_trials: int = 100_000):
    print(f"\n--- Parity Test: {name} ---")
    
    # 1. Generate via NumPy
    rng_np = np.random.default_rng(42)
    noise = Noise(t2_us=t2, shape=shape, visibility=vis, 
                  sigma_phi_rad=sigma_phi, correlation_shots=t_corr)
    
    np_features = []
    for _ in range(n_trials):
        c, n, rep_c = simulate(omega, noise, k, rng_np, return_repeat_counts=True)
        feat = extract_features(c, n, repeat_counts=rep_c)
        np_features.append(feat)
    np_features = np.array(np_features)
    np_mean = np_features.mean(axis=0)
    np_std = np_features.std(axis=0)

    # 2. Generate via PyTorch 
    # We temporarily override TRAIN_DIST in train_nn to force these exact parameters
    global TRAIN_DIST
    old_dist = dict(TRAIN_DIST)
    TRAIN_DIST["omega"] = (omega, omega)
    TRAIN_DIST["t2_us"] = (t2, t2)
    TRAIN_DIST["shape"] = (shape, shape)
    TRAIN_DIST["vis"] = (vis, vis)
    TRAIN_DIST["drift_prob"] = 1.0 if sigma_phi > 0 else 0.0
    TRAIN_DIST["sigma_phi"] = (sigma_phi, sigma_phi)
    TRAIN_DIST["t_corr"] = (t_corr, t_corr)
    TRAIN_DIST["k_vals"] = [k]
    
    # Generate in batches to avoid OOM
    batch_size = 2000
    n_batches = n_trials // batch_size
    pt_features = []
    
    torch.manual_seed(42)
    device = torch.device("cpu")
    for _ in range(n_batches):
        feat_pt, _ = _simulate_batch(batch_size, device)
        pt_features.append(feat_pt.numpy())
    pt_features = np.concatenate(pt_features, axis=0)
    pt_mean = pt_features.mean(axis=0)
    pt_std = pt_features.std(axis=0)
    
    # Restore TRAIN_DIST
    TRAIN_DIST.clear()
    TRAIN_DIST.update(old_dist)
    
    # 3. Compare means and stds
    # Compare pooled freq (0:6), norm budget (6:12), overdispersion (12:18), k=1 mask (18)
    diff_mean = np.abs(np_mean - pt_mean)
    diff_std = np.abs(np_std - pt_std)
    
    # Standard error of the mean is approx std / sqrt(n_trials)
    # We want agreement within ~3-5 standard errors.
    se_mean = np_std / math.sqrt(n_trials)
    
    max_z = (diff_mean / (se_mean + 1e-9)).max()
    print(f"Max Z-score of mean diff: {max_z:.2f} (Target < 4.0)")
    
    groups = {
        "Frequencies (0:6)": slice(0, 6),
        "Budget norm (6:12)": slice(6, 12),
        "Overdispersion (12:18)": slice(12, 18),
        "k=1 mask (18)": slice(18, 19)
    }
    
    for gname, slc in groups.items():
        nm = np_mean[slc]
        pm = pt_mean[slc]
        ns = np_std[slc]
        ps = pt_std[slc]
        print(f"{gname}:")
        print(f"  NP Mean: {nm.mean():.4f}  PT Mean: {pm.mean():.4f}  (Max diff: {np.abs(nm-pm).max():.4f})")
        print(f"  NP Std:  {ns.mean():.4f}  PT Std:  {ps.mean():.4f}  (Max diff: {np.abs(ns-ps).max():.4f})")

    assert max_z < 5.0, f"Parity check failed for {name}: {max_z} > 5.0"

if __name__ == "__main__":
    check_parity("Matched noise", omega=0.7, t2=10.0, shape=1.0, vis=0.9, sigma_phi=0.0, t_corr=50.0, k=8)
    check_parity("Drift 2 rad fast", omega=1.2, t2=10.0, shape=1.0, vis=0.9, sigma_phi=2.0, t_corr=20.0, k=16)
    check_parity("Shape p=2", omega=0.3, t2=2.5, shape=2.0, vis=0.6, sigma_phi=0.0, t_corr=50.0, k=4)
    print("\nAll parity tests passed!")
