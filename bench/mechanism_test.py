"""Check mechanism by binning trials by dispersion and comparing posterior widths."""
import numpy as np
from core.sim import Noise, simulate
from est.bayes import posterior, DEFAULT_GRID
import est.nn_infer as _nn

def compute_width(p, grid=_nn._bin_centers):
    cdf = np.cumsum(p)
    lb = grid[np.searchsorted(cdf, 0.025)]
    ub = grid[np.searchsorted(cdf, 0.975)]
    return ub - lb

def main():
    rng = np.random.default_rng(123)
    k = 16
    trials = 1000
    
    # 1. Collect Drift Trials
    noise_drift = Noise(sigma_phi_rad=3.0, correlation_shots=10)
    
    ods = []
    w_full = []
    w_abl = []
    
    print("Simulating Drift 3 rad trials...")
    for _ in range(trials):
        omega = rng.uniform(0.0, 1.759)
        counts, n, rep_c = simulate(omega, noise_drift, k, rng, return_repeat_counts=True)
        
        # Calculate mean overdispersion (features 12:18 are ODs in the 19-feat vector)
        feats_19 = _nn.extract_features(counts, n, repeat_counts=rep_c)
        od_mean = np.mean(feats_19[12:18])
        ods.append(od_mean)
        
        # Full NN Width
        p_full = _nn.posterior(counts, n, repeat_counts=rep_c, weights_path="artifacts/nn_weights.npz")
        w_full.append(compute_width(p_full))
        
        # Ablation NN Width
        p_abl = _nn.posterior(counts, n, repeat_counts=rep_c, weights_path="artifacts/nn_weights_ablation.npz")
        w_abl.append(compute_width(p_abl))
        
    ods = np.array(ods)
    w_full = np.array(w_full)
    w_abl = np.array(w_abl)
    
    median_od = np.median(ods)
    low_mask = ods <= median_od
    high_mask = ods > median_od
    
    print("\n--- Drift 3 rad Trials (Split by Trial OD) ---")
    print(f"Low Dispersion  (OD <= {median_od:.2f}):  Full Width = {np.mean(w_full[low_mask]):.4f}, Ablation Width = {np.mean(w_abl[low_mask]):.4f}")
    print(f"High Dispersion (OD >  {median_od:.2f}):  Full Width = {np.mean(w_full[high_mask]):.4f}, Ablation Width = {np.mean(w_abl[high_mask]):.4f}")
    
    print(f"\nWidth Change (High vs Low OD):")
    print(f"  Ablation NN expands by {np.mean(w_abl[high_mask]) / np.mean(w_abl[low_mask]):.2f}x")
    print(f"  Full NN expands by     {np.mean(w_full[high_mask]) / np.mean(w_full[low_mask]):.2f}x")
    
if __name__ == "__main__":
    main()
