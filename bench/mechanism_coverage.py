"""Check coverage across dispersion terciles."""
import numpy as np
from core.sim import Noise, simulate
from est.bayes import posterior, DEFAULT_GRID
import est.nn_infer as _nn

def main():
    rng = np.random.default_rng(999)
    k = 16
    trials = 2000
    
    noise_drift = Noise(sigma_phi_rad=3.0, correlation_shots=10)
    
    ods = []
    cov_full = []
    cov_abl = []
    
    print("Simulating Drift 3 rad trials (k=16)...")
    for _ in range(trials):
        omega = rng.uniform(0.0, 1.759)
        counts, n, rep_c = simulate(omega, noise_drift, k, rng, return_repeat_counts=True)
        
        feats_19 = _nn.extract_features(counts, n, repeat_counts=rep_c)
        od_mean = np.mean(feats_19[12:18])
        ods.append(od_mean)
        
        # Full NN
        p_full = _nn.posterior(counts, n, repeat_counts=rep_c, weights_path="artifacts/nn_weights.npz")
        cdf_full = np.cumsum(p_full)
        lb_f = _nn._bin_centers[np.searchsorted(cdf_full, 0.025)]
        ub_f = _nn._bin_centers[np.searchsorted(cdf_full, 0.975)]
        cov_full.append(lb_f <= omega <= ub_f)
        
        # Ablation NN
        p_abl = _nn.posterior(counts, n, repeat_counts=rep_c, weights_path="artifacts/nn_weights_ablation.npz")
        cdf_abl = np.cumsum(p_abl)
        lb_a = _nn._bin_centers[np.searchsorted(cdf_abl, 0.025)]
        ub_a = _nn._bin_centers[np.searchsorted(cdf_abl, 0.975)]
        cov_abl.append(lb_a <= omega <= ub_a)
        
    ods = np.array(ods)
    cov_full = np.array(cov_full)
    cov_abl = np.array(cov_abl)
    
    t1, t2 = np.percentile(ods, [33.33, 66.67])
    
    mask_low = ods <= t1
    mask_mid = (ods > t1) & (ods <= t2)
    mask_high = ods > t2
    
    print(f"\n--- Coverage by Dispersion Tercile ---")
    print(f"Low OD (<= {t1:.2f}):")
    print(f"  Full NN Coverage = {np.mean(cov_full[mask_low]):.3f}")
    print(f"  Ablation Coverage= {np.mean(cov_abl[mask_low]):.3f}")
    
    print(f"Mid OD ({t1:.2f} - {t2:.2f}):")
    print(f"  Full NN Coverage = {np.mean(cov_full[mask_mid]):.3f}")
    print(f"  Ablation Coverage= {np.mean(cov_abl[mask_mid]):.3f}")
    
    print(f"High OD (> {t2:.2f}):")
    print(f"  Full NN Coverage = {np.mean(cov_full[mask_high]):.3f}")
    print(f"  Ablation Coverage= {np.mean(cov_abl[mask_high]):.3f}")

if __name__ == "__main__":
    main()
