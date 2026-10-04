"""Calculate bootstrap CI for NN vs B3 and Full vs Ablation."""
import numpy as np
from core.sim import Noise, simulate
from est.bayes import posterior, DEFAULT_GRID
from core.physics import b_from_omega
import est.nn_infer as _nn

def bootstrap_ratio(errs_num, errs_den, trials=2000, n_boot=5000, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    boot_ratios = []
    for _ in range(n_boot):
        idx = rng.choice(trials, size=trials, replace=True)
        rmse_num = np.sqrt(np.mean(errs_num[idx]**2))
        rmse_den = np.sqrt(np.mean(errs_den[idx]**2))
        boot_ratios.append(rmse_num / rmse_den)
    return np.mean(boot_ratios), np.percentile(boot_ratios, 2.5), np.percentile(boot_ratios, 97.5)

def main():
    rng = np.random.default_rng(42)
    trials = 2000
    
    scenarios = [
        ("drift_2rad_fast", 4, Noise(sigma_phi_rad=2.0, correlation_shots=10)),
        ("drift_2rad_fast", 16, Noise(sigma_phi_rad=2.0, correlation_shots=10)),
        ("drift_3rad_fast", 4, Noise(sigma_phi_rad=3.0, correlation_shots=10)),
        ("drift_3rad_fast", 16, Noise(sigma_phi_rad=3.0, correlation_shots=10)),
    ]
    
    for name, k, noise in scenarios:
        print(f"\nScenario: {name}, k={k}")
        b3_errs = []
        full_errs = []
        abl_errs = []
        
        # We must collect errors sequentially for the same trials
        for _ in range(trials):
            omega = rng.uniform(0.0, 1.759)
            counts, n, rep_c = simulate(omega, noise, k, rng, return_repeat_counts=True)
            
            p_b3 = posterior(counts, n, "nuisance_shape")
            est_b3 = np.average(DEFAULT_GRID, weights=p_b3)
            b3_errs.append(b_from_omega(est_b3) - b_from_omega(omega))
            
            # Full NN (19 feats)
            p_full = _nn.posterior(counts, n, repeat_counts=rep_c, weights_path="artifacts/nn_weights.npz")
            est_full = np.average(_nn._bin_centers, weights=p_full)
            full_errs.append(b_from_omega(est_full) - b_from_omega(omega))
            
            # Abl NN (12 feats)
            p_abl = _nn.posterior(counts, n, repeat_counts=rep_c, weights_path="artifacts/nn_weights_ablation.npz")
            est_abl = np.average(_nn._bin_centers, weights=p_abl)
            abl_errs.append(b_from_omega(est_abl) - b_from_omega(omega))
            
        b3_errs = np.array(b3_errs)
        full_errs = np.array(full_errs)
        abl_errs = np.array(abl_errs)
        
        ratio_full_b3, ci_l, ci_u = bootstrap_ratio(full_errs, b3_errs, trials=trials, rng=rng)
        print(f"  Full NN / B3 RMSE Ratio:       {ratio_full_b3:.3f} [95% CI: {ci_l:.3f}, {ci_u:.3f}]")
        
        ratio_full_abl, ci_l_a, ci_u_a = bootstrap_ratio(full_errs, abl_errs, trials=trials, rng=rng)
        print(f"  Full NN / Ablation RMSE Ratio: {ratio_full_abl:.3f} [95% CI: {ci_l_a:.3f}, {ci_u_a:.3f}]")

if __name__ == "__main__":
    main()
