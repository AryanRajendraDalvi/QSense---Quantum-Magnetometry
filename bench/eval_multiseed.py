"""Evaluate multiple seeds for NN."""
import numpy as np
from core.sim import Noise, simulate
from est.bayes import posterior, DEFAULT_GRID
from core.physics import b_from_omega
import est.nn_infer as _nn
from pathlib import Path

def main():
    rng = np.random.default_rng(12345)
    trials = 2000
    
    seeds = [
        {"full": "artifacts/nn_weights.npz", "abl": "artifacts/nn_weights_ablation.npz"},
        {"full": "artifacts/nn_weights_full_20261008.npz", "abl": "artifacts/nn_weights_ablation_20261008.npz"},
    ]
    
    # Optional: check if other seeds exist in artifacts
    for seed_num in [20261005, 20261006, 20261007]:
        f_full = Path(f"artifacts/nn_weights_full_{seed_num}.npz")
        f_abl = Path(f"artifacts/nn_weights_ablation_{seed_num}.npz")
        if f_full.exists() and f_abl.exists():
            seeds.append({"full": str(f_full), "abl": str(f_abl)})
            
    print(f"Found {len(seeds)} trained seeds.")

    scenarios = [
        ("drift_2rad_fast", 4, Noise(sigma_phi_rad=2.0, correlation_shots=10)),
        ("drift_2rad_fast", 16, Noise(sigma_phi_rad=2.0, correlation_shots=10)),
        ("drift_3rad_fast", 4, Noise(sigma_phi_rad=3.0, correlation_shots=10)),
        ("drift_3rad_fast", 16, Noise(sigma_phi_rad=3.0, correlation_shots=10)),
    ]
    
    for name, k, noise in scenarios:
        print(f"\nScenario: {name}, k={k}")
        
        b3_errs = []
        full_errs = [[] for _ in seeds]
        abl_errs = [[] for _ in seeds]
        
        # Collect exactly same trials for all models
        for _ in range(trials):
            omega = rng.uniform(0.0, 1.759)
            counts, n, rep_c = simulate(omega, noise, k, rng, return_repeat_counts=True)
            
            p_b3 = posterior(counts, n, "nuisance_shape")
            est_b3 = np.average(DEFAULT_GRID, weights=p_b3)
            b3_errs.append(b_from_omega(est_b3) - b_from_omega(omega))
            
            for i, seed_dict in enumerate(seeds):
                p_full = _nn.posterior(counts, n, repeat_counts=rep_c, weights_path=seed_dict["full"])
                est_full = np.average(_nn._bin_centers, weights=p_full)
                full_errs[i].append(b_from_omega(est_full) - b_from_omega(omega))
                
                p_abl = _nn.posterior(counts, n, repeat_counts=rep_c, weights_path=seed_dict["abl"])
                est_abl = np.average(_nn._bin_centers, weights=p_abl)
                abl_errs[i].append(b_from_omega(est_abl) - b_from_omega(omega))
                
        b3_errs = np.array(b3_errs)
        rmse_b3 = np.sqrt(np.mean(b3_errs**2))
        
        nn_b3_ratios = []
        full_abl_ratios = []
        
        for i in range(len(seeds)):
            f_e = np.array(full_errs[i])
            a_e = np.array(abl_errs[i])
            rmse_f = np.sqrt(np.mean(f_e**2))
            rmse_a = np.sqrt(np.mean(a_e**2))
            
            nn_b3_ratios.append(rmse_f / rmse_b3)
            full_abl_ratios.append(rmse_f / rmse_a)
            
        print(f"  NN / B3 RMSE Ratio:        {np.mean(nn_b3_ratios):.3f} ± {np.std(nn_b3_ratios, ddof=1) if len(seeds) > 1 else 0.0:.3f}")
        print(f"  Full / Ablation RMSE Ratio:{np.mean(full_abl_ratios):.3f} ± {np.std(full_abl_ratios, ddof=1) if len(seeds) > 1 else 0.0:.3f}")
        print("  Raw ratios (Full/B3):", [f"{r:.3f}" for r in nn_b3_ratios])
        print("  Raw ratios (Full/Abl):", [f"{r:.3f}" for r in full_abl_ratios])

if __name__ == "__main__":
    main()
