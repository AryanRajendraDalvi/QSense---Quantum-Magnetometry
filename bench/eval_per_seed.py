"""Print per-seed table exactly matching the original test trials."""
import numpy as np
from core.sim import Noise, simulate
from est.bayes import posterior, DEFAULT_GRID
from core.physics import b_from_omega
import est.nn_infer as _nn
from pathlib import Path

def main():
    # MUST MATCH original bootstrap script seed (42) so the trials are identical!
    rng = np.random.default_rng(42)
    trials = 2000
    
    seeds = [
        {"name": "Seed 20261004 (Original)", "full": "artifacts/nn_weights.npz", "abl": "artifacts/nn_weights_ablation.npz"},
        {"name": "Seed 20261008 (New)", "full": "artifacts/nn_weights_full_20261008.npz", "abl": "artifacts/nn_weights_ablation_20261008.npz"},
    ]
    
    # Also check if user added others
    for seed_num in [20261005, 20261006, 20261007]:
        f_full = Path(f"artifacts/nn_weights_full_{seed_num}.npz")
        f_abl = Path(f"artifacts/nn_weights_ablation_{seed_num}.npz")
        if f_full.exists() and f_abl.exists():
            seeds.append({"name": f"Seed {seed_num}", "full": str(f_full), "abl": str(f_abl)})

    scenarios = [
        ("Drift 2 rad", 4, Noise(sigma_phi_rad=2.0, correlation_shots=10)),
        ("Drift 2 rad", 16, Noise(sigma_phi_rad=2.0, correlation_shots=10)),
        ("Drift 3 rad", 4, Noise(sigma_phi_rad=3.0, correlation_shots=10)),
        ("Drift 3 rad", 16, Noise(sigma_phi_rad=3.0, correlation_shots=10)),
    ]
    
    print(f"Per-Seed Table ({len(seeds)} Seeds) - Seed 42 Trials")
    print("=" * 80)
    
    for name, k, noise in scenarios:
        print(f"\n[{name}, k={k}]")
        
        b3_errs = []
        full_errs = [[] for _ in seeds]
        abl_errs = [[] for _ in seeds]
        
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
        
        for i, seed_dict in enumerate(seeds):
            f_e = np.array(full_errs[i])
            a_e = np.array(abl_errs[i])
            rmse_f = np.sqrt(np.mean(f_e**2))
            rmse_a = np.sqrt(np.mean(a_e**2))
            
            ratio_b3 = rmse_f / rmse_b3
            ratio_abl = rmse_f / rmse_a
            
            failure_b3 = "FAILURE" if ratio_b3 > 1.0 else "PASS"
            failure_abl = "FAILURE" if ratio_abl > 1.0 else "PASS"
            
            print(f"  {seed_dict['name']}:")
            print(f"    Full/B3 RMSE Ratio:       {ratio_b3:.3f} ({failure_b3})")
            print(f"    Full/Ablation RMSE Ratio: {ratio_abl:.3f} ({failure_abl})")

if __name__ == "__main__":
    main()
