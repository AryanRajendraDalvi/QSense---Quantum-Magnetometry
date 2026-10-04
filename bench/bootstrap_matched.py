"""Calculate CI for B3 vs B1 on matched noise."""
import numpy as np
from core.sim import Noise, simulate
from est.bayes import posterior, DEFAULT_GRID
from core.physics import b_from_omega

def main():
    rng = np.random.default_rng(42)
    trials = 2000
    k = 16
    noise = Noise(sigma_phi_rad=0.0) # matched
    
    b1_errs = []
    b3_errs = []
    
    for _ in range(trials):
        omega = rng.uniform(0.0, 1.759)
        counts, n = simulate(omega, noise, k, rng)
        
        p_b1 = posterior(counts, n, "assumed")
        est_b1 = np.average(DEFAULT_GRID, weights=p_b1)
        b1_errs.append(b_from_omega(est_b1) - b_from_omega(omega))
        
        p_b3 = posterior(counts, n, "nuisance_shape")
        est_b3 = np.average(DEFAULT_GRID, weights=p_b3)
        b3_errs.append(b_from_omega(est_b3) - b_from_omega(omega))
        
    b1_errs = np.array(b1_errs)
    b3_errs = np.array(b3_errs)
    
    boot_ratios = []
    for _ in range(5000):
        idx = rng.choice(trials, size=trials, replace=True)
        rmse_b3 = np.sqrt(np.mean(b3_errs[idx]**2))
        rmse_b1 = np.sqrt(np.mean(b1_errs[idx]**2))
        boot_ratios.append(rmse_b3 / rmse_b1)
        
    print(f"B3 / B1 RMSE Ratio: {np.mean(boot_ratios):.3f} [95% CI: {np.percentile(boot_ratios, 2.5):.3f}, {np.percentile(boot_ratios, 97.5):.3f}]")

if __name__ == "__main__":
    main()
