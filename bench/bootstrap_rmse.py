"""Calculate bootstrap CI for NN vs B3 RMSE ratio, and measure latency."""
import json
import time
import numpy as np
from core.sim import Noise, simulate
from est.bayes import posterior, DEFAULT_GRID
from core.physics import b_from_omega

def main():
    import est.nn_infer as _nn
    rng = np.random.default_rng(42)
    trials = 2000
    
    # Measure latency first
    print("Measuring latency...")
    # Warmup
    noise = Noise(sigma_phi_rad=2.0)
    for _ in range(5):
        counts, n, rep_c = simulate(0.5, noise, 16, rng, return_repeat_counts=True)
        _ = posterior(counts, n, "nuisance_shape")
        _ = _nn.posterior(counts, n, repeat_counts=rep_c, weights_path="artifacts/nn_weights.npz")
    
    t0 = time.perf_counter()
    for _ in range(100):
        counts, n, rep_c = simulate(0.5, noise, 16, rng, return_repeat_counts=True)
        _ = posterior(counts, n, "nuisance_shape")
    b3_time = (time.perf_counter() - t0) / 100 * 1000  # ms
    
    t0 = time.perf_counter()
    for _ in range(100):
        counts, n, rep_c = simulate(0.5, noise, 16, rng, return_repeat_counts=True)
        _ = _nn.posterior(counts, n, repeat_counts=rep_c, weights_path="artifacts/nn_weights.npz")
    nn_time = (time.perf_counter() - t0) / 100 * 1000  # ms
    print(f"Latency per estimate (k=16): B3 = {b3_time:.1f} ms, NN = {nn_time:.2f} ms")
    
    # Bootstrap CIs
    print("\nCalculating paired bootstrap CIs for RMSE ratio (NN / B3)...")
    scenarios = [
        ("drift_2rad_fast", 4, Noise(sigma_phi_rad=2.0, correlation_shots=10)),
        ("drift_2rad_fast", 16, Noise(sigma_phi_rad=2.0, correlation_shots=10)),
        ("drift_3rad_fast", 4, Noise(sigma_phi_rad=3.0, correlation_shots=10)),
        ("drift_3rad_fast", 16, Noise(sigma_phi_rad=3.0, correlation_shots=10)),
    ]
    
    for name, k, noise in scenarios:
        print(f"\nScenario: {name}, k={k}")
        nn_errs = []
        b3_errs = []
        for _ in range(trials):
            omega = rng.uniform(0.0, 1.759)
            counts, n, rep_c = simulate(omega, noise, k, rng, return_repeat_counts=True)
            
            p_b3 = posterior(counts, n, "nuisance_shape")
            est_b3 = np.average(DEFAULT_GRID, weights=p_b3)
            b3_errs.append(b_from_omega(est_b3) - b_from_omega(omega))
            
            p_nn = _nn.posterior(counts, n, repeat_counts=rep_c, weights_path="artifacts/nn_weights.npz")
            est_nn = np.average(_nn._bin_centers, weights=p_nn)
            nn_errs.append(b_from_omega(est_nn) - b_from_omega(omega))
            
        b3_errs = np.array(b3_errs)
        nn_errs = np.array(nn_errs)
        
        # Bootstrap
        n_boot = 5000
        boot_ratios = []
        for _ in range(n_boot):
            idx = rng.choice(trials, size=trials, replace=True)
            rmse_b3 = np.sqrt(np.mean(b3_errs[idx]**2))
            rmse_nn = np.sqrt(np.mean(nn_errs[idx]**2))
            boot_ratios.append(rmse_nn / rmse_b3)
            
        ratio_mean = np.mean(boot_ratios)
        ci_lower = np.percentile(boot_ratios, 2.5)
        ci_upper = np.percentile(boot_ratios, 97.5)
        
        print(f"  RMSE Ratio (NN/B3): {ratio_mean:.3f} [95% CI: {ci_lower:.3f}, {ci_upper:.3f}]")
        if ci_upper < 1.0:
            print("  -> NN significantly beats B3.")
        elif ci_lower > 1.0:
            print("  -> B3 significantly beats NN.")
        else:
            print("  -> NN matches B3 (difference not significant).")

if __name__ == "__main__":
    main()
