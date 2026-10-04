"""Evaluate ablation model specifically on the drift niche."""
import numpy as np
from core.sim import Noise, simulate
from est.bayes import posterior, DEFAULT_GRID
from core.physics import b_from_omega
import est.nn_infer as _nn

def main():
    rng = np.random.default_rng(42)
    trials = 2000
    
    # Load both models
    _nn._get_model("artifacts/nn_weights.npz")
    _nn._get_model("artifacts/nn_weights_ablation.npz")
    
    scenarios = [
        ("Drift 2 rad", 16, Noise(sigma_phi_rad=2.0, correlation_shots=10)),
        ("Drift 3 rad", 16, Noise(sigma_phi_rad=3.0, correlation_shots=10)),
    ]
    
    for label, k, noise in scenarios:
        print(f"\n--- {label}, k={k} ---")
        errs_b3 = []
        errs_nn = []
        errs_abl = []
        cov95_nn = 0
        cov95_abl = 0
        cov95_b3 = 0
        
        for _ in range(trials):
            omega = rng.uniform(0.0, 1.759)
            counts, n, rep_c = simulate(omega, noise, k, rng, return_repeat_counts=True)
            
            # B3
            p_b3 = posterior(counts, n, "nuisance_shape")
            est_b3 = np.average(DEFAULT_GRID, weights=p_b3)
            errs_b3.append(b_from_omega(est_b3) - b_from_omega(omega))
            cdf_b3 = np.cumsum(p_b3)
            lb_b3 = DEFAULT_GRID[np.searchsorted(cdf_b3, 0.025)]
            ub_b3 = DEFAULT_GRID[np.searchsorted(cdf_b3, 0.975)]
            if lb_b3 <= omega <= ub_b3: cov95_b3 += 1
            
            # Full NN
            p_nn = _nn.posterior(counts, n, repeat_counts=rep_c, weights_path="artifacts/nn_weights.npz")
            est_nn = np.average(_nn._bin_centers, weights=p_nn)
            errs_nn.append(b_from_omega(est_nn) - b_from_omega(omega))
            cdf_nn = np.cumsum(p_nn)
            lb_nn = _nn._bin_centers[np.searchsorted(cdf_nn, 0.025)]
            ub_nn = _nn._bin_centers[np.searchsorted(cdf_nn, 0.975)]
            if lb_nn <= omega <= ub_nn: cov95_nn += 1
            
            # Ablation NN
            p_abl = _nn.posterior(counts, n, repeat_counts=rep_c, weights_path="artifacts/nn_weights_ablation.npz")
            est_abl = np.average(_nn._bin_centers, weights=p_abl)
            errs_abl.append(b_from_omega(est_abl) - b_from_omega(omega))
            cdf_abl = np.cumsum(p_abl)
            lb_abl = _nn._bin_centers[np.searchsorted(cdf_abl, 0.025)]
            ub_abl = _nn._bin_centers[np.searchsorted(cdf_abl, 0.975)]
            if lb_abl <= omega <= ub_abl: cov95_abl += 1
            
        print(f"B3:       RMSE = {np.sqrt(np.mean(np.array(errs_b3)**2)):.4f} uT, cov95 = {cov95_b3/trials:.3f}")
        print(f"NN Abl:   RMSE = {np.sqrt(np.mean(np.array(errs_abl)**2)):.4f} uT, cov95 = {cov95_abl/trials:.3f}")
        print(f"NN Full:  RMSE = {np.sqrt(np.mean(np.array(errs_nn)**2)):.4f} uT, cov95 = {cov95_nn/trials:.3f}")

if __name__ == "__main__":
    main()
