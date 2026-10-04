import time
import numpy as np
from core.sim import Noise, simulate
from est.bayes import posterior, DEFAULT_GRID
from est.nn_infer import posterior as nn_posterior, _bin_centers

def main():
    rng = np.random.default_rng(42)
    omega = 1.0
    noise = Noise()
    counts, n, rep_c = simulate(omega, noise, 16, rng, return_repeat_counts=True)
    
    methods = [
        ("B1", lambda: posterior(counts, n, "assumed")),
        ("B2", lambda: posterior(counts, n, "nuisance")),
        ("B3", lambda: posterior(counts, n, "nuisance_shape")),
        ("NN", lambda: nn_posterior(counts, n, repeat_counts=rep_c, weights_path="artifacts/nn_weights.npz")),
    ]
    
    # Warmup
    for name, f in methods:
        for _ in range(5): f()
        
    for name, f in methods:
        t0 = time.time()
        n_iters = 100
        for _ in range(n_iters):
            f()
        dt = (time.time() - t0) / n_iters * 1e6
        print(f"{name}: {dt:.1f} us")

if __name__ == "__main__":
    main()
