# QSense

QSense is a simulator and estimator benchmark for single-qubit Ramsey magnetometry when real noise differs from the calibrated noise model.

## Run

```powershell
python -m pip install -r requirements.txt
python -m pytest
streamlit run app.py
```

The implementation includes the physical model, time-ordered correlated-drift simulator, local Fisher-information reference, and the B1/B2/B3 grid Bayesian estimators. NN training and the full paired benchmark harness are Sunday's milestones.

Run the Bayesian spike with `python -m bench.run_benchmark --trials 2000 --k 4 --output results/b1_b2_spike_2000.json`.

`fail_rate` is an aliasing-oriented metric: an estimate is catastrophic only when its error exceeds four times that trial's local reference σ.

## Pre-registered NN evaluation criteria (locked before training)

### Decision from B3 drift sweep (2000 trials, k=4 and k=16, seed 20261004)

B3 (shape-marginalized Bayesian) 95% coverage under fast correlated drift:

| Condition | k=4 cov95 | k=16 cov95 | k=16 cov68 |
|---|---:|---:|---:|
| drift 1 rad | 0.946 | 0.952 | 0.693 |
| drift 2 rad | 0.927 | 0.923 | 0.634 |
| drift 3 rad | 0.916 | 0.901 | 0.592 |

**Decision: The gate was marginal, but the NN niche remains.** The pre-registered rule required B3 to fall below 0.90 to cleanly validate the niche. At drift 3 rad, k=16, B3's 95% coverage is 0.901—in the gray zone. However, its 68% coverage drops dramatically to 0.592 (nominal 0.68), and its RMSE rises 74% compared to matched noise. The calibration headroom is small, but the accuracy damage is large. If the NN can detect over-dispersion and down-weight noisy long waits, accuracy is where it could win. Training proceeds.

### NN win criteria (evaluated only on the in-distribution niche)

| Scenario | Budget | NN counts as a win if |
|---|---|---|
| drift 2 rad fast | k=16 | cov95 ∈ [0.93, 0.97] **and** RMSE ≤ 1.1× best Bayesian |
| drift 3 rad fast | k=16 | cov95 ∈ [0.93, 0.97] **and** RMSE ≤ 1.1× best Bayesian |

If the NN misses **both** criteria it is reported as an ablation with inference speed as its only claim.

### Training distribution (revised, locked before training)

| Parameter | Range |
|---|---|
| ω (target) | U[0, 1.759] rad/µs |
| T2* | log-U[2.5, 40] µs |
| Shape p | U[1, 3] |
| Visibility V | U[0.6, 0.95] |
| Drift σφ | 50% none; else U[0, 3] rad |
| Correlation time Tc | log-U[10, 200] shots |
| Budget k | uniform over {1, 2, 4, 8, 16} |

### NN OOD stress tests (no pass/fail, reported separately)

- Visibility 0.5 (below training range)
- Drift 4 rad (above training range)
- T2* = 1.25 µs (below training range)
- Drift 3 rad + T2* = 5 µs (combined)

### NN inputs (19 features)

1. **6** pooled outcome-0 frequencies per tau: `f_j = k0_j / n_j`
2. **6** normalized budgets: `n_j / (16 × 10)`
3. **6** per-tau over-dispersion ratios: `var(repeat freq) / (f_j(1-f_j)/k + ε)`, clipped to [0, 10]
4. **1** k=1 mask (1.0 if k==1, else 0.0)

The beta-binomial over-dispersion likelihood is a classical alternative for the dispersion features; documented limitation, not part of Round 1.

### Inference latency reporting

Report median wall time per estimate (µs) for B1, B2, B3, and NN on the k=16 matched scenario. Speed is only the headline claim if the NN misses the calibration criteria.

## Units

All internal frequency units are rad/µs; displayed magnetic fields use µT with `gamma = 0.1759 rad/µs/µT`.

## Repo layout

```
qsense/
  app.py
  core/  physics.py  sim.py  crb.py
  est/   bayes.py  nn_infer.py
  train/ train_nn.py  export.py
  bench/ run_benchmark.py  metrics.py
  tests/ test_physics.py  test_qiskit_crosscheck.py  test_nn_export.py
  results/  artifacts/  figures/  sample_data/
  notebooks/reproduce.ipynb
  README.md  requirements.txt  kaggle_train.py
```

## Kaggle GPU training

Upload the whole repo to a Kaggle notebook with a GPU accelerator (T4 or P100).

```bash
python kaggle_train.py
```

Download `artifacts/nn_weights.npz` and `artifacts/nn_meta.json` and place them in the repo before running the full benchmark or the app.

## Benchmark commands

```powershell
# B1/B2 spike (2000 trials, k=4)
python -m bench.run_benchmark --trials 2000 --k 4 --output results/b1_b2_spike_2000.json

# B3 shape sweep
python -m bench.run_benchmark --trials 2000 --k 4 16 --suite harsh --include-b3 \
  --only shape_p2 shape_p3 shape_p2_t2_half shape_p3_t2_half shape_p2_t2_double shape_p3_t2_double \
  --output results/b3_shape_sweep_2000.json

# B3 drift/matched/visibility sweep
python -m bench.run_benchmark --trials 2000 --k 4 16 --suite harsh --include-b3 \
  --only matched drift_1rad_fast drift_2rad_fast drift_3rad_fast visibility_half \
  --output results/b3_drift_sweep_2000.json

# Full benchmark with NN (after training)
python -m bench.run_benchmark --trials 2000 --k 4 16 --suite harsh --include-b3 --include-nn \
  --output results/benchmark.json
```
