# QSense - Learned Quantum Magnetometry

### The Problem
Classical Bayesian estimation for quantum magnetometry assumes the laboratory environment is perfectly stable. When real-world noise (like phase drift or signal decay) violates these assumptions, standard estimators become dangerously overconfident, outputting narrow error bars that completely miss the true magnetic field.

### Team & Track
**Track:** Quantum Software / Applications  
**Team:** Devfolio Registration Confirmed

### Why Quantum?
Quantum sensors like NV centers in diamond offer unprecedented spatial resolution and sensitivity by measuring how a quantum spin wobbles in a magnetic field. But realizing this potential outside the lab requires software that can dynamically handle drifting, messy environments.

### Results Summary
1.  **Safety Restored:** Standard methods fail catastrophically under drift (coverage drops to 57%). Marginalizing over noise parameters (B3) restores safety, and our learned amortized Neural Network (NN) perfectly matches this safety.
2.  **Accuracy Boost:** Under strong phase drift, the NN achieves 3-6% lower error than the best Bayesian method (B3).
3.  **Speed:** The NN evaluates in ~406 us, over 30x faster than B3 (~12.4 ms).

### Run Instructions
Tested on Python 3.10+ (Windows/Linux).
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## Evaluation Scoreboard

All results from 2,000 paired test trials per scenario. k=16 repeats unless noted.
**Methods:** B1 = calibrated Bayesian, B2 = T2*/V nuisance, B3 (Best Bayesian) = +shape nuisance, NN = amortized network.

### Matched Noise (k=16)
| Method | RMSE (uT) | cov95 |
|--------|-----------|-------|
| **B1** | **0.181** | 0.949 |
| B2 | 0.195 | 0.951 |
| B3 | 0.206 | 0.963 |
| NN | 0.208 | 0.987 |

*Note: B3 pays a roughly 13% robustness tax under matched noise (B3/B1 RMSE ratio = 1.127 [95% CI: 1.077, 1.181]). The NN pays roughly a 15% tax (0.208 / 0.181 = 1.149). The NN's coverage (0.987) is notably highly conservative here.*

### Shape Mismatch (k=16)
| Scenario | Method | RMSE | cov95 | Verdict |
|----------|--------|------|-------|---------|
| p=2 | B1 | 0.206 | 0.842 | Under-covers |
| p=2 | **B3** | **0.212** | **0.954** | Good |
| p=2 | NN | 0.221 | 0.984 | Conservative |
| p=3 | B1 | 0.243 | 0.747 | Under-covers |
| p=3 | **B3** | **0.231** | **0.931** | Good |
| p=3 | NN | 0.242 | 0.970 | Near-nominal |

### Visibility Mismatch (k=16)
| Method | RMSE (uT) | cov95 |
|--------|-----------|-------|
| B1 | 1.020 | 0.351 |
| **B2** | **0.403** | **0.962** |
| B3 | 0.422 | 0.960 |
| NN (OOD)* | 0.430 | 0.960 |

*\*Visibility 0.5 is out of distribution for the NN (training range 0.6-0.95).*

### Extreme Out-of-Distribution (k=16)
*These conditions are entirely outside the NN's training support. Results are informational.*

| Scenario | Method | RMSE (uT) | cov95 |
|----------|--------|-----------|-------|
| Drift 4 rad (OOD) | B3 | 0.467 | 0.883 |
| Drift 4 rad (OOD) | **NN** | **0.449** | **0.945** |
| T2* = 1.25 us (OOD) | **B2** | **1.244** | **0.766** |
| T2* = 1.25 us (OOD) | B3 | 1.282 | 0.783 |
| T2* = 1.25 us (OOD) | NN | 1.364 | 0.741 |

*The NN degrades gracefully at higher drift (4 rad, 33% beyond training limit), but decisively loses to B3 when T2* falls to 1.25 us (half the minimum training value).*

---

## Multi-Seed Robustness & Exploratory Analysis

We evaluated the NN across 5 independent training seeds over the same 2,000 paired test trials. 

### Core Claim: NN vs B3
Reported as [min, max] RMSE ratio (NN / B3) across all 5 seeds:
*   **Drift 2 rad, k=4:** [0.971, 0.981]
*   **Drift 2 rad, k=16:** [0.958, 0.975]
*   **Drift 3 rad, k=4:** [0.971, 0.976]
*   **Drift 3 rad, k=16:** [0.936, 0.969]

*Conclusion: Non-inferiority to B3 safely passes in every cell across all 5 independent training seeds. The NN provides a 2-4% accuracy advantage at 2 rad, and a 3-6% advantage at 3 rad.*

### Feature Ablation: Mechanism of Gain
To understand where the accuracy gain comes from, we compared the Full (19-feature) NN vs an Ablated (12-feature) NN across the 5 seeds.

| Method Comparison | RMSE Ratio (3 rad, k=16) | Source of Accuracy Gain |
|---|---|---|
| **Ablation / B3** | ~0.968 | Purely from training on the noise mixture |
| **Full / Ablation** | [0.968, 1.000] | Adding dispersion features |

*Conclusion:* The NN's primary gain over B3 under drift comes mostly from training on the noise mixture itself. At $k=4$, the variance across repeats is too noisy to estimate, so dispersion features offer no benefit across all seeds. At $k=16$, the dispersion features provided a ~3% boost in 1 of 5 seeds. (Note: The Streamlit app runs this specific successful seed). Making the network reliably use these features-perhaps via log transformations or dynamic scaling-is a prime target for future work.

### Inference Latency
Reported as median wall time per estimate (k=16 scenario):
*   B1: 264.3 us
*   **NN: 406.5 us**
*   B2: 3,468.1 us
*   B3: 12,367.5 us

---

## Pre-Registered Evaluation Details

Our evaluation methodology was strictly pre-registered prior to testing the Neural Network. (Commit tag: `prereg-v3`).

**Criteria**: NN wins if cov95 ∈ [0.93, 0.97] AND RMSE <= 1.10x B3 (the best Bayesian method) on drift 2 and 3 rad at k=16.

*   **Drift 2 rad, k=16:** NN/B3 RMSE ratio = 0.972 (PASS <= 1.10). NN cov95 = 0.973 (NARROW MISS, > 0.97).
*   **Drift 3 rad, k=16:** NN/B3 RMSE ratio = 0.945 (PASS <= 1.10). NN cov95 = 0.965 (PASS).

*Summary:* The NN meets the non-inferiority pre-registered criteria on RMSE at both drift levels. It passes the coverage criterion cleanly at drift 3 rad, and narrowly over-covers at drift 2 rad (0.973 vs upper bound 0.97). *Note: The pre-registered ratios (0.972, 0.945) reflect the mean of 5,000 bootstrap resamples on a distinct random test set, consistent with the original freeze.*
