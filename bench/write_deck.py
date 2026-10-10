"""Rewrite deck.md"""

deck_content = """---
theme: default
class: text-center
highlighter: shiki
lineNumbers: false
---

# Headline Summary
Assumed-noise uncertainty is overconfident under mismatch. Marginalizing over nuisance parameters restores calibration at an accuracy cost. A learned estimator trained across a noise mixture matches that classical fix in matched noise, is modestly more accurate under strong drift, is conservative in calibration, and loses under extreme dephasing. Dispersion features helped in 1 of 5 seeds (exploratory).

---
layout: default
---

# QSense: Learned Estimation for Quantum Magnetometers Under Noise Mismatch
**Hackathon Track:** Quantum Software / Applications

Robust Bayesian estimation via amortized neural networks.

---
layout: default
---

# 1. The Problem: Noise Mismatch

Quantum magnetometers (like NV centers) estimate magnetic fields from phase accumulation.
Standard Bayesian estimators assume a perfectly known noise environment. 

* **The Reality:** Calibrated noise drifts over time. Shape, contrast, and dephasing times fluctuate.
* **The Danger:** When assumed noise doesn't match real noise, standard Bayesian estimators become **dangerously overconfident**.
* **Our Aim:** Show how standard estimators fail, and how an estimator trained across a broad noise distribution restores reliable uncertainty intervals.

---
layout: default
---

# 2. Quantum Sensing: The Ramsey Experiment

We model a single-qubit electron spin sensor (e.g., NV center) in a Ramsey sequence.
* **Parameter:** Magnetic field $B$ (measured via Larmor frequency $\omega$).
* **Protocol:** Prepare $|+\\rangle$, wait $\\tau$, apply $\\pi/2$ pulse, measure.
* **Signal:** $P(0 | \\omega, \\tau) = \\frac{1}{2} [ 1 + V \\cdot C(\\tau) \\cdot \\cos(\\omega\\tau) ]$
  * $V$: Visibility (contrast)
  * $C(\\tau)$: Decoherence envelope $\\exp(-(\\tau / T_2^*)^p)$

We use a standard ladder of wait times ($\\tau \\in \\{0.5, 1, 2, 4, 8, 16\\}\\ \\mu\\text{s}$) to resolve phase ambiguity while maintaining sensitivity. Total budget is controlled by repeat factor $k$.

---
layout: default
---

# 3. The Solution: Learned Marginalization

We evaluated three estimators across 2,000 paired trials under simulated noise mismatch:
1. **B1 (Assumed):** Point-calibrated Bayesian estimator (the naive baseline).
2. **B3 (Nuisance):** Best Bayesian estimator that marginalizes over a grid of $T_2^*$, $V$, and shape $p$.
3. **NN (Amortized):** A neural network trained on 41 million experiments across a wide distribution of noise parameters, outputting a 1024-bin posterior.

**Key Advantage:** The NN implicitly learns a robust prior over noise without the exponential computational cost of multi-dimensional Bayesian grid integration.

---
layout: default
---

# 4. Method: Features and Over-Dispersion

**How does the NN detect unmodeled noise?**
It uses **19 features** extracted purely from shot outcomes:
* 6 pooled frequencies $f_j$
* 6 normalized budget allocations
* 1 budget mask
* **6 Over-dispersion ratios**: Variance of the per-repeat frequency divided by expected binomial variance.

**Mechanism attribution (Ablation vs B3):**
* The ~3-6% accuracy gain over B3 comes *mostly from training on the noise mixture* (the 12-feature ablation without dispersion already beats B3). 
* The dispersion features provided a ~3% additional boost in **1 of 5 seeds** at k=16 (and no benefit at k=4 where variance is too noisy). The Streamlit app runs the specific seed that exploited these features.

---
layout: default
---

# 5. Pre-Registered Results: The Drift Niche

Our primary hypothesis: The NN beats B3 on accuracy and calibration under fast, correlated temporal drift.

**Results at Strong Drift (3 rad, $k=16$):**
* **B1 (Standard)**: Fails catastrophically. Coverage collapses to **57.0%** (target 95%).
* **B3 (Best Bayesian)**: Restores coverage to 90.3%, but struggles with accuracy (RMSE 0.374 $\\mu$T).
* **NN (Amortized)**: **Coverage 96.5%**, RMSE 0.353 $\\mu$T.

**Statistical Significance (5-Seed Evaluation):**
* NN / B3 RMSE ratio at 3 rad, k=16: **[0.936, 0.969]** across 5 seeds, both below 1. 
* At 2 rad, k=16: **[0.958, 0.975]** (2-4% gain). 
* Met the pre-registered criterion (non-inferiority). Exploratory analysis shows 3-6% lower RMSE at 3 rad robustly across training seeds.

---
layout: default
---

# 6. Comprehensive Benchmarking

Beyond drift, we tested shape mismatch, contrast degradation, and Out-of-Distribution (OOD) extremes.

* **Matched Noise:** B3 pays a ~13% robustness tax over B1 [95% CI 1.077, 1.181]. The NN pays roughly a 15% tax. *Note: NN cov95 is 0.987 (highly conservative here).*
* **Shape Mismatch:** B3 and NN both reliably cover >93%.
* **Visibility Collapse:** B1 coverage drops to 35%. NN holds 96%.
* **OOD Generalization:** At 4 rad drift (33% beyond training limit), the NN degrades gracefully and generalized better than B3 (cov95: 94.5% vs 88.3%).
* **Inference Latency:** Median wall time per estimate: B1: 264 $\\mu$s | NN: 406 $\\mu$s | B2: 3,468 $\\mu$s | B3: 12,367 $\\mu$s.

---
layout: default
---

# 7. QSense App & Users

We built a fully interactive Streamlit application to bring robust quantum sensing to researchers.
* **Run & Compare:** Simulate Ramsey sequences with dynamic noise and watch how the posteriors of B1, B2, B3, and the NN evolve.
* **Custom Data:** Upload your experimental CSVs and let the estimators find your magnetic field.
* **Target Users:** Quantum sensing experimentalists who want to drop in a robust estimator without tuning a bespoke Bayesian grid for their exact noise environment.

---
layout: default
---

# 8. Limits & Roadmap

**Limitations:**
* **Extreme OOD:** When $T_2^*$ drops to 1.25 $\\mu$s (half the minimum training value), the NN loses to B3. It relies on its training support.
* **Conservative Coverage:** The NN leans wide (cov95 ~ 97-98% in some ID scenarios) to hedge against noise.
* **Dispersion Utilization:** As seen in the ablation study, only 1 of 5 seeds learned to exploit the dispersion features effectively.

**Roadmap:**
1. **Feature Engineering:** Make the dispersion features reliably used across all seeds by experimenting with log transformations or dynamic scaling.
2. **Beta-Binomial Likelihood:** Compare the NN's learned dispersion handling against a formal Beta-Binomial Bayesian model.
3. **Hardware Deployment:** Test the 400 $\\mu$s inference on real NV center hardware loops.
"""

with open(r'C:\Users\ARYAN\.gemini\antigravity\brain\653fec1f-73bc-41b6-8f8f-59de643f75ff\deck.md', 'w', encoding='utf-8') as f:
    f.write(deck_content)
