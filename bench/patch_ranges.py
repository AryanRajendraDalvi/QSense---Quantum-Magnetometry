"""Patch README and Deck with ranges."""
import re

readme_path = 'README.md'
deck_path = r'C:\Users\ARYAN\.gemini\antigravity\brain\653fec1f-73bc-41b6-8f8f-59de643f75ff\deck.md'

with open(readme_path, 'r', encoding='utf-8') as f:
    readme = f.read()

# Replace the Multi-Seed section
new_multiseed = """### Multi-Seed Test Performance (NN vs B3)
We evaluated the NN across 2 independent training seeds over the same 2,000 paired test trials. Reported as [min, max] RMSE ratio (NN / B3) across seeds:
*   **Drift 2 rad, k=4:** [0.971, 0.978]
*   **Drift 2 rad, k=16:** [0.958, 0.975]
*   **Drift 3 rad, k=4:** [0.971, 0.974]
*   **Drift 3 rad, k=16:** [0.936, 0.969]

*Conclusion: Non-inferiority to B3 holds in every cell. At 3 rad, k=16, the NN is ~3-6% better than B3 in both seeds. At 2 rad, the advantage shrinks to near parity.*

### Feature Ablation (Multi-Seed)
We evaluated the Full (19-feature) NN vs an Ablation (12-feature) NN across the same 2 seeds. Reported as [min, max] RMSE ratio (Full / Ablation):
*   **k=4 (3 rad):** [0.996, 1.000]
*   **k=16 (3 rad):** [0.968, 1.000]

The k=4 result (no effect) is robust across seeds. The k=16 gain is seed-dependent, ranging from a ~3% accuracy boost in one seed to roughly parity (no benefit) in the other. (2 of 5 planned seeds completed)."""

readme = re.sub(r'### Multi-Seed Test Performance.*?\(~1\.4% ± 2\.0%\), though effect size varies by training seed\.', new_multiseed, readme, flags=re.DOTALL)

with open(readme_path, 'w', encoding='utf-8') as f:
    f.write(readme)

# Deck
with open(deck_path, 'r', encoding='utf-8') as f:
    deck = f.read()

deck = deck.replace('*(Ablation tests across multiple training seeds show dispersion features provide an accuracy boost at k=16 [ratio 0.986 ± 0.020], but zero benefit at low budgets like k=4 [ratio 0.999 ± 0.001] where variance across repeats is too noisy to estimate).*',
                    '*(Ablation tests show dispersion features provide a 0-3% accuracy boost at k=16 depending on seed, and zero benefit at low budgets like k=4 where variance across repeats is too noisy to estimate).*')

deck = deck.replace('* NN / B3 RMSE ratio is **0.951 ± 0.018** (measured across training seeds).',
                    '* NN / B3 RMSE ratio at 3 rad, k=16: **[0.936, 0.969]** across 2 seeds, both below 1. At 2 rad: near parity ([0.958, 0.975]).')

deck = deck.replace('Exploratory analysis shows about 5% lower RMSE at 3 rad robustly across training seeds.',
                    'Exploratory analysis shows 3-6% lower RMSE at 3 rad (2 of 5 planned seeds completed).')

with open(deck_path, 'w', encoding='utf-8') as f:
    f.write(deck)
