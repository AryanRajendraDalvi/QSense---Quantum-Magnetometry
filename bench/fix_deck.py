"""Fix deck wording."""
import re
with open(r'C:\Users\ARYAN\.gemini\antigravity\brain\653fec1f-73bc-41b6-8f8f-59de643f75ff\deck.md', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('*(Ablation tests and dispersion-binning are consistent with this mechanism).*',
                    '*(Ablation tests show dispersion features provide a ~3-4% accuracy boost at k=16. Limitation: they offer no benefit at low budgets like k=4 where variance across repeats is too noisy to estimate).*')

with open(r'C:\Users\ARYAN\.gemini\antigravity\brain\653fec1f-73bc-41b6-8f8f-59de643f75ff\deck.md', 'w', encoding='utf-8') as f:
    f.write(text)
