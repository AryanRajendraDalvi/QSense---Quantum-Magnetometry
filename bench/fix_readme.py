"""Fix README encoding."""
import re
with open('README.md', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('As', 'µs').replace('AT', 'µT').replace('I', 'σ').replace('I%', 'B').replace('II+', 'σ_φ').replace('^*', 'T_2^*').replace('A-', '-').replace('^^', '∈').replace('%', '≤').replace('?"', '—')

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(text)
