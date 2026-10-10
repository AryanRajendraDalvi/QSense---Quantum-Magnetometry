"""Clean Unicode in markdown files."""
import re
for fname in ['README.md', r'C:\Users\ARYAN\.gemini\antigravity\brain\653fec1f-73bc-41b6-8f8f-59de643f75ff\deck.md']:
    with open(fname, 'r', encoding='utf-8') as f:
        text = f.read()
    
    text = text.replace('In', 'In').replace('I', 'I').replace('% ', '% ').replace('', '')
    text = text.replace('(~3-4% lower RMSE)', '(about 3% [CI 2-4%] lower RMSE)')
    text = text.replace('about 14% over B1', 'about 13% [95% CI 8-18%] over B1')
    
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(text)
