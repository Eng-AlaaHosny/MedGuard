# this file normalizes names so matching stays consistent across sources
from __future__ import annotations
SALT_WORDS = {'hydrochloride', 'hcl', 'sodium', 'potassium', 'calcium', 'magnesium', 'acetate', 'phosphate', 'sulfate', 'sulphate', 'nitrate', 'chloride', 'bromide', 'iodide', 'tartrate', 'citrate', 'succinate', 'fumarate', 'gluconate', 'mesylate', 'besylate', 'tosylate', 'maleate'}

# this function is used to normalize kb text
def normalize_kb_text(name: str) -> str:
    if not name:
        return ''
    s = name.strip().lower()
    if '(' in s and ')' in s:
        import re
        s = re.sub('\\([^)]*\\)', ' ', s)
    for ch in [',', '.', ';', ':', "'", '"', '/', '\\', '+', '-', '_']:
        s = s.replace(ch, ' ')
    s = ' '.join(s.split())
    parts = [p for p in s.split() if p not in SALT_WORDS]
    s2 = ' '.join(parts).strip()
    return s2 if s2 else s
