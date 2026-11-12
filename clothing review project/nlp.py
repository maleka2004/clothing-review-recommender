"""Milestone I title + processed review count representation, shared at fit/predict."""
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parent
STOPWORDS = set((ROOT / 'data/stopwords_en.txt').read_text().split())
VOCAB = {line.rsplit(':', 1)[0] for line in (ROOT / 'data/vocab.txt').read_text().splitlines()}
PATTERN = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"

def title_tokens(text):
    return [w for w in re.findall(PATTERN, text.lower()) if len(w) > 1 and w not in STOPWORDS]

def review_tokens(text):
    # Fixed Task 1 vocabulary carries the original corpus frequency exclusions.
    return [w for w in re.findall(PATTERN, text.lower()) if w in VOCAB]

def features(title, review):
    return ' '.join(title_tokens(title) + review_tokens(review))
