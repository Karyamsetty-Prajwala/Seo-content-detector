# streamlit_app/utils/features.py
import re
from textstat import flesch_reading_ease

def _word_count(text):
    return len(text.split()) if text else 0

def _sentence_count(text):
    return max(1, len(re.findall(r'[.!?]+', text))) if text else 0

def _lexical_diversity(text):
    toks = text.split()
    return len(set(toks)) / len(toks) if toks else 0.0

def extract_features_from_text(text, title=None):
    wc = _word_count(text)
    sc = _sentence_count(text)
    try:
        readability = float(flesch_reading_ease(text)) if text else 0.0
    except Exception:
        readability = 0.0
    avg_sent_len = wc / sc if sc else 0.0
    lex_div = _lexical_diversity(text)

    return {
        "text": text,
        "title": title or "",
        "wordcount": int(wc),
        "sentence_count": int(sc),
        "avg_sentence_length": float(avg_sent_len),
        "flesch_reading_ease": float(readability),
        "lexical_diversity": float(lex_div)
    }
