# streamlit_app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# local utils
from utils.parser import fetch_html, extract_main_text
from utils.features import extract_features_from_text
from utils.scorer import QualityScorer

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

st.set_page_config(page_title="SEO Content Detector", layout="centered")
st.title("SEO Content Quality & Duplicate Detector")

# Sidebar controls
st.sidebar.header("Corpus & Settings")
use_default = st.sidebar.checkbox("Use provided corpus (data/extracted_content.csv)", value=True)
uploaded = st.sidebar.file_uploader("Or upload corpus CSV (.csv with 'url' and 'text' columns)", type=["csv"])
similarity_threshold = st.sidebar.slider("Similarity threshold", 0.0, 1.0, 0.35, 0.05)
top_k = st.sidebar.slider("Top K matches", 1, 10, 5)

# Load or prepare corpus
corpus_df = None
if use_default:
    for cand in ["data/extracted_content.csv", "data/data.csv"]:
        p = ROOT_DIR / cand
        if p.exists():
            try:
                corpus_df = pd.read_csv(p)
                st.sidebar.write(f"Loaded corpus: {cand} ({len(corpus_df)} rows)")
                break
            except Exception as e:
                st.sidebar.warning(f"Failed to load {cand}: {e}")
if uploaded is not None:
    try:
        uploaded_bytes = uploaded.read()
        tmp = pd.read_csv(io.BytesIO(uploaded_bytes))
        corpus_df = tmp
        st.sidebar.write(f"Uploaded corpus ({len(corpus_df)} rows)")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded file: {e}")


# --- Robust corpus normalization + diagnostics (paste into streamlit_app/app.py) ---
def normalize_corpus_text(df):
    """
    Return (df_with_text_col, used_column_name_or_None)
    Attempts case-insensitive exact matches, substring matches, then title+body fallback.
    """
    cols = list(df.columns)
    lower_to_orig = {c.lower().strip(): c for c in cols}

    # exact candidates in order of preference
    candidates = ['text','clean_bodytext','bodytext','clean_body_text','body','content','article','extracted_content','body_text','cleantext']
    # exact/case-insensitive match
    for cand in candidates:
        if cand in lower_to_orig:
            orig = lower_to_orig[cand]
            df['text'] = df[orig].astype(str)
            return df, orig

    # substring match (e.g., 'body', 'content', 'article' anywhere)
    for lower_name, orig in lower_to_orig.items():
        if any(tok in lower_name for tok in ('body','content','article','clean','text')):
            df['text'] = df[orig].astype(str)
            return df, orig

    # fallback: try title + some body-like column
    title_col = None
    body_col = None
    for lower_name, orig in lower_to_orig.items():
        if 'title' == lower_name:
            title_col = orig
        if any(tok in lower_name for tok in ('body','content','article')):
            body_col = orig
    if body_col:
        if title_col:
            df['text'] = (df[title_col].fillna('') + ". " + df[body_col].fillna('')).astype(str)
            return df, f"{title_col}+{body_col}"
        else:
            df['text'] = df[body_col].astype(str)
            return df, body_col

    # nothing usable found
    return None, None

# Use it (replace your earlier normalization)
if isinstance(corpus_df, pd.DataFrame):
    # show raw columns first (diagnostic)
    st.sidebar.markdown("**Corpus columns:**")
    try:
        st.sidebar.write(list(corpus_df.columns))
    except Exception:
        st.sidebar.write("Unable to list columns")

    # normalize
    corpus_df_normalized, used_col = normalize_corpus_text(corpus_df)
    if corpus_df_normalized is None:
        st.sidebar.error("Corpus has no usable text column — similarity disabled.")
        st.sidebar.info("If your file uses a non-standard column name, re-upload a CSV with a text column, or rename a column to 'text'.")
        corpus_df = None
    else:
        corpus_df = corpus_df_normalized
        st.sidebar.success(f"Using corpus text column: **{used_col}**")
        # show sample value for verification
        try:
            sample_len = corpus_df['text'].str.strip().astype(bool).sum()
            st.sidebar.write(f"Non-empty text rows: {sample_len} of {len(corpus_df)}")
            st.sidebar.write("Sample (first row, truncated):")
            st.sidebar.write(corpus_df['text'].astype(str).iloc[0][:300] if len(corpus_df)>0 else "")
        except Exception:
            pass
else:
    st.sidebar.info("No corpus DataFrame loaded yet.")

# --- After you've detected/assigned corpus_df['text'] = ...  add this block ---

# If the stored 'text' looks like HTML (starts with <!doctype or contains <html>), extract readable text
def _looks_like_html(s):
    if not isinstance(s, str) or len(s) < 20:
        return False
    s_low = s.strip().lower()
    return s_low.startswith("<!doctype") or s_low.startswith("<html") or "<html" in s_low[:200] or "<!doctype" in s_low[:200]

# only attempt extraction if there is any HTML-like row
try:
    sample_val = corpus_df['text'].astype(str).iloc[0] if len(corpus_df) > 0 else ""
except Exception:
    sample_val = ""

if _looks_like_html(sample_val):
    st.sidebar.info("Detected HTML in corpus text column — extracting body text now (this may take a moment).")
    # run extraction for all rows (consider caching or pre-saving if corpus large)
    cleaned = []
    for i, raw in enumerate(corpus_df['text'].astype(str)):
        try:
            # use the extract_main_text function you already have
            body = extract_main_text(raw)
            # fallback: if extraction empty, keep original shortened text
            if not body or len(body.strip()) < 50:
                cleaned.append(raw)   # keep original to avoid data loss
            else:
                cleaned.append(body)
        except Exception:
            cleaned.append(raw)
    corpus_df['text'] = cleaned
    st.sidebar.success("Extracted text for corpus (first row preview updated).")
    # (optional) show a small preview
    try:
        st.sidebar.write("Sample (first row, truncated):")
        st.sidebar.write(corpus_df['text'].iloc[0][:300])
    except Exception:
        pass
else:
    # not HTML; show the existing preview (as before)
    try:
        st.sidebar.write("Sample (first row, truncated):")
        st.sidebar.write(corpus_df['text'].astype(str).iloc[0][:300] if len(corpus_df)>0 else "")
    except Exception:
        pass
# --- end extraction block ---


# Initialize scorer (loads quality_model.pkl if present; uses rule fallback otherwise)
scorer = QualityScorer(model_path=APP_DIR / "models" / "quality_model.pkl")

# UI: URL input
st.subheader("Analyze a URL")
url = st.text_input("Enter article URL", "")
analyze = st.button("Analyze")

def build_tfidf(corpus_texts):
    vect = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')
    X = vect.fit_transform(corpus_texts)
    return vect, X

if analyze:
    if not url.strip():
        st.error("Please enter a URL.")
    else:
        with st.spinner("Fetching and analyzing..."):
            try:
                html = fetch_html(url)
                text = extract_main_text(html)
                features = extract_features_from_text(text)
                label, probs = scorer.predict_label(features)  # label str, probs dict
            except Exception as e:
                st.error(f"Failed to analyze URL: {e}")
                st.stop()

        # Show results
        st.markdown("### Quality Summary")
        st.write({
            "URL": url,
            "Title": features.get("title", ""),
            "Word count": features["wordcount"],
            "Readability (Flesch)": features["flesch_reading_ease"],
            "Avg sentence len": round(features["avg_sentence_length"], 2),
            "Lexical diversity": round(features["lexical_diversity"], 3),
            "Quality label": label
        })

        # Show text sample
        st.markdown("### Extracted Text (first 800 chars)")
        st.text(text[:800] + ("..." if len(text) > 800 else ""))

        # Similarity checks (if corpus available)
        if isinstance(corpus_df, pd.DataFrame) and len(corpus_df) > 0:
            st.markdown("### Similarity / Duplicate Check")
            corpus_texts = corpus_df['text'].fillna("").astype(str).tolist()
            corpus_urls = corpus_df['url'].fillna("").astype(str).tolist() if 'url' in corpus_df.columns else [f"row_{i}" for i in range(len(corpus_texts))]

            # build TF-IDF
            try:
                vect, X_corpus = build_tfidf(corpus_texts)
                X_q = vect.transform([text])
                if X_q.nnz == 0:
                    st.info("Query produced empty TF-IDF vector (too short or vocabulary mismatch). Try lower threshold or use embeddings.")
                else:
                    sims = cosine_similarity(X_q, X_corpus).flatten()
                    idx = np.argsort(-sims)[:top_k]
                    matches = [{"url": corpus_urls[i], "similarity": float(round(float(sims[i]),4)), "index": int(i)} for i in idx]
                    st.table(pd.DataFrame(matches))
                    above = [m for m in matches if m["similarity"] >= similarity_threshold]
                    if above:
                        st.success(f"{len(above)} matches above threshold ({similarity_threshold})")
                        st.table(pd.DataFrame(above))
                    else:
                        st.info(f"No matches exceeded the threshold {similarity_threshold}. Showing top {len(matches)} candidates.")
            except Exception as e:
                st.error(f"Similarity check failed: {e}")
        else:
            st.info("No corpus loaded for similarity checks (enable sidebar option or upload).")
# ---- robust corpus text normalization (paste into app.py) ----
if isinstance(corpus_df, pd.DataFrame):
    # List of candidate columns in order of preference
    candidates = ['text', 'clean_bodytext', 'bodytext', 'body', 'content', 'extracted_content', 'article']
    found = False
    for col in candidates:
        if col in corpus_df.columns:
            if col != 'text':
                corpus_df['text'] = corpus_df[col].astype(str)
            else:
                corpus_df['text'] = corpus_df['text'].astype(str)
            found = True
            break
    if not found:
        st.sidebar.warning("Corpus has no recognizable text column — similarity disabled.")
        corpus_df = None

st.markdown("---")
st.markdown("Notes: uses TF-IDF similarity and simple readability & length rules. For semantic similarity use precomputed embeddings.")
