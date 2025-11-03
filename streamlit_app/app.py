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

# Normalize corpus text column name if present
if isinstance(corpus_df, pd.DataFrame):
    # normalise column to text
    text_candidates = ['text','clean_bodytext','bodytext','body','content','article','parsed_text']
    found = False
    for col in text_candidates:
        if col in corpus_df.columns:
            corpus_df['text'] = corpus_df[col].astype(str)
            found = True
            break

    if not found:
        st.warning("Corpus has no usable text column — similarity disabled.")
        corpus_df = None



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
