SEO Content Detector

 1) Project Overview
This project automatically evaluates SEO content quality by extracting article text, measuring readability, word count, and lexical diversity, and predicting a “quality label” (High / Medium / Low). It also detects duplicate / near-duplicate pages using TF-IDF similarity. The goal is to create a lightweight, reproducible SEO scoring pipeline that can evaluate any URL in real-time.

2) Setup Instructions

git clone https://github.com/Karyamsetty-Prajwala/seo-content-detector
cd seo-content-detector
pip install -r requirements.txt
jupyter notebook notebooks/seo_pipeline.ipynb

3) Quick Start (local dataset run)
   ensure dataset is available at: data/data.csv
    open notebook: notebooks/seo_pipeline.ipynb
    run all cells

    run → run_interactive() in last cell
    → enter any URL
    → result prints JSON including similarity matches
   
4)  Deplloyed Streamlit URL: 
5) Key Decisions
TF-IDF used for similarity — lightweight + interpretable

BeautifulSoup used for HTML text extraction — avoids JS rendering complexity

Similarity threshold = 0.35 — based on histogram distribution of cosine distance in labeled data

Logistic Regression chosen because it achieved the best macro-F1 and lowest variance vs RF/XGBoost in cross-validation

6) Results Summary
Model	Accuracy	Macro-F1
Logistic Regression	0.62	0.58
RandomForest	0.47	0.44
Word Count Rule Baseline	0.33	0.25

typical duplicates detected per URL: ~1-3 if >= threshold

high quality examples: long NLP tutorials (~4k words, FRE ≈ 57)

thin content often < 300 words → labeled as Low

7) Limitations
TF-IDF cannot detect semantic duplicates if wording differs

no headless browser → JS-rendered content may not extract fully

model is small sample trained — class boundaries may vary by niche

