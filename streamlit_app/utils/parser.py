# streamlit_app/utils/parser.py
import requests, re
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SEO-Analyzer/1.0)"}

def fetch_html(url, timeout=10):
    """Fetch raw HTML text for a URL."""
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text

def extract_main_text(html):
    """Naive main-content extractor: prefer <article>, else join <p>."""
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article")
    if article and article.get_text(strip=True):
        text = article.get_text(separator=" ", strip=True)
    else:
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
        if paragraphs:
            text = "\n\n".join(paragraphs)
        else:
            text = soup.get_text(separator=" ", strip=True)
    # cleanup whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
