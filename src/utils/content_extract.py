import time
from typing import Optional, Dict

import tldextract
import requests
from requests.adapters import HTTPAdapter, Retry

from newspaper import Article as NPArticle

import trafilatura
from readability import Document



def _session(timeout: float, retries: int, max_redirects: int) -> requests.Session:
    s = requests.Session()
    s.max_redirects = max_redirects
    retry = Retry(total=retries, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    })
    s.request = (lambda f: (lambda *a, **kw: f(*a, timeout=timeout, **kw)))(s.request)  # type: ignore
    return s


def _domain(url: str) -> str:
    ext = tldextract.extract(url)
    return ".".join([p for p in [ext.domain, ext.suffix] if p])


def _quality(text: str) -> Dict[str, float]:
    text = (text or "").strip()
    n = len(text)
    alpha = sum(c.isalpha() for c in text)
    num_links = text.count("http") + text.count("www.")
    boiler = 0.0 if n == 0 else max(0.0, 1.0 - (alpha / max(1, n)))
    return {"extracted_length": n, "boilerplate_ratio": boiler, "num_links": float(num_links)}


def fetch_and_extract(url: str, *, timeout: float = 6.0, retries: int = 1, max_redirects: int = 3) -> Dict:
    start = time.time()
    sess = _session(timeout, retries, max_redirects)
    status_code: Optional[int] = None
    html = None
    error = None
    try:
        resp = sess.get(url, allow_redirects=True)
        status_code = resp.status_code
        if 200 <= status_code < 300:
            html = resp.text
    except Exception as e:
        error = str(e)

    extracted = {
        "url": url,
        "status_code": status_code,
        "extracted_text": "",
        "extracted_title": "",
        "language": None,
        "published_time": None,
        "domain": _domain(url),
        "duration_ms": int((time.time() - start) * 1000),
        "extractor_used": None,
        "error": error,
    }

    # Early exit if no HTML
    if not html:
        return {**extracted, **_quality("")}

    # 1) newspaper3k
    if NPArticle is not None:
        try:
            art = NPArticle(url)
            art.set_html(html)
            art.parse()
            text = (art.text or "").strip()
            title = (art.title or "").strip()
            q = _quality(text)
            if q["extracted_length"] >= 200 and q["boilerplate_ratio"] <= 0.6:
                return {**extracted, **q, "extracted_text": text, "extracted_title": title, "extractor_used": "newspaper3k"}
        except Exception:
            pass

    # 2) trafilatura
    if trafilatura is not None:
        try:
            text = trafilatura.extract(html, include_formatting=False, favor_recall=True) or ""
            title = trafilatura.extract_title(html) or ""
            q = _quality(text)
            if q["extracted_length"] >= 200 and q["boilerplate_ratio"] <= 0.6:
                return {**extracted, **q, "extracted_text": text, "extracted_title": title, "extractor_used": "trafilatura"}
        except Exception:
            pass

    # 3) readability-lxml
    if Document is not None:
        try:
            doc = Document(html)
            title = (doc.short_title() or "").strip()
            text = (doc.summary() or "").strip()
            # crude HTML tag removal
            import re
            text = re.sub(r"<[^>]+>", " ", text)
            q = _quality(text)
            if q["extracted_length"] >= 200 and q["boilerplate_ratio"] <= 0.7:
                return {**extracted, **q, "extracted_text": text, "extracted_title": title, "extractor_used": "readability"}
        except Exception:
            pass

    # fallback: return weakest result
    return {**extracted, **_quality("")}


