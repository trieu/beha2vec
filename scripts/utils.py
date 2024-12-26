import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging

logger = logging.getLogger(__name__)

# Simple in-memory cache to store (page_theme, page_type) for each URL
classification_cache = {}

def classification_cache_get(url: str):
    return classification_cache.get(url)

def classification_cache_set(url: str, result):
    classification_cache[url] = result

def scrape_page(url: str) -> str:
    """
    Scrapes page content (with retries) and returns partial text.
    """
    s = requests.Session()
    r = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    s.mount("http://", HTTPAdapter(max_retries=r))
    s.mount("https://", HTTPAdapter(max_retries=r))

    try:
        resp = s.get(url, timeout=5)
        resp.raise_for_status()
        return resp.text[:2000]
    except Exception as e:
        logger.warning(f"Scraping fail for {url}: {e}")
        return "No content"
