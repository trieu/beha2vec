import pandas as pd
import logging
from transformers import pipeline
from tqdm import tqdm
from scripts.utils import scrape_page, classification_cache_get, classification_cache_set

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles GA4 data preprocessing:
      1. Filters rows to 'page_view'.
      2. Renames columns (user_id, timestamp, pageview_URL).
      3. Optionally classifies page_location -> (page_theme, page_type).
      4. Sorts by timestamp for user sequences.
    """

    def __init__(self, theme_labels=None, type_labels=None):
        self.theme_labels = theme_labels
        self.type_labels = type_labels
        self.data = None

        if self.theme_labels and self.type_labels:
            self.theme_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            self.type_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        else:
            self.theme_classifier = None
            self.type_classifier = None

    def load_csv(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Loading GA4 CSV from {file_path}")
    
        # Specify dtype for specific columns to avoid DtypeWarning
        dtypes = {
            "event_name": str,
            "user_pseudo_id": str,
            "event_timestamp": int,
            "event_date": str,
            "page_location": str
        }
    
        # Read CSV with low_memory=False and dtype specification
        df = pd.read_csv(file_path, dtype=dtypes)
    
        # Filter to only include rows where event_name is 'page_view'
        df = df[df["event_name"] == "page_view"].copy()
    
        # Keep only relevant columns
        df = df[["user_pseudo_id", "event_timestamp", "event_date", "page_location"]]
    
        # Rename columns for consistency
        df.rename(columns={
            "user_pseudo_id": "user_id",
            "event_timestamp": "timestamp",
            "page_location": "pageview_URL"
        }, inplace=True)
    
        self.data = df
        return self.data


    def extract_page_info(self, output_file: str) -> str:
        if not (self.theme_classifier and self.type_classifier):
            raise ValueError("Cannot extract page info without theme and type labels.")

        urls = self.data["pageview_URL"].unique().tolist()
        logger.info(f"{len(urls)} unique URLs to classify.")

        url_map = {}
        # Use tqdm to show progress for classifying multiple URLs
        for url in tqdm(urls, desc="Classifying URLs"):
            cached = classification_cache_get(url)
            if cached:
                url_map[url] = cached
                continue
            try:
                content = scrape_page(url)
                t = self.theme_classifier(content, self.theme_labels)['labels'][0]
                p = self.type_classifier(content, self.type_labels)['labels'][0]
                url_map[url] = (t, p)
                classification_cache_set(url, (t, p))
            except Exception as e:
                logger.warning(f"Classification failed for {url}: {e}")
                url_map[url] = ("UnknownTheme", "UnknownType")

        self.data["page_theme"] = self.data["pageview_URL"].apply(lambda u: url_map[u][0])
        self.data["page_type"]  = self.data["pageview_URL"].apply(lambda u: url_map[u][1])

        self.data.to_csv(output_file, index=False)
        logger.info(f"Enriched data -> {output_file}")
        return output_file

    def generate_user_sequences(self, use_page_theme=False, use_page_type=False, output_file: str="user_sequences.csv") -> str:
        if self.data is None:
            raise ValueError("No GA4 data loaded.")
        self.data["timestamp"] = pd.to_numeric(self.data["timestamp"], errors="coerce")

        cols = ["user_id", "timestamp", "pageview_URL"]
        if use_page_theme:
            cols.append("page_theme")
        if use_page_type:
            cols.append("page_type")

        seqs = (
            self.data[cols]
            .groupby("user_id", as_index=False)
            .apply(lambda x: x.sort_values("timestamp"))
            .reset_index(drop=True)
        )
        seqs.to_csv(output_file, index=False)
        logger.info(f"User sequences -> {output_file}")
        return output_file
