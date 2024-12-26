import torch
import pandas as pd
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generates an embedding per user from the trained model.
    """
    def __init__(self, model, url_vocab_size=1000, theme_vocab_size=50, max_seq_len=512, use_page_theme=False, use_page_type=False):
        self.model = model
        self.model.eval()
        self.url_vocab_size = url_vocab_size
        self.theme_vocab_size = theme_vocab_size
        self.max_seq_len = max_seq_len
        self.use_page_theme = use_page_theme
        self.use_page_type = use_page_type

    def generate_embeddings(self, user_sequences_csv, output_prefix="user"):
        df = pd.read_csv(user_sequences_csv)
        user_dict = {}
        for _, row in df.iterrows():
            uid = row['user_id']
            if uid not in user_dict:
                user_dict[uid] = []
            user_dict[uid].append(row)

        results = {}
        # Wrap user iteration with tqdm for progress
        for uid in tqdm(user_dict.keys(), desc="Generating embeddings for users"):
            rows = user_dict[uid]
            url_ids = [hash(r['pageview_URL']) % self.url_vocab_size for r in rows][:self.max_seq_len]
            url_seq = torch.tensor([url_ids], dtype=torch.long)

            theme_seq, type_vec = None, None
            if self.use_page_theme and 'page_theme' in rows[0]:
                th = [hash(r['page_theme']) % self.theme_vocab_size for r in rows][:self.max_seq_len]
                theme_seq = torch.tensor([th], dtype=torch.long)

            if self.use_page_type and 'page_type' in rows[0]:
                tv = [0,0,0]
                for rr in rows:
                    pg = rr['page_type'].lower()
                    if "blog" in pg: tv[0]+=1
                    elif "service" in pg: tv[1]+=1
                    elif "home" in pg: tv[2]+=1
                type_vec = torch.tensor([tv], dtype=torch.float)

            with torch.no_grad():
                emb = self.model(url_seq, theme_seq, type_vec).squeeze(0).numpy()
            results[uid] = emb

        out_file = f"{output_prefix}_url_embeddings.csv"
        df_out = pd.DataFrame.from_dict(results, orient='index')
        df_out.index.name = 'user_id'
        df_out.to_csv(out_file, header=False)
        logger.info(f"Embeddings -> {out_file}")
        return {"url": results}
