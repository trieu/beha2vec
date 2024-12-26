import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EmbeddingCombiner:
    """
    Combines multiple embeddings (url, theme, type) with given weights.
    """
    def __init__(self, weights=None):
        self.weights = weights if weights else {'url':0.5,'theme':0.3,'type':0.2}

    def combine_embeddings(self, embeddings, output_file="user_combined_embeddings.csv"):
        user_ids = set()
        for k in embeddings:
            user_ids.update(embeddings[k].keys())
        combined={}
        for uid in user_ids:
            vec=None
            for k in ['url','theme','type']:
                if k in embeddings and uid in embeddings[k]:
                    w = self.weights.get(k, 0.0)
                    if vec is None: vec = w*embeddings[k][uid]
                    else: vec += w*embeddings[k][uid]
            if vec is None: vec = np.zeros(128,dtype=np.float32)
            combined[uid] = vec
        dfc = pd.DataFrame.from_dict(combined,orient='index')
        dfc.index.name='user_id'
        dfc.to_csv(output_file, header=False)
        logger.info(f"Combined -> {output_file}")
        return combined
