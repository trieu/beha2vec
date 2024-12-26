import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import logging
import random
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from scripts.user_behavior_transformer import UserBehaviorTransformer

logger = logging.getLogger(__name__)

class TripletDataset(Dataset):
    """
    Builds (anchor, positive, negative) triplets from user sequences.
    Each user_id's data is split in half: anchor vs. positive.
    A negative comes from a different user.
    """

    def __init__(
        self,
        user_sequences_csv: str,
        url_vocab_size: int = 1000,
        theme_vocab_size: int = 50,
        use_theme: bool = False,
        use_type: bool = False,
        max_seq_len: int = 512
    ):
        # Read CSV with columns: user_id, timestamp, pageview_URL (+ optionally page_theme, page_type)
        self.df = pd.read_csv(user_sequences_csv)
        self.url_vocab_size = url_vocab_size
        self.theme_vocab_size = theme_vocab_size
        self.use_theme = use_theme
        self.use_type = use_type
        self.max_seq_len = max_seq_len
        
        # user_dict: user_id -> list of rows
        self.user_dict = {}
        for _, row in self.df.iterrows():
            uid = row['user_id']
            if uid not in self.user_dict:
                self.user_dict[uid] = []
            self.user_dict[uid].append(row)

        # Sort each user's rows by timestamp
        for uid in self.user_dict:
            self.user_dict[uid].sort(key=lambda x: x['timestamp'])

        # All distinct users
        self.all_users = list(self.user_dict.keys())

    def __len__(self):
        return len(self.all_users)

    def __getitem__(self, idx):
        """
        Returns: (anchor_input, positive_input, negative_input)
        Each input is (url_seq, theme_seq, type_vec).
        """
        uid = self.all_users[idx]
        data = self.user_dict[uid]

        # Split data -> anchor vs. positive
        if len(data) < 2:
            anchor_rows = data
            positive_rows = data
        else:
            mid = len(data) // 2
            anchor_rows = data[:mid]
            positive_rows = data[mid:]

        # Negative from a different user
        neg_uid = random.choice(self.all_users)
        while neg_uid == uid and len(self.all_users) > 1:
            neg_uid = random.choice(self.all_users)
        negative_rows = self.user_dict[neg_uid]

        anchor_input   = self._build_input(anchor_rows)
        positive_input = self._build_input(positive_rows)
        negative_input = self._build_input(negative_rows)

        return (anchor_input, positive_input, negative_input)

    def _build_input(self, rows):
        """
        Returns (url_seq, theme_seq, type_vec) for each set of rows.
        - url_seq: shape [seq_len], never None
        - theme_seq: shape [seq_len], never None (could be empty)
        - type_vec: shape [3], never None
        """
        if len(rows) == 0:
            # Return placeholders if user has no data
            return (
                torch.tensor([], dtype=torch.long),          # url_seq
                torch.tensor([], dtype=torch.long),          # theme_seq
                torch.zeros(3, dtype=torch.float)            # type_vec
            )

        # URL sequence
        url_ids = [hash(r['pageview_URL']) % self.url_vocab_size for r in rows]
        url_ids = url_ids[: self.max_seq_len]
        url_seq = torch.tensor(url_ids, dtype=torch.long)

        # Theme sequence
        theme_seq = torch.tensor([], dtype=torch.long)
        if self.use_theme and 'page_theme' in rows[0]:
            t = [hash(r['page_theme']) % self.theme_vocab_size for r in rows]
            t = t[: self.max_seq_len]
            theme_seq = torch.tensor(t, dtype=torch.long)

        # Type vector
        type_vec = torch.zeros(3, dtype=torch.float)
        if self.use_type and 'page_type' in rows[0]:
            tv = [0, 0, 0]
            for rr in rows:
                ptype = rr['page_type'].lower()
                if "blog" in ptype:
                    tv[0] += 1
                elif "service" in ptype:
                    tv[1] += 1
                elif "home" in ptype:
                    tv[2] += 1
            type_vec = torch.tensor(tv, dtype=torch.float)

        return (url_seq, theme_seq, type_vec)

def triplet_collate_fn(batch):
    """
    Collates variable-length sequences from TripletDataset.
    Each batch item => ((urlA, themeA, typeA),(urlB, themeB, typeB),(urlC, themeC, typeC))
    We'll pad url_seq & theme_seq, and stack type_vec.
    """
    anchors, positives, negatives = zip(*batch)  # Unpack into 3 lists

    # Debug print (uncomment to investigate shapes):
    # print(f"[DEBUG] Batch size: {len(batch)} anchors: {len(anchors)}")

    def pad_triplet(triplets):
        url_seqs   = [x[0] for x in triplets]
        theme_seqs = [x[1] for x in triplets]
        type_vecs  = [x[2] for x in triplets]

        # Pad URLs
        padded_urls = pad_sequence(url_seqs, batch_first=True, padding_value=0)
        # Pad themes
        padded_themes = pad_sequence(theme_seqs, batch_first=True, padding_value=0)
        # Stack type
        stacked_type = torch.stack(type_vecs, dim=0)

        return (padded_urls, padded_themes, stacked_type)

    anchor_out   = pad_triplet(anchors)
    positive_out = pad_triplet(positives)
    negative_out = pad_triplet(negatives)

    return anchor_out, positive_out, negative_out

class TransformerModelTrainer:
    """
    Trainer for the Transformer with Triplet Margin Loss.
    """
    def __init__(
        self,
        url_vocab_size,
        theme_vocab_size=None,
        type_dim=None,
        embedding_dim=128,
        n_heads=4,
        n_layers=2,
        combined_dim=128,
        learning_rate=1e-4,
        max_seq_len=512
    ):
        self.model = UserBehaviorTransformer(
            url_vocab_size, theme_vocab_size, type_dim,
            embedding_dim, n_heads, n_layers, combined_dim, max_seq_len
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.TripletMarginLoss(margin=1.0)

    def train(self, dataloader, epochs, output_file="model.pth"):
        """
        For each epoch, iter over dataloader => anchor, positive, negative,
        compute triplet loss.
        """
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for anchor, positive, negative in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                # anchor => (url_seq, theme_seq, type_vec)
                #   url_seq => [B, seq_len], theme_seq => [B, seq_len], type_vec => [B,3]
                # p = (url_seq, theme_seq, type_vec)
                a = self.model(*anchor)  # => self.model(anchor[0], anchor[1], anchor[2])
                p = self.model(*positive)
                n = self.model(*negative)

                loss = self.criterion(a, p, n)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        torch.save(self.model.state_dict(), output_file)
        logger.info(f"Model saved -> {output_file}")

    def load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval()
        logger.info(f"Loaded model from {file_path}")
