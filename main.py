import logging
import pandas
import sys
import os
import json
from torch.utils.data import DataLoader
from dotenv import load_dotenv


# Import your script modules
from scripts.data_preprocessing import DataPreprocessor
from scripts.model_training import (
    TripletDataset, 
    TransformerModelTrainer,
    triplet_collate_fn
)
from scripts.embedding_generation import EmbeddingGenerator
from scripts.embedding_combination import EmbeddingCombiner
from scripts.clustering_analysis import ClusterAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_theme_labels_from_file(file_path: str) -> list:
    """
    Reads lines from a text file, stripping whitespace,
    and returns them as a list of strings.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def write_preprocessing_config(use_theme: bool, use_type: bool, config_file: str):
    """
    Stores the user choices in a JSON file for subsequent stages.
    """
    data = {
        "use_theme": use_theme,
        "use_type": use_type
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

def read_preprocessing_config(config_file: str):
    """
    Reads the user choices (use_theme, use_type) from a JSON file.
    If the file doesn't exist or is invalid, returns (False, False) by default.
    """
    if not os.path.exists(config_file):
        return (False, False)
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (data.get("use_theme", False), data.get("use_type", False))
    except Exception:
        return (False, False)

def main():
    # If you have OPENAI_API_KEY in a .env file, uncomment:
    # load_dotenv("config.env")

    # 1. Load theme labels from /data/input/theme_labels.txt
    theme_labels_file = "data/input/theme_labels.txt"
    theme_labels = load_theme_labels_from_file(theme_labels_file)

    # 2. Type labels
    type_labels = ["Blog", "Service Page", "Homepage"]

    # 3. Ask user for the stage (1..5)
    stage_input = input("Stage (1: preprocess, 2: train, 3: embed, 4: combine, 5: cluster, 6: cluster analysis): ")
    try:
        stage = int(stage_input)
    except ValueError:
        stage = 1


    stage = max(1, min(stage, 6))

    # We'll store user choices in data/output/preprocessing_config.json
    config_file = "data/output/preprocessing_config.json"

    # If stage=1, ask about theme/type, otherwise read from config
    if stage == 1:
        use_theme = (input("Use page_theme? (y/n): ") == 'y')
        use_type = (input("Use page_type? (y/n): ") == 'y')
        write_preprocessing_config(use_theme, use_type, config_file)
    else:
        use_theme, use_type = read_preprocessing_config(config_file)

    # Now run from 'stage' up to 6 in sequence
    if stage <= 1:
        dp = DataPreprocessor(
            theme_labels if use_theme else None, 
            type_labels if use_type else None
        )
        dp.load_csv("data/input/ga4_data.csv")
        if use_theme or use_type:
            dp.extract_page_info("data/output/enriched_data.csv")
        dp.generate_user_sequences(use_theme, use_type, "data/output/user_sequences.csv")

    if stage <= 2:
        # Use the custom collate function for variable-length sequences
        ds = TripletDataset(
            user_sequences_csv="data/output/user_sequences.csv",
            url_vocab_size=1000,
            theme_vocab_size=50,
            use_theme=use_theme,
            use_type=use_type,
            max_seq_len=512
        )
        dl = DataLoader(
            ds,
            batch_size=2,
            shuffle=True,
            collate_fn=triplet_collate_fn  # <-- IMPORTANT
        )
        trainer = TransformerModelTrainer(
            url_vocab_size=1000,
            theme_vocab_size=50 if use_theme else None,
            type_dim=3 if use_type else None,
            embedding_dim=128,
            n_heads=4,
            n_layers=2,
            combined_dim=128,
            learning_rate=1e-4,
            max_seq_len=512
        )
        trainer.train(dl, 5, "data/output/model.pth")

    if stage <= 3:
        trainer = TransformerModelTrainer(
            url_vocab_size=1000,
            theme_vocab_size=50 if use_theme else None,
            type_dim=3 if use_type else None,
            embedding_dim=128,
            n_heads=4,
            n_layers=2,
            combined_dim=128,
            learning_rate=1e-4,
            max_seq_len=512
        )
        trainer.load_model("data/output/model.pth")
        gen = EmbeddingGenerator(
            model=trainer.model,
            url_vocab_size=1000,
            theme_vocab_size=50,
            max_seq_len=512,
            use_page_theme=use_theme,
            use_page_type=use_type
        )
        gen.generate_embeddings("data/output/user_sequences.csv", "data/output/user")

    if stage <= 4:
        import pandas as pd
        dfu = pd.read_csv("data/output/user_url_embeddings.csv", header=None)
        url_emb = {}
        for i, row in dfu.iterrows():
            uid = str(row[0])
            v = row[1:].values
            url_emb[uid] = v
        comb = EmbeddingCombiner({'url': 0.5, 'theme': 0.3, 'type': 0.2})
        comb.combine_embeddings({'url': url_emb}, "data/output/user_combined_embeddings.csv")

    if stage <= 5:
        n = int(input("Number of clusters: "))
        ana = ClusterAnalyzer(n)
        ana.fit_and_predict("data/output/user_combined_embeddings.csv", "data/output/clustered_users.csv")
        ana.generate_cluster_samples(
            clustered_data_file="data/output/clustered_users.csv",
            raw_data_file="data/input/ga4_data.csv",
            sample_output_file="data/output/cluster_samples.csv",
            n_samples=5
        )

    if stage == 6:
        cluster_file = "data/output/clustered_users.csv"
        sample_file = "data/output/cluster_samples.csv"
        quantitative_file = "data/output/quantitative_analysis.csv"
        overall_summary_file = "data/output/overall_analysis.txt"
        cluster_description_file = "data/output/cluster_descriptions.txt"

        # Determine the number of clusters if not provided
        if not 'n' in locals() or n is None:
            clustered_df = pandas.read_csv(cluster_file)
            n = clustered_df['cluster'].nunique()
            logger.info(f"Determined number of clusters from file: {n}")

        # Instantiate ClusterAnalyzer
        ana = ClusterAnalyzer(n)

        # Step 1: Generate descriptions for individual clusters
        logger.info("Generating descriptions for individual clusters...")
        descriptions = ana.describe_clusters(sampled_data_file=sample_file, gpt_output_file=cluster_description_file)
        for cluster_id, description in descriptions.items():
            print(f"\nCluster {cluster_id}:\n{description}\n")

        # Step 2: Perform quantitative analysis for each cluster
        logger.info("Performing quantitative analysis for each cluster...")
        ana.perform_quantitative_analysis(
            clustered_data_file=cluster_file,
            raw_data_file="data/input/ga4_data.csv",
            output_file=quantitative_file
        )

        # Step 3: Generate overall comparison summary
        logger.info("Generating overall comparison summary...")
        ana.generate_overall_analysis(
            quantitative_file=quantitative_file,
            description_file=cluster_description_file,
            output_file=overall_summary_file
        )
        logger.info(f"Overall analysis completed and saved to {overall_summary_file}")

        # Step 4: Interactive Q&A session
        logger.info("Starting interactive Q&A session...")
        ana.perform_interactive_qna(
            sample_file=sample_file,
            quantitative_file=quantitative_file,
            overall_file=overall_summary_file
        )



if __name__ == "__main__":
    main()
