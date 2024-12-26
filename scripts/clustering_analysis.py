import os
import time
import logging
import random
import pandas as pd
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, OpenAIError

# Load environment variables from config.env
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.env')
load_dotenv(CONFIG_PATH)

logger = logging.getLogger(__name__)

class ClusterAnalyzer:
    """
    Performs K-Means clustering on user embeddings and manages cluster-related operations.
    """

    def __init__(self, n_clusters, max_retries=3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.max_retries = max_retries

        # Load the OpenAI API key from the environment
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY is not set in config.env. GPT-4 descriptions will not work.")

        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def fit_and_predict(self, embeddings_file="user_combined_embeddings.csv", output_file="clustered_users.csv"):
        df = pd.read_csv(embeddings_file, header=None, index_col=0)
        user_ids = df.index.tolist()
        X = df.values

        labels = self.kmeans.fit_predict(X)

        result_df = pd.DataFrame({"user_id": user_ids, "cluster": labels})
        result_df.to_csv(output_file, index=False)

        logger.info(f"Cluster assignments saved to: {output_file}")
        return result_df

    def generate_cluster_samples(self, clustered_data_file, raw_data_file, sample_output_file, n_samples=5):
        relevant_columns = [
            "event_date", "event_timestamp", "user_pseudo_id", "page_location",
            "page_title", "page_referrer", "engagement_time_msec", "device_browser", "geo_country"
        ]

        clustered_df = pd.read_csv(clustered_data_file)
        raw_df = pd.read_csv(raw_data_file)
        raw_df = raw_df[raw_df["event_name"] == "page_view"]
        raw_df = raw_df[relevant_columns]

        samples = []
        for c in range(self.n_clusters):
            cluster_members = clustered_df[clustered_df['cluster'] == c]['user_id'].tolist()
            if not cluster_members:
                logger.info(f"No members in cluster {c}. Skipping sampling.")
                continue

            sample_size = min(n_samples, len(cluster_members))
            sampled_users = random.sample(cluster_members, sample_size)
            subset_raw = raw_df[raw_df['user_pseudo_id'].isin(sampled_users)]

            if subset_raw.empty:
                logger.warning(f"No matching user records found for cluster {c}.")
                continue

            subset_raw["cluster"] = c
            samples.append(subset_raw)

        if samples:
            sampled_df = pd.concat(samples, ignore_index=True)
            sampled_df.to_csv(sample_output_file, index=False)
            logger.info(f"Cluster samples saved to: {sample_output_file}")
        else:
            logger.warning("No samples generated. Check cluster and raw data.")

    def perform_quantitative_analysis(self, clustered_data_file, raw_data_file, output_file):
        clustered_df = pd.read_csv(clustered_data_file)
        raw_df = pd.read_csv(raw_data_file)
        raw_df = raw_df[raw_df["event_name"] == "page_view"]

        quantitative_results = []
        for c in range(self.n_clusters):
            cluster_members = clustered_df[clustered_df['cluster'] == c]['user_id'].tolist()
            cluster_data = raw_df[raw_df['user_pseudo_id'].isin(cluster_members)]

            avg_page_views = cluster_data.groupby('user_pseudo_id').size().mean()
            top_urls = cluster_data['page_location'].value_counts().head(3).to_dict()
            top_browsers = cluster_data['device_browser'].value_counts().head(3).to_dict()
            avg_engagement = cluster_data['engagement_time_msec'].mean()

            quantitative_results.append({
                "cluster": c,
                "avg_page_views": avg_page_views,
                "top_urls": top_urls,
                "top_browsers": top_browsers,
                "avg_engagement_time": avg_engagement
            })

        pd.DataFrame(quantitative_results).to_csv(output_file, index=False)
        logger.info(f"Quantitative analysis saved to: {output_file}")
        return quantitative_results

    def describe_clusters(self, sampled_data_file, gpt_output_file):
        sampled_df = pd.read_csv(sampled_data_file)

        descriptions = {}
        with open(gpt_output_file, "w", encoding="utf-8") as f:
            for c in range(self.n_clusters):
                cluster_data = sampled_df[sampled_df['cluster'] == c]
                if cluster_data.empty:
                    description = f"Cluster {c} is empty or has no sampled data."
                else:
                    subset_text = self._convert_records_to_text(cluster_data)
                    prompt = (
                        f"Describe the typical behavior of users in cluster {c}.\n"
                        f"Here is a sample of user records:\n\n"
                        f"{subset_text}\n\n"
                        f"Provide a concise and detailed summary."
                    )
                    description = self._call_gpt4(prompt)

                descriptions[c] = description
                f.write(f"Cluster {c}:\n{description}\n\n")

        logger.info(f"Cluster descriptions saved to: {gpt_output_file}")
        return descriptions

    def generate_overall_analysis(self, quantitative_file, description_file, output_file):
        """
        Combines the cluster quantitative analysis and GPT-4 descriptions
        into a single comparative summary and saves it to a text file.
        """
        quantitative_df = pd.read_csv(quantitative_file)

        with open(description_file, "r", encoding="utf-8") as desc_file:
            descriptions = desc_file.read()

        prompt = (
            f"Based on the following quantitative analysis:\n\n"
            f"{quantitative_df.to_string(index=False)}\n\n"
            f"And these cluster descriptions:\n\n"
            f"{descriptions}\n\n"
            f"Provide a comparative summary highlighting key differences, similarities, and insights across clusters."
        )

        overall_analysis = self._call_gpt4(prompt)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(overall_analysis)

        logger.info(f"Overall comparative analysis saved to: {output_file}")

    def perform_interactive_qna(self, sample_file, quantitative_file, overall_file):
        """
        Interactive Q&A session using GPT-4, incorporating sample data and analysis.
        """
        if not self.client:
            logger.warning("OpenAI client is not configured. Q&A session cannot proceed.")
            return

        sample_data = pd.read_csv(sample_file)
        quantitative_data = pd.read_csv(quantitative_file)
        with open(overall_file, "r", encoding="utf-8") as f:
            overall_summary = f.read()

        conversation_context = []

        print("\nInteractive Q&A Session. Type 'exit' to end the session.")
        while True:
            user_question = input("Ask a question about the clusters: ")
            if user_question.lower() == "exit":
                print("Exiting Q&A session.")
                break

            prompt = (
                f"Using the following data:\n\n"
                f"Sample Data:\n{sample_data.to_string(index=False)}\n\n"
                f"Quantitative Analysis:\n{quantitative_data.to_string(index=False)}\n\n"
                f"Overall Summary:\n{overall_summary}\n\n"
                f"Answer the user's question:\n{user_question}"
            )

            conversation_context.append({"role": "user", "content": user_question})

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a behavioral analysis expert."},
                        *conversation_context,
                        {"role": "user", "content": prompt},
                    ]
                )
                reply = response.choices[0].message.content
                print(f"Answer: {reply}")
                conversation_context.append({"role": "assistant", "content": reply})
            except Exception as e:
                logger.error(f"Error during Q&A: {e}")
                print("Could not process your question. Please try again.")

    def _convert_records_to_text(self, dataframe):
        records = dataframe.to_dict(orient='records')
        formatted_records = [", ".join(f"{key}: {value}" for key, value in record.items()) for record in records]
        return "\n".join(formatted_records)

    def _call_gpt4(self, prompt):
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a behavioral analysis expert."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                logger.warning(f"Rate limit exceeded (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(10)
            except OpenAIError as e:
                logger.error(f"OpenAI API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(2)
            except Exception as e:
                logger.error(f"Unexpected error during GPT-4 call (attempt {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(2)

        return "Failed to retrieve GPT-4 description after multiple retries."
