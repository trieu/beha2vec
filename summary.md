# Beha2Vec — Using Transformers to Analyze User & Customer Behaviour

Analyzing user behavior by examining event-related data is a fascinating area of study, particularly in the realms of digital marketing, customer experience, and sales. As users, we are accustomed to receiving personalized recommendations from almost every digital system we interact with. These recommendations are often built upon the analysis of behavioral sequences recorded from our digital activities. Scrolling through Instagram, reading a post, purchasing an item, interacting with a company’s sales representative, or moving between familiar locations — all these actions form a narrative of our behavior and enable systems to construct a story around us.

Despite the fact that even the smallest companies now collect significant amounts of data about their users and customers — whether through CRM systems, mobile apps, or web analytics tools — the analyses performed on these datasets often remain similar to those conducted years ago. This holds true even in a world where we are increasingly accustomed to discussions about AGI (Artificial General Intelligence) and breakthroughs related to AI agents and LLMs (Large Language Models).

Today’s state-of-the-art models for text and music generation, which underpin many generative AI applications we use daily, emerged from efforts to predict sequences (whether words or musical notes) with high accuracy by learning from large datasets. The cornerstone technologies enabling these breakthroughs are Transformer models and the Attention mechanism.

As a small experiment, I wondered whether the same logic used for language could be applied to the analysis of user behavior. Just as we “play” with predicting the next word in a sentence to create wonders like text generation or tools that describe language, such as embeddings, could we try to predict the next page view, the next purchase, or the next user action using a sufficiently large dataset?

The experiment described here does not claim to be scientifically rigorous or suitable for production use. It should instead be seen as an educational exercise — a way to explore the tools and logic behind training models like those we use every day.

---

## What Are Embeddings and Transformers?

### Embeddings
In machine learning, embeddings are dense vector representations of data points. They reduce complex, high-dimensional data into lower-dimensional spaces, making patterns more apparent and enabling a wide range of applications typical of vectorial spaces.

> The classical embedding “visualization” for words. Imagine having the same for people, based on their behavioral patterns.

For users, embeddings represent their behaviors, where:
- **Close vectors** indicate similar users (in terms of similar behavior).
- **Distant vectors** suggest different behaviors.

### Transformers
Transformers are the backbone of modern NLP; their power lies in the attention mechanism, which allows models to focus on the most relevant parts of a sequence while processing it. Key concepts include:
- **Self-Attention:** Captures relationships between sequence elements (words in a sentence or events in a user’s journey).
- **Positional Encoding:** Adds order to sequences, crucial for time-series data like user behaviors.

---

## The Goal
To test this approach, I used the old-fashioned Google Merchandise Store dataset, a demo dataset available through Kaggle. It contains 2+ GiB of user navigation events tracked via GA4. These include anonymous `user_pseudo_id` (cookie IDs) tied to actions like page views and purchases.

> Note: Using more stable IDs like PIIs (Hashed Emails) or fingerprinting-based/persistent IDs would dramatically increase performance. However, this was a quick experiment built during a holiday break!

---

## The Process

1. **Data Preprocessing**
   - Filtered for `page_view` events.
   - Enriched the dataset by extracting themes and types from `page_location` using zero-shot classification.
   - Sorted events by `event_timestamp` to create sequential user histories.

2. **Model Training & Embedding Generation**
   - Built a custom Transformer model tailored for user behavior:
     - **Inputs:** URL sequences, optional themes, and types.
     - **Embedding Layers:** Encoded these inputs into dense vectors.
     - **Positional Encoding:** Preserved sequence order.
     - **Output:** A single embedding vector representing a user.
   - Used **triplet loss** to train the model:
     - **Anchor:** A sequence from a user.
     - **Positive:** Another sequence from the same user.
     - **Negative:** A sequence from a different user.

3. **Clustering and Insights**
   - Clustered users using K-Means based on embeddings.
   - For each cluster:
     - Performed quantitative analysis (e.g., top pages, average engagement times).
     - Used OpenAI’s GPT to generate descriptive insights.
   - Created a chat-like interface for querying insights about clusters.

---

## Insights

### Similarities Across Clusters:
1. **Browser Usage:** Chrome dominates, followed by Safari, particularly in Clusters 1 and 4.
2. **Direct Navigation:** Most users access the site directly, indicating strong brand loyalty.
3. **Engagement Times:** Low or unrecorded engagement times across clusters.
4. **International Audience:** Users hail from diverse countries.

### Key Differences:
1. **Page Views:** Cluster 0 has the highest average page views (40.6), indicating exploration intent.
2. **Product Interest:** Clusters 4 and 5 show diverse product interests, while Clusters 1 and 2 are more niche.
3. **Geographic Variance:** Cluster 1 is primarily North American; others show wider diversity.
4. **Technical Issues:** Cluster 0 frequently encounters "Page Unavailable" errors.

---

## Explore the Code and Contribute
The full code is available on GitHub: [pdellov/beha2vec](https://github.com/pdellov/beha2vec). The repository includes:
- Documentation.
- Modular pipeline for preprocessing, model training, embedding generation, and clustering analysis.
- Instructions to set up and run the tool.
