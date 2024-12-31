# Beha2Vec — Using Transformers to Analyze User & Customer Behaviour

full blog: https://pdellov.medium.com/beha2vec-using-transformers-to-analyze-user-customer-behaviour-34d9f45b652a

Analyzing user behavior by examining event-related data is a fascinating area of study, particularly in the realms of digital marketing, customer experience, and sales. As users, we are accustomed to receiving personalized recommendations from almost every digital system we interact with. These recommendations are often built upon the analysis of behavioral sequences recorded from our digital activities. Scrolling through Instagram, reading a post, purchasing an item, interacting with a company’s sales representative, or moving between familiar locations — all these actions form a narrative of our behavior and enable systems to construct a story around us.

Despite the fact that even the smallest companies now collect significant amounts of data about their users and customers — whether through CRM systems, mobile apps, or web analytics tools — the analyses performed on these datasets often remain similar to those conducted years ago. This holds true even in a world where we are increasingly accustomed to discussions about AGI (Artificial General Intelligence) and breakthroughs related to AI agents and LLMs (Large Language Models).

Today’s state-of-the-art models for text and music generation, which underpin many generative AI applications we use daily, emerged from efforts to predict sequences (whether words or musical notes) with high accuracy by learning from large datasets. The cornerstone technologies enabling these breakthroughs are Transformer models and the Attention mechanism.

As a small experiment, I wondered whether the same logic used for language could be applied to the analysis of user behavior. Just as we “play” with predicting the next word in a sentence to create wonders like text generation or tools that describe language, such as embeddings, could we try to predict the next page view, the next purchase, or the next user action using a sufficiently large dataset?

The experiment described here does not claim to be scientifically rigorous or suitable for production use. It should instead be seen as an educational exercise — a way to explore the tools and logic behind training models like those we use every day.

What Are Embeddings and Transformers?
Embeddings
In machine learning, embeddings are dense vector representations of data points. They reduce complex, high-dimensional data into lower-dimensional spaces, making patterns more apparent and enabling a wide range of applications typical of vectorial spaces.


The classical embedding “visualization” for words. Imagine to have the same for people, based on their behavioural patterns.
For users, embeddings represent their behaviours, where:

Close vectors indicate similar users (when we say similar, we think about “similar behaviour”.
Distant vectors suggest different behaviors.
Transformers
Transformers are the backbone of modern NLP; their power lies in the attention mechanism, which allows models to focus on the most relevant parts of a sequence while processing it. Key concepts include:

Self-Attention: Captures relationships between sequence elements (words in a sentence or events in a user’s journey).
Positional Encoding: Adds order to sequences, crucial for time-series data like user behaviors.
The Goal
To test this approach, I used the old-fashioned Google Merchandise Store dataset, a demo dataset available through Kaggle. It is basically the dataset created over the Google Merchandise Store, which Google itself uses to demo its Google Analytics product. It contains 2+ GiB of user navigation events tracked via GA4. These include anonymous user_pseudo_id (cookie IDs) tied to actions like page views and purchases.

For those who are aware of the limits of Cookie IDs (even if they are first party cookie), the answer is YES: building the model upon more stable IDs, like PIIs (Hashed Emails) or fingerprinting based/persistent IDs will dramatically increase the performance. But I built this tool between a Salt Cod Fritter and Panettone, while my kids were sleeping and I was still slowly processing the Christmas dinner wine. Everything can be improved :D

The Process
Build a model to predict user behavior based on event sequences.
Use the model to generate embeddings that represent each user.
Cluster users into groups based on their embeddings.
Use LLMs (OpenAI) to describe and interpret the clusters.
While this project focuses on building embeddings and clustering users, the same vectors could enable advanced tasks like:

Identifying paths to convert “idle users” into “active buyers.”
Designing marketing strategies informed by the vector transformations between user states.
The project involved several stages, implemented as a modular pipeline:

1. Data Preprocessing
Starting with flattened GA4 data, I filtered for page_view events and enriched the dataset by:

Extracting themes and types from page_location using zero-shot classification.
Sorting events by event_timestamp to create sequential user histories.
2. Model Training & Embedding Generation
I designed a custom Transformer model tailored for user behavior:

Inputs: URL sequences, optional themes, and types.
Embedding Layers: Encoded these inputs into dense vectors.
Positional Encoding: Ensured sequence order was preserved.
Output: A single embedding vector representing a user.
To ensure embeddings captured meaningful relationships, I used triplet loss:

Anchor: A sequence from a user.
Positive: Another sequence from the same user.
Negative: A sequence from a different user.
The model learned to position the anchor closer to the positive than the negative, creating a coherent embedding space.

3. Clustering and Insights
Using K-Means, I clustered users based on their embeddings. For each cluster:

Quantitative analysis identified patterns like top pages or average engagement times.
Descriptive analysis using OpenAI’s GPT-4o provided human-readable summaries.
I also added a chat-like interface for querying insights about clusters — so at the end of the analysis you can basically interact with your data and you clusters asking the tools queries like “Build a buyer persona for Cluster 0” and other funny things.

Chatting with data to extract some information in natural language. Here I’m just using OpenAI API, but clusters are made with the model.
Below you will find an example of the analysis produced in a run over 6 clusters created with K-Means over the embedding space:

The analysis of user behavior across these clusters highlights distinct patterns that manifest in browsing habits, engagement, geographic diversity, and intent. Here’s a comparative summary highlighting the key differences, similarities, and insights across the clusters:

Similarities Across Clusters:

1. Browser Usage: Chrome is the predominant browser across all clusters, indicating a strong user base on this platform. Safari is the second most used browser, particularly evident in Clusters 1 and 4, signifying a secondary platform preference likely attributable to iOS devices.

2. Direct Navigation: A common pattern is the lack of referrer information, indicating that users across clusters are accessing the site directly, perhaps through bookmarks or direct URL inputs, suggesting a committed user base or recognized brand loyalty.

3. Engagement Times: Engagement times are generally low or not recorded across most clusters, suggesting either rapid browsing behavior or possible deficiencies in engagement tracking.

4. International Audience: Users hail from a wide array of countries in most clusters, reflecting the global reach and appeal of the Google Merchandise Store.

Key Differences:

1. Page Views and Engagement Patterns: Cluster 0 stands out with the highest average page views (40.6) indicating extensive browsing, yet with brief engagement time, suggesting exploration rather than conversion intent. Cluster 1 users have lower page views (1.3) but the highest average engagement time (19.5), pointing towards focused browsing with interest in specific products like YouTube merchandise.

2. Product Interest and Navigation Focus: Cluster 4 and Cluster 5 show a broad interest in a variety of merchandise categories, from apparel to lifestyle products, while Cluster 1 and Cluster 2 have more niche interest, particularly in YouTube merchandise and apparel, respectively. Cluster 3 users primarily explore lifestyle and accessory sections, hinting at a distinct consumer intent towards lifestyle merchandise.

3. Geographic Variance: Cluster 1 primarily consists of North American users, contrasting with other clusters which show wider international diversity. Clusters like Cluster 5 illustrate a diverse geographic spread, from Europe to North America.

4. Technical Details and Error Patterns: Cluster 0 users frequently encounter ‘Page Unavailable’ errors, which could hinder conversion, while **Cluster 5** users experience page availability issues specifically related to product URLs.

5. Repeat Visits and Checkout Interaction: Cluster 0 shows intermittent interactions with checkout and cart pages with hints of purchase intent, unlike other clusters where such data is not prominently highlighted.

Insights:

Purchase Intent vs. Browsing: There is a clear distinction between clusters focused on browsing (e.g., Cluster 0, Cluster 5) and those showing potential purchase intent or product-specific interest (e.g., Cluster 1).

Global Reach with Localized Interests: While global interest is high, clusters like Cluster 1 indicate localized interest, particularly in North American users drawn to YouTube merchandise.

Potential for Targeted Marketing: Clusters can be targeted differently based on their interests and browsing behaviors. For example, Cluster 1 could be targeted with YouTube merchandise promotions, while a broader range of product types might appeal to Cluster 4 or 5 users.

Optimization Opportunities: The consistent use of direct navigation suggests a strong brand recognition but also highlights opportunities to enhance referral traffic. Addressing engagement tracking and resolving page errors (as noted in Clusters 0 and 5) could also improve user experience and potential conversion rates.

Overall, these clusters reveal diverse user behaviors, presenting opportunities for tailored marketing strategies and experience optimization to bolster engagement and conversion at the Google Merchandise Store.
