# **Beha2Vec: User Behavioral Analysis Tool**

**This README.md is generated with the help of ChatGPT and the OpenAI `gpt-4o` model, used to expedite the development process.**

**Output data are samples from a first run (only the lightest has pushed), they will be overwritten in the next run.**

## **Overview**
This tool is designed to analyze user behavior by leveraging **transformer-based models**, which are typically employed in natural language processing (NLP). The project innovatively applies transformers to user behavioral analysis, enabling the creation of **user embeddings**—dense numerical representations of user actions.

### **Why Transformers for Behavioral Analysis?**
Transformers have transformed NLP by capturing dependencies in sequential data through their attention mechanisms. Here, they are adapted for behavioral analysis to:
- Generate embeddings that represent user behaviors.
- Enable clustering and segmentation based on these embeddings.
- Provide a scalable solution for analyzing large datasets and uncovering intricate patterns.

By mapping user actions into a shared embedding space, we achieve:
- **Better Similarity Modeling**: Users with similar behaviors are closer in the embedding space.
- **Enhanced Insights**: Embeddings distill user behavior for easier interpretability.
- **Scalability**: Pre-computed embeddings can power real-time recommendations and personalization.

---

## **Theoretical Basis**
Transformers, such as BERT and GPT, excel in modeling sequential data. This project develops a custom transformer architecture, the **UserBehaviorTransformer**, tailored for analyzing behavioral data.

### **Transformer Model Used**
- **Input Modalities**:
  - **URL Sequences**: Encoded as hashed IDs and embedded into dense vectors.
  - **Theme Sequences** (optional): Captures categories of interest, e.g., "Lifestyle" or "Apparel."
  - **Type Features** (optional): Encodes page types, e.g., "Blog," "Service Page," or "Homepage."
- **Architecture**:
  - Positional encoding for time-ordered behaviors.
  - Multi-head attention layers to capture patterns.
  - Triplet margin loss for meaningful embedding learning.

---

## **Tool Features**

### **1. Preprocessing**
- Filters **GA4 flattened data** for `page_view` events.
- Enriches data with page themes and types (optional).
- Prepares time-ordered user sequences.

### **2. Embedding Generation**
- Uses the **UserBehaviorTransformer** to encode user behaviors into embeddings.
- Supports optional inputs for themes and types.

### **3. Clustering**
- Applies **K-Means** to embeddings for grouping users by behavior.
- Generates samples for analysis from each cluster.

### **4. Quantitative Analysis**
- Measures metrics like:
  - **Average Page Views**
  - **Top URLs**
  - **Browser Preferences**
  - **Engagement Time**

### **5. Qualitative Analysis**
- Leverages GPT-4 to:
  - Describe clusters in natural language.
  - Compare clusters, highlighting differences and similarities.

### **6. Interactive Q&A**
- Chat-like interface for querying insights about clusters.
- Combines GPT-4 with quantitative and qualitative data for answers.

---

## **Components**

### **1. Data Preprocessing**
#### Logical Process:
1. Filters `page_view` events in GA4 data.
2. Extracts relevant columns: `user_id`, `timestamp`, `pageview_URL`.
3. Enriches data with page themes and types using zero-shot classification (optional).
4. Outputs time-sorted user sequences.

#### Notes:
- Initially designed for GA4 data from the Google Merchandise Store but adaptable for other datasets.
- Production use can replace `user_pseudo_id` with a persistent identifier for long-term tracking.

---

### **2. Embedding Generation**
#### Logical Process:
1. Inputs user sequences and encodes them using the `UserBehaviorTransformer`.
2. Outputs dense embeddings for clustering.

#### Transformer Architecture:
- **Inputs**:
  - URL sequences embedded via dense layers.
  - Optional themes and types for embedding enrichment.
- **Training**:
  - **Triplet Loss**: Optimized for meaningful embedding by minimizing anchor-positive distance and maximizing anchor-negative distance.

---

### **3. Clustering Analysis**
#### Logical Process:
1. Uses K-Means to group users by embeddings.
2. Samples representative users for qualitative analysis.
3. Performs quantitative analysis to summarize user behaviors.

#### GPT-4 Integration:
- Generates natural language summaries of clusters.
- Provides comparative insights and powers Q&A sessions.

---

### **4. Embedding Combination**
#### Logical Process:
1. Combines embeddings (e.g., URL, theme, type) using weighted averaging.
2. Outputs a consolidated embedding per user.

#### Notes:
- Default weights (`URL: 0.5`, `Theme: 0.3`, `Type: 0.2`) can be customized.

---

### **5. Model Training**
#### Logical Process:
1. Splits data into triplets: Anchor, Positive, and Negative.
2. Optimizes embeddings using **Triplet Margin Loss**.

#### Loss Function:
- Encourages proximity between anchor and positive embeddings while distancing negatives.

---

### **6. Utilities**
- Handles URL classification via scraping and caching.
- Uses retry logic for robust classification requests.

---

## **Setup Instructions**

### 1. **Clone the Repository**
```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. **Set Up a Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Configure OpenAI API Key**
1. Rename `config-sample.env` to `config.env`.
2. Add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-<your-key>
   ```

### 5. **Download ga4_data**
Download your flattened data and put them into `input` folder, naming it `ga4_data.csv`
You can use this dataset (or follow that file structure): https://www.kaggle.com/datasets/hashiromer/ga4-merchandise-store-flattened-bigquery

---

## **Usage**

### **First Run**
Start from Step 1:
```bash
python main.py
```
- Follow prompts to configure options.

### **Subsequent Runs**
Start from any intermediate step:
1. **1:** Preprocess data.
2. **2:** Train the model.
3. **3:** Generate embeddings.
4. **4:** Combine embeddings.
5. **5:** Perform clustering.
6. **6:** Analyze clusters.

---

## **Production Notes**
- Replace `user_pseudo_id` with persistent identifiers for extended user tracking.
- Integrate real-time data pipelines for dynamic embedding generation.

---

## **Interactive Q&A**
After completing cluster analysis:
1. Enter a Q&A session by providing questions about clusters.
2. Answers are generated by GPT-4 using the analysis and data.

---