# 🌍 Cultural Trend Prediction using Reddit Data

This project predicts and visualizes **emerging cultural trends** from Reddit posts using **Natural Language Processing (NLP)**, **clustering**, and **keyword extraction** techniques. It identifies which topics are gaining traction and generates **interactive insights** like growth ratios, trending keywords, and word clouds for each cluster.

---

## 🚀 Features

- 🔍 **Data Collection:** Gathers Reddit cultural discussions using the Reddit API (PRAW).
- 🧹 **Preprocessing:** Cleans text data — removes URLs, emojis, stopwords, punctuation, and converts to lowercase.
- 🧠 **Embedding & Clustering:** Uses `SentenceTransformer` to create embeddings and `KMeans` for clustering similar discussions.
- 🧩 **Trend Detection:** Detects the fastest-growing clusters using engagement metrics like growth ratio.
- 🗝️ **Keyword Extraction:** Extracts representative keywords using **KeyBERT** for better interpretability.
- 🎨 **Visualization:** Word clouds and growth reports to visualize rising cultural discussions.
- 🌐 **Deployment Ready:** Streamlit-compatible structure for future deployment as an interactive dashboard.

