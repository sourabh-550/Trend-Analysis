# # %%
# !pip uninstall -y sentence-transformers
# !pip install -q sentence-transformers==2.2.2
# !pip install -q transformers==4.41.2
# !pip install -q --upgrade huggingface_hub==0.23.4
# !pip install -q sentence-transformers==2.2.2


# %%
# Load all libraries, set parameters, and prepare notebook environment

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# clustering
from sklearn.cluster import KMeans

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# install keybert if needed
# !pip install keybert > /dev/null
from keybert import KeyBERT
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Parameters (tune these)
CLEANED_CSV = "/kaggle/input/dataset01/cleaned_data.csv"
EMB_PATH = "text_embeddings.npy"
CLUSTERED_CSV = "clustered_reddit_data.csv"
TREND_REPORT = "trend_report.csv"


N_CLUSTERS = 12
RECENT_DAYS = 7
ENGAGEMENT_WEIGHT_COMMENTS = 2
TOP_K_KEYWORDS = 12

# Display settings
pd.set_option('display.max_colwidth', 200)
sns.set_style('whitegrid')


# %%
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# %%
from sentence_transformers import SentenceTransformer

try:
    # Try the main model
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print("⚠️ Issue loading MiniLM. Falling back to alternative model.", e)
    # Use a working local mirror (will auto-download)
    model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

print("✅ Model loaded successfully!")


# %% [markdown]
# ## Load Dataset

# %%
# Read the cleaned CSV and ensure columns are valid.

assert os.path.exists(CLEANED_CSV), f"File not found: {CLEANED_CSV}"

df = pd.read_csv(CLEANED_CSV)
print("Initial shape:", df.shape)
print("Columns:", df.columns.tolist())

# Ensure required columns exist
for col in ['subreddit','processed_text','score','num_comments','created_utc']:
    if col not in df.columns:
        raise ValueError(f"Required column missing: {col}")

# Parse datetime
df['created_utc'] = pd.to_datetime(df['created_utc'], errors='coerce')
df = df.dropna(subset=['processed_text']).reset_index(drop=True)

print("After cleaning:", df.shape)
df.head(3)


# %% [markdown]
# ## Generate or Load Embeddings

# %%
# Compute text embeddings using SentenceTransformer or load precomputed ones.
EMBEDDING_MODEL="all-MiniLM-L6-v2"
if os.path.exists(EMB_PATH):
    print("Loading existing embeddings:", EMB_PATH)
    embeddings = np.load(EMB_PATH)
else:
    print("Computing embeddings using:", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = df['processed_text'].astype(str).tolist()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    np.save(EMB_PATH, embeddings)
    print("Saved embeddings to:", EMB_PATH)

print("Embeddings shape:", embeddings.shape)


# %% [markdown]
# ## Clustering

# %%
# Apply KMeans to group similar topics based on semantic embeddings.

print("Running KMeans with", N_CLUSTERS, "clusters")
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
labels = km.fit_predict(embeddings)
df['cluster'] = labels

print("Cluster distribution:")
print(df['cluster'].value_counts().sort_index())

df.to_csv(CLUSTERED_CSV, index=False)
print("Saved clustered data to:", CLUSTERED_CSV)


# %% [markdown]
# ## Trend Detection Logic

# %%
# Detect which clusters are gaining momentum recently.

df['engagement'] = df['score'].fillna(0) + ENGAGEMENT_WEIGHT_COMMENTS * df['num_comments'].fillna(0)

now = df['created_utc'].max() if df['created_utc'].notna().any() else pd.Timestamp.now()
recent_thresh = now - pd.Timedelta(days=RECENT_DAYS)
print("Current time:", now, "| Recent threshold:", recent_thresh)

df_recent = df[df['created_utc'] >= recent_thresh]
print("Recent posts:", len(df_recent))

recent_stats = df_recent.groupby('cluster').agg(
    recent_count=('processed_text','count'),
    recent_engagement=('engagement','sum')
).reset_index()

baseline_start = recent_thresh - pd.Timedelta(days=4*RECENT_DAYS)
df_prev = df[(df['created_utc'] >= baseline_start) & (df['created_utc'] < recent_thresh)]
prev_stats = df_prev.groupby('cluster').agg(prev_count=('processed_text','count')).reset_index()

trend_df = recent_stats.merge(prev_stats, on='cluster', how='left').fillna(0)
trend_df['growth_ratio'] = (trend_df['recent_count'] + 1) / (trend_df['prev_count'] + 1)
trend_df = trend_df.sort_values(['growth_ratio','recent_count'], ascending=[False, False]).reset_index(drop=True)
trend_df['rank'] = trend_df.index + 1

trend_df.head(12)


# %% [markdown]
# ## Keyword Extraction (KeyBERT)

# %%
# Extract multi-word topics (semantic phrases) instead of random tokens.

kw_model = KeyBERT(model='all-MiniLM-L6-v2')
cluster_keywords = []

for cluster_id in sorted(df['cluster'].unique()):
    cluster_texts = df[df['cluster'] == cluster_id]['processed_text'].dropna().tolist()
    combined_text = " ".join(cluster_texts)
    if not combined_text.strip():
        cluster_keywords.append({'cluster': cluster_id, 'top_keywords': ''})
        continue

    keywords = kw_model.extract_keywords(
        combined_text,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        top_n=TOP_K_KEYWORDS
    )

    filtered_keywords = [
        kw for kw, score in keywords
        if len(kw.split()) > 1 or kw.lower() not in ENGLISH_STOP_WORDS
    ]

    cluster_keywords.append({
        'cluster': cluster_id,
        'top_keywords': ", ".join(filtered_keywords)
    })

# Merge keywords into trend_df
keywords_df = pd.DataFrame(cluster_keywords)
trend_df = trend_df.merge(keywords_df, on='cluster', how='left')
trend_df.to_csv(TREND_REPORT, index=False)

print("✅ Trend report saved to:", TREND_REPORT)
trend_df.head(10)


# %% [markdown]
# ## Identify and Inspect Top Trends

# %%
TOP_N = 6
top_trending = trend_df.sort_values(['growth_ratio','recent_count'], ascending=[False, False]).head(TOP_N)

print("🔥 Top Trending Topics:\n")
for idx, row in top_trending.iterrows():
    print(f"Rank {row['rank']}: Cluster {int(row['cluster'])}")
    print(f"Growth Ratio: {row['growth_ratio']:.2f} | Recent Posts: {int(row['recent_count'])}")
    print("Top Keywords:", row['top_keywords'])
    print("-"*80)


# %% [markdown]
# ## Visualization (Trend Graph)

# %%
df['date'] = pd.to_datetime(df['created_utc']).dt.date
trend_time = df.groupby(['date','cluster']).size().unstack(fill_value=0)

top_clusters = df['cluster'].value_counts().head(6).index.tolist()
plt.figure(figsize=(12,6))
for c in top_clusters:
    series = trend_time[c] if c in trend_time.columns else pd.Series(0, index=trend_time.index)
    series.rolling(window=3, min_periods=1).mean().plot(label=f"Cluster {c}")
plt.legend()
plt.title("Cluster Trends Over Time (3-day Rolling Mean)")
plt.xlabel("Date")
plt.ylabel("Post Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Word Cloud for Top Cluster

# %%
def show_wordcloud_for_cluster(cluster_id, max_words=150):
    texts = " ".join(df[df['cluster']==cluster_id]['processed_text'].astype(str).tolist())
    if not texts.strip():
        print("No text for cluster", cluster_id)
        return
    wc = WordCloud(width=900, height=400, background_color='white', collocations=False).generate(texts)
    plt.figure(figsize=(14,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"WordCloud - Cluster {cluster_id}")
    plt.show()

top_cluster_id = int(top_trending.iloc[0]['cluster'])
show_wordcloud_for_cluster(top_cluster_id)

top_cluster_posts = df[df['cluster']==top_cluster_id][['subreddit','processed_text','score','num_comments']].head(10)
print(f"🔝 Sample Posts from Cluster {top_cluster_id}")
display(top_cluster_posts)


# %% [markdown]
# ## Save Final Clustered Data

# %%
df['cluster_keywords'] = df['cluster'].apply(
    lambda c: trend_df.loc[trend_df['cluster']==c, 'top_keywords'].values[0]
    if c in trend_df['cluster'].values else ""
)

outpath = "clustered_reddit_with_keywords.csv"
df.to_csv(outpath, index=False)
print("✅ Saved clustered data with semantic keywords to:", outpath)


# %%
df = pd.read_csv("/kaggle/working/clustered_reddit_with_keywords.csv")
df.head(10)

# %%
trending = df.groupby('cluster_keywords').size().sort_values(ascending=False).head(10)
print(trending)


# %%



