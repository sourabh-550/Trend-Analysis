# reddit_collector.py
import praw
import pandas as pd
import re
import os
import time
from datetime import datetime
from tqdm import tqdm
from prawcore.exceptions import RequestException, ResponseException, ServerError

# ---------- Reddit API Setup ----------
reddit = praw.Reddit(
    client_id="rgcMgNkZC-dPMMh4j-gNPA",
    client_secret="B8Y_xrQi0oMCm3kzW-VEQdCmg5elRA",
    user_agent="Impressive-Roll-5854"
)

# ---------- Configuration ----------
TARGET_SUBS = [
    "fashion", "memes", "slang", "streetwear",
    "popculture", "aesthetics", "trendingsubreddits"
]

LIMIT = 500  # ✅ posts per subreddit (safe limit)
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "reddit_cultural_data.csv")

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Helper Functions ----------
def clean_text(text):
    """Clean and normalize post text"""
    text = re.sub(r"http\S+", "", text)        # remove links
    text = re.sub(r"[^A-Za-z\s]", "", text)    # remove special chars
    text = re.sub(r"\s+", " ", text)           # normalize spaces
    return text.strip().lower()

def fetch_reddit_posts(subreddit_name, limit):
    """Fetch posts safely from a subreddit"""
    subreddit = reddit.subreddit(subreddit_name)
    posts = []

    # Use .top() instead of .hot() for stability and less API stress
    for post in subreddit.top(limit=limit, time_filter="month"):
        try:
            # Filter low-quality posts
            if post.score < 20 or len(post.selftext) < 50:
                continue

            posts.append({
                "subreddit": subreddit_name,
                "title": clean_text(post.title),
                "text": clean_text(post.selftext),
                "score": post.score,
                "num_comments": post.num_comments,
                "created_utc": datetime.fromtimestamp(post.created_utc),
                "url": post.url
            })
        except Exception as e:
            print(f"⚠️ Error parsing post in {subreddit_name}: {e}")
            continue
    return posts

# ---------- Main Data Collection ----------
def collect_reddit_data():
    all_posts = []
    print("🚀 Starting Reddit data collection...\n")

    for sub in tqdm(TARGET_SUBS, desc="Collecting from subreddits"):
        try:
            posts = fetch_reddit_posts(sub, LIMIT)
            all_posts.extend(posts)
            print(f"✅ Collected {len(posts)} posts from r/{sub}")

            # Wait between subreddits to avoid rate limit
            time.sleep(5)

        except (RequestException, ResponseException, ServerError) as e:
            print(f"⏳ API limit or network error on {sub}: {e}")
            print("Sleeping for 60 seconds before retrying...")
            time.sleep(60)
            continue
        except Exception as e:
            print(f"❌ Unexpected error fetching from {sub}: {e}")
            time.sleep(10)
            continue

    return all_posts

# ---------- Save Data ----------
def save_clean_data(posts):
    df = pd.DataFrame(posts)

    if df.empty:
        print("⚠️ No posts collected. Try again later or check credentials.")
        return

    # Clean dataset
    df.drop_duplicates(subset=['text'], inplace=True)
    df = df[df['text'].str.strip() != ""]

    # Save or append
    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates(subset=["text"])

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Saved {len(df)} total clean posts to '{OUTPUT_FILE}'")

# ---------- Run Script ----------
if __name__ == "__main__":
    posts = collect_reddit_data()
    save_clean_data(posts)
    print("\n✅ Data collection complete. Safe & high-quality dataset ready!")
