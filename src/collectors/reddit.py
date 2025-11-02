# reddit_bulk_collector.py
import praw
import pandas as pd
import re
import os
import time
from datetime import datetime
from tqdm import tqdm
import random

# ---------- Reddit API Setup ----------
reddit = praw.Reddit(
    client_id="rgcMgNkZC-dPMMh4j-gNPA",
    client_secret="B8Y_xrQi0oMCm3kzW-VEQdCmg5elRA",
    user_agent="Impressive-Roll-5854"
)

# ---------- Configuration ----------
TARGET_SUBS = [
    "AskReddit", "OutOfTheLoop", "trendingsubreddits", "upcomingtrends", "internetculture",
    "popculturechat", "entertainment", "television", "movies", "Celebs", "Futurology",
    "NewsOfTheWeird", "BrandNewSentence", "MemeEconomy",
    "memes", "dankmemes", "wholesomememes", "MemeTemplatesOfficial", "ComedyCemetery",
    "shitposting", "DeepFriedMemes", "okbuddyretard", "2meirl4meirl", "HolUp", "funny",
    "facepalm", "TikTokCringe", "Instagramreality", "InternetIsBeautiful",
    "fashion", "streetwear", "OutfitCritique", "malefashionadvice", "femalefashionadvice",
    "aesthetics", "ThriftStoreHauls", "MakeupAddiction", "BeautyGuruChatter",
    "Haulternative", "SkincareAddiction", "Y2KFashion", "VintageFashion",
    "Music", "hiphopheads", "popheads", "WeAreTheMusicMakers", "InternetStars",
    "InfluencerNews", "YouTube", "TikTok", "ShortsCreators", "Instagram", "CelebDrama",
    "SocialMedia", "gaming", "pcgaming", "xbox", "PlayStation", "NintendoSwitch",
    "PUBGMobile", "Valorant", "Fortnite", "Minecraft", "Genshin_Impact", "roblox",
    "Twitch", "StreamerNews", "livestreamfail", "technology", "ArtificialIntelligence",
    "MachineLearning", "DataScience", "PromptCraft", "ChatGPT", "StableDiffusion",
    "GenerativeAI", "TechNews", "futureology", "gadgets", "computers", "GenZ",
    "millennials", "selfimprovement", "LifeProTips", "AskMen", "AskWomen", "relationships",
    "dating_advice", "antiwork", "WorkReform", "WorkplaceDrama", "productivity",
    "UnpopularOpinion", "changemyview", "CasualConversation", "anime", "SoloLeveling",
    "Jujutsukaisen", "Marvel", "DC_Cinematic", "NetflixBestOf", "StreamersGoneWild",
    "influencersnark", "trendwatch", "BrandNewSentence", "unexpected", "nextfuckinglevel",
    "interestingasfuck"
]


SORT_METHODS = ["hot", "top", "new"]
LIMIT_PER_METHOD = 600  # per subreddit per method
DATA_PATH = "data/reddit_bulk_data.csv"

# ---------- Ensure data folder exists ----------
os.makedirs("data", exist_ok=True)

# ---------- Helper Functions ----------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def fetch_posts(subreddit_name, method, limit, existing_ids):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []

    try:
        if method == "hot":
            submissions = subreddit.hot(limit=limit)
        elif method == "top":
            submissions = subreddit.top(time_filter="week", limit=limit)
        elif method == "new":
            submissions = subreddit.new(limit=limit)
        else:
            return []

        for post in submissions:
            if post.id in existing_ids:
                continue

            if post.score < 20 or post.num_comments < 5:
                continue

            text_content = post.selftext if post.selftext else post.title
            cleaned = clean_text(text_content)

            if len(cleaned.split()) < 6:
                continue

            posts.append({
                "id": post.id,
                "subreddit": subreddit_name,
                "sort_method": method,
                "title": post.title,
                "text": cleaned,
                "score": post.score,
                "num_comments": post.num_comments,
                "created_utc": datetime.fromtimestamp(post.created_utc),
                "url": post.url
            })

            time.sleep(random.uniform(0.3, 0.8))  # human-like delay

    except Exception as e:
        print(f"❌ Error fetching from r/{subreddit_name} [{method}]: {e}")

    return posts

# ---------- Load Existing Data ----------
if os.path.exists(DATA_PATH):
    existing_df = pd.read_csv(DATA_PATH)
    if "id" in existing_df.columns:
        existing_ids = set(existing_df["id"].astype(str))
        print(f"📦 Loaded {len(existing_ids)} existing post IDs.")
    else:
        existing_ids = set()
else:
    existing_df = pd.DataFrame()
    existing_ids = set()

# ---------- Bulk Collection ----------
all_posts = []
for sub in tqdm(TARGET_SUBS, desc="Collecting trending data"):
    for method in SORT_METHODS:
        new_posts = fetch_posts(sub, method, LIMIT_PER_METHOD, existing_ids)
        print(f"✅ {len(new_posts)} new posts fetched from r/{sub} [{method}]")
        all_posts.extend(new_posts)
        time.sleep(random.uniform(2.5, 4.5))  # safe delay between subreddit batches

# ---------- Merge & Save ----------
if all_posts:
    new_df = pd.DataFrame(all_posts)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.drop_duplicates(subset=["id"], inplace=True)
    combined_df.to_csv(DATA_PATH, index=False)
    print(f"\n✅ Added {len(new_df)} new posts. Total: {len(combined_df)} saved to {DATA_PATH}")
else:
    print("\n😴 No new posts found this run.")
