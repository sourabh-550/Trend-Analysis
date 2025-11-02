import os
import sys
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

files = [
    "reddit_bulk_data.csv",
    "reddit_cultural_data0.csv",
    "reddit_cultural_data1.csv"
]

# Keep only files that actually exist; warn about missing ones.
existing_files = [f for f in files if os.path.exists(f)]
missing_files = [f for f in files if f not in existing_files]
if missing_files:
    print(f"Warning: the following input files were not found and will be skipped: {missing_files}")

if not existing_files:
    print("Error: no input files found. Checked files:", files)
    sys.exit(1)

df = pd.concat([pd.read_csv(f) for f in existing_files], ignore_index=True)
df.drop_duplicates(inplace=True)
df.dropna(subset=["title", "text"], inplace=True)
print(df.shape)
df["content"] = df["title"].fillna('') + " " + df["text"].fillna('')


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # remove URLs
    text = re.sub(r"\[.*?\]", "", text)             # remove [brackets]
    text = re.sub(r"[^a-z\s]", "", text)            # keep only alphabets
    text = re.sub(r"\s+", " ", text).strip()        # remove extra spaces
    return text

df["clean_text"] = df["content"].apply(clean_text)


nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
custom_stopwords = {"reddit", "im", "dont", "cant", "ive", "amp", "like", "get", "people", "also"}
stop_words.update(custom_stopwords)

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

df["processed_text"] = df["clean_text"].apply(preprocess_text)

df = df[df["processed_text"].str.split().str.len() > 3]

df = df[["subreddit", "processed_text", "score", "num_comments", "created_utc"]]
os.makedirs("data", exist_ok=True)
df.to_csv("data/cleaned_for_trend_analysis.csv", index=False)
print("✅ Cleaned dataset saved!")
