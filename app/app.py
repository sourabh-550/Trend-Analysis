# app.py — Cultural Trend Explorer
from pathlib import Path
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# ------------------- SETTINGS -------------------
st.set_page_config(
    page_title="Cultural Trend Explorer",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Dark violet theme CSS ---
PRIMARY = "#8A2BE2"  # violet
ACCENT = "#B882FF"
BG = "#0b0b0f"
TEXT = "#e6e6ee"

st.markdown(
    f"""
    <style>
        :root{{ --primary: {PRIMARY}; --accent: {ACCENT}; --bg: {BG}; --text: {TEXT}; }}
        .stApp {{ background-color: var(--bg); color: var(--text); }}
        .css-1d391kg {{ color: var(--text); }}
        .stButton>button {{ background-color: var(--primary); color: white; }}
        .stSlider>div>div>div>div {{ background: linear-gradient(90deg, var(--primary), var(--accent)); }}
        h1, h2, h3, h4, h5 {{ color: var(--accent) }}
        .metric-label {{ color: var(--text); }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------- HELPERS & DATA LOADING -------------------
BASE_DIR = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def load_data():
    """Load CSV files located in same directory as app.py."""
    trend_path = BASE_DIR / "trend_report.csv"
    clustered_path = BASE_DIR / "clustered_reddit_with_keywords.csv"

    if not trend_path.exists() or not clustered_path.exists():
        missing = []
        if not trend_path.exists():
            missing.append(str(trend_path.name))
        if not clustered_path.exists():
            missing.append(str(clustered_path.name))
        raise FileNotFoundError("Missing files: " + ", ".join(missing))

    trend_df = pd.read_csv(trend_path)
    clustered_df = pd.read_csv(clustered_path)
    return trend_df, clustered_df


def make_wordcloud(text, width=800, height=300, colormap="Purples"):
    if not text or str(text).strip() == "":
        return None
    wc = WordCloud(
        width=width,
        height=height,
        background_color="black",
        colormap=colormap,
        prefer_horizontal=0.9,
        collocations=False,
    ).generate(text)
    return wc


# ------------------- MAIN APP -------------------
def main():
    st.title("📊 Cultural Trend Explorer")
    st.caption("Analyze trending cultural clusters from Reddit — beginner-friendly dashboard 💜")

    # Load data
    try:
        trend_df, clustered_df = load_data()
    except FileNotFoundError as e:
        st.error(f"❌ {e}.\n\nPlease place both CSV files next to this app (`{BASE_DIR}`).")
        st.info("Expected files: `trend_report.csv`, `clustered_reddit_with_keywords.csv`")
        return

    # Validate columns
    required_trend_cols = {"cluster", "recent_count", "recent_engagement", "prev_count", "growth_ratio", "rank", "top_keywords"}
    required_cluster_cols = {"subreddit", "processed_text", "score", "num_comments", "created_utc", "cluster", "engagement", "date", "cluster_keywords"}

    if not required_trend_cols.issubset(set(trend_df.columns)):
        st.error("`trend_report.csv` is missing expected columns.")
        return
    if not required_cluster_cols.issubset(set(clustered_df.columns)):
        st.error("`clustered_reddit_with_keywords.csv` is missing expected columns.")
        return

    # Ensure cluster columns are strings
    trend_df["cluster_str"] = trend_df["cluster"].astype(str)
    clustered_df["cluster_str"] = clustered_df["cluster"].astype(str)

    # ------------------- OVERVIEW -------------------
    st.markdown("---")
    st.subheader("Overview — Top Clusters by Growth")

    top_n = st.slider("Number of top clusters to show", min_value=3, max_value=15, value=5)
    top_clusters = trend_df.sort_values("growth_ratio", ascending=False).head(top_n)

    # Beginner-friendly bar chart
    fig, ax = plt.subplots(figsize=(9, max(3, top_n * 0.6)), facecolor=BG)
    sns.set_style("darkgrid", {"axes.facecolor": BG, "figure.facecolor": BG})
    palette = sns.color_palette("Purples", n_colors=len(top_clusters))
    sns.barplot(x="growth_ratio", y="cluster_str", data=top_clusters, palette=palette, ax=ax)
    ax.set_xlabel("Growth Ratio", color=TEXT)
    ax.set_ylabel("Cluster", color=TEXT)
    ax.set_title("Top Clusters by Growth Ratio", color=ACCENT)
    for spine in ax.spines.values():
        spine.set_color("#222")
    ax.tick_params(colors=TEXT)
    st.pyplot(fig)
    st.caption("Higher growth ratio = faster recent growth compared to previous period.")

    # ------------------- CLUSTER EXPLORATION -------------------
    st.markdown("---")
    st.subheader("Explore Clusters")
    st.caption("Select clusters to see their topics, keywords, and word cloud visualization.")

    with st.expander("What is a cluster? 🤔", expanded=False):
        st.write(
            "A cluster groups similar Reddit posts by theme or topic. "
            "Each cluster has a numeric ID and represents one trending theme — "
            "for example, fashion, AI, gaming, or sports. You can explore them below."
        )

    cluster_options = trend_df["cluster_str"].tolist()
    default_sel = top_clusters["cluster_str"].tolist()[:3]
    selected = st.multiselect("Select clusters to explore", options=cluster_options, default=default_sel)

    if not selected:
        st.info("Pick at least one cluster to inspect.")
        return

    # ------------------- CLUSTER DETAILS -------------------
    for c in selected:
        row = trend_df[trend_df["cluster_str"] == c].iloc[0]
        c_posts = clustered_df[clustered_df["cluster_str"] == c]

        # --- Determine topic ---
        topic = row.get("top_keywords", "")
        if not topic or pd.isna(topic):
            topic = ", ".join(c_posts["cluster_keywords"].dropna().astype(str).unique()[:5])
        if not topic:
            topic = "N/A"

        # Clean & shorten topic
        short_topic = "N/A"
        try:
            if topic and not pd.isna(topic):
                first_chunk = re.split(r'[;,/|\-]', str(topic))[0]
                words = first_chunk.strip().split()
                short_topic = " ".join(words[:2]).title()
        except Exception:
            short_topic = "N/A"

        # --- Display ---
        st.markdown(f"### 🟣 Cluster {c} — Related Topic: **{short_topic}**")
        st.markdown(f"**Top Keywords:** _{row.get('top_keywords', '')}_")
        st.metric(label="Growth Ratio", value=f"{row.get('growth_ratio', 0):.2f}")
        st.caption(f"Recent engagement: {row.get('recent_engagement', 'N/A')} | Previous count: {row.get('prev_count', 'N/A')}")

        # --- Word Cloud ---
        text_blob = " ".join(c_posts["processed_text"].dropna().astype(str).tolist())
        wc = make_wordcloud(text_blob)
        if wc is not None:
            fig, ax = plt.subplots(figsize=(10, 3), facecolor=BG)
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            st.caption("Word cloud — larger words appear more often in this cluster.")
        else:
            st.info("No textual posts available to build a word cloud for this cluster.")

        st.markdown("---")

    st.markdown("\n\n💜 Made with Streamlit — Cultural Trend Visualization for Everyone.")


if __name__ == "__main__":
    main()
