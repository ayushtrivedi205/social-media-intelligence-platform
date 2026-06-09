import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu
engagement_model = joblib.load("models/engagement_model.pkl")
platform_encoder = joblib.load("models/platform_encoder.pkl")
topic_encoder = joblib.load("models/topic_encoder.pkl")

st.info(
    """
    AI-Powered Social Media Intelligence Platform

    Features:
    • Sentiment Analysis
    • Engagement Prediction
    • Viral Post Prediction
    • Campaign Analytics
    • Toxicity Monitoring
    """
)

# Load dataset
df = pd.read_csv("data/Social Media Engagement Dataset.csv")

# Load model
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Page title
st.title("AI-Powered Social Media Analytics Platform")

st.write("Final Year Data Science Project")

# -------------------
# Dataset Metrics
# -------------------

st.header("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Posts", len(df))
col2.metric("Likes", int(df["likes_count"].sum()))
col3.metric("Shares", int(df["shares_count"].sum()))
col4.metric("Comments", int(df["comments_count"].sum()))

# -------------------
# Sentiment Chart
# -------------------

st.header("Sentiment Distribution")

st.bar_chart(df["sentiment_label"].value_counts())

# -------------------
# Platform Chart
# -------------------

st.header("Platform Distribution")

st.bar_chart(df["platform"].value_counts())

# -------------------
# Live Prediction
# -------------------

st.header("Live Sentiment Prediction")

user_text = st.text_area(
    "Enter a social media post"
)

if st.button("Predict"):

    transformed_text = vectorizer.transform([user_text])

    prediction = model.predict(transformed_text)

    st.success(
        f"Predicted Sentiment: {prediction[0]}"
    )

st.header("Engagement Rate Predictor")

platform = st.selectbox(
    "Platform",
    list(platform_encoder.classes_)
)

topic = st.selectbox(
    "Topic Category",
    list(topic_encoder.classes_)
)

sentiment_score = st.slider(
    "Sentiment Score",
    -1.0,
    1.0,
    0.0
)

toxicity_score = st.slider(
    "Toxicity Score",
    0.0,
    1.0,
    0.1
)

likes = st.number_input(
    "Likes",
    min_value=0,
    value=100
)

shares = st.number_input(
    "Shares",
    min_value=0,
    value=20
)

comments = st.number_input(
    "Comments",
    min_value=0,
    value=10
)

impressions = st.number_input(
    "Impressions",
    min_value=1,
    value=1000
)

if st.button("Predict Engagement"):

    platform_value = platform_encoder.transform([platform])[0]
    topic_value = topic_encoder.transform([topic])[0]

    prediction = engagement_model.predict([[
        platform_value,
        topic_value,
        sentiment_score,
        toxicity_score,
        likes,
        shares,
        comments,
        impressions
    ]])

    st.success(
        f"Predicted Engagement Rate: {prediction[0]:.4f}"
    )    

with st.sidebar:

    selected = option_menu(
        "Navigation",
        [
            "Dashboard",
            "Sentiment Analysis",
            "Engagement Predictor",
            "Viral Predictor",
            "Campaign Analytics"
        ],
        icons=[
            "house",
            "emoji-smile",
            "graph-up",
            "rocket",
            "bar-chart"
        ],
        default_index=0
    )

if selected == "Dashboard":

    st.title("AI Social Media Intelligence Platform")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Posts", len(df))
    col2.metric("Likes", int(df["likes_count"].sum()))
    col3.metric("Shares", int(df["shares_count"].sum()))
    col4.metric("Comments", int(df["comments_count"].sum()))

    st.subheader("Sentiment Distribution")
    st.bar_chart(df["sentiment_label"].value_counts())

    st.subheader("Platform Distribution")
    st.bar_chart(df["platform"].value_counts())

if selected == "Campaign Analytics":

    st.title("Campaign Analytics")

    campaign_data = (
        df.groupby("campaign_name")
        ["engagement_rate"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(campaign_data.head(10))

if selected == "Toxicity Analysis":

    st.title("Toxicity Analysis")

    st.subheader("Average Toxicity By Platform")

    toxicity = (
        df.groupby("platform")
        ["toxicity_score"]
        .mean()
    )

    st.bar_chart(toxicity)

from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = " ".join(df["text_content"].astype(str))

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(text)

fig, ax = plt.subplots()
ax.imshow(wordcloud)
ax.axis("off")

st.pyplot(fig)

topics = (
    df["topic_category"]
    .value_counts()
    .head(10)
)

st.bar_chart(topics)

platforms = (
    df["platform"]
    .value_counts()
)

st.bar_chart(platforms)

