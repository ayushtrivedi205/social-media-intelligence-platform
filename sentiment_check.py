import pandas as pd

df = pd.read_csv("data/Social Media Engagement Dataset.csv")

print(df["sentiment_label"].value_counts())