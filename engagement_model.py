import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/Social Media Engagement Dataset.csv")

# Create encoders
platform_encoder = LabelEncoder()
topic_encoder = LabelEncoder()

df["platform_encoded"] = platform_encoder.fit_transform(df["platform"])
df["topic_encoded"] = topic_encoder.fit_transform(df["topic_category"])

# Features
X = df[
    [
        "platform_encoded",
        "topic_encoded",
        "sentiment_score",
        "toxicity_score",
        "likes_count",
        "shares_count",
        "comments_count",
        "impressions"
    ]
]

# Target
y = df["engagement_rate"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

score = r2_score(y_test, predictions)

print("R2 Score:", score)

# Save everything
joblib.dump(model, "models/engagement_model.pkl")
joblib.dump(platform_encoder, "models/platform_encoder.pkl")
joblib.dump(topic_encoder, "models/topic_encoder.pkl")

print("Engagement Model Saved")