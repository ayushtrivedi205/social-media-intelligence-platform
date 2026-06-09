import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/Social Media Engagement Dataset.csv")

# Create viral column
average_engagement = df["engagement_rate"].mean()

df["viral"] = (
    df["engagement_rate"] > average_engagement
).astype(int)

# Encode text categories
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
        "impressions"
    ]
]

# Target
y = df["viral"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Save
joblib.dump(model, "models/viral_model.pkl")

print("Viral Model Saved")