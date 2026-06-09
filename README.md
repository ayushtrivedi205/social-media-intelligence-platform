# AI-Powered Social Media Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![License](https://img.shields.io/badge/License-Academic-green)

## 📌 Project Overview

The **AI-Powered Social Media Intelligence Platform** is a Data Science and Machine Learning project designed to analyze social media content, predict engagement, identify viral posts, and provide actionable insights through an interactive dashboard.

The system uses Natural Language Processing (NLP), Machine Learning, and Data Visualization techniques to help understand user sentiment, engagement patterns, campaign performance, and content virality.

---

## 🎯 Objectives

* Analyze social media posts and user engagement.
* Perform sentiment analysis using NLP.
* Predict engagement rates of posts.
* Predict whether a post is likely to go viral.
* Analyze campaign performance.
* Monitor toxicity levels in social media content.
* Provide insights through an interactive Streamlit dashboard.

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Plotly
* Matplotlib
* WordCloud
* Joblib

---

## 📂 Dataset

### Dataset Details

* Total Records: **12,000**
* Total Features: **28**
* Domain: **Social Media Analytics**

### Important Features

```text
platform
text_content
hashtags
keywords
sentiment_score
sentiment_label
emotion_type
toxicity_score
likes_count
shares_count
comments_count
impressions
engagement_rate
campaign_name
campaign_phase
```

---

## 🏗 Project Architecture

```text
Dataset
   │
   ▼
Data Preprocessing
   │
   ▼
Feature Engineering
   │
   ├── Sentiment Analysis Model
   │          │
   │          ▼
   │   Sentiment Prediction
   │
   ├── Engagement Prediction Model
   │          │
   │          ▼
   │   Engagement Rate Prediction
   │
   └── Viral Prediction Model
              │
              ▼
       Viral Post Prediction

              │
              ▼

      Streamlit Dashboard
```

---

## 🤖 Machine Learning Models

### 1. Sentiment Analysis

**Algorithm:**

* Logistic Regression

**Technique:**

* TF-IDF Vectorization

**Target:**

```text
sentiment_label
```

**Accuracy:**

```text
93.83%
```

---

### 2. Engagement Rate Prediction

**Algorithm:**

* Random Forest Regressor

**Target:**

```text
engagement_rate
```

**Performance:**

```text
R² Score = 0.9906
```

---

### 3. Viral Post Prediction

**Algorithm:**

* Random Forest Classifier

**Target:**

```text
viral
```

**Accuracy:**

```text
95.50%
```

---

## 📊 Dashboard Features

### Dashboard Overview

* Total Posts
* Total Likes
* Total Shares
* Total Comments

### Sentiment Analysis

* Live sentiment prediction
* Sentiment distribution visualization

### Engagement Prediction

* Predict engagement rate based on post metrics

### Viral Post Prediction

* Predict whether a post is likely to become viral

### Campaign Analytics

* Top-performing campaigns
* Campaign engagement comparison

### Toxicity Analysis

* Platform-wise toxicity analysis
* Content safety insights

---

## 📁 Project Structure

```text
SocialMediaAnalyticsProject

│
├── data/
│   └── Social Media Engagement Dataset.csv
│
├── models/
│   ├── sentiment_model.pkl
│   ├── vectorizer.pkl
│   ├── engagement_model.pkl
│   ├── viral_model.pkl
│   ├── platform_encoder.pkl
│   └── topic_encoder.pkl
│
├── app/
│   └── dashboard.py
│
├── eda.py
├── sentiment_model.py
├── engagement_model.py
├── viral_model.py
│
├── screenshots/
│
├── requirements.txt
│
└── README.md
```

---

## ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/social-media-intelligence-platform.git
```

Move into the project directory:

```bash
cd social-media-intelligence-platform
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Project

Start the Streamlit dashboard:

```bash
```

Open in browser:

```text
https://social-media-intelligence-platform-8jhvhxu9ej7tjs5l7klkbn.streamlit.app/
```

---

## 📈 Results

| Model                 | Performance     |
| --------------------- | --------------- |
| Sentiment Analysis    | 93.83% Accuracy |
| Engagement Prediction | 0.9906 R² Score |
| Viral Prediction      | 95.50% Accuracy |

---

## 🔮 Future Enhancements

* Real-time social media API integration
* Deep Learning (LSTM/BERT) sentiment analysis
* Multi-language sentiment detection
* Influencer identification system
* Social media trend forecasting
* Automated campaign recommendations

---

## 👨‍💻 Author

**Ayush Trivedi**

Final Year Data Science Project

---

## 📜 License

This project is developed for educational and academic purposes.
