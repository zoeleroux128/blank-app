import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from nltk.tokenize import word_tokenize
import json

# Set Streamlit Title
st.title("Sentiment Analysis Dashboard")

# Text Input Section
st.subheader("Input Text for Analysis")
user_input = st.text_area("Enter your text here:")

# File Upload Section
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

def analyze_sentiment(text):
    """Function to analyze sentiment and provide classification & confidence score."""
    analysis = TextBlob(text)
    sentiment_score = analysis.sentiment.polarity
    if sentiment_score > 0:
        return "Positive", sentiment_score
    elif sentiment_score < 0:
        return "Negative", sentiment_score
    else:
        return "Neutral", sentiment_score

def extract_keywords(text):
    """Function to extract keywords influencing sentiment."""
    return word_tokenize(text)

# Process Input Data
if user_input:
    sentiment, confidence = analyze_sentiment(user_input)
    keywords = extract_keywords(user_input)
    
    st.write(f"**Sentiment:** {sentiment}")
    st.write(f"**Confidence Score:** {confidence:.2f}")
    st.write(f"**Keywords:** {', '.join(keywords)}")

# Process Batch Data from File Upload
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df["Sentiment"], df["Confidence"] = zip(*df["text"].apply(analyze_sentiment))
    st.dataframe(df)

    # Visualization
    fig = px.histogram(df, x="Sentiment", title="Sentiment Distribution")
    st.plotly_chart(fig)

    # Export Options
    json_data = df.to_json(orient="records")
    st.download_button("Download JSON", json_data, file_name="sentiment_results.json")
    df.to_csv("sentiment_results.csv")
    st.download_button("Download CSV", df.to_csv(index=False), file_name="sentiment_results.csv")
