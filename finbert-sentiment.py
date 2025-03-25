import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from scipy.special import softmax

# Load FinBERT model
MODEL_NAME = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def get_finbert_sentiment(text):
    """Returns normalized sentiment score (0-100) for given text using FinBERT."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits.numpy()[0]
    probs = softmax(logits)  # Convert logits to probabilities
    sentiment_classes = ["negative", "neutral", "positive"]

    # Weighted sum to normalize into 0-100 scale
    score = probs[0] * 20 + probs[1] * 50 + probs[2] * 80  
    return score, sentiment_classes[np.argmax(probs)]

def news_analyst(state):
    """Analyzes news sentiment using FinBERT and generates a sentiment score."""
    news_df = state["news"]
    news_texts = []
    
    for _, row in news_df.iterrows():
        news_texts.append(f"{row.title}\n{row.body}")

    scores = []
    sentiments = []
    
    for text in news_texts[:20]:  # Process only the first 20 articles
        score, sentiment = get_finbert_sentiment(text)
        scores.append(score)
        sentiments.append(sentiment)
    
    avg_score = np.mean(scores) if scores else 50  # Default neutral score
    explanation = f"News articles are mostly {sentiments.count('positive')} positive, {sentiments.count('neutral')} neutral, and {sentiments.count('negative')} negative."
    
    return {"news_analyst_report": f"Sentiment Score: {avg_score:.2f}/100\n{explanation}"}
