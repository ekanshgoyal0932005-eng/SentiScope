
from transformers import pipeline

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)


def analyze_text(text):
    if not text or not text.strip():
        raise ValueError("Please enter some text.")

    sentiment = sentiment_model(text[:512])[0]

    emotions = emotion_model(text[:512])[0]
    emotions = sorted(
        emotions,
        key=lambda item: item["score"],
        reverse=True
    )

    return {
        "sentiment": sentiment["label"],
        "sentiment_score": round(sentiment["score"] * 100, 2),
        "emotion": emotions[0]["label"],
        "emotion_score": round(emotions[0]["score"] * 100, 2),
        "all_emotions": emotions
    }
