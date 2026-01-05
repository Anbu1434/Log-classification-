import joblib
from sentence_transformers import SentenceTransformer

model_embedding = SentenceTransformer("all-MiniLM-L6-v2")
model_classifier = joblib.load("models/log_classifier.joblib")


def classify_with_bert(log_message: str):
    embedding = model_embedding.encode([log_message])
    probabilities = model_classifier.predict_proba(embedding)[0]

    confidence = max(probabilities)

    if confidence < 0.5:
        return None  # ⚠️ allow fallback

    return model_classifier.predict(embedding)[0]
