import os
import joblib
from sentence_transformers import SentenceTransformer

# ------------------ PATH SETUP (CRITICAL FIX) ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "log_classifier.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# ------------------ LOAD MODELS ------------------
model_embedding = SentenceTransformer("all-MiniLM-L6-v2")
model_classifier = joblib.load(MODEL_PATH)

# ------------------ CLASSIFIER FUNCTION ------------------
def classify_with_bert(log_message: str):
    embedding = model_embedding.encode([log_message])
    probabilities = model_classifier.predict_proba(embedding)[0]

    confidence = max(probabilities)

    if confidence < 0.5:
        return None  # allow fallback (regex / LLM)

    return model_classifier.predict(embedding)[0]
