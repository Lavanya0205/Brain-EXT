import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model once at startup
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Correct relative paths
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")

classifier = joblib.load(os.path.join(MODEL_DIR, "lobe_classifier.joblib"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))


def classify_text(text: str):

    embedding = embedder.encode([text])

    probabilities = classifier.predict_proba(embedding)[0]

    best_index = int(np.argmax(probabilities))
    predicted_lobe = label_encoder.inverse_transform([best_index])[0]
    confidence = float(probabilities[best_index])

    return {
        "predicted_lobe": predicted_lobe,
        "confidence": round(confidence, 3)
    }