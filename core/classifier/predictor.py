import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load trained classifier and label encoder
classifier = joblib.load("core/classifier/lobe_classifier.joblib")
label_encoder = joblib.load("core/classifier/label_encoder.joblib")


def classify_text(text: str):
    """
    Predict lobe and confidence for a given query
    """

    # Convert text → embedding
    embedding = embedder.encode([text])  # shape: (1, 384)

    # Predict probabilities
    probabilities = classifier.predict_proba(embedding)[0]

    # Get best prediction
    best_index = int(np.argmax(probabilities))
    predicted_lobe = label_encoder.inverse_transform([best_index])[0]
    confidence = float(probabilities[best_index])

    return {
        "predicted_lobe": predicted_lobe,
        "confidence": round(confidence, 3)
    }

