import numpy as np
from core.embeddings.text_embedder import embed_text
from core.classifier.predictor import classify_text
from core.config.lobes import LOBE_TEXT

# Precompute lobe embeddings (runs once at import)
LOBE_EMBEDDINGS = {
    lobe: embed_text(text)
    for lobe, text in LOBE_TEXT.items()
}

def normalize_confidence(conf: float) -> float:
    """
    Maps raw confidence (≈0.3-0.7) to human-readable scale (0-1)
    """
    if conf <= 0.3:
        return 0.3
    if conf >= 0.85:
        return 0.95
    return round((conf - 0.3) / (0.85 - 0.3), 3)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_confidence(sim_scores: dict) -> float:
    """Normalize similarity scores into a confidence"""
    values = np.array(list(sim_scores.values()))
    exp_vals = np.exp(values - np.max(values))
    probs = exp_vals / exp_vals.sum()
    return float(np.max(probs))


def hybrid_route(query: str):
    #  Classifier prediction
    clf_result = classify_text(query)
    clf_lobe = clf_result["predicted_lobe"]
    clf_conf = float(clf_result["confidence"])

    #  Embedding similarity
    query_emb = embed_text(query)

    sim_scores = {
        lobe: cosine_similarity(query_emb, emb)
        for lobe, emb in LOBE_EMBEDDINGS.items()
    }

    best_sim_lobe = max(sim_scores, key=sim_scores.get)

    #  Decide final lobe
    final_lobe = clf_lobe
    if clf_conf < 0.6 and sim_scores[best_sim_lobe] > sim_scores[clf_lobe] + 0.15:
        final_lobe = best_sim_lobe

    # Compute confidence correctly
    # Confidence calibration 
    if clf_conf >= 0.7:
        final_confidence = clf_conf
    else:
        final_confidence = (clf_conf * 0.7) + (sim_scores[final_lobe] * 0.3)
        final_confidence = min(0.95, max(0.3, final_confidence))
        sorted_scores = sorted(sim_scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1]
        if margin > 0.08:
            final_confidence += 0.1
        elif margin > 0.12:
            final_confidence += 0.2

    final_confidence = min(final_confidence, 0.95)
    final_confidence = normalize_confidence(final_confidence)

    # Return response
    return {
    "query": query,
    "selected_lobe": final_lobe,
    "confidence": final_confidence,
    "classifier_confidence": round(float(clf_conf), 3),
    "embedding_scores": {
        k: round(float(v), 3) for k, v in sim_scores.items()
    }
}

