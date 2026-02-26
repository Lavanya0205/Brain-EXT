import numpy as np
from core.embeddings.text_embedder import embed_text
from core.classifier.predictor import classify_text
from core.config.lobes import LOBE_TEXT
from core.router.action_router import decide_action
from core.memory.manager_memory import update_memory, get_memory_context
from core.user.user_model import user_model
from core.user.user_adapter import adapt_action
from core.memory.manager_memory import update_memory, retrieve_similar
from core.memory.semantic_memory import search_memory

# Precompute lobe embeddings
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
    values = np.array(list(sim_scores.values()))
    exp_vals = np.exp(values - np.max(values))
    probs = exp_vals / exp_vals.sum()
    return float(np.max(probs))

def hybrid_route(query: str):

    # Classifier prediction
    clf_result = classify_text(query)
    clf_lobe = clf_result["predicted_lobe"]
    clf_conf = float(clf_result["confidence"])

    # Initialize defaults
    final_lobe = clf_lobe
    final_confidence = clf_conf

    # Embedding similarity scoring
    query_emb = embed_text(query)
    sim_scores = {
        lobe: cosine_similarity(query_emb, emb)
        for lobe, emb in LOBE_EMBEDDINGS.items()
    }
    # Rank similarities
    sorted_lobes = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)
    best_sim_lobe, best_sim_score = sorted_lobes[0]

    # Decide final lobe (Hybrid logic)
    if clf_conf < 0.6 and best_sim_score > sim_scores[clf_lobe] + 0.15:
        final_lobe = best_sim_lobe

    # Compute final confidence
    if clf_conf < 0.7:
        final_confidence = (clf_conf * 0.7) + (sim_scores[final_lobe] * 0.3)
    final_confidence = min(max(final_confidence, 0.35), 0.95)
    final_confidence = normalize_confidence(final_confidence)

    user_bias_lobe = user_model.get_dominant_lobe()

    if final_confidence < 0.6 and user_bias_lobe:
        final_lobe = user_bias_lobe

    memory_used = search_memory(query, final_lobe, top_k=3)
    best_similarity = 0
    if memory_used:
     best_similarity = memory_used[0]["score"]
    final_confidence += best_similarity * 0.05
    final_confidence = min(final_confidence, 0.98)
    action = decide_action(
        final_lobe,
        final_confidence,
        memory_used
    )
    action = adapt_action(action, final_confidence)
    user_model.update(
        lobe=final_lobe,
        action=action,
        confidence=final_confidence
    )
    update_memory(
        query=query,
        lobe=final_lobe,
        action=action,
        confidence=final_confidence
    )
    return {
        "query": query,
        "selected_lobe": final_lobe,
        "confidence": round(final_confidence, 3),
        "action": action,
        "classifier_confidence": round(clf_conf, 3),
        "embedding_scores": {
            k: round(float(v), 3) for k, v in sim_scores.items()
        },
        "memory_used": memory_used[:2]  
    }
