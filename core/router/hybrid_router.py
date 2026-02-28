import numpy as np
from core.embeddings.text_embedder import embed_text
from core.classifier.predictor import classify_text
from core.config.lobes import LOBE_TEXT
from core.router.action_router import decide_action
from core.memory.manager_memory import update_memory
from core.memory.semantic_memory import search_memory
from core.user.user_model import user_model
from core.user.user_adapter import adapt_action
from core.LLM.brain_llm import generate_response

# Precompute Lobe Embeddings
LOBE_EMBEDDINGS = {
    lobe: embed_text(text)
    for lobe, text in LOBE_TEXT.items()
}
# Utility Functions
def normalize_confidence(conf: float) -> float:
    if conf <= 0.3:
        return 0.3
    if conf >= 0.85:
        return 0.95
    return round((conf - 0.3) / (0.85 - 0.3), 3)


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# MAIN HYBRID ROUTER
def hybrid_route(query: str):

    # Classifier Prediction
    clf_result = classify_text(query)
    clf_lobe = clf_result["predicted_lobe"]
    clf_conf = float(clf_result["confidence"])

    final_lobe = clf_lobe
    final_confidence = clf_conf

    # Embedding Similarity Scoring
    query_emb = embed_text(query)

    sim_scores = {
        lobe: cosine_similarity(query_emb, emb)
        for lobe, emb in LOBE_EMBEDDINGS.items()
    }

    sorted_lobes = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)
    best_sim_lobe, best_sim_score = sorted_lobes[0]

    # Hybrid Lobe Decision
    if clf_conf < 0.6 and best_sim_score > sim_scores[clf_lobe] + 0.15:
        final_lobe = best_sim_lobe

    # Confidence Blending
    if clf_conf < 0.7:
        final_confidence = (clf_conf * 0.7) + (sim_scores[final_lobe] * 0.3)

    final_confidence = min(max(final_confidence, 0.35), 0.95)
    final_confidence = normalize_confidence(final_confidence)

    # User Bias Influence
    user_bias_lobe = user_model.get_dominant_lobe()

    if final_confidence < 0.6 and user_bias_lobe:
        final_lobe = user_bias_lobe

    # Retrieve Semantic Memory (RAG)
    memory_used = search_memory(query, final_lobe, top_k=3)

    best_similarity = 0
    if memory_used:
        best_similarity = memory_used[0]["score"]

    final_confidence += best_similarity * 0.05
    final_confidence = min(final_confidence, 0.98)

    # Decide Action
    action = decide_action(
        final_lobe,
        final_confidence,
        memory_used
    )

    action = adapt_action(action, final_confidence)

    # Update User Model
    user_model.update(
        lobe=final_lobe,
        action=action,
        confidence=final_confidence
    )
    # LLM CONTEXT BUILDING (RAG + Brain State)
    context_text = ""

    if memory_used:
        context_text = "\nRelevant past memories:\n"
        for m in memory_used:
            context_text += f"- {m['text']} (similarity: {round(m['score'], 3)})\n"

    prompt = f"""
You are an advanced cognitive AI system.

User Question:
{query}

Selected Brain Lobe:
{final_lobe}

Confidence Level:
{round(final_confidence, 3)}

{context_text}

Answer clearly, intelligently, and in a structured way.
"""

    # Generate LLM response FIRST
    llm_response = generate_response(prompt)

    # Store Query + Response in Memory & DB
    update_memory(
        query=query,
        response=llm_response,
        lobe=final_lobe,
        action=action,
        confidence=final_confidence
    )
    # Final Response
    return {
        "query": query,
        "selected_lobe": final_lobe,
        "confidence": round(final_confidence, 3),
        "action": action,
        "classifier_confidence": round(clf_conf, 3),
        "embedding_scores": {
            k: round(float(v), 3) for k, v in sim_scores.items()
        },
        "memory_used": memory_used[:2],
        "response": llm_response
    }