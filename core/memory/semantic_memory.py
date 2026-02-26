from typing import List, Dict
import numpy as np
from core.embeddings.text_embedder import embed_text

MEMORY_BANK: Dict[str, List[Dict]] = {
    "frontal": [],
    "temporal": [],
    "parietal": [],
    "occipital": []
}

def add_memory(text: str, lobe: str, action: str, confidence: float):

    embedding = embed_text(text)

    memory = {
        "text": text,
        "embedding": embedding,
        "lobe": lobe,
        "action": action,
        "confidence": confidence
    }

    MEMORY_BANK[lobe].append(memory)
    return memory


def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def search_memory(query: str, lobe: str, top_k: int = 3, threshold: float = 0.6):

    if not MEMORY_BANK[lobe]:
        return []

    query_embedding = embed_text(query)
    results = []

    for memory in MEMORY_BANK[lobe]:
        score = cosine_similarity(query_embedding, memory["embedding"])

        if score >= threshold:
            results.append({
                "text": memory["text"],
                "score": score,
                "action": memory["action"],
                "confidence": memory["confidence"]
            })

    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]