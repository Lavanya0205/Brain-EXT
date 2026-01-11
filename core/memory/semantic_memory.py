from typing import List, Dict
import numpy as np

from core.embeddings.text_embedder import embed_text

# Temporary in-memory storage (brain prototype)
MEMORY_BANK: List[Dict] = []

def add_memory(text: str):
    """
    Store a memory with its embedding
    """
    embedding = embed_text(text)

    memory = {
        "text": text,
        "embedding": embedding
    }

    MEMORY_BANK.append(memory)
    return memory

def cosine_similarity(vec1, vec2):
    """
    Measure similarity between two vectors
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def search_memory(query: str, top_k: int = 3):
    """
    Search memories using semantic similarity
    """
    if not MEMORY_BANK:
        return []

    query_embedding = embed_text(query)
    results = []

    for memory in MEMORY_BANK:
        score = cosine_similarity(
            query_embedding,
            memory["embedding"]
        )
        results.append({
            "text": memory["text"],
            "score": score
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
