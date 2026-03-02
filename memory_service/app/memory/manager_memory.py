import numpy as np
from app.embeddings.text_embedder import embed_text

memory_store = []  # simple in-memory store

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def add_memory(query, response):
    embedding = embed_text(query)

    memory_store.append({
        "query": query,
        "response": response,
        "embedding": embedding
    })

    return {"message": "Memory stored"}

def retrieve_similar(query, top_k=3):
    query_embedding = embed_text(query)

    scored = []

    for item in memory_store:
        similarity = cosine_similarity(query_embedding, item["embedding"])
        scored.append((similarity, item))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [item for _, item in scored[:top_k]]

def get_memory_context():
    return memory_store