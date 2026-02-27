import numpy as np
from core.memory.short_term import ShortTermMemory
from core.memory.long_term import LongTermMemory
from core.graph.entity_extractor import extract_entities
from core.graph.graph_store import knowledge_graph
from core.embeddings.text_embedder import embed_text
from core.memory.semantic_memory import add_memory
from core.database.mongo import vector_collection

short_memory = {
    "frontal": ShortTermMemory(),
    "temporal": ShortTermMemory(),
    "parietal": ShortTermMemory(),
    "occipital": ShortTermMemory()
}

long_memory = {
    "frontal": LongTermMemory(),
    "temporal": LongTermMemory(),
    "parietal": LongTermMemory(),
    "occipital": LongTermMemory()
}

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def update_memory(query, lobe, action, confidence):

    embedding = embed_text(query)
    vector_collection.insert_one({
    "user_id": "demo_user", 
    "lobe": lobe,
    "text": query,
    "embedding": embedding,
    "action": action,
    "confidence": confidence
})

    short_memory[lobe].add({
        "query": query,
        "embedding": embedding,
        "lobe": lobe,
        "action": action,
        "confidence": confidence
    })
    add_memory(query, lobe, action, confidence)

    long_memory[lobe].update_lobe(lobe)

    if confidence < 0.5:
        long_memory[lobe].increment_confusion()

    entities = extract_entities(query)

    if entities:
        knowledge_graph.add_concepts(entities)
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                knowledge_graph.connect(entities[i], entities[j])

def retrieve_similar(query, lobe, top_k=3):

    query_embedding = embed_text(query)

    memories = short_memory[lobe].get()

    scored = []

    for item in memories:
        if "embedding" in item:
            score = cosine_similarity(query_embedding, item["embedding"])
            scored.append((score, item))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [item for _, item in scored[:top_k]]


def get_memory_context(lobe=None):
    if lobe:
        return {
            "short_term_memory": short_memory[lobe].get(),
            "long_term_memory": long_memory[lobe].data
        }

    return {
        lobe_name: {
            "short_term_memory": short_memory[lobe_name].get(),
            "long_term_memory": long_memory[lobe_name].data
        }
        for lobe_name in short_memory
    }