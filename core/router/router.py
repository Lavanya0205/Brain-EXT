import numpy as np
from core.embeddings.text_embedder import embed_text
from core.router.lobe_examples import LOBE_EXAMPLES


def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def route_query(query: str):
    query_embedding = embed_text(query)
    scores = {}

    for lobe, data in LOBE_EXAMPLES.items():
        example_scores = []

        for example in data["examples"]:
            example_embedding = embed_text(example)
            score = cosine_similarity(query_embedding, example_embedding)
            example_scores.append(score)

        scores[lobe] = max(example_scores)

    selected_lobe = max(scores, key=scores.get)

    return {
        "query": query,
        "selected_lobe": selected_lobe,
        "confidence": scores[selected_lobe],
        "all_scores": scores
    }
