import faiss
import numpy as np

class LobeVectorIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.vectors = []

    def add(self, vector):
        vec = np.array([vector]).astype("float32")
        self.index.add(vec)
        self.vectors.append(vector)

    def search(self, query_vector, k=3):
        q = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(q, k)
        return indices[0]