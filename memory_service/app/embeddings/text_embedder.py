from sentence_transformers import SentenceTransformer

# Load model once at startup
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str):
    """
    Convert text into a semantic vector
    """
    embedding = model.encode(text)
    return embedding.tolist()
