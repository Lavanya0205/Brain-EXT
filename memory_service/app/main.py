from fastapi import FastAPI
from pydantic import BaseModel
from app.memory.manager_memory import add_memory, get_memory_context
from app.memory.semantic_memory import add_memory,search_memory
from app.embeddings.text_embedder import embed_text

app = FastAPI(title="Memory Service")

# -----------------------------
# Request / Response Models
# -----------------------------

class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embedding: list
    dimension: int

class MemoryAddRequest(BaseModel):
    text: str
    lobe: str
    action: str
    confidence: float


class MemorySearchRequest(BaseModel):
    query: str
    lobe: str
    top_k: int = 3

# Routes-

@app.get("/health")
def health_check():
    return {"status": "Memory service running"}


@app.post("/embed", response_model=EmbedResponse)
def create_embedding(request: EmbedRequest):
    vector = embed_text(request.text)
    return {
        "embedding": vector,
        "dimension": len(vector)
    }


@app.post("/memory/add")
def add_memory_api(request: MemoryAddRequest):
    return add_memory(
        text=request.text,
        lobe=request.lobe,
        action=request.action,
        confidence=request.confidence
    )

@app.post("/memory/search")
def search_memory_api(request: MemorySearchRequest):
    results = search_memory(
        query=request.query,
        lobe=request.lobe,
        top_k=request.top_k
    )
    return {
        "query": request.query,
        "results": results
    }


@app.get("/memory")
def memory():
    return get_memory_context()