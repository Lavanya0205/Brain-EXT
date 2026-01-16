from fastapi import FastAPI
from pydantic import BaseModel
from core.memory.semantic_memory import add_memory, search_memory
from core.embeddings.text_embedder import embed_text
from core.router.router import route_query
from core.classifier.predictor import classify_text
from core.router.hybrid_router import hybrid_route

app = FastAPI(title="Super Memory ML Service")

# ---------- Request / Response ----------
class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embedding: list
    dimension: int

class MemoryAddRequest(BaseModel):
    text: str

class MemorySearchRequest(BaseModel):
    query: str
    top_k: int = 3

class RouteRequest(BaseModel):
    query: str

class ClassifyRequest(BaseModel):
    query: str

class HybridRouteRequest(BaseModel):
    query: str



# ---------- Routes ----------
@app.get("/health")
def health_check():
    return {"status": "ML service is running"}

@app.post("/embed", response_model=EmbedResponse)
def create_embedding(request: EmbedRequest):
    vector = embed_text(request.text)
    return {
        "embedding": vector,
        "dimension": len(vector)
    }
@app.post("/memory/add")
def add_memory_api(request: MemoryAddRequest):
    memory = add_memory(request.text)
    return {
        "message": "Memory added successfully",
        "memory": memory["text"]
    }

@app.post("/memory/search")
def search_memory_api(request: MemorySearchRequest):
    results = search_memory(request.query, request.top_k)
    return {
        "query": request.query,
        "results": results
    }

@app.post("/route")
def route_api(request: RouteRequest):
    return route_query(request.query)

@app.post("/classify")
def classify_query(request: ClassifyRequest):
    return classify_text(request.query)

@app.post("/hybrid/route")
def route(request: RouteRequest):
    return hybrid_route(request.query)
