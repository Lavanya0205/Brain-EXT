from fastapi import FastAPI
from pydantic import BaseModel
from core.memory.semantic_memory import add_memory, search_memory
from core.embeddings.text_embedder import embed_text
from core.router.router import route_query
from core.classifier.predictor import classify_text
from core.router.hybrid_router import hybrid_route
from core.memory.manager_memory import get_memory_context
from core.user.user_model import UserModel
from core.user.user_model import user_model
from core.dream.background_worker import BackgroundTasks
from core.dream.background_worker import process_upload
from core.graph.graph_store import knowledge_graph
from core.translation.translator import translate_text
from pydantic import BaseModel

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

class TranslationRequest(BaseModel):
    text: str
    target_language: str



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

@app.get("/memory")
def memory():
    return get_memory_context()

@app.get("/memory/{lobe}")
def check_lobe_memory(lobe: str):
    return get_memory_context(lobe)

@app.get("/user/profile")
def get_user_profile():
    return user_model.summary()

@app.post("/user/reset")
def reset_user():
    user_model.__init__()  # reinitialize
    return {"message": "User profile reset"}

@app.post("/upload")
def upload_file(file_text: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_upload, file_text)
    return {"status": "Processing in background"}

@app.get("/graph")
def get_graph():
    print("Current nodes:", list(knowledge_graph.graph.nodes))
    print("Current edges:", list(knowledge_graph.graph.edges))

    return {
        "nodes": list(knowledge_graph.graph.nodes),
        "edges": list(knowledge_graph.graph.edges)
    }
@app.post("/translate")
def translate(request: TranslationRequest):
    return translate_text(
        text=request.text,
        target_language=request.target_language
    )