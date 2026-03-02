from fastapi import FastAPI
from pydantic import BaseModel
from app.classifier.predictor import classify_text
from app.route.hybrid_router import hybrid_route

app = FastAPI(title="Routing Service")


class RouteRequest(BaseModel):
    query: str

class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "Routing service running"}


@app.post("/classify")
def classify(request: RouteRequest):
    return classify_text(request.query)


@app.post("/route")
def route(request: QueryRequest):
    return hybrid_route(request.query)