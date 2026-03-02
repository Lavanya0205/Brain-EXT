from fastapi import FastAPI
from pydantic import BaseModel
from app.translation.translator import translate_text
from app.llm.brain_llm import generate_response

app = FastAPI(title="LLM Service")


# ---------- Request Models ----------

class GenerateRequest(BaseModel):
    prompt: str


class TranslationRequest(BaseModel):
    text: str
    target_language: str


# ---------- Routes ----------

@app.get("/health")
def health_check():
    return {"status": "LLM service is running"}


@app.post("/generate")
def generate(request: GenerateRequest):
    response = generate_response(request.prompt)
    return {"response": response}


@app.post("/translate")
def translate(request: TranslationRequest):
    return translate_text(
        text=request.text,
        target_language=request.target_language
    )