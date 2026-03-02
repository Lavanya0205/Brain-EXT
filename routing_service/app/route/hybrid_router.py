import os
import httpx
from app.classifier.predictor import classify_text


LLM_URL = os.getenv("https://llm-translator-93s4.onrender.com")
MEMORY_URL = os.getenv("https://memory-lhf4.onrender.com")
OCR_URL = os.getenv("https://ocr-fl0e.onrender.com")


def hybrid_route(query: str):

    # Classify
    clf_result = classify_text(query)
    lobe = clf_result["predicted_lobe"]
    confidence = clf_result["confidence"]

    # Ask memory service
    memory_response = httpx.post(
        f"{MEMORY_URL}/memory/search",
        json={
            "query": query,
            "lobe": lobe,
            "top_k": 3
        }
    )

    memory_data = memory_response.json()

    if memory_data.get("results"):
        return {
            "source": "memory",
            "lobe": lobe,
            "confidence": confidence,
            "response": memory_data
        }

    # If no strong memory → call LLM service
    llm_response = httpx.post(
        f"{LLM_URL}/generate",
        json={
            "query": query,
            "lobe": lobe,
            "confidence": confidence
        }
    )

    return {
        "source": "llm",
        "lobe": lobe,
        "confidence": confidence,
        "response": llm_response.json()
    }