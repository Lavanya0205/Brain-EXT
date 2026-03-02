from fastapi import FastAPI, UploadFile, File
import shutil
import os
from ocr.ocr_engine import extract_text_from_image

app = FastAPI(title="OCR Service")

@app.get("/health")
def health_check():
    return {"status": "OCR service running"}

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):

    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = extract_text_from_image(file_path)

    os.remove(file_path)

    return result
