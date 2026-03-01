import easyocr
import cv2
import numpy as np
import re
from langdetect import detect
from core.translation.translator import translate_text
from core.LLM.brain_llm import generate_response

reader = easyocr.Reader(
    ['en', 'hi', 'ta', 'te', 'bn', 'mr', 'gu', 'pa', 'ml', 'kn', 'or'],
    gpu=False
)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    return thresh

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def summarize_text(text):
    prompt = f"""
Summarize the following extracted text clearly and concisely:

{text}
"""
    return generate_response(prompt)

def extract_text_from_image(image_path):

    processed_img = preprocess_image(image_path)

    results = reader.readtext(processed_img)

    extracted_text = " ".join([res[1] for res in results])
    cleaned = clean_text(extracted_text)

    # Average confidence
    confidence_scores = [res[2] for res in results]
    avg_confidence = round(sum(confidence_scores)/len(confidence_scores), 3) if confidence_scores else 0

    # Language detection
    try:
        detected_lang = detect(cleaned)
    except:
        detected_lang = "unknown"

    # Auto translate to English if not English
    translated = None
    if detected_lang != "en" and detected_lang != "unknown":
        translation_result = translate_text(cleaned, "english")
        translated = translation_result.get("translated_text")

    # LLM Summary
    summary = summarize_text(cleaned)

    return {
        "raw_text": extracted_text,
        "cleaned_text": cleaned,
        "detected_language": detected_lang,
        "translated_text": translated,
        "summary": summary,
        "confidence_avg": avg_confidence
    }