from app.llm.brain_llm import generate_response


SUPPORTED_LANGUAGES = {
    "hindi": "Hindi",
    "bengali": "Bengali",
    "tamil": "Tamil",
    "telugu": "Telugu",
    "marathi": "Marathi",
    "gujarati": "Gujarati",
    "punjabi": "Punjabi",
    "malayalam": "Malayalam",
    "kannada": "Kannada",
    "odia": "Odia",
    "english": "English",
    "hinglish": "Hinglish (Hindi written in English script)"
}


def translate_text(text: str, target_language: str):

    target_language = target_language.lower()

    if target_language not in SUPPORTED_LANGUAGES:
        return {
            "error": f"Unsupported language. Supported languages: {list(SUPPORTED_LANGUAGES.keys())}"
        }

    language_name = SUPPORTED_LANGUAGES[target_language]

    prompt = f"""
You are a professional multilingual translator.

TASK:
Translate the following text into {language_name}.

RULES:
- Preserve original meaning.
- Maintain tone.
- Do NOT add extra explanation.
- Output ONLY the translated text.
- Do NOT wrap in quotes.

TEXT:
{text}
"""

    translated = generate_response(prompt)

    return {
        "original_text": text,
        "target_language": language_name,
        "translated_text": translated.strip()
    }