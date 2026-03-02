import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

api_key = os.getenv("GROQ_API")

if not api_key:
    raise ValueError("GROQ_API not found.")

client = Groq(api_key=api_key)

def generate_response(prompt: str):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-8b-instant",
    )

    return chat_completion.choices[0].message.content