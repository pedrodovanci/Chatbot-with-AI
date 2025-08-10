import os, requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"

def ask_gemini(texto: str) -> str:
    payload = {"contents": [{"parts": [{"text": texto}]}]}
    r = requests.post(URL, json=payload, timeout=40)
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]
