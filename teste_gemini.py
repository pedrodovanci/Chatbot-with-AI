from vertexai import init
from vertexai.generative_models import GenerativeModel

init(project="chat-clinico-n8n", location="us-central1")
m = GenerativeModel("gemini-1.0-pro-001")     # começa com 1.0 garantido
r = m.generate_content("Diga 'ok' em uma palavra.")
print(r.text)
