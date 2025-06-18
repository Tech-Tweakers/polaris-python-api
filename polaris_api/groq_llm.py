import os
import requests

import logging
from colorama import Fore, Style, init

init(autoreset=True)

# Logging igual ao main
def log_info(message: str):
    logging.info(f"🔹 {message}")

def log_success(message: str):
    logging.info(f"✅ {message}")

def log_warning(message: str):
    logging.warning(f"⚠️ {message}")

def log_error(message: str):
    logging.error(f"❌ {message}")


class GroqLLM:
    def __init__(self, api_key: str, model: str = "llama3-70b-8192"):
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def load(self):
        log_info("🔌 Polaris conectado ao motor Groq.")
        log_success(f"✅ Modelo configurado: {self.model}")

    def close(self):
        log_info("🛑 Encerrando conexão simbólica com a Groq.")

    def invoke(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Você é Polaris, um assistente inteligente."},
                {"role": "user", "content": prompt}
            ],
            "temperature": float(os.getenv("TEMPERATURE", 0.2)),
            "top_p": float(os.getenv("TOP_P", 0.7)),
            "max_tokens": 1024,
            "stop": ["<|eot_id|>"]
        }

        try:
            log_info(f"📤 Enviando prompt para Groq...")
            response = requests.post(self.endpoint, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            log_success(f"🧠 Resposta Groq recebida com sucesso.")
            return content
        except Exception as e:
            log_error(f"❌ Erro na inferência via Groq: {e}")
            return "Erro ao consultar o modelo Groq."
