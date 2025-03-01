import os
import logging
import requests
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv

# Carregar variáveis do .env (caso esteja usando)
load_dotenv()

# Configurações
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7892223046:AAFyfB9HHMOtZKAeIEnGomc6tkdQFJKsH7s")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
POLARIS_API_URL = "http://localhost:8000/inference/"  # Endpoint da Polaris

# Inicializar FastAPI
app = FastAPI()

# Logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class TelegramMessage(BaseModel):
    update_id: int
    message: dict

@app.post("/telegram-webhook/")
async def telegram_webhook(update: TelegramMessage):
    """Recebe mensagens do Telegram e responde via Polaris"""
    chat_id = update.message["chat"]["id"]
    text = update.message.get("text", "")

    log.info(f"📩 Mensagem recebida de {chat_id}: {text}")

    if text.startswith("/start"):
        reply_text = "🤖 Olá! Eu sou a Polaris, sua assistente privada. Como posso ajudar?"
    else:
        # Enviar a pergunta para a Polaris
        polaris_response = requests.post(POLARIS_API_URL, json={"prompt": text})
        response_data = polaris_response.json()
        reply_text = response_data.get("resposta", "⚠️ Erro ao processar a resposta.")

    # Enviar a resposta de volta para o Telegram
    send_message(chat_id, reply_text)
    return {"status": "ok"}

def send_message(chat_id, text):
    """Envia uma mensagem para um chat no Telegram"""
    url = f"{TELEGRAM_API_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, json=payload)

if __name__ == "__main__":
    log.info("🚀 Iniciando Telegram Bot Handler...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
