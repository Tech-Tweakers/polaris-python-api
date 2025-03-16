import os
import logging
import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_API_URL = os.getenv("TELEGRAM_API_URL")

POLARIS_API_URL = os.getenv("POLARIS_API_URL")

app = FastAPI()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class TelegramMessage(BaseModel):
    update_id: int
    message: dict


@app.post("/telegram-webhook/")
async def telegram_webhook(update: TelegramMessage):
    chat_id = update.message["chat"]["id"]
    text = update.message.get("text", "")

    log.info(f"📩 Mensagem recebida de {chat_id}: {text}")

    if text.startswith("/start"):
        reply_text = "🤖 Olá! Meu nome é Polaris e sou sua assistente privada. Como posso ajudar?"
    else:
        polaris_response = requests.post(
            POLARIS_API_URL, json={"prompt": text, "session_id": str(chat_id)}
        )

        if polaris_response.status_code == 200:
            response_data = polaris_response.json()
            reply_text = response_data.get(
                "resposta", "⚠️ Erro ao processar a resposta."
            )
        else:
            reply_text = "⚠️ Erro ao se comunicar com a Polaris."

    # Enviar a resposta de volta para o Telegram
    send_message(chat_id, reply_text)
    return {"status": "ok"}


def send_message(chat_id, text):
    url = f"{TELEGRAM_API_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        requests.post(url, json=payload, timeout=120)  # 🔥 Tempo máximo de resposta: 10s
    except requests.Timeout:
        log.warning(f"⚠️ Timeout ao tentar responder {chat_id}.")
    except requests.RequestException as e:
        log.error(f"❌ Erro ao enviar mensagem para {chat_id}: {e}")


if __name__ == "__main__":
    log.info("🚀 Iniciando Telegram Bot Handler...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
