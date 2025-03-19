import os
import logging
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackContext,
)

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
POLARIS_API_URL = os.getenv("POLARIS_API_URL")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


async def start(update: Update, context: CallbackContext):
    """Comando /start"""
    await update.message.reply_text(
        "🤖 Olá! Meu nome é Polaris e sou sua assistente privada. Como posso ajudar?"
    )


async def handle_message(update: Update, context: CallbackContext):
    """Manipula mensagens enviadas pelo usuário"""
    chat_id = update.message.chat_id
    text = update.message.text

    log.info(f"📩 Mensagem recebida de {chat_id}: {text}")

    try:
        response = requests.post(
            POLARIS_API_URL,
            json={"prompt": text, "session_id": str(chat_id)},
            timeout=30,
        )
        response.raise_for_status()
        resposta = response.json().get("resposta", "⚠️ Erro ao processar a resposta.")
    except requests.exceptions.RequestException as e:
        log.error(f"Erro na requisição: {e}")
        resposta = "⚠️ Erro ao se comunicar com a Polaris."

    log.info(f"📤 Resposta enviada para {chat_id}: {resposta}")
    await update.message.reply_text(resposta)


def main():
    """Inicia o bot"""
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .read_timeout(240)
        .write_timeout(240)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("🚀 Polaris Bot iniciado com sucesso!")
    app.run_polling()


if __name__ == "__main__":
    main()
