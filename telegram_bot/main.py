import os
import logging
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

# 🔧 Carregar variáveis de ambiente
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
POLARIS_API_URL = os.getenv("POLARIS_API_URL")

# 📝 Configuração de logs
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# 🚀 Inicializa a aplicação do bot
app = Application.builder().token(TELEGRAM_TOKEN).read_timeout(240).write_timeout(240).build()


async def start(update: Update, context: CallbackContext):
    """Comando /start"""
    await update.message.reply_text("🤖 Olá! Meu nome é Polaris e sou sua assistente privada. Como posso ajudar?")


async def handle_message(update: Update, context: CallbackContext):
    """Manipula mensagens enviadas pelo usuário"""
    chat_id = update.message.chat_id
    text = update.message.text

    log.info(f"📩 Mensagem recebida de {chat_id}: {text}")

    # 🔥 Enviar para Polaris
    response = requests.post(POLARIS_API_URL, json={"prompt": text, "session_id": str(chat_id)})

    if response.status_code == 200:
        resposta = response.json().get("resposta", "⚠️ Erro ao processar a resposta.")
    else:
        resposta = "⚠️ Erro ao se comunicar com a Polaris."

    # 🔥 Responde ao usuário
    await update.message.reply_text(resposta)


def main():
    """Inicia o bot"""
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("🚀 Bot está rodando...")
    app.run_polling()


if __name__ == "__main__":
    main()
