import os
import logging
import requests
import whisper
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackContext,
)

# Carrega variáveis de ambiente
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
POLARIS_API_URL = os.getenv("POLARIS_API_URL")

# Logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Carrega modelo Whisper local
log.info("🧠 Carregando modelo Whisper...")
model = whisper.load_model("base")  # você pode trocar por "small", "medium" ou "large"


async def start(update: Update, context: CallbackContext):
    """Comando /start"""
    await update.message.reply_text(
        "🤖 Olá! Eu sou a Polaris, sua assistente privada.\n"
        "Me mande uma mensagem de texto ou um áudio e eu te respondo com amor e inteligência. 💫"
    )


async def handle_message(update: Update, context: CallbackContext):
    """Manipula mensagens de texto"""
    chat_id = update.message.chat_id
    text = update.message.text

    log.info(f"📩 Texto recebido de {chat_id}: {text}")

    try:
        response = requests.post(
            POLARIS_API_URL,
            json={"prompt": text, "session_id": str(chat_id)},
            timeout=360,
        )
        response.raise_for_status()
        resposta = response.json().get("resposta", "⚠️ Erro ao processar a resposta.")
    except requests.exceptions.RequestException as e:
        log.error(f"Erro na requisição: {e}")
        resposta = "⚠️ Erro ao se comunicar com a Polaris."

    log.info(f"📤 Resposta enviada para {chat_id}: {resposta}")
    await update.message.reply_text(resposta)


async def handle_audio(update: Update, context: CallbackContext):
    """Manipula mensagens de voz ou áudio"""
    chat_id = update.message.chat_id
    file = update.message.voice or update.message.audio

    if not file:
        await update.message.reply_text("⚠️ Não consegui encontrar o áudio.")
        return

    log.info(f"🎙️ Áudio recebido de {chat_id}")

    file_id = file.file_id
    new_file = await context.bot.get_file(file_id)

    os.makedirs("audios", exist_ok=True)
    file_path = f"audios/audio_{chat_id}_{file_id}.ogg"
    await new_file.download_to_drive(file_path)

    log.info(f"📥 Áudio salvo em {file_path}")
    await update.message.reply_text("🎧 Transcrevendo o áudio...")

    try:
        result = model.transcribe(file_path)
        texto_transcrito = result["text"].strip()
        log.info(f"📝 Transcrição de {chat_id}: {texto_transcrito}")

        await update.message.reply_text(f"🗣️ Transcrição:\n\n{texto_transcrito}")

        # Envia para a Polaris, como se fosse uma mensagem de texto
        response = requests.post(
            POLARIS_API_URL,
            json={"prompt": texto_transcrito, "session_id": str(chat_id)},
            timeout=360,
        )
        response.raise_for_status()
        resposta = response.json().get("resposta", "⚠️ Erro ao processar a resposta.")

        log.info(f"📤 Resposta da Polaris para {chat_id}: {resposta}")
        await update.message.reply_text(resposta)

    except Exception as e:
        log.error(f"Erro ao transcrever ou enviar para a Polaris: {e}")
        await update.message.reply_text("⚠️ Ocorreu um erro ao processar o áudio.")


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
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))

    log.info("🚀 Polaris Bot iniciado com sucesso!")
    app.run_polling()


if __name__ == "__main__":
    main()
