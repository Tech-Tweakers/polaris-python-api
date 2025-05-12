import os
import logging
import requests
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackContext,
)
from TTS.api import TTS

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
POLARIS_API_URL = os.getenv("POLARIS_API_URL")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

log.info("🧠 Carregando modelo Whisper (faster-whisper)...")
model = WhisperModel("small", compute_type="int8")

log.info("🗣️  Carregando modelo de voz...")
tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/your_tts",
    progress_bar=True,
    gpu=False,
)
print("🔊 Vozes disponíveis:", tts.speakers)


def gerar_audio(texto: str, path: str):
    """Gera arquivo de voz a partir de texto"""
    tts.tts_to_file(
        text=texto,
        file_path=path,
        speaker="female-pt-4\n",
        language="pt-br",
        speed=2,
    )


async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "🤖 Olá! Eu sou a Polaris, seja bem vinda(o)!\n"
        "Me mande uma mensagem de texto ou um áudio e eu te respondo o quanto antes! 💫"
    )


async def handle_message(update: Update, context: CallbackContext):
    chat_id = update.message.chat_id
    text = update.message.text
    log.info(f"📩 Texto recebido de {chat_id}: {text}")

    try:
        response = requests.post(
            POLARIS_API_URL,
            json={"prompt": text, "session_id": str(chat_id)},
            timeout=8000,
        )
        response.raise_for_status()
        resposta = response.json().get("resposta", "⚠️ Erro ao processar a resposta.")
    except requests.exceptions.RequestException as e:
        log.error(f"Erro na requisição: {e}")
        resposta = "⚠️ Erro ao se comunicar com a Polaris."

    log.info(f"📤 Resposta para {chat_id}: {resposta}")
    await update.message.reply_text(resposta)


async def handle_audio(update: Update, context: CallbackContext):
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
    # await update.message.reply_text("🎧 Transcrevendo o áudio...")

    try:
        segments, info = model.transcribe(file_path)
        texto_transcrito = " ".join([seg.text for seg in segments]).strip()
        log.info(f"📝 Idioma detectado: {info.language}")
        log.info(f"📝 Transcrição: {texto_transcrito}")

        # await update.message.reply_text(f"🗣️ Transcrição:\n\n{texto_transcrito}")

        response = requests.post(
            POLARIS_API_URL,
            json={"prompt": texto_transcrito, "session_id": str(chat_id)},
            timeout=8000,
        )
        response.raise_for_status()
        resposta = response.json().get("resposta", "⚠️ Erro ao processar a resposta.")

        log.info(f"📤 Resposta da Polaris: {resposta}")
        # await update.message.reply_text(resposta)

        audio_path = f"audios/resposta_{chat_id}.wav"
        gerar_audio(resposta, audio_path)
        await update.message.reply_voice(voice=open(audio_path, "rb"))

    except Exception as e:
        log.error(f"Erro no fluxo de áudio: {e}")
        await update.message.reply_text("⚠️ Erro ao processar o áudio.")


async def handle_pdf(update: Update, context: CallbackContext):
    chat_id = update.message.chat_id
    document = update.message.document

    if not document:
        await update.message.reply_text("⚠️ Não consegui encontrar o arquivo.")
        return

    log.info(f"📄 Documento recebido de {chat_id}: {document.file_name}")

    file_id = document.file_id
    new_file = await context.bot.get_file(file_id)

    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{document.file_name}"
    await new_file.download_to_drive(file_path)

    log.info(f"📥 PDF salvo em {file_path}")
    await update.message.reply_text("📂 Processando o PDF, aguarde...")

    try:
        with open(file_path, "rb") as f:
            files = {
                "file": (document.file_name, f, "application/pdf"),
                "session_id": (None, str(chat_id)),
            }
            response = requests.post(
                POLARIS_API_URL.replace("/inference/", "/upload-pdf/"),
                files=files,
                timeout=8000,
            )
            response.raise_for_status()
            result = response.json()

        await update.message.reply_text(f"✅ PDF processado com sucesso!")
        log.info(f"📤 Upload para Polaris OK: {result}")

    except Exception as e:
        log.error(f"Erro no upload do PDF: {e}")
        await update.message.reply_text("⚠️ Erro ao processar o PDF.")


def main():
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .read_timeout(8000)
        .write_timeout(8000)
        .connect_timeout(8000)
        .pool_timeout(8000)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))

    log.info("🚀 Polaris Bot com texto, áudio e PDF ativados!")
    app.run_polling()


if __name__ == "__main__":
    main()
