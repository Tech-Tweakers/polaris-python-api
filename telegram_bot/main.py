import os
import uuid
import shutil
import logging
import requests
import subprocess
import threading
import time

from faster_whisper import WhisperModel
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from TTS.api import TTS
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.tts.models.xtts import XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from tts_router import gerar_audio
from prometheus_client import (
    CollectorRegistry,
    Gauge,
    Counter,
    Summary,
    push_to_gateway,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response


# Segurança de unpickling
add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
load_dotenv()

# Variáveis de ambiente
COQUI_SPEAKER_WAV = os.getenv("COQUI_SPEAKER_WAV", "polaris-voice.wav")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
POLARIS_API_URL = os.getenv("POLARIS_API_URL", "http://192.168.1.104:8000/inference/")

# Logs
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

registry = CollectorRegistry()

integration_total = Counter(
    "integration_requests_total",
    "Número total de requisições de integração processadas",
    ["endpoint", "session_id"],
    registry=registry,
)

integration_failures = Counter(
    "integration_failures_total",
    "Número total de falhas nas integrações",
    ["endpoint", "session_id"],
    registry=registry,
)

integration_duration = Summary(
    "integration_duration_seconds",
    "Duração da requisição da integração em segundos",
    ["endpoint", "session_id"],
    registry=registry,
)

# Inicialização dos modelos
log.info("🧠 Carregando modelo Whisper...")
whisper = WhisperModel("small", compute_type="int8")

TTS_ENGINE = os.getenv("TTS_ENGINE", "coqui").lower()
tts = None

if TTS_ENGINE == "coqui":
    log.info("🗣️  Carregando modelo TTS local (Coqui XTTS)...")
    tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=False,
        gpu=False,
    )
else:
    log.info(f"🗣️  TTS local desativado. Engine selecionada: {TTS_ENGINE.upper()}")


os.makedirs("audios", exist_ok=True)


# Utilitários de TTS
def limpar_texto(texto: str) -> str:
    return (
        texto.replace("...", ".")
        .replace("—", "")
        .replace("“", "")
        .replace("”", "")
        .strip()
    )


# Handlers Telegram
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Olá! Me mande texto, áudio ou PDF que eu te respondo!"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    text = update.message.text
    log.info(f"📩 Texto de {chat_id}: {text}")

    try:
        response = requests.post(
            POLARIS_API_URL,
            json={"prompt": text, "session_id": str(chat_id)},
            timeout=10,
        )
        resposta = response.json().get("resposta", "⚠️ Erro ao processar resposta.")
    except Exception as e:
        log.error(f"Erro: {e}")
        resposta = "⚠️ Erro ao se comunicar com a Polaris."

    await update.message.reply_text(resposta)


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    file = update.message.voice or update.message.audio

    if not file:
        await update.message.reply_text("⚠️ Áudio não encontrado.")
        return

    new_file = await context.bot.get_file(file.file_id)
    file_path = f"audios/audio_{chat_id}_{file.file_id}.ogg"
    await new_file.download_to_drive(file_path)

    try:
        segments, _ = whisper.transcribe(file_path, language="pt")
        texto = " ".join([seg.text for seg in segments]).strip()

        response = requests.post(
            POLARIS_API_URL,
            json={"prompt": texto, "session_id": str(chat_id)},
            timeout=10,
        )
        resposta = response.json().get("resposta", "⚠️ Erro ao processar resposta.")

        output_audio = f"audios/resposta_{chat_id}.mp3"
        gerar_audio(resposta, output_audio)
        await update.message.reply_voice(voice=open(output_audio, "rb"))
    except Exception as e:
        log.error(f"Erro: {e}")
        await update.message.reply_text("⚠️ Erro ao processar o áudio.")


async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    doc = update.message.document
    if not doc:
        await update.message.reply_text("⚠️ Arquivo não encontrado.")
        return

    file_path = f"uploads/{doc.file_name}"
    os.makedirs("uploads", exist_ok=True)
    new_file = await context.bot.get_file(doc.file_id)
    await new_file.download_to_drive(file_path)

    with open(file_path, "rb") as f:
        files = {
            "file": (doc.file_name, f, "application/pdf"),
            "session_id": (None, str(chat_id)),
        }
        r = requests.post(
            POLARIS_API_URL.replace("/inference/", "/upload-pdf/"), files=files
        )

    if r.status_code == 200:
        await update.message.reply_text("✅ PDF processado com sucesso!")
    else:
        await update.message.reply_text("⚠️ Erro ao processar PDF.")


# FastAPI Microserviço Local (para o frontend)
api = FastAPI()
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.post("/audio-inference/")
async def audio_inference(audio: UploadFile, session_id: str = Form(...)):
    endpoint = "/audio-inference/"
    integration_total.labels(endpoint=endpoint, session_id=session_id).inc()
    erro = False
    start_time = time.time()

    uid = str(uuid.uuid4())
    user_audio_name = f"user_{uid}.webm"
    input_path = os.path.join("audios", user_audio_name)
    wav_path = f"temp_{uid}.wav"
    mp3_path = f"audios/resposta_{uid}.mp3"

    try:
        with open(input_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        subprocess.run(["ffmpeg", "-y", "-i", input_path, wav_path], check=True)
        segments, _ = whisper.transcribe(wav_path, language="pt")
        texto = " ".join([seg.text for seg in segments]).strip()

        res = requests.post(
            POLARIS_API_URL, json={"prompt": texto, "session_id": session_id}
        )
        resposta = res.json().get("resposta", "Erro na Polaris")

        gerar_audio(resposta, mp3_path)

        PUBLIC_URL = os.getenv(
            "PUBLIC_URL", "https://fixtures-respective-condo-width.trycloudflare.com"
        )
        return {
            "resposta": resposta,
            "tts_audio_url": f"{PUBLIC_URL}/audio/{os.path.basename(mp3_path)}",
            "user_audio_url": f"{PUBLIC_URL}/audio/{user_audio_name}",
        }

    except Exception as e:
        erro = True
        integration_failures.labels(endpoint=endpoint, session_id=session_id).inc()
        return JSONResponse(status_code=500, content={"erro": str(e)})

    finally:
        elapsed = time.time() - start_time
        integration_duration.labels(endpoint=endpoint, session_id=session_id).observe(elapsed)
        try:
            push_to_gateway(
                "http://10.10.10.20:9091",  # ajuste conforme seu pushgateway real
                job="polaris-integrations",
                registry=registry,
            )
            log.info("📊 Métricas da integração enviadas com sucesso!")
        except Exception as push_error:
            log.warning(f"⚠️ Falha ao enviar métricas: {push_error}")

from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

@api.get("/metrics")
def metrics():
    try:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        log_error(f"Erro ao expor métricas: {e}")
        raise HTTPException(status_code=500, detail="Erro ao gerar métricas Prometheus")


from mimetypes import guess_type


@api.get("/audio/{filename}")
def get_audio(filename: str):
    path = os.path.join("audios", filename)
    if os.path.exists(path):
        media_type, _ = guess_type(path)
        return FileResponse(path, media_type=media_type or "application/octet-stream")
    return JSONResponse(status_code=404, content={"erro": "Arquivo não encontrado"})


def rodar_api():
    uvicorn.run(api, host="0.0.0.0", port=8010)


def main():
    threading.Thread(target=rodar_api, daemon=True).start()

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .read_timeout(10)
        .write_timeout(10)
        .connect_timeout(10)
        .pool_timeout(10)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))

    log.info("🚀 Polaris Telegram e API /audio-inference/ ativados!")
    app.run_polling()


if __name__ == "__main__":
    main()
