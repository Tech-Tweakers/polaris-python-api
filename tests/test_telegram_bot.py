import os
import pytest
import requests_mock
from dotenv import load_dotenv
from telegram import Update, Message, Voice
from telegram.ext import CallbackContext
from telegram.ext import MessageHandler, filters
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from telegram_bot.main import start, handle_message, handle_audio, handle_pdf

load_dotenv()
POLARIS_API_URL = os.getenv("POLARIS_API_URL", "http://mocked-api/polaris")

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_update():
    message = MagicMock(spec=Message)
    message.chat_id = 12345
    message.text = "Teste Polaris"
    message.reply_text = AsyncMock()

    update = MagicMock(spec=Update)
    update.message = message
    return update


@pytest.fixture
def mock_context():
    return MagicMock(spec=CallbackContext)


async def test_start(mock_update, mock_context):
    await start(mock_update, mock_context)
    mock_update.message.reply_text.assert_called_once_with(
        "🤖 Olá! Eu sou a Polaris, sua assistente privada.\n"
        "Me mande uma mensagem de texto ou um áudio e eu te respondo com amor e inteligência. 💫"
    )


async def test_handle_message_success(mock_update, mock_context):
    with requests_mock.Mocker() as mock_requests:
        mock_requests.post(POLARIS_API_URL, json={"resposta": "Resposta da Polaris"})
        await handle_message(mock_update, mock_context)
        mock_update.message.reply_text.assert_called_once_with("Resposta da Polaris")


async def test_handle_message_error(mock_update, mock_context):
    with requests_mock.Mocker() as mock_requests:
        mock_requests.post(POLARIS_API_URL, status_code=500)
        await handle_message(mock_update, mock_context)
        mock_update.message.reply_text.assert_called_once_with(
            "⚠️ Erro ao se comunicar com a Polaris."
        )


async def test_handle_message_no_response(mock_update, mock_context):
    with requests_mock.Mocker() as mock_requests:
        mock_requests.post(POLARIS_API_URL, json={})
        await handle_message(mock_update, mock_context)
        mock_update.message.reply_text.assert_called_once_with(
            "⚠️ Erro ao processar a resposta."
        )


@patch("telegram_bot.main.requests.post")
async def test_handle_pdf_success(mock_post, mock_context):
    # Mock resposta da Polaris no upload
    mock_post.return_value.json.return_value = {
        "message": "PDF processado com sucesso!"
    }
    mock_post.return_value.raise_for_status = lambda: None

    # Simula uma mensagem com documento
    document = MagicMock()
    document.file_id = "file123"
    document.file_name = "teste.pdf"

    message = MagicMock(spec=Message)
    message.chat_id = 12345
    message.document = document
    message.reply_text = AsyncMock()

    update = MagicMock(spec=Update)
    update.message = message

    # Simula download do arquivo
    file_mock = AsyncMock()
    file_mock.download_to_drive = AsyncMock()
    mock_context.bot.get_file = AsyncMock(return_value=file_mock)

    # Simula envio do arquivo de upload
    with patch("builtins.open", mock_open(read_data=b"pdfcontent")):
        await handle_pdf(update, mock_context)

    # Verificações
    message.reply_text.assert_any_call("📂 Processando o PDF, aguarde...")
    message.reply_text.assert_any_call("✅ PDF processado com sucesso!")


@patch("telegram_bot.main.requests.post")
async def test_handle_pdf_success(mock_post, mock_context):
    # Mock resposta da Polaris no upload
    mock_post.return_value.json.return_value = {
        "message": "PDF processado com sucesso!"
    }
    mock_post.return_value.raise_for_status = lambda: None

    # Simula uma mensagem com documento
    document = MagicMock()
    document.file_id = "file123"
    document.file_name = "teste.pdf"

    message = MagicMock(spec=Message)
    message.chat_id = 12345
    message.document = document
    message.reply_text = AsyncMock()

    update = MagicMock(spec=Update)
    update.message = message

    # Simula download do arquivo
    file_mock = AsyncMock()
    file_mock.download_to_drive = AsyncMock()
    mock_context.bot.get_file = AsyncMock(return_value=file_mock)

    # Simula envio do arquivo de upload
    with patch("builtins.open", mock_open(read_data=b"pdfcontent")):
        await handle_pdf(update, mock_context)

    # Verificações
    message.reply_text.assert_any_call("📂 Processando o PDF, aguarde...")
    message.reply_text.assert_any_call("✅ PDF processado com sucesso!")


@patch("telegram_bot.main.gerar_audio")
@patch("telegram_bot.main.model.transcribe")
@patch("telegram_bot.main.requests.post")
async def test_handle_audio_transcription_success(
    mock_post, mock_transcribe, mock_gerar_audio, mock_context
):
    # Mock transcrição e resposta da Polaris
    audio_text = "isto é um teste de áudio"
    mock_transcribe.return_value = {"text": audio_text}
    mock_post.return_value.json.return_value = {"resposta": "Resposta da Polaris"}
    mock_post.return_value.raise_for_status = lambda: None

    # Simula uma mensagem com voz
    message = MagicMock(spec=Message)
    message.chat_id = 12345
    message.voice = MagicMock(spec=Voice, file_id="abc123")
    message.audio = None
    message.reply_text = AsyncMock()

    update = MagicMock(spec=Update)
    update.message = message

    # Simula download do arquivo de áudio
    file_mock = AsyncMock()
    file_mock.download_to_drive = AsyncMock()
    mock_context.bot.get_file = AsyncMock(return_value=file_mock)

    # Simula envio de áudio de resposta
    with patch("builtins.open", mock_open(read_data=b"audio")):
        await handle_audio(update, mock_context)

    # Verificações
    assert message.reply_text.call_count == 3
    message.reply_text.assert_any_call("🎧 Transcrevendo o áudio...")
    message.reply_text.assert_any_call(f"🗣️ Transcrição:\n\n{audio_text}")
    message.reply_text.assert_any_call("Resposta da Polaris")
    mock_gerar_audio.assert_called_once_with(
        "Resposta da Polaris", "audios/resposta_12345.wav"
    )
