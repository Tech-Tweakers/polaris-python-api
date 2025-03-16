import os
import pytest
import requests
import asyncio
from dotenv import load_dotenv
from aioresponses import aioresponses
from telegram import Update, Message, Chat
from telegram.ext import Application, CallbackContext
from unittest.mock import AsyncMock, MagicMock
from telegram_bot.main import start, handle_message

# 🔧 Carregar variáveis de ambiente
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
POLARIS_API_URL = os.getenv("POLARIS_API_URL")

# Configuração do pytest para testes assíncronos
pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_update():
    """Cria um mock de Update do Telegram"""
    message = MagicMock(spec=Message)
    message.chat_id = 12345
    message.text = "Teste Polaris"
    message.reply_text = AsyncMock()

    update = MagicMock(spec=Update)
    update.message = message

    return update


@pytest.fixture
def mock_context():
    """Cria um mock de CallbackContext do Telegram"""
    return MagicMock(spec=CallbackContext)


async def test_start(mock_update, mock_context):
    """Testa se o comando /start retorna a resposta correta"""
    await start(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_once_with(
        "🤖 Olá! Meu nome é Polaris e sou sua assistente privada. Como posso ajudar?"
    )


async def test_handle_message_success(mock_update, mock_context):
    """Testa se a resposta da Polaris é processada corretamente"""
    with aioresponses() as mock_requests:
        mock_requests.post(
            POLARIS_API_URL, payload={"resposta": "Resposta da Polaris"}
        )

        await handle_message(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once_with("Resposta da Polaris")


async def test_handle_message_error(mock_update, mock_context):
    """Testa se o bot lida corretamente com erro na API da Polaris"""
    with aioresponses() as mock_requests:
        mock_requests.post(POLARIS_API_URL, status=500)

        await handle_message(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once_with(
            "⚠️ Erro ao se comunicar com a Polaris."
        )


async def test_handle_message_no_response(mock_update, mock_context):
    """Testa quando a Polaris retorna uma resposta vazia"""
    with aioresponses() as mock_requests:
        mock_requests.post(POLARIS_API_URL, payload={})  # Resposta sem "resposta"

        await handle_message(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once_with(
            "⚠️ Erro ao processar a resposta."
        )


if __name__ == "__main__":
    pytest.main()
