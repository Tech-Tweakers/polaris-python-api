import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from telegram_bot.main import app, send_message

# Cliente de teste para FastAPI
client = TestClient(app)

# Definição das URLs usadas no mock
POLARIS_API_URL = "http://mock-api:8000/inference/"


@pytest.fixture
def mock_polaris_response():
    """Mocka a resposta da API Polaris"""
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"resposta": "Resposta da Polaris"}
        mock_post.return_value = mock_response
        yield mock_post  # Retorna o mock para o teste


@pytest.fixture
def mock_telegram_response():
    """Mocka a resposta do envio de mensagem do Telegram"""
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        yield mock_post  # Retorna o mock para o teste


def test_start_command(mock_telegram_response):
    """Testa se o comando /start retorna a resposta correta sem chamadas externas"""
    payload = {
        "update_id": 123,
        "message": {"chat": {"id": 456}, "text": "/start"},
    }

    response = client.post("/telegram-webhook/", json=payload)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    # Garante que a chamada foi feita para qualquer URL que termine com /sendMessage
    mock_telegram_response.assert_called_once()
    called_url = mock_telegram_response.call_args[0][0]  # Obtém a URL chamada
    assert called_url.endswith("/sendMessage"), f"Chamada inesperada: {called_url}"


@patch("requests.post")
def test_message_forwarded_to_polaris(mock_post):
    """Testa se uma mensagem normal é enviada corretamente para a Polaris e a resposta é reenviada ao Telegram"""
    
    # Criamos dois mocks diferentes
    mock_polaris_response = MagicMock()
    mock_polaris_response.status_code = 200
    mock_polaris_response.json.return_value = {"resposta": "Resposta da Polaris"}

    mock_telegram_response = MagicMock()
    mock_telegram_response.status_code = 200

    # O primeiro mock será para a Polaris, o segundo para o Telegram
    mock_post.side_effect = [mock_polaris_response, mock_telegram_response]

    payload = {
        "update_id": 123,
        "message": {"chat": {"id": 456}, "text": "Qual a capital da França?"},
    }

    response = client.post("/telegram-webhook/", json=payload)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    # Testa se a requisição foi feita para a Polaris API
    mock_post.assert_any_call(
        POLARIS_API_URL,  
        json={"prompt": "Qual a capital da França?", "session_id": "456"}
    )

    # Testa se a resposta foi enviada ao Telegram sem verificar o token na URL
    telegram_call_url = mock_post.call_args_list[1][0][0]  # Obtém a URL da segunda chamada
    assert telegram_call_url.endswith("/sendMessage"), f"Chamada inesperada: {telegram_call_url}"

    # Garante que **duas** chamadas foram feitas (Polaris + Telegram)
    assert mock_post.call_count == 2


@patch("requests.post")
def test_polaris_error_handling(mock_post):
    """Testa o tratamento de erro caso a API Polaris falhe"""

    # Mock da resposta de erro da Polaris
    mock_polaris_response = MagicMock()
    mock_polaris_response.status_code = 500
    mock_polaris_response.json.return_value = {}

    # Mock da resposta do Telegram
    mock_telegram_response = MagicMock()
    mock_telegram_response.status_code = 200

    # O primeiro mock será para a Polaris, o segundo para o Telegram
    mock_post.side_effect = [mock_polaris_response, mock_telegram_response]

    payload = {
        "update_id": 123,
        "message": {"chat": {"id": 456}, "text": "Quem descobriu o Brasil?"},
    }

    response = client.post("/telegram-webhook/", json=payload)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    # Testa se a requisição foi feita para a Polaris API
    mock_post.assert_any_call(
        POLARIS_API_URL,  
        json={"prompt": "Quem descobriu o Brasil?", "session_id": "456"}
    )

    # Testa se a resposta de erro foi enviada ao Telegram sem validar o token na URL
    telegram_call_url = mock_post.call_args_list[1][0][0]  # Obtém a URL da segunda chamada
    assert telegram_call_url.endswith("/sendMessage"), f"Chamada inesperada: {telegram_call_url}"

    # Garante que **duas** chamadas foram feitas (Polaris + Telegram)
    assert mock_post.call_count == 2


def test_send_message_function(mock_telegram_response):
    """Testa se a função send_message faz a requisição correta ao Telegram"""
    send_message(1234, "Teste de envio")

    # Garante que a chamada foi feita para qualquer URL que termine com /sendMessage
    mock_telegram_response.assert_called_once()
    called_url = mock_telegram_response.call_args[0][0]  # Obtém a URL chamada
    assert called_url.endswith("/sendMessage"), f"Chamada inesperada: {called_url}"
