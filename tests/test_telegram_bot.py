import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from telegram_bot.main import app, send_message

# Cliente de teste para FastAPI
client = TestClient(app)


@pytest.fixture
def mock_polaris_response():
    """Mocka a resposta da API Polaris"""
    with patch("telegram_bot.main.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"resposta": "Resposta da Polaris"}
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def mock_telegram_response():
    """Mocka a resposta do envio de mensagem do Telegram"""
    with patch("telegram_bot.main.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        yield mock_post


@patch("telegram_bot.main.requests.post")
def test_message_forwarded_to_polaris(mock_post):
    """Testa se uma mensagem normal é enviada corretamente para a Polaris e a resposta é reenviada ao Telegram"""

    # Criamos dois mocks diferentes
    mock_polaris_response = MagicMock()
    mock_polaris_response.status_code = 200
    mock_polaris_response.json.return_value = {"resposta": "Resposta da Polaris"}

    mock_telegram_response = MagicMock()
    mock_telegram_response.status_code = 200

    # O primeiro mock será para a Polaris, o segundo para o Telegram
    mock_post.side_effect = lambda *args, **kwargs: (
        mock_polaris_response if "inference" in args[0] else mock_telegram_response
    )

    payload = {
        "update_id": 123,
        "message": {"chat": {"id": 456}, "text": "Qual a capital da França?"},
    }

    response = client.post("/telegram-webhook/", json=payload)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    print("Chamadas registradas:", mock_post.call_args_list)  # Debugging

    # Captura a URL que foi chamada de fato para evitar erro de hardcoded
    if mock_post.call_args_list:
        polaris_call_url = mock_post.call_args_list[0][0][0]  # URL da primeira chamada
        polaris_call_data = mock_post.call_args_list[0][1]["json"]  # JSON enviado
        assert (
            "/inference/" in polaris_call_url
        ), f"Chamada inesperada: {polaris_call_url}"
        assert polaris_call_data == {
            "prompt": "Qual a capital da França?",
            "session_id": "456",
        }, f"Payload inesperado: {polaris_call_data}"

        # Testa se a resposta foi enviada ao Telegram sem validar o token na URL
        telegram_call_url = mock_post.call_args_list[1][0][
            0
        ]  # Obtém a URL da segunda chamada
        assert telegram_call_url.endswith(
            "/sendMessage"
        ), f"Chamada inesperada: {telegram_call_url}"

        # Garante que **duas** chamadas foram feitas (Polaris + Telegram)
        assert mock_post.call_count == 2
    else:
        pytest.fail("Nenhuma chamada a requests.post foi registrada.")
