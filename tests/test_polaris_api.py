import pytest
from httpx import AsyncClient
from polaris_api.main import app
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

@pytest.mark.asyncio
async def test_api_running():
    """Testa se a API inicializa corretamente"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 404  # O endpoint root não está definido

@pytest.mark.asyncio
async def test_inference_valid_prompt():
    """Testa a inferência com um prompt válido"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        payload = {"prompt": "Qual é a capital da França?", "session_id": "user_123"}
        response = await client.post("/inference/", json=payload)
        assert response.status_code == 200
        assert "resposta" in response.json()
        assert isinstance(response.json()["resposta"], str)

@pytest.mark.asyncio
async def test_inference_missing_prompt():
    """Testa a inferência sem fornecer um prompt (deve falhar)"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        payload = {"session_id": "user_456"}  # Sem prompt
        response = await client.post("/inference/", json=payload)
        assert response.status_code == 422  # FastAPI retorna erro de validação

@pytest.mark.asyncio
async def test_inference_invalid_session_id():
    """Testa a inferência com session_id inválido"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        payload = {"prompt": "Teste erro", "session_id": None}
        response = await client.post("/inference/", json=payload)
        assert response.status_code == 422  # FastAPI deve recusar o request

@pytest.mark.asyncio
async def test_inference_long_prompt():
    """Testa um prompt longo para verificar a robustez"""
    long_prompt = "Teste " * 1000  # Criando um prompt muito longo
    async with AsyncClient(app=app, base_url="http://test") as client:
        payload = {"prompt": long_prompt, "session_id": "long_user"}
        response = await client.post("/inference/", json=payload)
        assert response.status_code == 200
        assert "resposta" in response.json()

@pytest.mark.asyncio
async def test_model_not_loaded():
    """Testa a resposta caso o modelo não esteja carregado"""
    from polaris_api import llm  # Importa a instância do modelo

    llm.close()  # Força o modelo a fechar antes do teste
    async with AsyncClient(app=app, base_url="http://test") as client:
        payload = {"prompt": "Teste sem modelo", "session_id": "user_789"}
        response = await client.post("/inference/", json=payload)
        assert response.status_code == 500
        assert response.json()["detail"] == "Modelo não carregado!"
    
    llm.load()  # Recarrega o modelo após o teste
