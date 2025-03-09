# 📘 Documentação da Arquitetura - Polaris

## 📌 Introdução
Polaris é um sistema modular que permite a automação de interações via Telegram e APIs internas. Esta documentação segue o **modelo C4**, estruturando a arquitetura do projeto para melhor compreensão e escalabilidade.

---

## 📊 Diagrama de Arquitetura

A seguir, apresentamos o diagrama de contêineres representando os principais componentes do Polaris e suas interações:

![Diagrama de Contêineres](diag2.png)

---

## 📜 Componentes do Sistema

### 🚀 Polaris API
API principal baseada em **FastAPI** que recebe e processa requisições.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/ping")
def ping():
    return {"message": "Polaris API online!"}
```

### 🤖 Telegram Bot
Bot que interage com usuários e envia requisições para a Polaris API.

```python
from fastapi import FastAPI
import requests

app = FastAPI()

@app.post("/telegram-webhook/")
def webhook(update: dict):
    chat_id = update["message"]["chat"]["id"]
    text = update["message"].get("text", "")
    response = requests.post("http://polaris-api:8000/inference/", json={"prompt": text})
    return {"status": "ok"}
```

---

## 🐳 Infraestrutura Docker

### 📄 **docker-compose.yml**
Define a infraestrutura do sistema Polaris com **MongoDB, API e Bot Telegram**.

```yaml
version: '3.8'
services:
  mongodb:
    image: mongo:latest
    container_name: mongodb
    restart: always
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: admin123
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    networks:
      - polaris_network

  polaris-api:
    build:
      context: .
      dockerfile: ./polaris_api/Dockerfile
    container_name: polaris-api
    restart: always
    ports:
      - "8000:8000"
    depends_on:
      - mongodb
    environment:
      - MONGO_URI=mongodb://admin:admin123@mongodb:27017/polaris_db?authSource=admin
    networks:
      - polaris_network

  telegram-bot:
    build:
      context: .
      dockerfile: ./telegram-bot/Dockerfile
    container_name: telegram-bot
    restart: always
    ports:
      - "8001:8001"
    depends_on:
      - polaris-api
    environment:
      - POLARIS_API_URL=http://polaris-api:8000/inference/
    networks:
      - polaris_network

networks:
  polaris_network:
    driver: bridge
```

### 📄 **Makefile**
Facilita a automação de comandos do projeto.

```makefile
PYTHON = python3
PIP = pip
DOCKER_COMPOSE = docker-compose
BLACK = black

.PHONY: install format test docker-build docker-up docker-down version

install:
	$(PIP) install -r polaris_api/requirements.txt
	$(PIP) install -r telegram_bot/requirements.txt

format:
	$(BLACK) polaris_api telegram_bot tests

test:
	PYTHONPATH=./ pytest tests

docker-build:
	$(DOCKER_COMPOSE) build

docker-up:
	$(DOCKER_COMPOSE) up -d

docker-down:
	$(DOCKER_COMPOSE) down

version:
	git tag $(shell date +"v%Y.%m.%d-%H%M%S")
	git push origin --tags
```

---

## 📜 Conclusão
O Polaris é uma arquitetura modular e escalável baseada em **FastAPI, MongoDB e Docker**, permitindo fácil implantação e manutenção. 🚀

