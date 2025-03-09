# 📘 Documentação da Arquitetura - Polaris

## 📌 Introdução
Polaris é um sistema modular que permite a automação de interações via Telegram e APIs internas. Esta documentação segue o **modelo C4**, estruturando a arquitetura do projeto para melhor compreensão e escalabilidade.

---

## 📊 Nível 1: Diagrama de Contexto
Mostra a visão geral do sistema e seus principais atores.

```plantuml
@startuml
!include <C4/C4_Container>

Person(Usuario, "Usuário")
System(Polaris, "Sistema Polaris", "Plataforma de Assistente Virtual")
System_Ext(Telegram, "Telegram", "Canal de Comunicação")
System_Ext(LLM, "Modelo de Linguagem", "Processamento de Respostas")

Usuario --> Telegram : Envia mensagens
Telegram --> Polaris : Encaminha mensagens
Polaris --> LLM : Processa as mensagens e gera respostas
LLM --> Polaris : Retorna respostas geradas
Polaris --> Telegram : Envia resposta ao usuário
@enduml
```

---

## 📦 Nível 2: Diagrama de Contêineres
Detalha os principais contêineres do sistema e como eles se comunicam.

```plantuml
@startuml
!include <C4/C4_Container>

Person(Usuario, "Usuário")
System_Boundary(Polaris, "Sistema Polaris") {
    Container(Backend, "FastAPI Backend", "Python", "Processa requisições")
    Container(TelegramBot, "Telegram Bot", "Python", "Interação com Telegram")
    Container(Database, "MongoDB", "Banco de Dados", "Armazena histórico")
}

System_Ext(Telegram, "Telegram")
System_Ext(LLM, "Modelo de Linguagem")

Usuario --> Telegram : Envia mensagens
Telegram --> TelegramBot : Encaminha mensagens
TelegramBot --> Backend : Processa requisição
Backend --> Database : Armazena dados
Backend --> LLM : Envia requisição
LLM --> Backend : Retorna resposta
Backend --> TelegramBot : Responde ao usuário
TelegramBot --> Telegram : Envia resposta
@enduml
```

---

## 🏗️ Nível 3: Diagrama de Componentes
Este nível detalha os módulos internos da Polaris.

```plantuml
@startuml
!include <C4/C4_Component>

Container_Boundary(Backend, "FastAPI Backend") {
    Component(API, "API Polaris", "FastAPI", "Gerencia requisições")
    Component(LLMClient, "Cliente LLM", "Python", "Comunicação com modelo de linguagem")
    Component(DBManager, "Gerenciador de DB", "Python", "Persistência de dados")
}

API --> LLMClient : Envia prompt
LLMClient --> API : Retorna resposta
API --> DBManager : Armazena histórico
@enduml
```

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

Agora, qualquer desenvolvedor pode entender e contribuir rapidamente com o projeto! 😃