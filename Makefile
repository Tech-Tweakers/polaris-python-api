PYTHON = python3
PIP = pip
DOCKER_COMPOSE = docker-compose
BLACK = black

.PHONY: help install format test docker-build docker-up docker-down

help:
	@echo "Comandos disponíveis:"
	@echo "  make install        -> Instala dependências do projeto"
	@echo "  make format         -> Formata o código com Black"
	@echo "  make test           -> Roda os testes unitários"
	@echo "  make docker-build   -> Constrói as imagens Docker"
	@echo "  make docker-up      -> Sobe os containers Docker"
	@echo "  make docker-down    -> Para e remove os containers Docker"

install:
	$(PIP) install --upgrade pip
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

