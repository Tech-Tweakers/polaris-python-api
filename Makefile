# Variáveis de ambiente
DEPLOY_PATH := $(shell pwd)
PYTHON := python3
PIP := pip3
MODEL_DIR := $(DEPLOY_PATH)/models
MODEL_URL := https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf?download=true  # Exemplo de URL

# ------------------------------------------------------------------------------------------
# 🛠️ Configuração inicial
# ------------------------------------------------------------------------------------------
.PHONY: setup
setup:
	@echo "📦 Instalando dependências globais..."
	sudo apt update && sudo apt install -y python3-pip jq wget
	$(PIP) install --upgrade pip
	@echo "✅ Setup inicial concluído!"

# ------------------------------------------------------------------------------------------
# 📦 Instalar dependências do projeto
# ------------------------------------------------------------------------------------------
.PHONY: install
install:
	@echo "📦 Instalando dependências do projeto..."
	$(PIP) install -r polaris_api/requirements.txt
	$(PIP) install -r telegram_bot/requirements.txt
	@echo "✅ Dependências instaladas!"

# ------------------------------------------------------------------------------------------
# 🤖 Baixar modelo LLaMA 3
# ------------------------------------------------------------------------------------------
.PHONY: download-model
download-model:
	@echo "📥 Baixando modelo LLaMA 3..."
	mkdir -p $(MODEL_DIR)
	wget -c $(MODEL_URL) -O $(MODEL_DIR)/llama3-7B.safetensors
	@echo "✅ Modelo LLaMA 3 baixado em $(MODEL_DIR)!"

# ------------------------------------------------------------------------------------------
# 🚀 Rodar API
# ------------------------------------------------------------------------------------------
.PHONY: start-api
start-api:
	@echo "🚀 Iniciando API..."
	cd polaris_api && $(PYTHON) main.py
	@echo "✅ API rodando!"

# ------------------------------------------------------------------------------------------
# 🤖 Rodar Telegram Bot
# ------------------------------------------------------------------------------------------
.PHONY: start-bot
start-bot:
	@echo "🤖 Iniciando Telegram Bot..."
	cd telegram_bot && $(PYTHON) main.py
	@echo "✅ Telegram Bot rodando!"

# ------------------------------------------------------------------------------------------
# 🌍 Configurar Ngrok + Webhook Telegram
# ------------------------------------------------------------------------------------------
.PHONY: setup-ngrok
setup-ngrok:
	@echo "🌐 Configurando Ngrok..."
	bash polaris_setup/scripts/setup_ngrok.sh
	@echo "✅ Ngrok e Webhook do Telegram configurados!"

# ------------------------------------------------------------------------------------------
# 🔄 Rodar tudo
# ------------------------------------------------------------------------------------------
.PHONY: start-all
start-all:
	@echo "🔄 Iniciando tudo..."
	make start-api &
	make start-bot &
	@echo "✅ Todos os serviços iniciados!"

# ------------------------------------------------------------------------------------------
# 🛑 Parar todos os processos
# ------------------------------------------------------------------------------------------
.PHONY: stop-all
stop-all:
	@echo "🛑 Parando todos os serviços..."
	pkill -f "python3 main.py"
	@echo "✅ Todos os processos parados!"

# ------------------------------------------------------------------------------------------
# 🔄 Reiniciar tudo
# ------------------------------------------------------------------------------------------
.PHONY: restart-all
restart-all:
	@echo "🔄 Reiniciando tudo..."
	make stop-all
	sleep 2
	make start-all
	@echo "✅ API e Telegram Bot reiniciados!"
