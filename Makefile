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
# 🐳 Subir MongoDB e Mongo Express (Docker Compose)
# ------------------------------------------------------------------------------------------
.PHONY: start-db
start-db:
	@echo "🐳 Iniciando MongoDB e Mongo Express..."
	cd polaris_setup/ && docker-compose up -d
	@echo "✅ MongoDB e Mongo Express rodando!"

# ------------------------------------------------------------------------------------------
# 🛑 Parar MongoDB e Mongo Express
# ------------------------------------------------------------------------------------------
.PHONY: stop-db
stop-db:
	@echo "🛑 Parando MongoDB e Mongo Express..."
	cd polaris_setup/ && docker-compose down
	@echo "✅ MongoDB e Mongo Express parados!"

# ------------------------------------------------------------------------------------------
# 🔄 Reiniciar MongoDB e Mongo Express
# ------------------------------------------------------------------------------------------
.PHONY: restart-db
restart-db:
	@echo "🔄 Reiniciando MongoDB e Mongo Express..."
	make stop-db
	sleep 2
	make start-db
	@echo "✅ Banco de dados reiniciado!"

# ------------------------------------------------------------------------------------------
# 🔄 Rodar tudo incluindo banco de dados
# ------------------------------------------------------------------------------------------
.PHONY: start-all
start-all:
	@echo "🔄 Iniciando tudo..."
	make start-db
	make start-api &
	make start-bot &
	@echo "✅ Todos os serviços iniciados!"

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
	@if [ -f telegram_bot/.env ]; then \
		export TELEGRAM_TOKEN=$$(grep "^TELEGRAM_TOKEN" telegram_bot/.env | cut -d '=' -f2); \
		export TELEGRAM_API_URL="https://api.telegram.org/bot$${TELEGRAM_TOKEN}"; \
	else \
		echo "⚠️  .env do Telegram Bot não encontrado!"; \
		exit 1; \
	fi
	sudo apt install -y ffmpeg
	cd telegram_bot && $(PYTHON) main.py
	@echo "✅ Telegram Bot rodando!"


# ------------------------------------------------------------------------------------------
# 🌍 Configurar Ngrok + Webhook Telegram
# ------------------------------------------------------------------------------------------
.PHONY: setup-ngrok
setup-ngrok:
	@echo "🌐 Configurando Ngrok..."
	bash polaris_setup/setup_ngrok.sh
	@echo "✅ Ngrok e Webhook do Telegram configurados!"

# ------------------------------------------------------------------------------------------
# 📝 Criar .env da API se não existir
# ------------------------------------------------------------------------------------------
.PHONY: create-env-api
create-env-api:
	@echo "📝 Verificando .env da API..."
	@if [ ! -f polaris_api/.env ]; then \
		echo "⚠️  .env da API não encontrado! Criando um novo..."; \
		touch polaris_api/.env; \
		echo "MODEL_PATH=\"../models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf\"" >> polaris_api/.env; \
		echo "NUM_CORES=16" >> polaris_api/.env; \
		echo "MODEL_CONTEXT_SIZE=4096" >> polaris_api/.env; \
		echo "MODEL_BATCH_SIZE=8" >> polaris_api/.env; \
		echo "" >> polaris_api/.env; \
		echo "# Configuração de histórico" >> polaris_api/.env; \
		echo "MONGODB_HISTORY=2" >> polaris_api/.env; \
		echo "LANGCHAIN_HISTORY=10" >> polaris_api/.env; \
		echo "" >> polaris_api/.env; \
		echo "# Hiperparâmetros do modelo" >> polaris_api/.env; \
		echo "TEMPERATURE=0.3" >> polaris_api/.env; \
		echo "TOP_P=0.7" >> polaris_api/.env; \
		echo "TOP_K=70" >> polaris_api/.env; \
		echo "FREQUENCY_PENALTY=3" >> polaris_api/.env; \
		echo "" >> polaris_api/.env; \
		echo "# Configuração do MongoDB" >> polaris_api/.env; \
		echo "MONGO_URI=\"mongodb://admin:admin123@localhost:27017/polaris_db?authSource=admin\"" >> polaris_api/.env; \
		echo "✅ .env da API criado! Edite-o para ajustar os valores."; \
	else \
		echo "✅ .env da API já existe!"; \
	fi

# ------------------------------------------------------------------------------------------
# 📝 Criar .env do Telegram Bot se não existir
# ------------------------------------------------------------------------------------------
.PHONY: create-env-bot
create-env-bot:
	@echo "📝 Verificando .env do Telegram Bot..."
	@if [ ! -f telegram_bot/.env ]; then \
		echo "⚠️  .env do Bot não encontrado! Criando um novo..."; \
		touch telegram_bot/.env; \
		echo "TELEGRAM_TOKEN=\"0000000000:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\"" >> telegram_bot/.env; \
		echo "POLARIS_API_URL=\"http://192.168.2.48:8000/inference/\"" >> telegram_bot/.env; \
		echo "✅ .env do Telegram Bot criado! Edite-o para ajustar os valores."; \
	else \
		echo "✅ .env do Telegram Bot já existe!"; \
	fi


# ------------------------------------------------------------------------------------------
# 🌐 Configuração do Ngrok e Webhook do Telegram
# ------------------------------------------------------------------------------------------

# 🌍 Iniciar o Ngrok e configurar o Webhook do Telegram (executa o script com variáveis exportadas)
.PHONY: setup-ngrok
setup-ngrok:
	@echo "🌐 Exportando variáveis e iniciando Ngrok..."
	@if [ -f telegram_bot/.env ]; then \
		export TELEGRAM_BOT_PORT=8000; \
		export TELEGRAM_TOKEN=$$(grep "^TELEGRAM_TOKEN" telegram_bot/.env | cut -d '=' -f2); \
		bash polaris_setup/scripts/setup_ngrok.sh; \
		echo "✅ Ngrok e Webhook configurados!"; \
	else \
		echo "⚠️ .env do Telegram Bot não encontrado! Certifique-se de rodar 'make create-env-bot' primeiro."; \
		exit 1; \
	fi

# 🛑 Parar Ngrok
.PHONY: stop-ngrok
stop-ngrok:
	@echo "🛑 Parando Ngrok..."
	@pkill -f ngrok || true
	@echo "✅ Ngrok parado!"

# 🔄 Reiniciar Ngrok
.PHONY: restart-ngrok
restart-ngrok:
	@echo "🔄 Reiniciando Ngrok..."
	make stop-ngrok
	sleep 2
	make setup-ngrok
	@echo "✅ Ngrok reiniciado!"


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
