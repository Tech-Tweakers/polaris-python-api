# Usa uma imagem base do Python
FROM python:3.10-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Define a variável de ambiente para evitar criação de cache do pip
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VENV_PATH="/app/venv"

# Instala dependências básicas do sistema para o FAISS
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O Meta-Llama-3-8B-Instruct.Q4_0.gguf https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_0.gguf

# Cria um ambiente virtual para evitar conflitos de pacotes
RUN python -m venv $VENV_PATH

# Ativa o ambiente virtual e instala as dependências
COPY requirements.txt /app/requirements.txt
RUN . $VENV_PATH/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install -r requirements.txt

# Copia os arquivos do projeto para o container
COPY . /app

# Expõe a porta 8000
EXPOSE 8000

# Comando para iniciar a API usando o ambiente virtual
CMD ["/app/venv/bin/uvicorn", "polaris_api:app", "--host", "0.0.0.0", "--port", "8000"]
