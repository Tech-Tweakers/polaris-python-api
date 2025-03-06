FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VENV_PATH="/app/venv"

RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Criar e ativar o ambiente virtual
RUN python -m venv $VENV_PATH

# Definir o Python do virtualenv como padrão
ENV PATH="$VENV_PATH/bin:$PATH"

# Copiar os arquivos de dependências primeiro para evitar reinstalação desnecessária
COPY ./polaris_api/requirements.txt /app/requirements.txt

# Instalar dependências
RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r requirements.txt

# Copiar o restante da aplicação
COPY ./polaris_api/ /app

EXPOSE 8000

# Definir o comando padrão
CMD ["python", "main.py"]
