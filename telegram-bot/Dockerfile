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

RUN python -m venv $VENV_PATH

COPY telegram-bot/requirements.txt /app/requirements.txt
RUN . $VENV_PATH/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install -r requirements.txt

COPY telegram-bot /app

EXPOSE 8001

CMD ["/app/venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
