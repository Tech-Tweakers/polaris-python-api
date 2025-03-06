FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VENV_PATH="/app/venv"

RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv $VENV_PATH

COPY ./telegram_bot/requirements.txt /app/requirements.txt

RUN . $VENV_PATH/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install -r requirements.txt

COPY ./telegram_bot/ /app

EXPOSE 8001

CMD ["python", "main.py"]
