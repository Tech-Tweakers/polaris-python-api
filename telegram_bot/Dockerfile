FROM python:3.10-slim AS builder

WORKDIR /app

COPY telegram_bot/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt --target /app/deps

FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /app/deps /app/deps

COPY telegram_bot/ /app

ENV PYTHONPATH=/app:/app/deps

EXPOSE 8000

CMD ["python", "main.py"]
