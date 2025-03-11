FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY polaris_api/requirements.txt /app/requirements.txt

# Instala a libgomp
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt --target /app/deps

FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /app/deps /app/deps

COPY polaris_api/ /app

ENV PYTHONPATH=/app:/app/deps

EXPOSE 8000

CMD ["python", "main.py"]
