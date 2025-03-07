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

ENV PATH="$VENV_PATH/bin:$PATH"

COPY ./polaris_api/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r requirements.txt

COPY ./polaris_api/ /app

EXPOSE 8000

CMD ["python", "main.py"]
