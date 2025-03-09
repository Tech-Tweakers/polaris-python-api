# Usa a imagem base do Python
FROM python:3.10-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia apenas o requirements.txt primeiro (melhor para cache)
COPY polaris_api/requirements.txt /app/requirements.txt

# Instala as dependências
RUN pip install --no-cache-dir --upgrade pip && pip install -r requirements.txt

# Agora copia o resto do código
COPY polaris_api/ /app

# Garante que o Python enxergue a pasta
ENV PYTHONPATH=/app

# Expõe a porta 8000
EXPOSE 8000

# Comando para rodar a API
CMD ["python", "main.py"]
