# Usa a imagem base do Python
FROM python:3.10-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia apenas o requirements.txt primeiro (isso melhora o cache do Docker)
COPY polaris_api/requirements.txt /app/requirements.txt

# Instala as dependências antes de copiar o resto do código
RUN pip install --no-cache-dir --upgrade pip && pip install -r requirements.txt

# Agora copia o resto do código
COPY polaris_api/ /app

# Garante que o Python enxergue os módulos corretamente
ENV PYTHONPATH=/app

# Expõe a porta 8000 para a API
EXPOSE 8000

# Comando para rodar a API
CMD ["python", "main.py"]
