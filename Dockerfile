# Usa uma imagem base do Python
FROM python:3.10

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia os arquivos do projeto para o container
COPY . /app

# Instala as dependências do projeto
RUN pip install --no-cache-dir --upgrade pip && \
    pip install fastapi uvicorn llama-cpp-python motor

# Expõe a porta 8000
EXPOSE 8000

# Comando para iniciar a API
CMD ["uvicorn", "polaris_api:app", "--host", "0.0.0.0", "--port", "8000"]
