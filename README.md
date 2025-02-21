# Polaris AI v2

## 🚀 Sobre o Projeto

Polaris é um assistente inteligente desenvolvido utilizando *FastAPI*, *Llama*, *LangChain* e *MongoDB* para oferecer inferências contextuais aprimoradas. Ele possui memória de conversação e armazenamento de memórias de usuários, permitindo respostas mais precisas e personalizadas ao longo do tempo.

## ✨ Recursos Principais
- 🌍 **API REST**: Desenvolvida em *FastAPI* para alta performance.
- 🧠 **Modelo LLaMA**: Utiliza *Meta-Llama-3-8B-Instruct* para inferências inteligentes.
- 🗄️ **Memória Persistente**: Armazena informações importantes no *MongoDB*.
- 🔄 **Memória Temporária**: Utiliza *LangChain Memory* para manter contexto recente.
- 🔥 **Inferência Eficiente**: Implementa otimizações para respostas rápidas e relevantes.
- 🔧 **Logging Estruturado**: Sistema de logs para rastreamento detalhado.
- 🔓 **CORS Middleware**: Suporte a conexões de diversas origens.

## 🏗️ Tecnologias Utilizadas

- **[FastAPI](https://fastapi.tiangolo.com/)** - Framework para desenvolvimento rápido de APIs
- **[Llama (llama-cpp)](https://github.com/ggerganov/llama.cpp)** - Implementação eficiente de modelos LLaMA
- **[LangChain](https://python.langchain.com/)** - Gerenciamento de memória e histórico de conversação
- **[MongoDB](https://www.mongodb.com/)** - Banco de dados NoSQL para persistência de informações
- **[ChromaDB](https://www.trychroma.com/)** - Vetorizador de memórias contextuais
- **[Uvicorn](https://www.uvicorn.org/)** - Servidor ASGI para execução do FastAPI

## 📦 Instalação e Execução

### 📌 Requisitos
- Python 3.9+
- MongoDB rodando na porta padrão (ou configurado em `MONGO_URI`)
- Modelo LLaMA baixado para `./models/`
- Dependências Python instaladas

### ⚙️ Passos para Configuração
1. Clone o repositório:
   ```sh
   git clone https://github.com/Tech-Tweakers/polaris-python-api.git
   cd polaris
   ```

2. Crie e ative um ambiente virtual (opcional, mas recomendado):
   ```sh
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate  # Windows
   ```

3. Instale as dependências:
   ```sh
   pip install -r requirements.txt
   ```

4. Execute a Docker Composer:
   ```sh
   docker-compose up --build -d
   ```

5. Execute a API:
   ```sh
   python polaris_api.py
   ```

## 🔥 Endpoints Disponíveis

### 🔹 Inferência com o Modelo
**POST** `/inference/`
- **Descrição**: Envia um prompt para o modelo e recebe uma resposta inteligente.
- **Body (JSON)**:
  ```json
  {
    "prompt": "Qual é a capital da França?",
    "session_id": "usuario123"
  }
  ```
- **Resposta (JSON)**:
  ```json
  {
    "resposta": "A capital da França é Paris."
  }
  ```

## 🛠️ Estrutura do Projeto
```
polaris/
│── models/                  # Modelos LLaMA
│── chroma_db/               # Base de dados vetorizada do ChromaDB
│── logs/                    # Arquivos de log
│── main.py                  # Código principal
│── requirements.txt         # Dependências
│── polaris_prompt.txt       # Prompt de configuração do modelo
│── keywords.txt             # Palavras-chave para memórias importantes
```

## 📌 Considerações Finais

Polaris foi projetado para ser um assistente poderoso e altamente escalável. Com suporte a memórias persistentes e contexto dinâmico, ele se torna um sistema de IA altamente interativo e personalizável.

💡 Se desejar contribuir com melhorias, sinta-se à vontade para abrir um *Pull Request*! 🚀