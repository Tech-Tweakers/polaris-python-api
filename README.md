# Polaris AI v2

## 🚀 Sobre o Projeto

Polaris é um assistente inteligente desenvolvido utilizando *FastAPI*, *Llama.cpp*, *LangChain* e *MongoDB* para oferecer inferências contextuais aprimoradas. Ele possui memória de conversação e armazenamento de memórias de usuários, permitindo respostas mais precisas e personalizadas ao longo do tempo.

## ✨ Recursos Principais

- 🌍 **API REST**: Desenvolvida em *FastAPI* para alta performance.
- 🤖 **Integração com Telegram**: O bot do Telegram interage diretamente com a Polaris para responder perguntas em tempo real.
- 🧠 **Modelo LLaMA**: Utiliza *Meta-Llama-3-8B-Instruct* para respostas inteligentes e contextuais.
- 🗄️ **Memória Persistente**: Armazena informações importantes no *MongoDB*.
- 🔄 **Memória Temporária**: Utiliza *LangChain Memory* e *ChromaDB* para manter contexto recente.
- 🔥 **Inferência Eficiente**: Implementa otimizações para respostas rápidas e relevantes.
- 🔧 **Logging Estruturado**: Sistema de logs para rastreamento detalhado.

## 🏗️ Tecnologias Utilizadas

- [**FastAPI**](https://fastapi.tiangolo.com/) - Framework para desenvolvimento rápido de APIs
- [**Llama.cpp**](https://github.com/ggerganov/llama.cpp) - Implementação eficiente de modelos LLaMA
- [**LangChain**](https://python.langchain.com/) - Gerenciamento de memória e histórico de conversação
- [**MongoDB**](https://www.mongodb.com/) - Banco de dados NoSQL para persistência de informações
- [**ChromaDB**](https://www.trychroma.com/) - Vetorizador de memórias contextuais para busca eficiente
- [**Uvicorn**](https://www.uvicorn.org/) - Servidor ASGI para execução do FastAPI
- [**Telegram Bot API**](https://core.telegram.org/bots/api) - API de bots do Telegram para interatividade em tempo real

## 📦 Instalação e Execução

### 📌 Requisitos

- Python 3.9+
- MongoDB rodando na porta padrão (ou configurado em `MONGO_URI`)
- Modelo LLaMA baixado para `./models/`
- Dependências Python instaladas
- **Token do Telegram Bot** configurado em variáveis de ambiente

### ⚙️ Passos para Configuração

1. Clone o repositório:

   ```sh
   git clone https://github.com/Tech-Tweakers/polaris-python-api.git
   cd polaris-python
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

4. Execute o Docker Composer (se necessário para MongoDB e ChromaDB):

   ```sh
   docker-compose up --build -d
   ```

5. **Inicie a API Polaris**:

   ```sh
   python polaris-api/polaris_api.py
   ```

6. **Inicie o Bot do Telegram**:

   ```sh
   python telegram-bot/main.py
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

### 🔹 Webhook do Telegram

**POST** `/telegram-webhook/`

- **Descrição**: Recebe mensagens do Telegram e retorna respostas geradas pela Polaris.
- **Body (JSON)**:
  ```json
  {
    "update_id": 123456,
    "message": {
      "chat": { "id": 987654321 },
      "text": "Olá, Polaris!"
    }
  }
  ```
- **Resposta (JSON)**:
  ```json
  {
    "status": "ok"
  }
  ```

## 🛠️ Estrutura do Projeto

```
polaris-python/
│── polaris-api/              # Código principal da API
│   │── Dockerfile
│   │── keywords.txt          # Palavras-chave para memórias importantes
│   │── polaris_api.py        # Código principal da API Polaris
│   │── polaris_prompt.txt    # Prompt de configuração do modelo
│   │── requirements.txt      # Dependências da API
│
│── telegram-bot/             # Módulo do bot do Telegram
│   │── Dockerfile
│   │── main.py               # Código principal do bot
│   │── requirements.txt      # Dependências do bot
│
│── .gitignore
│── docker-compose.yml        # Configuração do Docker Compose
│── README.md                 # Documentação do projeto
```

## 📌 Considerações Finais

Polaris foi projetado para ser um assistente poderoso e altamente escalável. Atualmente, a inferência é feita diretamente no backend Python, mas há planos de migrar para um servidor LLaMA.cpp dedicado para maior eficiência.

Além disso, a integração com o **Telegram Bot** permite interatividade em tempo real, ampliando a usabilidade do projeto.

💡 Se desejar contribuir com melhorias, sinta-se à vontade para abrir um *Pull Request*! 🚀