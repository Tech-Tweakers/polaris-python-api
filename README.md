# 🌟 Polaris AI v2 - Assistente Virtual Inteligente

## 📌 Sobre o Projeto
Polaris é um **assistente inteligente** que interage com os usuários via **Telegram**, processando mensagens e fornecendo respostas contextuais utilizando um **modelo de linguagem avançado**. O sistema é baseado em **FastAPI, Llama.cpp, LangChain e MongoDB**, garantindo escalabilidade e precisão nas respostas.

---

## 🚀 Funcionalidades
✅ **Interação via Telegram** – Polaris recebe mensagens e retorna respostas inteligentes.  
✅ **API baseada em FastAPI** – Interface eficiente para comunicação com o backend.  
✅ **Inferência via LLaMA** – Utiliza *Meta-Llama-3-8B-Instruct* para gerar respostas contextuais.  
✅ **Memória Persistente** – Armazena informações importantes no *MongoDB*.  
✅ **Memória Temporária** – Utiliza *LangChain Memory* e *ChromaDB* para contexto recente.  
✅ **Infraestrutura Docker** – Fácil deploy e escalabilidade.  
✅ **Arquitetura modular** – Separação clara entre API, modelo e Bot do Telegram.  
✅ **Logging Estruturado** – Logs detalhados para rastreamento eficiente.  
✅ **Configuração customizável** – Hiperparâmetros ajustáveis via `.env`.  

---

## 🏗️ Arquitetura
Polaris segue o **modelo C4**, organizado nos seguintes módulos:
- **Polaris API** – Processa requisições e interage com o modelo de linguagem.
- **Telegram Bot** – Interface para comunicação com os usuários.
- **MongoDB** – Banco de dados para armazenamento de históricos.
- **LLaMA Model** – Motor de inferência para respostas contextuais.
- **Docker** – Infraestrutura para execução dos serviços.

📖 **[Documentação completa](./docs/README.md)**

---

## 🔧 Como Executar o Projeto
### **1️⃣ Clonar o Repositório**
```bash
git clone https://github.com/seu-usuario/polaris.git
cd polaris
```

### **2️⃣ Criar um Bot no Telegram**
Para conectar o Polaris ao Telegram, siga estes passos:
1. Acesse o **Telegram** e procure por `@BotFather`.
2. Envie o comando `/newbot` e siga as instruções.
3. Escolha um nome e um nome de usuário único para o bot.
4. Após a criação, o BotFather fornecerá um **TOKEN de API**.
5. Copie esse token e adicione no arquivo `.env` conforme o próximo passo.

### **3️⃣ Configurar Variáveis de Ambiente**
Crie um arquivo `.env` e adicione as configurações necessárias:
```env
# Configuração do modelo
MODEL_PATH="../models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
NUM_CORES=16
MODEL_CONTEXT_SIZE=4096
MODEL_BATCH_SIZE=8

# Configuração de histórico
MONGODB_HISTORY=2
LANGCHAIN_HISTORY=10

# Hiperparâmetros do modelo
TEMPERATURE=0.3
TOP_P=0.7
TOP_K=70
FREQUENCY_PENALTY=3

# Configuração do MongoDB
MONGO_URI="mongodb://admin:admin123@localhost:27017/polaris_db?authSource=admin"
```

### **4️⃣ Subir os Containers com Docker**
```bash
docker-compose up -d --build
```

### **5️⃣ Testar a API**
Acesse no navegador ou use `curl`:
```bash
curl http://localhost:8000/ping
```
Saída esperada:
```json
{"message": "Polaris API online!"}
```

### **6️⃣ Testar o Bot do Telegram**
Envie uma mensagem para o bot e verifique a resposta!

---

## 🧪 Executar Testes
Para rodar os testes unitários:
```bash
make test
```

---

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

---

## 📜 Estrutura do Projeto
```bash
📂 polaris
├── 📂 polaris_api          # API FastAPI
├── 📂 telegram_bot         # Bot do Telegram
├── 📂 models               # Modelos LLaMA para inferência
├── 📂 tests                # Testes unitários
├── 📜 docker-compose.yml   # Infraestrutura Docker
├── 📜 Makefile             # Automação de comandos
└── 📜 README.md            # Documentação inicial
```

---

## 📌 Tecnologias Utilizadas
- **Python 3.10**
- **FastAPI**
- **Llama.cpp**
- **LangChain**
- **MongoDB**
- **ChromaDB**
- **Docker & Docker Compose**
- **PlantUML (Documentação C4)**

---

## 📄 Licença
Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para mais detalhes.
