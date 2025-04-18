
# 🌟 Polaris AI v2 - Assistente Virtual Inteligente

## 📌 Sobre o Projeto
Polaris é um **assistente inteligente** que interage com os usuários via **Telegram**, processando mensagens e fornecendo respostas contextuais utilizando o modelo **llama3**. O sistema é baseado em **FastAPI, Llama.cpp, LangChain e MongoDB**, garantindo escalabilidade e precisão nas respostas.

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
### **Clonar o Repositório**
```bash
git clone https://github.com/Tech-Tweakers/polaris-python-api.git
cd polaris
```

### **Criar um Bot no Telegram**
Para conectar o Polaris ao Telegram, siga estes passos:
1. Acesse o **Telegram** e procure por `@BotFather`.
2. Envie o comando `/newbot` e siga as instruções.
3. Escolha um nome e um nome de usuário único para o bot.
4. Após a criação, o BotFather fornecerá um **TOKEN de API**.
5. Copie esse token e adicione no arquivo `.env` conforme o próximo passo.

### **Configurar o Bot na pasta `telegram_bot/.env`**
```env
TELEGRAM_API_URL="https://api.telegram.org/bot00000000000:AAFCCCCCCCCCCCCBBJKsH7s"
POLARIS_API_URL="http://192.168.2.48:8000/inference/"
```

### **Configurar Variáveis de Ambiente Polaris API **
Crie um arquivo `polaris_api/.env` e adicione as configurações necessárias:
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

### **Subir os Containers do MongoDB e MongoDB Express com Docker**
```bash
cd polaris_setup
docker-compose up -d --build
```

### **Testar a API**
Utilizando `curl`:
```bash
curl -X POST http://localhost:8000/inference/ \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Qual é a capital da França?", "session_id": "teste123"}'
```
Saída esperada:
```json
{
  "resposta": "A capital da França é Paris."
}
```

### **Testar o Bot do Telegram**
Envie uma mensagem para o bot e verifique a resposta!

---

## 🗂️ Estrutura do Projeto
```bash
📂 polaris_api
│   ├── main.py           # Lógica da API FastAPI
│   ├── .env              # Variáveis de ambiente
│   ├── polaris_prompt.txt
│   └── requirements.txt
📂 telegram_bot
│   ├── main.py           # Lógica do Bot
│   ├── .env
│   └── requirements.txt
📂 polaris_setup
│   ├── docker-compose.yml
│   ├── setup_ngrok.sh
│   └── polaris-os-tunner.sh
📂 models                 # Modelos LLaMA
📂 docs                   # Documentação
📂 tests                  # Testes automatizados
📄 Makefile               # Comandos automatizados
📄 local-setup.sh         # Menu interativo de instalação
```

---

## 🛠️ Setup Interativo e Automação

Para facilitar a instalação e o uso da Polaris, o projeto conta com um menu interativo (`local-setup.sh`) e um `Makefile` com comandos automatizados. Essa abordagem reduz a complexidade da configuração manual e garante que todos os componentes sejam inicializados corretamente.

### ✅ Pré-requisitos
- Python 3.10+
- pip
- docker e docker-compose

### 📦 Instalação com menu interativo
```bash
./local-setup.sh
```

### 📋 Opções disponíveis

| Opção | Descrição |
|-------|-----------|
| 1     | Instala dependências via `pip` |
| 2     | Cria arquivos `.env` para API e Bot |
| 3     | Baixa o modelo Meta-LLaMA-3 |
| 4     | Sobe MongoDB e Mongo Express |
| 5     | Configura Ngrok e Webhook |
| 6     | Inicia API |
| 7     | Inicia Bot |
| 8     | Inicia tudo |
| 9     | Para tudo |
| 10    | Reinicia tudo |

### ⚙️ Comandos manuais úteis
```bash
make install
make create-env-api
make create-env-bot
make download-model
make start-db
make start-api
make start-bot
make start-all
make stop-all
```

---

## 📄 Licença
Este projeto está licenciado sob a **MIT License**.