import asyncio
import os
import traceback
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from llama_cpp import Llama
from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager
import uvicorn
import multiprocessing

# 🔹 Configuração do Logging (Colorido e Organizado)
class LogColors:
    INFO = "\033[94m"   # Azul
    SUCCESS = "\033[92m" # Verde
    WARNING = "\033[93m" # Amarelo
    ERROR = "\033[91m"   # Vermelho
    RESET = "\033[0m"    # Reset de cor

def log_info(message: str):
    print(f"{LogColors.INFO}🔹 {message}{LogColors.RESET}")

def log_success(message: str):
    print(f"{LogColors.SUCCESS}✅ {message}{LogColors.RESET}")

def log_warning(message: str):
    print(f"{LogColors.WARNING}⚠️ {message}{LogColors.RESET}")

def log_error(message: str):
    print(f"{LogColors.ERROR}❌ {message}{LogColors.RESET}")

# 🔹 Configurações Gerais
MODEL_PATH = "./models/Meta-Llama-3-8B-Instruct.Q2_K.gguf"
PROMPT_FILE = "polaris_prompt.txt"
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_URI = f"mongodb://admin:adminpassword@{MONGO_HOST}:27017/"
DATABASE_NAME = "polaris_db"

# 🔹 Ajustes de Desempenho para Máquinas Fracas
NUM_CORES = 8
MODEL_CONTEXT_SIZE = 512  # 🔥 Para evitar consumo excessivo de RAM
MODEL_BATCH_SIZE = 32  # 🔥 Ajustado para balancear performance

# 🔹 Inicializa o banco de dados (opcional)
client = None
collection = None
mongo_available = False

async def connect_mongo():
    """Tenta conectar ao MongoDB e atualizar o status global."""
    global client, collection, mongo_available
    log_info("🔹 Tentando conectar ao MongoDB...")
    try:
        client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        await client.server_info()  # Testa conexão
        collection = client[DATABASE_NAME]["inferences"]
        mongo_available = True
        log_success("✅ Conectado ao MongoDB!")
    except Exception as e:
        mongo_available = False
        log_warning(f"⚠️ MongoDB não disponível! Rodando em modo offline. Erro: {e}")

# 🔹 Modelo de dados para API
class InferenceRequest(BaseModel):
    prompt: str
    stop_words: Optional[List[str]] = None
    temperature: Optional[float] = 0.9
    top_p: Optional[float] = 0.5
    top_k: Optional[int] = 40
    frequency_penalty: Optional[float] = 2.0
    presence_penalty: Optional[float] = 1.5
    max_tokens: Optional[int] = 64
    session_id: Optional[str] = None

# 🔹 Classe Singleton do Modelo
class LlamaLLM:
    _instance = None

    def __new__(cls, model_path: str, prompt_file: str):
        if cls._instance is None:
            cls._instance = super(LlamaLLM, cls).__new__(cls)
            cls._instance.model_path = model_path
            cls._instance.prompt_base = cls.read_prompt(prompt_file)
            cls._instance.llm = None
        return cls._instance

    @staticmethod
    def read_prompt(file_path: str, max_length: int = 300) -> str:
        """Lê o arquivo de prompt e limita o tamanho"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                prompt = file.read().strip()
                return " ".join(prompt.split()[:max_length]) if len(prompt.split()) > max_length else prompt
        except Exception as e:
            log_error(f"Erro ao ler o arquivo {file_path}: {e}")
            return "Você é um assistente de IA útil. Responda com clareza."

    def load(self):
        """Carrega o modelo Llama apenas uma vez e faz uma chamada de aquecimento"""
        if self.llm is None:
            try:
                log_info(f"🔹 Carregando modelo de: {self.model_path}...")
                self.llm = Llama(
                    model_path=self.model_path,
                    verbose=True,
                    n_threads=NUM_CORES,
                    n_ctx=2048,
                    n_ctx_per_seq=1024,
                    batch_size=MODEL_BATCH_SIZE
                )
                log_success("✅ Modelo carregado com sucesso!")

                # 🔥 Chamada de aquecimento
                log_info("☀️ Esquentando o modelo...")
                warmup_response = self.call("Acorda Polaris, já amanheceu!")
                log_success(f"🌞 Polaris acordou! Resposta de aquecimento: {warmup_response}")

            except Exception as e:
                log_error(f"Erro ao carregar o modelo: {e}\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail="Erro ao carregar o modelo Llama")

    def call(self, user_prompt: str, **kwargs) -> str:
        """Chama o modelo e retorna a resposta utilizando os parâmetros do front"""
        if self.llm is None:
            raise HTTPException(status_code=500, detail="Modelo ainda não carregado!")

        try:
            full_prompt = f"{self.prompt_base}\n\nPergunta: {user_prompt}"
            log_info(f"📩 Prompt enviado ao modelo:\n{full_prompt}")

            start_time = datetime.now()
            
            response = self.llm(
                full_prompt,
                stop=kwargs.get("stop_words", ["Pergunta:", "Pergunte:", "\n```\n"]),
                max_tokens=kwargs.get("max_tokens", 128),  # Agora usa o valor do front se enviado
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 50),
                frequency_penalty=kwargs.get("frequency_penalty", 1.0),
                presence_penalty=kwargs.get("presence_penalty", 1.2)
            )

            elapsed_time = (datetime.now() - start_time).total_seconds()

            raw_answer = response["choices"][0]["text"].strip() if response and "choices" in response and response["choices"] else "[Erro: Modelo retornou resposta vazia]"
            log_success(f"🎯 Resposta do modelo em {elapsed_time:.2f}s: {raw_answer}")
            return raw_answer
        except Exception as e:
            log_error(f"Erro durante a inferência: {e}\n{traceback.format_exc()}")
            return "[Erro: Falha na inferência]"

# 🔹 Lifespan Handler para Inicialização
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa a API corretamente e evita DeprecationWarning."""
    log_info("🚀 Iniciando a API...")
    await connect_mongo()
    llm.load()
    log_success("🔥 API pronta para uso!")
    yield
    log_warning("🛑 API sendo desligada...")

# 🔹 Inicializa a API com Lifespan
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Instancia o modelo
llm = LlamaLLM(model_path=MODEL_PATH, prompt_file=PROMPT_FILE)

@app.post("/inference/")
async def inference(request: InferenceRequest):
    """Gera resposta e mantém contexto no MongoDB"""
    try:
        session_data = await collection.find_one({"session_id": request.session_id}) if mongo_available else None

        # Resgatar histórico da conversa, se existir
        history = session_data.get("history", []) if session_data else []

        # Adicionar a pergunta ao histórico
        history.append({"role": "user", "content": request.prompt})

        # Passar histórico para o modelo
        context_prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])

        answer = llm.call(context_prompt)

        # Adicionar resposta ao histórico
        history.append({"role": "bot", "content": answer})

        if mongo_available:
            await collection.update_one(
                {"session_id": request.session_id},
                {"$set": {"history": history}},
                upsert=True
            )
            log_success("💾 Histórico da sessão atualizado no MongoDB.")

        return {"resposta": answer}
    except Exception as e:
        log_error(f"Erro inesperado: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Erro interno na API")

if __name__ == "__main__":
    log_success("🔥 Iniciando servidor FastAPI...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
