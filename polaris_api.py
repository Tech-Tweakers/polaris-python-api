import asyncio
import os
import time
import traceback
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from llama_cpp import Llama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from pymongo import MongoClient
import uvicorn

# 🔹 Configuração do log
class LogColors:
    INFO = "\033[94m"
    SUCCESS = "\033[92m"
    WARNING = "\033[93m"
    ERROR = "\033[91m"
    RESET = "\033[0m"

def log_info(message: str): logging.info(f"{LogColors.INFO}🔹 {message}{LogColors.RESET}")
def log_success(message: str): logging.info(f"{LogColors.SUCCESS}✅ {message}{LogColors.RESET}")
def log_warning(message: str): logging.warning(f"{LogColors.WARNING}⚠️ {message}{LogColors.RESET}")
def log_error(message: str): logging.error(f"{LogColors.ERROR}❌ {message}{LogColors.RESET}")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔹 Configuração do modelo
MODEL_PATH = "./models/Meta-Llama-3-8B-Instruct.Q4_0.gguf"

NUM_CORES = 6
MODEL_CONTEXT_SIZE = 2048
MODEL_BATCH_SIZE = 8

# 🔹 Conexão com o MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["polaris_db"]
collection = db["user_memory"]

# 🔹 Inicializa FastAPI
app = FastAPI()

# 🔹 Inicializa LangChain Memory
log_info("🔹 Configurando memória do LangChain...")

embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedder)

# 🔹 Memória de conversação curta
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# 🔹 Classe para Llama (Agora um `Runnable`)
class LlamaRunnable:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.llm = None

    def load(self):
        if self.llm is None:
            log_info("🔹 Carregando modelo LLaMA...")
            self.llm = Llama(model_path=self.model_path, n_threads=NUM_CORES, n_ctx=4096, batch_size=MODEL_BATCH_SIZE)
            log_success("✅ Modelo LLaMA carregado com sucesso!")

    def close(self):
        """ Fecha corretamente o modelo LLaMA para evitar erro de finalização. """
        if self.llm is not None:
            log_info("🔹 Fechando o modelo LLaMA...")
            del self.llm  # Remove o objeto para liberar a memória
            self.llm = None
            log_success("✅ Modelo LLaMA fechado com sucesso!")

    def invoke(self, messages):
        if self.llm is None:
            log_error("❌ Erro: Modelo não carregado!")
            raise HTTPException(status_code=500, detail="Modelo não carregado!")

        # 🔹 Construir prompt baseado no histórico
        context = "\n".join([msg["content"] if isinstance(msg, dict) else msg.content for msg in messages])

        # 🔹 Formatar prompt para IA
        full_prompt = f"Usuário: {context}\nPolaris:"

        # 🔹 Mede tempo de inferência
        start_time = time.time()
        response = self.llm(full_prompt, stop=["\n"], max_tokens=100, echo=False)
        end_time = time.time()

        # 🔹 Correção do cálculo do tempo de inferência
        elapsed_time = end_time - start_time
        formatted_time = f"{int(elapsed_time // 3600):02}:{int((elapsed_time % 3600) // 60):02}:{elapsed_time % 60:06.3f}"
        log_info(f"⚡ Tempo de inferência: {formatted_time}")

        # 🔹 Verifica se a resposta do modelo existe antes de acessá-la
        if "choices" in response and response["choices"]:
            return response["choices"][0]["text"].strip()

        log_error("❌ Erro: Resposta do modelo vazia ou inválida!")
        return "Erro ao gerar resposta."

llm = LlamaRunnable(model_path=MODEL_PATH)

# 🔹 Carrega o modelo antes do primeiro uso
@app.on_event("startup")
async def startup_event():
    llm.load()

@app.on_event("shutdown")
async def shutdown_event():
    llm.close()

# 🔹 Modelo de inferência
class InferenceRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = "default_session"

# 🔹 Recuperar memórias do MongoDB
def get_memories():
    memories = collection.find().sort("timestamp", -1).limit(5)
    return [mem["text"] for mem in memories]

# 🔹 Recuperar contexto semântico do ChromaDB
def get_similar_memories(prompt):
    retrieved_docs = vectorstore.similarity_search(prompt, k=3)
    return [doc.page_content for doc in retrieved_docs]

# 🔹 Armazenar informações no MongoDB
def save_to_mongo(user_input):
    doc = {"text": user_input, "timestamp": datetime.utcnow()}
    collection.insert_one(doc)

@app.post("/inference/")
async def inference(request: InferenceRequest):
    session_id = request.session_id or "default_session"

    # 🔹 Recupera memórias de longo prazo
    mongo_memories = get_memories()
    chroma_memories = get_similar_memories(request.prompt)
    memory_short = memory.load_memory_variables({})["history"]

    if not isinstance(memory_short, list):
        memory_short = []

    # 🔹 Monta contexto com MongoDB + ChromaDB + Memória curta
    context = "\n".join(mongo_memories + chroma_memories + [msg["content"] for msg in memory_short])

    # 🔹 Faz a inferência com base no histórico atualizado
    resposta = llm.invoke([{"role": "user", "content": context}])

    # 🔹 Salva resposta na memória curta
    memory.save_context({"input": request.prompt}, {"output": resposta})

    # 🔹 Verifica se o prompt contém informações importantes para armazenar no MongoDB
    if any(kw in request.prompt.lower() for kw in [
        "meu nome é", "eu moro em", "eu gosto de", "minha profissão é",
        "tenho um amigo chamado", "minha esposa se chama", "viajei para",
        "meu objetivo é", "meu sonho é", "eu torço para"
    ]):
        save_to_mongo(request.prompt)

    return {"resposta": resposta}

# 🔹 Aplicando CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Rodar servidor
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
