import asyncio
import os
import time  # 📌 Importado para medir tempo de inferência
import traceback
import logging
import faiss
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from llama_cpp import Llama
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
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

# 🔹 Configuração do modelo e FAISS
MODEL_PATH = "./models/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
FAISS_INDEX_PATH = "./faiss/faiss_memory.index"

NUM_CORES = 6
MODEL_CONTEXT_SIZE = 512
MODEL_BATCH_SIZE = 8
VECTOR_DIM = 384  # Dimensão correta para MiniLM

# 🔹 Inicializa FastAPI primeiro para evitar erro de rota
app = FastAPI()

# 🔹 Inicializar FAISS e Embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
    log_success("🔹 FAISS carregado do disco!")
else:
    index = faiss.IndexFlatL2(VECTOR_DIM)
    log_warning("⚠️ Nenhuma memória FAISS encontrada, criando um novo índice.")

# 🔹 Banco de dados local para histórico textual
database = []

# 🔹 Modelo de inferência
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

# 🔹 Função para formatar tempo em MM:SS.sss
def format_time(milliseconds):
    seconds = milliseconds / 1000
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes:02}:{seconds:06.3f}"  # Exemplo: 00:02.345 (2 segundos e 345ms)

# 🔹 Classe para Llama
class LlamaLLM:
    _instance = None
    def __new__(cls, model_path: str):
        if cls._instance is None:
            cls._instance = super(LlamaLLM, cls).__new__(cls)
            cls._instance.model_path = model_path
            cls._instance.llm = None
        return cls._instance

    def load(self):
        if self.llm is None:
            log_info("🔹 Carregando modelo LLaMA...")
            self.llm = Llama(model_path=self.model_path, n_threads=NUM_CORES, n_ctx=512, batch_size=MODEL_BATCH_SIZE)
            log_success("✅ Modelo LLaMA carregado com sucesso!")

    def call(self, user_prompt: str, **kwargs) -> str:
        if self.llm is None:
            log_error("❌ Erro: Modelo não carregado!")
            raise HTTPException(status_code=500, detail="Modelo não carregado!")

        start_time = time.time()  # 📌 Marca o tempo antes da inferência
        full_prompt = f"Usuário: {user_prompt}\nPolaris:"
        response = self.llm(full_prompt, stop=["\n"], max_tokens=100, echo=False)
        end_time = time.time()  # 📌 Marca o tempo após a inferência

        inference_time_ms = (end_time - start_time) * 1000  # Tempo em milissegundos
        formatted_time = format_time(inference_time_ms)  # Converte para MM:SS.sss

        log_info(f"⚡ Tempo de inferência: {formatted_time}")  # 📌 Log formatado

        return response["choices"][0]["text"].strip()

llm = LlamaLLM(model_path=MODEL_PATH)

# 🔹 Carrega o modelo antes do primeiro uso
@app.on_event("startup")
async def startup_event():
    llm.load()

# 🔹 Memória FAISS
def salvar_conversa(user_input, resposta):
    resumo = f"Usuário: {user_input} | Polaris: {resposta}"
    embedding = embedder.encode([resumo]).astype(np.float32)
    
    index.add(embedding)
    database.append(resumo)

    faiss.write_index(index, FAISS_INDEX_PATH)
    log_success("✅ Conversa salva no FAISS!")

def buscar_contexto(consulta):
    if len(database) == 0:
        return []  # Evita erro se FAISS estiver vazio
    
    embedding = embedder.encode([consulta]).astype(np.float32)
    distancias, indices = index.search(embedding, k=5)
    
    return [database[i] for i in indices[0] if i < len(database) and i >= 0]

# 🔹 Rota principal para inferência
@app.post("/inference/")
async def inference(request: InferenceRequest):
    contexto = buscar_contexto(request.prompt)
    contexto_formatado = "\n".join([f"Usuário: {c.split('|')[0].strip()} | Polaris: {c.split('|')[1].strip()}" for c in contexto if '|' in c])
    
    prompt_final = f"{contexto_formatado}\nUsuário: {request.prompt}\nPolaris:"
    resposta = llm.call(prompt_final).strip()

    salvar_conversa(request.prompt, resposta)

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
