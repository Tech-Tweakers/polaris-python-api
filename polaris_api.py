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
LOG_FILE = "polaris.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def log_info(message: str): logging.info(f"🔹 {message}")
def log_success(message: str): logging.info(f"✅ {message}")
def log_warning(message: str): logging.warning(f"⚠️ {message}")
def log_error(message: str): logging.error(f"❌ {message}")

# 🔹 Configuração do modelo
MODEL_PATH = "./models/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
NUM_CORES = 6
MODEL_CONTEXT_SIZE = 4096
MODEL_BATCH_SIZE = 8

# 🔹 Conexão com o MongoDB
MONGO_URI = "mongodb://admin:admin123@localhost:27017/polaris_db?authSource=admin"
client = MongoClient(MONGO_URI)
db = client["polaris_db"]
collection = db["user_memory"]

# 🔹 Inicializa FastAPI
app = FastAPI()

# 🔹 Inicializa LangChain Memory
log_info("🔹 Configurando memória do LangChain...")

embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedder)
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# 🔹 Classe para Llama
class LlamaRunnable:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.llm = None

    def load(self):
        try:
            if self.llm is None:
                log_info("🔹 Carregando modelo LLaMA...")
                self.llm = Llama(model_path=self.model_path, n_threads=NUM_CORES, n_ctx=MODEL_CONTEXT_SIZE, batch_size=MODEL_BATCH_SIZE, verbose=False)
                log_success("✅ Modelo LLaMA carregado com sucesso!")
        except Exception as e:
            log_error(f"❌ Erro ao carregar o modelo LLaMA: {str(e)}")
            raise e

    def close(self):
        if self.llm is not None:
            log_info("🔹 Fechando o modelo LLaMA...")
            del self.llm
            self.llm = None
            log_success("✅ Modelo LLaMA fechado com sucesso!")

    def invoke(self, prompt: str):
        if self.llm is None:
            log_error("❌ Erro: Modelo não carregado!")
            raise HTTPException(status_code=500, detail="Modelo não carregado!")

        log_info(f"📜 Enviando prompt ao modelo:\n{prompt}")

        start_time = time.time()
        response = self.llm(prompt, stop=["\n"], max_tokens=100, echo=False)
        end_time = time.time()

        elapsed_time = end_time - start_time
        log_info(f"⚡ Tempo de inferência: {elapsed_time:.3f} segundos")

        if "choices" in response and response["choices"]:
            resposta = response["choices"][0]["text"].strip()
            log_success(f"✅ Resposta gerada pelo modelo: {resposta}")
            return resposta

        log_error("❌ Erro: Resposta do modelo vazia ou inválida!")
        return "Erro ao gerar resposta."

llm = LlamaRunnable(model_path=MODEL_PATH)

@app.on_event("startup")
async def startup_event():
    llm.load()

@app.on_event("shutdown")
async def shutdown_event():
    llm.close()

class InferenceRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = "default_session"

# 🔹 Recuperar memórias do MongoDB
def get_memories():
    memories = collection.find().sort("timestamp", -1).limit(5)
    texts = [mem["text"] for mem in memories]
    log_info(f"📌 Recuperadas {len(texts)} memórias do MongoDB.")
    return texts

# 🔹 Recuperar contexto do ChromaDB
def get_similar_memories(prompt):
    retrieved_docs = vectorstore.similarity_search(prompt, k=3)
    docs = [doc.page_content for doc in retrieved_docs]
    log_info(f"📌 Recuperadas {len(docs)} memórias semânticas do ChromaDB.")
    return docs

# 🔹 Armazenar informações no MongoDB
def save_to_mongo(user_input):
    """Salva informações no MongoDB, evitando entradas duplicadas."""
    try:
        # Verifica se a entrada já existe no banco
        existing_entry = collection.find_one({"text": user_input})
        if existing_entry:
            log_warning(f"⚠️ Entrada duplicada detectada, não será salva: {user_input}")
            return  # Se já existe, não salva de novo

        # Caso não exista, insere no banco
        doc = {"text": user_input, "timestamp": datetime.utcnow()}
        result = collection.insert_one(doc)
        if result.inserted_id:
            log_success(f"✅ Informação armazenada no MongoDB: {user_input}")
    except Exception as e:
        log_error(f"❌ Erro ao salvar no MongoDB: {str(e)}")

# 🔹 Carrega o prompt de instrução do arquivo
def load_prompt_from_file(file_path="polaris_prompt.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        log_warning(f"⚠️ Arquivo {file_path} não encontrado! Usando um prompt padrão.")
        return """\
        ### Instruções:
        Você é Polaris, um assistente inteligente.
        Responda de forma clara e objetiva, utilizando informações do histórico e memórias disponíveis.
        Se não souber a resposta, seja honesto e não invente informações.

        Agora, aqui está a conversa atual:
        """

def load_keywords_from_file(file_path="keywords.txt"):
    """Carrega a lista de palavras-chave do arquivo especificado."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            keywords = [line.strip().lower() for line in file.readlines() if line.strip()]
            log_info(f"📂 Palavras-chave carregadas do arquivo ({len(keywords)} palavras).")
            return keywords
    except FileNotFoundError:
        log_warning(f"⚠️ Arquivo {file_path} não encontrado! Usando palavras-chave padrão.")
        return ["meu nome é", "eu moro em", "eu gosto de"]


from langchain.schema import HumanMessage, AIMessage

@app.post("/inference/")
async def inference(request: InferenceRequest):
    session_id = request.session_id or "default_session"
    log_info(f"📥 Nova solicitação de inferência: {request.prompt}")

    keywords = load_keywords_from_file()

    if any(kw in request.prompt.lower() for kw in keywords):
        save_to_mongo(request.prompt)

    # 🔹 Recupera memórias de longo prazo
    mongo_memories = get_memories()
    chroma_memories = get_similar_memories(request.prompt)
    memory_short = memory.load_memory_variables({})["history"]

    if not isinstance(memory_short, list):
        memory_short = []

    # 🔹 Organiza a memória curta corretamente
    short_memory_formatted = "\n".join(
        [f"Usuário: {msg.content}" if isinstance(msg, HumanMessage) else f"Polaris: {msg.content}" for msg in memory_short]
    )

    # 🔹 Constrói o contexto separado por blocos
    context_pieces = []
    
    if mongo_memories:
        context_pieces.append("📌 Memória do Usuário (extraída do banco de dados):\n" + "\n".join(mongo_memories))
    
    if chroma_memories:
        context_pieces.append("📌 Informações relevantes (ChromaDB):\n" + "\n".join(chroma_memories))
    
    if short_memory_formatted:
        context_pieces.append("📌 Conversa recente:\n" + short_memory_formatted)

    # 🔹 Junta tudo em um contexto bem formatado
    context = "\n\n".join(context_pieces)

    # 🔹 Carrega o prompt de instrução
    prompt_instrucoes = load_prompt_from_file()

    # 🔹 Constrói o prompt final
    full_prompt = f"""{prompt_instrucoes}

--- CONTEXTO ---
{context}

--- CONVERSA ATUAL ---
Usuário: {request.prompt}

Polaris:"""

    log_info(f"📜 Prompt final gerado:\n{full_prompt}")

    # 🔹 Faz a inferência com base no prompt ajustado
    resposta = llm.invoke(full_prompt)

    # 🔹 Salva a resposta na memória curta
    memory.save_context({"input": request.prompt}, {"output": resposta})

    # 🔹 Verifica se o input contém informações importantes para armazenar no MongoDB
    if any(kw in request.prompt.lower() for kw in ["meu nome é", "eu moro em", "eu gosto de"]):
        save_to_mongo(request.prompt)

    return {"resposta": resposta}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
