import time
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
from llama_cpp import Llama
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from pymongo import MongoClient
import uvicorn

from colorama import Fore, Style, init

init(autoreset=True)

TEXT_COLOR = Fore.LIGHTCYAN_EX
STAR_COLOR = Fore.YELLOW
DIM_STAR = Fore.LIGHTBLACK_EX

# ASCII ART do Polaris com cores
LOGO = f"""
       {STAR_COLOR}*{Style.RESET_ALL}        .       .   *    .
  .        .    {TEXT_COLOR}POLARIS AI v2{Style.RESET_ALL}       .
       {STAR_COLOR}*{Style.RESET_ALL}        .       *    .  
    .      *       .        .
 {STAR_COLOR}*{Style.RESET_ALL}      .     *         .     
     .     .        .   *    
"""

# Exibir o logo no terminal
print(LOGO)

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

MODEL_PATH = "./models/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
NUM_CORES = 6
MODEL_CONTEXT_SIZE = 4096
MODEL_BATCH_SIZE = 8

MONGODB_HISTORY = 10
LANGCHAIN_HISTORY = 6

# Hiperparâmetros ajustáveis
TEMPERATURE = 0.7  # Criatividade do modelo (0.0 = determinístico, 1.0 = criativo)
TOP_P = 0.9  # Nucleus sampling (ajusta diversidade)
TOP_K = 50  # Limita os k tokens mais prováveis
FREQUENCY_PENALTY = 2.0  # Penaliza repetições

MONGO_URI = "mongodb://admin:admin123@localhost:27017/polaris_db?authSource=admin"
client = MongoClient(MONGO_URI)
db = client["polaris_db"]
collection = db["user_memory"]

app = FastAPI()

log_info("Configurando memória do LangChain...")

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedder)

history = ChatMessageHistory()
memory = ConversationBufferMemory(chat_memory=history, return_messages=True)  # ✅ memory_key REMOVIDO


class LlamaRunnable:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.llm = None

    def load(self):
        try:
            if self.llm is None:
                log_info("Carregando modelo LLaMA...")
                self.llm = Llama(
                    model_path=self.model_path, 
                    n_threads=NUM_CORES, 
                    n_ctx=MODEL_CONTEXT_SIZE, 
                    batch_size=MODEL_BATCH_SIZE, 
                    verbose=False,
                    use_mlock=True
                )
                log_success("Modelo LLaMA carregado com sucesso!")
        except Exception as e:
            log_error(f"Erro ao carregar o modelo LLaMA: {str(e)}")
            raise e

    def close(self):
        if self.llm is not None:
            log_info("Fechando o modelo LLaMA...")
            del self.llm
            self.llm = None
            log_success("Modelo LLaMA fechado com sucesso!")

    def invoke(self, prompt: str):
        if self.llm is None:
            log_error("Erro: Modelo não carregado!")
            raise HTTPException(status_code=500, detail="Modelo não carregado!")

        log_info(f"📜 Enviando prompt ao modelo:\n{prompt}")

        start_time = time.time()
        response = self.llm(
            prompt, 
            stop=["\n", "---"], 
            max_tokens=1024, 
            echo=False,
            temperature=TEMPERATURE,  # 🔥 Aplicando temperatura
            top_p=TOP_P,  # 🔥 Aplicando nucleus sampling
            top_k=TOP_K,  # 🔥 Aplicando top_k sampling
            repeat_penalty=FREQUENCY_PENALTY  # 🔥 Aplicando penalidade de repetição
        )
        end_time = time.time()

        elapsed_time = end_time - start_time
        log_info(f"⚡ Tempo de inferência: {elapsed_time:.3f} segundos")

        if "choices" in response and response["choices"]:
            resposta = response["choices"][0]["text"].strip()
            log_success(f"Resposta gerada pelo modelo: {resposta}")
            return resposta

        log_error("Erro: Resposta do modelo vazia ou inválida!")
        return "Erro ao gerar resposta."

llm = LlamaRunnable(model_path=MODEL_PATH)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Evento de startup
    llm.load()
    yield  # Permite que a aplicação rode enquanto a inicialização ocorre
    # Evento de shutdown
    llm.close()

app = FastAPI(lifespan=lifespan)

class InferenceRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = "default_session"

def get_memories():
    memories = collection.find().sort("timestamp", -1).limit(MONGODB_HISTORY)
    texts = [mem["text"] for mem in memories]
    log_info(f"📌 Recuperadas {len(texts)} memórias do MongoDB.")
    return texts

def get_recent_memories():
    history = memory.load_memory_variables({})["history"]

    if not isinstance(history, list):
        return []

    recent_memories = "\n".join(
        [f"Usuário: {msg.content}" if isinstance(msg, HumanMessage) else f"Polaris: {msg.content}" for msg in history]
    )

    log_info(f"📌 Recuperadas {len(history)} mensagens da memória temporária do LangChain.")
    return recent_memories


def save_to_langchain_memory(user_input, response):
    try:
        memory.save_context({"input": user_input}, {"output": response})
        history = memory.load_memory_variables({})["history"]

        if len(history) > LANGCHAIN_HISTORY:
            log_warning("Memória temporária cheia, removendo mensagens mais antigas...")
            memory.clear()
            for i in range(len(history) - LANGCHAIN_HISTORY, len(history)):
                entry = history[i]
                if isinstance(entry, HumanMessage):
                    memory.save_context({"input": entry.content}, {"output": ""})
                elif isinstance(entry, AIMessage):
                    memory.save_context({"input": "", "output": entry.content})

        log_success("Memória temporária do LangChain atualizada com sucesso!")

    except Exception as e:
        log_error(f"Erro ao salvar na memória temporária do LangChain: {str(e)}")

def save_to_mongo(user_input):
    try:
        existing_entry = collection.find_one({"text": user_input})
        if existing_entry:
            log_warning(f"Entrada duplicada detectada, não será salva: {user_input}")
            return

        doc = {"text": user_input, "timestamp": datetime.utcnow()}
        result = collection.insert_one(doc)
        if result.inserted_id:
            log_success(f"Informação armazenada no MongoDB: {user_input}")

    except Exception as e:
        log_error(f"Erro ao salvar no MongoDB: {str(e)}")

def load_prompt_from_file(file_path="polaris_prompt.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        log_warning(f"Arquivo {file_path} não encontrado! Usando um prompt padrão.")
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
        log_warning(f"Arquivo {file_path} não encontrado! Usando palavras-chave padrão.")
        return ["meu nome é", "eu moro em", "eu gosto de"]

def trim_langchain_memory():
    try:
        history = memory.load_memory_variables({})["history"]

        if not isinstance(history, list):
            return

        if len(history) > LANGCHAIN_HISTORY:
            log_warning("Memória temporária cheia, removendo mensagens mais antigas...")
            memory.chat_memory.messages = history[-LANGCHAIN_HISTORY:]

        log_info("📂 Memória temporária ajustada sem perda de formato!")

    except Exception as e:
        log_error(f"Erro ao ajustar memória temporária do LangChain: {str(e)}")

from langchain.schema import HumanMessage, AIMessage

@app.post("/inference/")
async def inference(request: InferenceRequest):
    session_id = request.session_id or "default_session"
    log_info(f"📥 Nova solicitação de inferência: {request.prompt}")

    keywords = load_keywords_from_file()

    if any(kw in request.prompt.lower() for kw in keywords):
        save_to_mongo(request.prompt)

    trim_langchain_memory()

    mongo_memories = get_memories()
    recent_memories = get_recent_memories()

    context_pieces = []
    if mongo_memories:
        context_pieces.append("📌 Memória do Usuário:\n" + "\n".join(mongo_memories))
    if recent_memories:
        context_pieces.append("📌 Conversa recente:\n" + recent_memories)

    context = "\n\n".join(context_pieces)
    prompt_instrucoes = load_prompt_from_file()
    full_prompt = f"""{prompt_instrucoes}

--- CONTEXTO ---
{context}

--- CONVERSA ATUAL ---
Usuário: {request.prompt}

Polaris:"""

    resposta = llm.invoke(full_prompt)
    memory.save_context({"input": request.prompt}, {"output": resposta})

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
