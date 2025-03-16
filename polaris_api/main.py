import time
import logging
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
from pydantic import Field, validator
from llama_cpp import Llama
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from pymongo import MongoClient
import uvicorn
import os
from colorama import Fore, Style, init

init(autoreset=True)
TEXT_COLOR = Fore.LIGHTCYAN_EX
STAR_COLOR = Fore.YELLOW
DIM_STAR = Fore.LIGHTBLACK_EX
LOGO = f"""
       {STAR_COLOR}*{Style.RESET_ALL}        .       *    .  
    .      *       .        .
       {STAR_COLOR}*{Style.RESET_ALL}        .       .   *    .
  .        .  {TEXT_COLOR}POLARIS AI v2{Style.RESET_ALL}        .
       {STAR_COLOR}*{Style.RESET_ALL}        .        *     .  
    .       *        .        .
 {STAR_COLOR}*{Style.RESET_ALL}      .     *         .     
     .     .        .    *    
"""
print(LOGO)

LOG_FILE = "polaris.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)


def log_info(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"🔹 [{timestamp}] {message}")


def log_success(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"✅ [{timestamp}] {message}")


def log_warning(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.warning(f"⚠️ [{timestamp}] {message}")


def log_error(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.error(f"❌ [{timestamp}] {message}")


load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
NUM_CORES = int(os.getenv("NUM_CORES", 4))
MODEL_CONTEXT_SIZE = int(os.getenv("MODEL_CONTEXT_SIZE", 512))
MODEL_BATCH_SIZE = int(os.getenv("MODEL_BATCH_SIZE", 8))

MONGODB_HISTORY = int(os.getenv("MONGODB_HISTORY", 0))
LANGCHAIN_HISTORY = int(os.getenv("LANGCHAIN_HISTORY", 0))

TEMPERATURE = float(os.getenv("TEMPERATURE", 0.5))
TOP_P = float(os.getenv("TOP_P", 0.7))
TOP_K = int(os.getenv("TOP_K", 40))
FREQUENCY_PENALTY = int(os.getenv("FREQUENCY_PENALTY", 3))

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["polaris_db"]
collection = db["user_memory"]
memory_store = {}

app = FastAPI()

log_info("Configurando memória do LangChain...")

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedder)

history = ChatMessageHistory()
memory = ConversationBufferMemory(chat_memory=history, return_messages=True)


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
                    use_mlock=True,
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
            stop=["---"],
            max_tokens=1024,
            echo=False,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repeat_penalty=FREQUENCY_PENALTY,
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
    llm.load()
    store_instruction_prompt()  # Armazena o prompt vetorizado ao iniciar
    yield
    llm.close()


app = FastAPI(lifespan=lifespan)


class InferenceRequest(BaseModel):
    prompt: str = Field(
        ..., min_length=1, description="Texto de entrada para inferência"
    )
    session_id: Optional[str] = "default_session"

    @validator("prompt")
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError("⚠️ O prompt não pode ser vazio.")
        return v


def get_memories(session_id):
    """Recupera memórias armazenadas no MongoDB, limitando a um período de 30 dias."""
    cutoff_date = datetime.utcnow() - timedelta(
        days=30
    )  # Recupera apenas os últimos 30 dias
    memories = (
        collection.find({"session_id": session_id, "timestamp": {"$gte": cutoff_date}})
        .sort("timestamp", -1)
        .limit(MONGODB_HISTORY)
    )

    texts = [mem["text"] for mem in memories]
    log_info(
        f"📌 {len(texts)} memórias recuperadas para sessão {session_id} (últimos 30 dias)."
    )

    return texts


def get_recent_memories(session_id):
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferMemory(
            chat_memory=ChatMessageHistory(), return_messages=True
        )

    history = memory_store[session_id].load_memory_variables({})["history"]

    if not isinstance(history, list):
        return []

    recent_memories = "\n".join(
        [
            (
                f"Usuário: {msg.content}"
                if isinstance(msg, HumanMessage)
                else f"Polaris: {msg.content}"
            )
            for msg in history
        ]
    )

    log_info(
        f"📌 Recuperadas {len(history)} mensagens da memória temporária do LangChain."
    )
    return recent_memories


def save_to_langchain_memory(user_input, response, session_id):
    try:
        if session_id not in memory_store:
            memory_store[session_id] = ConversationBufferMemory(
                chat_memory=ChatMessageHistory(), return_messages=True
            )

        memory_store[session_id].save_context(
            {"input": user_input}, {"output": response}
        )
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


def save_to_mongo(user_input, session_id):
    try:
        existing_entry = collection.find_one(
            {"text": user_input, "session_id": session_id}
        )
        if existing_entry:
            log_warning(
                f"Entrada duplicada detectada para sessão {session_id}, não será salva: {user_input}"
            )
            return

        doc = {
            "text": user_input,
            "session_id": session_id,
            "timestamp": datetime.utcnow(),
        }
        result = collection.insert_one(doc)
        if result.inserted_id:
            log_success(
                f"Informação armazenada no MongoDB para sessão {session_id}: {user_input}"
            )

    except Exception as e:
        log_error(f"Erro ao salvar no MongoDB: {str(e)}")


def store_instruction_prompt():
    """Lê o prompt do arquivo, armazena no LangChain e só atualiza se houver mudança."""
    prompt_path = "polaris_prompt.txt"
    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            new_prompt = file.read().strip()
    except FileNotFoundError:
        log_warning(f"Arquivo {prompt_path} não encontrado! Usando um prompt padrão.")
        new_prompt = """\
        ### Instruções:
        Você é Polaris, um assistente inteligente.
        Responda de forma clara e objetiva, utilizando informações do histórico e memórias disponíveis.
        Se não souber a resposta, seja honesto e não invente informações.

        Agora, aqui está a conversa atual:
        """

    # Verifica se já temos o prompt armazenado no vetorstore
    existing_docs = vectorstore.similarity_search("INSTRUCTION_PROMPT", k=1)
    if existing_docs and existing_docs[0].page_content == new_prompt:
        log_info("📌 O prompt já está atualizado no LangChain. Nenhuma alteração necessária.")
        return

    # Se o prompt mudou, armazenamos no vetorstore
    vectorstore.add_texts([new_prompt], metadatas=[{"type": "instruction_prompt"}])
    log_success("✅ Prompt de instrução armazenado/vetorizado no LangChain!")


def get_instruction_prompt():
    """Recupera o prompt vetorizado armazenado no LangChain."""
    docs = vectorstore.similarity_search("INSTRUCTION_PROMPT", k=1)
    if docs:
        return docs[0].page_content

    log_warning("⚠️ Nenhum prompt encontrado no LangChain! Usando fallback.")
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
            keywords = [
                line.strip().lower() for line in file.readlines() if line.strip()
            ]
            log_info(
                f"📂 Palavras-chave carregadas do arquivo ({len(keywords)} palavras)."
            )
            return keywords
    except FileNotFoundError:
        log_warning(
            f"Arquivo {file_path} não encontrado! Usando palavras-chave padrão."
        )
        return ["meu nome é", "eu moro em", "eu gosto de"]


def trim_langchain_memory(session_id):
    """Garante que a memória temporária do LangChain não ultrapasse o limite configurado."""
    if session_id in memory_store:
        history = memory_store[session_id].chat_memory.messages

        if len(history) > LANGCHAIN_HISTORY:
            log_warning(
                f"🧹 Memória cheia para sessão {session_id}, removendo mensagens antigas..."
            )
            memory_store[session_id].chat_memory.messages = history[-LANGCHAIN_HISTORY:]

            log_info(
                f"📂 Memória ajustada, mantendo as últimas {LANGCHAIN_HISTORY} mensagens."
            )


from langchain.schema import HumanMessage, AIMessage


@app.post("/inference/")
async def inference(request: InferenceRequest):
    session_id = request.session_id
    log_info(
        f"📥 Nova solicitação de inferência para sessão {session_id}: {request.prompt}"
    )

    keywords = load_keywords_from_file()

    if any(kw in request.prompt.lower() for kw in keywords):
        save_to_mongo(request.prompt, session_id)

    trim_langchain_memory(session_id)

    mongo_memories = get_memories(session_id)
    recent_memories = get_recent_memories(session_id)

    context_pieces = []
    if mongo_memories:
        context_pieces.append("📌 Memória do Usuário:\n" + "\n".join(mongo_memories))
    if recent_memories:
        context_pieces.append("📌 Conversa recente:\n" + recent_memories)

    prompt_instrucoes = get_instruction_prompt()

    context = "\n\n".join(context_pieces)
    full_prompt = f"""{prompt_instrucoes}

--- CONTEXTO ---
{context}

--- CONVERSA ATUAL ---
Usuário: {request.prompt}

Polaris:"""

    log_info(f"📝 Prompt final enviado ao modelo:\n{full_prompt}")

    resposta = llm.invoke(full_prompt)
    save_to_langchain_memory(request.prompt, resposta, session_id)

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
