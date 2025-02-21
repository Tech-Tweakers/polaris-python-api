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
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
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

# 🔹 Inicializa FastAPI
app = FastAPI()

# 🔹 Inicializa LangChain Memory
log_info("🔹 Configurando memória do LangChain...")

embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedder)

# 🔹 Memória de conversação curta
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# 🔹 Classe para Llama (Agora um `Runnable`)
class LlamaRunnable(Runnable):
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

        # 🔹 Corrigindo extração da última mensagem e garantindo que `user_prompt` sempre existe
        user_prompt = "Desculpe, não encontrei uma pergunta válida."  # Default

        if isinstance(messages, list) and messages:  # Garante que `messages` é uma lista e não está vazia
            full_prompt = "\n".join(
                [msg["content"] if isinstance(msg, dict) else msg.content for msg in messages]
            )  # <- Correção aqui
            user_prompt = messages[-1]["content"] if isinstance(messages[-1], dict) else messages[-1].content  # <- Correção aqui
        else:
            full_prompt = user_prompt

        # 🔹 Adiciona um prefixo ao prompt
        full_prompt = f"Usuário: {full_prompt}\nPolaris:"

        # 🔹 Mede tempo de inferência
        start_time = time.time()
        response = self.llm(full_prompt, stop=["\n"], max_tokens=100, echo=False)
        end_time = time.time()

        # 🔹 Correção do cálculo do tempo de inferência
        elapsed_time = end_time - start_time  # Tempo decorrido em segundos
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60

        formatted_time = f"{hours:02}:{minutes:02}:{seconds:06.3f}"
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

# 🔹 Função para interagir com a IA
@app.post("/inference/")
async def inference(request: InferenceRequest):
    session_id = request.session_id or "default_session"

    # 🔹 Obtém o histórico salvo na memória curta
    history = memory.load_memory_variables({})["history"]
    
    # 🔹 Certifica-se de que `history` é uma lista antes de modificar
    if not isinstance(history, list):
        history = []
    
    history.append({"role": "user", "content": request.prompt})

    # 🔍 **Passo 2: Recuperar memória de longo prazo da ChromaDB**
    retrieved_docs = vectorstore.similarity_search(request.prompt, k=1)
    if retrieved_docs:
        log_info(f"🔍 Memória recuperada: {retrieved_docs[0].page_content}")
        # Adiciona a memória recuperada ao histórico para reforçar a resposta
        history.append({"role": "system", "content": f"Lembre-se: {retrieved_docs[0].page_content}"})

    # 🔹 Faz a inferência com base no histórico atualizado
    resposta = llm.invoke(history)

    # 🔹 Salva resposta na memória curta
    memory.save_context({"input": request.prompt}, {"output": resposta})

    # 🔹 Categorias expandidas de palavras-chave para detectar informações importantes
    KEYWORDS = [
        # 🔹 Identidade
        "meu nome é", "sou conhecido como", "me chamam de", "meu apelido é",
        "eu tenho", "eu sou um", "eu sou uma", "minha idade é", "tenho anos",
        "trabalho como", "minha profissão é", "sou formado em", "me formei em",
        "minha religião é", "sou religioso", "sou ateu", "sou agnóstico", "acredito em",
        "sou brasileiro", "minha nacionalidade é", "falo", "aprendi a falar",
        "minha crença é", "eu sigo", "minha filosofia de vida é",

        # 🔹 Localização
        "eu moro em", "eu vivo em", "sou de", "nasci em", "minha cidade é", "meu bairro é",
        "moro no estado de", "meu endereço é", "eu frequento", "costumo ir em",
        "trabalho em", "estudo em", "minha escola é", "minha faculdade é", "minha universidade é",

        # 🔹 Preferências
        "eu gosto de", "adoro", "sou fã de", "meu hobby é", "prefiro", 
        "minha comida favorita é", "meu prato favorito é", "minha bebida favorita é",
        "minha banda favorita é", "meu filme favorito é", "meu livro favorito é",
        "meu autor favorito é", "minha série favorita é", "meu esporte favorito é",
        "eu torço para", "meu time é", "minha cor favorita é", "eu amo", "eu odeio",
        "sou vegetariano", "sou vegano", "não como", "sou alérgico a", "eu não gosto de",
        "meu animal favorito é", "tenho um pet", "meu cachorro se chama", "meu gato se chama",
        
        # 🔹 Relacionamentos
        "tenho um amigo chamado", "minha mãe se chama", "meu pai se chama",
        "meu filho se chama", "minha filha se chama", "minha esposa se chama",
        "meu marido se chama", "meu namorado se chama", "minha namorada se chama",
        "estou solteiro", "estou casado", "estou noivo", "estou divorciado",
        "tenho irmãos", "meu irmão se chama", "minha irmã se chama",
        
        # 🔹 Eventos e experiências
        "viajei para", "já estive em", "fui para", "participei de", 
        "me formei em", "trabalhei em", "estudei em", "morei em", "já morei em",
        "meu primeiro emprego foi", "fiz um intercâmbio", "já dei aula de",
        "já apresentei", "já ganhei um prêmio", "já fui voluntário em",
        
        # 🔹 Rotina e hábitos
        "acordo às", "durmo às", "trabalho das", "estudo das", "tenho aula de",
        "costumo acordar", "tenho o hábito de", "minha rotina é", "todos os dias eu",
        "eu sempre faço", "normalmente eu", "costumo fazer", "nos finais de semana eu",
        "tenho o costume de", "minha manhã é", "minha noite é",
        
        # 🔹 Saúde e bem-estar
        "faço academia", "pratico esportes", "corro", "ando de bicicleta",
        "tenho pressão alta", "tenho diabetes", "sou intolerante a", "sou alérgico a",
        "estou em tratamento para", "faço dieta", "não como", "sou fitness",
        "meu médico recomendou", "eu medito", "eu faço yoga", "eu bebo muita água",
        
        # 🔹 Objetivos e sonhos
        "meu sonho é", "quero aprender a", "quero viajar para", "meu objetivo é",
        "planejo me mudar para", "quero me formar em", "quero trabalhar com",
        "quero morar em", "pretendo abrir um negócio", "meu desejo é", "gostaria de",
        
        # 🔹 Personalidade e comportamentos
        "tenho medo de", "sou ansioso", "me considero uma pessoa", "eu sou tímido",
        "eu sou extrovertido", "sou uma pessoa organizada", "sou bagunceiro",
        "eu costumo procrastinar", "eu sou perfeccionista", "eu sou impulsivo",
        "eu gosto de desafios", "eu evito conflitos", "sou muito emocional",
        "tenho dificuldade em", "sou bom em", "sou especialista em",
        
        # 🔹 Tecnologia e uso de internet
        "eu uso redes sociais", "tenho conta no Instagram", "uso Twitter",
        "trabalho com tecnologia", "sou programador", "sou desenvolvedor",
        "uso Linux", "uso Windows", "uso Mac", "meu celular é um", "meu notebook é um",
        "meu jogo favorito é", "jogo videogame", "tenho um PlayStation", "tenho um Xbox",
        "gosto de assistir lives", "sigo canais no YouTube",
    ]

    # 🔹 Verifica se o prompt contém informações importantes para armazenar
    if any(kw in request.prompt.lower() for kw in KEYWORDS):
        vectorstore.add_texts([request.prompt])
        log_info(f"📌 Informação armazenada na memória de longo prazo: {request.prompt}")

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
