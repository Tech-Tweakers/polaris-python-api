import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from llama_cpp import Llama
from motor.motor_asyncio import AsyncIOMotorClient  # 🔥 Conexão com MongoDB
from datetime import datetime
from contextlib import asynccontextmanager

# 🔹 Configurações
MODEL_PATH = "./models/Meta-Llama-3-8B-Instruct.Q2_K.gguf"
PROMPT_FILE = "polaris_prompt.txt"
MONGO_URI = "mongodb://admin:adminpassword@mongodb:27017/"
DATABASE_NAME = "polaris_db"

# 🔹 Conexão com MongoDB
client = AsyncIOMotorClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db["inferences"]

# 🔹 Função para ler o arquivo de prompt
def read_prompt_file(file_path: str, max_length: int = 500) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt = file.read().strip()
            prompt_tokens = prompt.split()
            if len(prompt_tokens) > max_length:
                prompt = " ".join(prompt_tokens[:max_length])
            return prompt
    except Exception as e:
        print(f"❌ Erro ao ler o arquivo {file_path}: {e}")
        return "Você é um assistente de IA útil. Responda com clareza."

# 🔹 Modelo de dados para API
class InferenceRequest(BaseModel):
    prompt: str
    stop_words: Optional[List[str]] = None
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.5
    top_k: Optional[int] = 40
    frequency_penalty: Optional[float] = 2.0
    presence_penalty: Optional[float] = 1.5
    max_tokens: Optional[int] = 512
    session_id: Optional[str] = None

# 🔹 Classe Singleton do Modelo
class LlamaLLM:
    _instance = None

    def __new__(cls, model_path: str, prompt_file: str):
        if cls._instance is None:
            cls._instance = super(LlamaLLM, cls).__new__(cls)
            cls._instance.model_path = model_path
            cls._instance.prompt_base = read_prompt_file(prompt_file)
            cls._instance.llm = None
        return cls._instance

    def load(self):
        """Carrega o modelo Llama apenas uma vez"""
        if self.llm is None:
            try:
                print(f"🔄 Carregando modelo de: {self.model_path}...")
                self.llm = Llama(model_path=self.model_path, verbose=True, n_ctx=8192)
                print("✅ Modelo carregado com sucesso!")
            except Exception as e:
                print(f"❌ Erro ao carregar o modelo: {e}")
                raise HTTPException(status_code=500, detail="Erro ao carregar o modelo Llama")

    def call(self, user_prompt: str, **kwargs) -> str:
        """Chama o modelo e retorna a resposta"""
        if self.llm is None:
            raise HTTPException(status_code=500, detail="Modelo ainda não carregado!")

        try:
            full_prompt = f"{self.prompt_base}\n\nPergunta: {user_prompt}"
            print(f"🔹 Prompt enviado ao modelo:\n{full_prompt}")

            response = self.llm(
                full_prompt,
                stop=["Pergunta:", "Pergunte:"],  
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.4),
                top_p=kwargs.get("top_p", 0.5),
                top_k=kwargs.get("top_k", 40),
                frequency_penalty=kwargs.get("frequency_penalty", 2.0),
                presence_penalty=kwargs.get("presence_penalty", 1.5)
            )

            if "choices" in response and response["choices"]:
                raw_answer = response["choices"][0]["text"].strip()
            else:
                raw_answer = "[Erro: Modelo retornou resposta vazia]"

            return raw_answer
        except Exception as e:
            print(f"❌ Erro durante a inferência: {e}")
            return "[Erro: Falha na inferência]"

# 🔹 Gerenciamento de inicialização usando Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo na inicialização da API e fecha conexões no desligamento"""
    print("🚀 Iniciando a API...")
    llm.load()
    yield
    print("🛑 API sendo desligada...")

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
    """Gera resposta e salva no MongoDB"""
    try:
        answer = llm.call(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty
        )

        # 🔥 Salvando no MongoDB
        inference_data = {
            "prompt": request.prompt,
            "resposta": answer,
            "timestamp": datetime.utcnow()
        }
        await collection.insert_one(inference_data)

        return {"resposta": answer}
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        raise HTTPException(status_code=500, detail="Erro interno na API")

@app.get("/historico/")
async def get_historico():
    """Retorna as inferências armazenadas no MongoDB"""
    try:
        docs = await collection.find().to_list(100)  # 🔥 Pega até 100 registros
        return docs
    except Exception as e:
        print(f"❌ Erro ao buscar histórico: {e}")
        raise HTTPException(status_code=500, detail="Erro ao buscar histórico")
