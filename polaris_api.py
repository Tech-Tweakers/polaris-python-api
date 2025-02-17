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

class LogColors:
    INFO = "\033[94m"
    SUCCESS = "\033[92m"
    WARNING = "\033[93m"
    ERROR = "\033[91m"
    RESET = "\033[0m"

def log_info(message: str):
    print(f"{LogColors.INFO}🔹 {message}{LogColors.RESET}")

def log_success(message: str):
    print(f"{LogColors.SUCCESS}✅ {message}{LogColors.RESET}")

def log_warning(message: str):
    print(f"{LogColors.WARNING}⚠️ {message}{LogColors.RESET}")

def log_error(message: str):
    print(f"{LogColors.ERROR}❌ {message}{LogColors.RESET}")

MODEL_PATH = "./models/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
MONGO_URI = f"mongodb://admin:adminpassword@localhost:27017/"
DATABASE_NAME = "polaris_db"
NUM_CORES = 6
MODEL_BATCH_SIZE = 8

client = None
collection = None
mongo_available = False

async def connect_mongo():
    global client, collection, mongo_available
    try:
        client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        await client.server_info()
        collection = client[DATABASE_NAME]["inferences"]
        mongo_available = True
        log_success("✅ Conectado ao MongoDB!")
    except Exception as e:
        mongo_available = False
        log_warning(f"⚠️ MongoDB não disponível! Erro: {e}")

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

class LlamaLLM:
    _instance = None
    def __new__(cls, model_path: str, prompt_file: str):
        if cls._instance is None:
            cls._instance = super(LlamaLLM, cls).__new__(cls)
            cls._instance.model_path = model_path
            cls._instance.prompt_base = "<|im_start|>system\nVocê é Polaris, uma assistente virtual amigável e útil. Responda apenas o que o usuário perguntar, sem completar frases.\n<|im_end|>"
            cls._instance.llm = None
        return cls._instance

    def load(self):
        if self.llm is None:
            self.llm = Llama(model_path=self.model_path, n_threads=NUM_CORES, batch_size=MODEL_BATCH_SIZE)

    def call(self, user_prompt: str, **kwargs) -> str:
        if self.llm is None:
            raise HTTPException(status_code=500, detail="Modelo não carregado!")
        full_prompt = (
            f"<|im_start|>system\n{self.prompt_base}\n<|im_end|>\n<|im_start|>user\n{user_prompt}\n<|im_end|>\n<|im_start|>assistant"
        )
        response = self.llm(full_prompt, stop=["<|im_end|>"], max_tokens=100, echo=False)
        return response["choices"][0]["text"].strip().split("\\n")[0]

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_mongo()
    llm.load()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = LlamaLLM(model_path=MODEL_PATH, prompt_file="polaris_prompt.txt")

@app.post("/inference/")
async def inference(request: InferenceRequest):
    session_data = await collection.find_one({"session_id": request.session_id}) if mongo_available else None
    history = session_data.get("history", []) if session_data else []
    history.append({"role": "User", "content": request.prompt})
    context_prompt = "\n".join([f"{msg['content']}" for msg in history])
    answer = llm.call(context_prompt)
    history.append({"role": "Polaris", "content": answer})
    if mongo_available:
        await collection.update_one({"session_id": request.session_id}, {"$set": {"history": history}}, upsert=True)
    return {"resposta": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
