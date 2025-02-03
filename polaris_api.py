import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
from typing import Optional, List

# Configuração
MODEL_PATH = "../../models/Meta-Llama-3-8B-Instruct.Q2_K.gguf"
PROMPT_FILE = "polaris_prompt.txt"

# Função para ler o arquivo de prompt
def read_prompt_file(file_path: str, max_length: int = 500) -> str:
    """Lê o prompt base do arquivo e limita seu tamanho para evitar estouro de tokens."""
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

# Modelo de dados para API
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

# Criamos uma instância única global para evitar múltiplas cargas do modelo
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

    async def warmup_model(self):
        """Executa um warm-up em background após carregar o modelo"""
        print("🔥 Iniciando warm-up em background...")
        await asyncio.sleep(1)  # Simula um pequeno atraso antes de rodar o warm-up
        try:
            response = self.llm("Qual é a capital da França?", max_tokens=32, temperature=0.7, top_p=1.0, top_k=50)
            print(f"🔥 Warm-up completo! Resposta: {response['choices'][0]['text'].strip()}")
        except Exception as e:
            print(f"⚠️ Erro no warm-up: {e}")

    def call(self, user_prompt: str, **kwargs) -> str:
        """Chama o modelo e retorna a resposta"""
        if self.llm is None:
            raise HTTPException(status_code=500, detail="Modelo ainda não carregado!")

        try:
            # 🔥 Agora o prompt sempre inclui o contexto do `polaris_prompt.txt`
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

            print(f"🔹 Resposta bruta do modelo: {response}")

            if "choices" in response and response["choices"]:
                raw_answer = response["choices"][0]["text"].strip()
            else:
                raw_answer = "[Erro: Modelo retornou resposta vazia]"

            if raw_answer == "":
                raw_answer = "[Erro: O modelo não respondeu nada útil]"

            print(f"🔹 Resposta final filtrada: {raw_answer}")

            return raw_answer
        except Exception as e:
            print(f"❌ Erro durante a inferência: {e}")
            return "[Erro: Falha na inferência]"

# Instancia a API
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Criamos a instância global do modelo, mas **ainda não carregamos ele**
llm = LlamaLLM(model_path=MODEL_PATH, prompt_file=PROMPT_FILE)

@app.on_event("startup")
async def startup_event():
    """Carrega o modelo apenas uma vez quando a API inicia e executa o warm-up"""
    llm.load()
    asyncio.create_task(llm.warmup_model())

@app.post("/inference/")
async def inference(request: InferenceRequest):
    """Endpoint para gerar resposta"""
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
        return {"resposta": answer}
    except HTTPException as e:
        print(f"❌ Erro no endpoint: {e}")
        raise e
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        raise HTTPException(status_code=500, detail="Erro interno na API")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("polaris_api:app", host="0.0.0.0", port=8000, reload=True, workers=1)
