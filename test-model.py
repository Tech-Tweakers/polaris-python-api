from llama_cpp import Llama

# Defina o caminho correto do modelo
model_path = "../../models/Meta-Llama-3-8B-Instruct.Q2_K.gguf"

print(f"🔄 Carregando modelo de: {model_path}...")
try:
    llm = Llama(model_path=model_path, verbose=True, n_ctx=8192)
    print("✅ Modelo carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar o modelo: {e}")
    exit(1)

# Teste simples para ver se o modelo está respondendo
prompt = "Qual é a capital do Brasil?"
print(f"\n🔹 Enviando prompt: {prompt}")

try:
    response = llm(prompt, max_tokens=256, temperature=0.7, top_p=0.9, top_k=50)
    print("\n🔹 Resposta gerada:")
    print(response)
except Exception as e:
    print(f"❌ Erro durante a inferência: {e}")
