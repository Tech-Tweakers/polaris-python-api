import argparse
from llama_cpp import Llama
from typing import Optional, List


class LlamaLLM:
    def __init__(self, model_path: str):
        """
        Inicializa o modelo Llama com o caminho do modelo fornecido.
        """
        self.llm = Llama(model_path=model_path, verbose=False, n_ctx=8192)  # Aumentando o contexto para 8192 tokens

    def call(self, prompt: str, stop: Optional[List[str]] = None, max_tokens: int = 1024,
             temperature: float = 0.4, top_p: float = 0.9, top_k: int = 40,
             frequency_penalty: float = 2.5, presence_penalty: float = 1.5) -> str:
        """
        Chama o modelo com o prompt fornecido e retorna a resposta.
        A geração do texto é interrompida assim que a stop word for encontrada.
        Parâmetros adicionais controlam a geração do texto.
        """
        response = self.llm(
            prompt,
            stop=stop or [],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        return response["choices"][0]["text"].strip()

    def close(self):
        """
        Fecha o modelo e libera os recursos.
        """
        self.llm.close()


# Leitura do arquivo de prompt
def read_prompt_file(file_path: str) -> str:
    """
    Lê o conteúdo do arquivo de prompt e retorna como string.
    Se ocorrer erro, retorna uma string vazia.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        print(f"Erro ao ler o arquivo {file_path}: {e}")
        return ""


def main():
    # Argumentos da linha de comando
    parser = argparse.ArgumentParser(description="Executar o modelo Llama com um arquivo de prompt.")
    parser.add_argument("-m", "--model", type=str, default="../../models/Meta-Llama-3-8B-Instruct.Q2_K.gguf", help="Caminho do modelo")
    parser.add_argument("-f", "--file", type=str, default="polaris_prompt.txt", help="Caminho do arquivo de prompt")
    parser.add_argument("--temperature", type=float, default=0.4, help="Controla a aleatoriedade da resposta")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus sampling)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--frequency_penalty", type=float, default=2.5, help="Penalidade de frequência")
    parser.add_argument("--presence_penalty", type=float, default=1.5, help="Penalidade de presença")
    args = parser.parse_args()

    # Leitura do arquivo de prompt
    prompt = read_prompt_file(args.file)
    if not prompt:
        print("Erro: O arquivo de prompt está vazio ou não pôde ser lido.")
        return  # Encerra o script caso o prompt não seja lido corretamente

    # Inicializa o modelo
    llm = LlamaLLM(model_path=args.model)

    # Definir stop words (palavras de parada)
    stop_words = ["Pergunta:", "Resposta:"]

    try:
        # Ajuste do prompt para ser mais claro
        question = "Me mostra um exemplo de hello world em golang?"
        full_prompt = f"""Você é um assistente virtual chamado Polaris. Você é especializado em fornecer respostas claras e concisas sobre desenvolvimento de software. 
                        Pergunta: {question}
                        Resposta: """

        # Chama o modelo com o prompt e stop words, e usa parâmetros avançados
        answer = llm.call(
            full_prompt,
            stop=stop_words,
            max_tokens=1024,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty
        )

        # Exibe a resposta
        print(f"Resposta: {answer}")
    finally:
        # Libera recursos do modelo
        llm.close()


if __name__ == "__main__":
    main()
