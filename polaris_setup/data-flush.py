import os
import shutil
from pymongo import MongoClient
from dotenv import load_dotenv

# Carrega variáveis
load_dotenv(dotenv_path="../polaris_api/.env")

MONGO_URI = os.getenv("MONGO_URI")
CHROMA_DB_PATH = "../polaris_api/chroma_db"

# Conecta ao MongoDB
client = MongoClient(MONGO_URI)
db = client["polaris_db"]
collection = db["user_memory"]


def flush_mongo():
    """Apaga todos os documentos do MongoDB"""
    result = collection.delete_many({})
    print(f"🗑️  MongoDB limpo! {result.deleted_count} documentos apagados.")


def flush_chroma():
    """Apaga toda a pasta de vetores"""
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)
        print("🗑️  Vetores do ChromaDB apagados!")
    else:
        print("ℹ️  Nenhuma pasta chroma_db encontrada.")


def flush_geral():
    """Flush geral: MongoDB + Chroma"""
    flush_mongo()
    flush_chroma()
    print("✅ Polaris zerada e pronta pra nova jornada!")


if __name__ == "__main__":
    print(
        """
==== Polaris Flush Tool ====

[1] Limpar apenas conversas (MongoDB)
[2] Limpar apenas vetores (ChromaDB)
[3] Limpar tudo (Mongo + Chroma)
"""
    )
    opcao = input("Escolha a opção: ").strip()

    if opcao == "1":
        flush_mongo()
    elif opcao == "2":
        flush_chroma()
    elif opcao == "3":
        flush_geral()
    else:
        print("⚠️ Opção inválida.")
