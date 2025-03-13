import requests
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime, timedelta
import os

# 🔧 Configuração (token vem dos secrets do GitHub)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Pega o token do ambiente do GitHub Actions
OWNER = "Tech-Tweakers"
REPO = "polaris-python-api"
BRANCH = "main" 

HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

# 🔧 Parâmetros para definir retrabalho
REWORK_THRESHOLD = 3  # Número mínimo de alterações para contar como retrabalho
REWORK_DAYS = 21  # Período máximo para considerar um ofensor recente

# 🔧 Estrutura para armazenar mudanças no código
file_changes = defaultdict(lambda: defaultdict(lambda: {"count": 0, "dates": set()}))

# 🔧 Contadores das métricas
rework_changes_total = 0  # Todas as alterações repetidas
rework_changes_recent = 0  # Apenas alterações nos últimos REWORK_DAYS dias
total_changes = 0  # Total de alterações no código


def get_commits(owner, repo, branch):
    """Obtém a lista completa de commits do branch."""
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params = {"sha": branch, "per_page": 100}
    commits = []
    
    print(f"📥 Buscando commits no branch '{branch}' de {owner}/{repo}...")

    while url:
        response = requests.get(url, headers=HEADERS, params=params)
        
        if response.status_code != 200:
            raise Exception(f"❌ Erro ao buscar commits: {response.json()}")
        
        data = response.json()
        commits.extend(data)
        
        # Verifica se há mais páginas de commits
        url = response.links.get('next', {}).get('url')
    
    print(f"✅ {len(commits)} commits encontrados!")
    return commits


def get_commit_changes(owner, repo, sha):
    """Obtém os arquivos e linhas modificadas em um commit."""
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print(f"⚠️ Falha ao buscar detalhes do commit {sha}")
        return None
    
    data = response.json()
    files = data.get("files", [])
    
    changes = {}
    for file in files:
        filename = file["filename"]
        patch = file.get("patch", "")
        
        if patch:
            changed_lines = set()
            for line in patch.split("\n"):
                if line.startswith("+") and not line.startswith("+++"):  # Linhas adicionadas
                    changed_lines.add(line)
                elif line.startswith("-") and not line.startswith("---"):  # Linhas removidas
                    changed_lines.add(line)
            
            changes[filename] = changed_lines
    
    return changes


def analyze_rework(commits):
    """Analisa quantas vezes as mesmas linhas foram alteradas para calcular retrabalho."""
    global rework_changes_total, rework_changes_recent, total_changes
    rework_rate_data = []

    print("\n📊 Analisando commits para calcular retrabalho...")

    for i, commit in enumerate(commits, 1):
        sha = commit["sha"]
        author = commit["commit"]["author"]["name"]
        date = commit["commit"]["author"]["date"]
        message = commit["commit"]["message"].lower()

        print(f"\n🔹 [{i}/{len(commits)}] Analisando commit {sha[:7]} de {author} ({date})")
        print(f"   📜 Mensagem: {message}")

        changes = get_commit_changes(OWNER, REPO, sha)

        if not changes:
            print("   ⚠️ Não foi possível obter detalhes do commit. Pulando...")
            continue

        commit_date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ")  # Convertendo string ISO para datetime

        for file, lines in changes.items():
            for line in lines:
                file_changes[file][line]["count"] += 1
                file_changes[file][line]["dates"].add(commit_date)
                total_changes += 1

    # 🔍 Verifica retrabalho com e sem limite de tempo
    for file, lines in file_changes.items():
        for line, info in lines.items():
            # 🔥 1️⃣ MÉTRICA DE REWORK GERAL (independente de tempo)
            if info["count"] >= REWORK_THRESHOLD:
                rework_changes_total += 1

            # 🔥 2️⃣ MÉTRICA DE REWORK DENTRO DE 21 DIAS
            recent_dates = {d for d in info["dates"] if d >= datetime.utcnow() - timedelta(days=REWORK_DAYS)}
            if len(recent_dates) >= REWORK_THRESHOLD:
                rework_changes_recent += 1

    # Cálculo das taxas de retrabalho
    rework_rate_total = (rework_changes_total / total_changes) * 100 if total_changes > 0 else 0
    rework_rate_recent = (rework_changes_recent / total_changes) * 100 if total_changes > 0 else 0

    print("\n📊 RESULTADO FINAL:")
    print(f"   🔢 Total de alterações no código: {total_changes}")
    print(f"   🔄 Alterações repetidas (Rework Geral): {rework_changes_total}")
    print(f"   📊 Rework Rate Geral: {rework_rate_total:.2f}%")
    print(f"   🔄 Alterações repetidas nos últimos {REWORK_DAYS} dias: {rework_changes_recent}")
    print(f"   📊 Rework Rate nos últimos {REWORK_DAYS} dias: {rework_rate_recent:.2f}%\n")

    # Salvar os dados em JSON
    rework_rate_data.append({"data": datetime.utcnow().strftime("%Y-%m-%d"), "rework_rate": rework_rate_total})
    with open("rework_rate.json", "w") as f:
        json.dump(rework_rate_data, f, indent=4)

    # 🔥 GERAÇÃO DO GRÁFICO
    df = pd.DataFrame(rework_rate_data)

    plt.figure(figsize=(10, 5))
    plt.plot(df["data"], df["rework_rate"], marker="o", linestyle="-", color="b")
    plt.xlabel("Data")
    plt.ylabel("Rework Rate (%)")
    plt.title("Evolução do Rework Rate ao longo do tempo")
    plt.xticks(rotation=45)
    plt.grid()

    # Salvar o gráfico como imagem
    plt.savefig("rework_rate.png", dpi=300)
    print("📊 Gráfico de Rework Rate salvo como rework_rate.png")


if __name__ == "__main__":
    try:
        commits = get_commits(OWNER, REPO, BRANCH)
        analyze_rework(commits)
        print("✅ Análise concluída com sucesso!")
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
