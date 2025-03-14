import requests
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime, timedelta
import os
import argparse

# 🔧 Configuração (token vem dos secrets do GitHub)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
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
        date = commit["commit"]["author"]["date"]

        print(f"\n🔹 [{i}/{len(commits)}] Analisando commit {sha[:7]} ({date})")

        changes = get_commit_changes(OWNER, REPO, sha)

        if not changes:
            print("   ⚠️ Não foi possível obter detalhes do commit. Pulando...")
            continue

        commit_date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ")

        for file, lines in changes.items():
            for line in lines:
                file_changes[file][line]["count"] += 1
                file_changes[file][line]["dates"].add(commit_date)
                total_changes += 1

    # 🔍 Verifica retrabalho com e sem limite de tempo
    for file, lines in file_changes.items():
        for line, info in lines.items():
            if info["count"] >= REWORK_THRESHOLD:
                rework_changes_total += 1

            recent_dates = {d for d in info["dates"] if d >= datetime.utcnow() - timedelta(days=REWORK_DAYS)}
            if len(recent_dates) >= REWORK_THRESHOLD:
                rework_changes_recent += 1

    rework_rate_total = (rework_changes_total / total_changes) * 100 if total_changes > 0 else 0
    rework_rate_recent = (rework_changes_recent / total_changes) * 100 if total_changes > 0 else 0

    # 🔥 JSON para armazenar TODOS os dados e permitir consultas flexíveis
    json_file = "rework_analysis.json"
    today = datetime.utcnow().strftime("%Y-%m-%d")

    def load_json(filename):
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print(f"⚠️ Erro ao carregar {filename}, recriando arquivo...")
                    return []
        return []

    def save_json(filename, data):
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    rework_data = load_json(json_file)

    new_entry = {
        "data": today,
        "total_changes": total_changes,
        "rework_changes_total": rework_changes_total,
        "rework_rate_total": rework_rate_total,
        "rework_changes_recent": rework_changes_recent,
        "rework_rate_recent": rework_rate_recent,
    }

    if isinstance(rework_data, list):
        existing_dates = {entry["data"] for entry in rework_data if "data" in entry}
        if today not in existing_dates:
            rework_data.append(new_entry)
            save_json(json_file, rework_data)
            print(f"📊 JSON atualizado com histórico completo para análises: {json_file}")

    # 🔥 GERAR O GRÁFICO COMPLETO
    df = pd.DataFrame(rework_data)
    df["data"] = pd.to_datetime(df["data"])
    df = df.sort_values("data")

    date_range = pd.date_range(start=df["data"].min(), end=datetime.utcnow().strftime("%Y-%m-%d"))
    df = df.set_index("data").reindex(date_range, method="ffill").reset_index()
    df.rename(columns={"index": "data"}, inplace=True)
    df["data"] = df["data"].dt.strftime("%Y-%m-%d")

    plt.figure(figsize=(12, 6))
    plt.plot(df["data"], df["rework_rate_total"], marker="o", linestyle="-", color="b", label="Rework Rate Geral")
    plt.plot(df["data"], df["rework_rate_recent"], marker="o", linestyle="--", color="r", label="Rework Rate (Últimos 21 dias)")

    plt.xticks(rotation=45, ticks=df["data"][::max(1, len(df) // 10)])
    plt.xlabel("Data")
    plt.ylabel("Rework Rate (%)")
    plt.title("Evolução do Rework Rate ao longo do tempo")
    plt.grid()
    plt.legend()

    plt.savefig("rework_rate.png", dpi=300)
    print("📊 Gráfico de Rework Rate salvo como rework_rate.png")


if __name__ == "__main__":
    commits = get_commits("Tech-Tweakers", "polaris-python-api", "main")
    analyze_rework(commits)
