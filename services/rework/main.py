import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime, timedelta
import os

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OWNER = os.getenv("OWNER")
REPO = os.getenv("REPO")

HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

REWORK_THRESHOLD = 3
REWORK_DAYS = 21

json_file = "rework_analysis.json"


def load_json(filename):
    """Carrega JSON existente ou cria um novo."""
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Erro ao carregar {filename}, recriando arquivo...")
                return []
    return []


def save_json(filename, data):
    """Salva os dados no JSON."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


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
    """Armazena no JSON cada commit com todas as informações necessárias para cálculo de métricas futuras."""
    rework_data = load_json(json_file)

    existing_shas = {entry["sha"] for entry in rework_data if "sha" in entry}

    for i, commit in enumerate(commits, 1):
        sha = commit["sha"]
        date = commit["commit"]["author"]["date"]
        commit_date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ")

        if sha in existing_shas:
            print(f"🔄 Commit {sha[:7]} já analisado, pulando...")
            continue

        print(f"\n🔹 [{i}/{len(commits)}] Processando commit {sha[:7]} ({date})")

        changes = get_commit_changes(OWNER, REPO, sha)
        if not changes:
            continue

        total_changes = sum(len(lines) for lines in changes.values())

        rework_changes_total = sum(1 for file in changes for line in changes[file] if len(changes[file]) >= REWORK_THRESHOLD)
        rework_changes_recent = sum(
            1 for file in changes for line in changes[file]
            if len(changes[file]) >= REWORK_THRESHOLD and commit_date >= datetime.utcnow() - timedelta(days=REWORK_DAYS)
        )

        rework_rate_total = (rework_changes_total / total_changes) * 100 if total_changes > 0 else 0
        rework_rate_recent = (rework_changes_recent / total_changes) * 100 if total_changes > 0 else 0

        new_entry = {
            "data": date[:10],  # Apenas YYYY-MM-DD
            "sha": sha,
            "total_changes": total_changes,
            "rework_changes_total": rework_changes_total,
            "rework_rate_total": rework_rate_total,
            "rework_changes_recent": rework_changes_recent,
            "rework_rate_recent": rework_rate_recent,
            "arquivos_modificados": list(changes.keys())
        }

        rework_data.append(new_entry)

    save_json(json_file, rework_data)
    print(f"📊 JSON atualizado com histórico completo de commits: {json_file}")


def load_json(filename):
    """Carrega os dados do JSON."""
    with open(filename, "r") as f:
        return json.load(f)

def generate_graph():
    """Gera um gráfico com base no JSON existente."""
    rework_data = load_json(json_file)

    # 📌 Criar um DataFrame a partir dos dados
    df = pd.DataFrame(rework_data)

    # 📌 Converter a data para formato datetime e ordenar
    df["data"] = pd.to_datetime(df["data"])
    df = df.sort_values("data")

    # 📌 Remover duplicatas mantendo o último valor registrado para cada data
    df = df.drop_duplicates(subset="data", keep="last")

    # 📌 Criar um intervalo contínuo de datas desde o primeiro commit até hoje
    date_range = pd.date_range(start=df["data"].min(), end=datetime.utcnow().strftime("%Y-%m-%d"))

    # 📌 Preencher dias vazios com o último valor conhecido
    df = df.set_index("data").reindex(date_range, method="ffill").reset_index()
    df.rename(columns={"index": "data"}, inplace=True)
    df["data"] = df["data"].dt.strftime("%Y-%m-%d")

    # 📌 Aplicar média móvel para suavizar oscilações extremas
    df["rework_rate_total"] = df["rework_rate_total"].rolling(window=3, min_periods=1).mean()
    df["rework_rate_recent"] = df["rework_rate_recent"].rolling(window=3, min_periods=1).mean()

    # 📊 Criar o gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(df["data"], df["rework_rate_total"], marker="o", linestyle="-", color="b", label="Rework Rate Geral")
    plt.plot(df["data"], df["rework_rate_recent"], marker="o", linestyle="--", color="r", label="Rework Rate (Últimos 21 dias)")

    # 📌 Melhorar visualização do eixo X
    plt.xticks(rotation=45, ticks=df["data"][::max(1, len(df) // 10)])
    
    # 📌 Labels e título
    plt.xlabel("Data")
    plt.ylabel("Rework Rate (%)")
    plt.title("Evolução do Rework Rate ao longo do tempo")
    plt.grid()
    plt.legend()

    # 📌 Salvar gráfico
    plt.savefig("rework_rate.png", dpi=300)
    print("📊 Gráfico salvo como rework_rate.png")


if __name__ == "__main__":
    commits = get_commits(OWNER, REPO, "main")
    analyze_rework(commits)
    generate_graph()
#