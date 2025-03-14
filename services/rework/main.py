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

json_file = f"rework_analysis_{REPO}.json"


def load_json(filename):
    """Carrega JSON existente ou cria um novo como uma lista vazia."""
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    print(f"⚠️ {filename} estava no formato errado. Recriando...")
            except json.JSONDecodeError:
                print(f"⚠️ Erro ao carregar {filename}, recriando arquivo...")

    with open(filename, "w") as f:
        json.dump([], f, indent=4)
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

        url = response.links.get("next", {}).get("url")

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
                if line.startswith("+") and not line.startswith(
                    "+++"
                ):  # Linhas adicionadas
                    changed_lines.add(line)
                elif line.startswith("-") and not line.startswith(
                    "---"
                ):  # Linhas removidas
                    changed_lines.add(line)

            changes[filename] = changed_lines

    return changes


def analyze_rework(commits):
    """Analisa commits e salva no JSON."""
    rework_data = load_json(json_file)  # Agora sempre retorna uma lista

    existing_shas = {entry["sha"] for entry in rework_data if "sha" in entry}

    total_rework_rate = 0
    total_rework_rate_recent = 0
    total_commits = 0

    total_lines_analyzed = 0
    total_lines_rework = 0
    total_lines_rework_recent = 0

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

        rework_changes_total = sum(
            1
            for file in changes
            for line in changes[file]
            if len(changes[file]) >= REWORK_THRESHOLD
        )
        rework_changes_recent = sum(
            1
            for file in changes
            for line in changes[file]
            if len(changes[file]) >= REWORK_THRESHOLD
            and commit_date >= datetime.utcnow() - timedelta(days=REWORK_DAYS)
        )

        rework_rate_total = (
            (rework_changes_total / total_changes) * 100 if total_changes > 0 else 0
        )
        rework_rate_recent = (
            (rework_changes_recent / total_changes) * 100 if total_changes > 0 else 0
        )

        new_entry = {
            "data": date[:10],  # Apenas YYYY-MM-DD
            "sha": sha,
            "total_changes": total_changes,
            "rework_changes_total": rework_changes_total,
            "rework_rate_total": rework_rate_total,
            "rework_changes_recent": rework_changes_recent,
            "rework_rate_recent": rework_rate_recent,
            "arquivos_modificados": list(changes.keys()),
        }

        rework_data.append(new_entry)

        # Somando os resultados para o print final
        total_rework_rate += rework_rate_total
        total_rework_rate_recent += rework_rate_recent
        total_commits += 1

        total_lines_analyzed += total_changes
        total_lines_rework += rework_changes_total
        total_lines_rework_recent += rework_changes_recent

    save_json(json_file, rework_data)
    print(f"📊 JSON atualizado com histórico completo de commits: {json_file}")

    # Exibir as métricas no final
    if total_commits > 0:
        average_rework_rate = total_rework_rate / total_commits
        average_rework_rate_recent = total_rework_rate_recent / total_commits

        print("\n📊 **RESULTADOS FINAIS:**")
        print(f"🔹 Total de Commits analisados: {total_commits}")
        print(f"🔹 Total de Linhas Analisadas: {total_lines_analyzed}")
        print(f"🔹 Total de Linhas de Retrabalho: {total_lines_rework}")
        print(
            f"🔹 Total de Linhas de Retrabalho nos últimos {REWORK_DAYS} dias: {total_lines_rework_recent}"
        )
        print(f"🔹 Rework Rate Geral: {average_rework_rate:.2f}%")
        print(
            f"🔹 Rework Rate nos últimos {REWORK_DAYS} dias: {average_rework_rate_recent:.2f}%"
        )
    else:
        print("⚠️ Nenhum commit foi analisado.")


if __name__ == "__main__":
    commits = get_commits(OWNER, REPO, "main")
    analyze_rework(commits)
