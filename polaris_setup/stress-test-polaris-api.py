import time
import asyncio
import aiohttp
import statistics
import random

# === CONFIGURAÇÕES ===
URL = "https://538f59746d28.ngrok-free.app/inference"
PAYLOAD = {
    "prompt": "Olá Polaris, me diga uma curiosidade!"
}

CARGAS_PROGRESSIVAS = [10, 20, 30, 50]
CONCORRENCIA_MAXIMA = 10
INTERVALO_ENTRE_REQS = (0.1, 0.5)  # segundos entre as requisições dentro de uma onda
COOLDOWN_ENTRE_ONDAS = 5  # segundos de pausa entre as ondas

# === MÉTRICAS GLOBAIS ===
latencies = []
erros_total = {
    "timeout": 0,
    "500": 0,
    "429": 0,
    "outras": 0,
    "sucesso": 0
}

# === FUNÇÃO DE TESTE INDIVIDUAL ===
# NOVO
VERBOSE = True
respostas = []

# Atualizado
async def fazer_requisicao(session, i, sem):
    async with sem:
        try:
            await asyncio.sleep(random.uniform(*INTERVALO_ENTRE_REQS))
            start = time.perf_counter()
            async with session.post(URL, json=PAYLOAD, timeout=10) as resp:
                duracao = time.perf_counter() - start
                status = resp.status
                corpo = await resp.text()

                respostas.append({
                    "id": i,
                    "status": status,
                    "tempo": round(duracao, 2),
                    "resposta": corpo[:200]  # Limita tamanho da string
                })

                if status == 200:
                    latencies.append(duracao)
                    erros_total["sucesso"] += 1
                elif status == 500:
                    erros_total["500"] += 1
                elif status == 429:
                    erros_total["429"] += 1
                else:
                    erros_total["outras"] += 1

        except asyncio.TimeoutError:
            erros_total["timeout"] += 1
        except aiohttp.ClientError as e:
            erros_total["outras"] += 1
            respostas.append({
                "id": i,
                "status": "erro",
                "tempo": 0,
                "resposta": f"Erro de conexão: {str(e)}"
            })

# === EXECUÇÃO POR ONDA ===
async def executar_onda(qtd_requisicoes):
    sem = asyncio.Semaphore(CONCORRENCIA_MAXIMA)
    async with aiohttp.ClientSession() as session:
        tasks = [fazer_requisicao(session, i, sem) for i in range(qtd_requisicoes)]
        await asyncio.gather(*tasks)

# === PROGRAMA PRINCIPAL ===
async def main():
    print("\n🚀 Iniciando stress test progressivo...\n")
    total_inicio = time.perf_counter()

    for onda, carga in enumerate(CARGAS_PROGRESSIVAS, start=1):
        print(f"\n🌊 Onda {onda}: Disparando {carga} requisições...")
        inicio = time.perf_counter()
        await executar_onda(carga)
        duracao = time.perf_counter() - inicio
        print(f"✅ Onda {onda} concluída em {duracao:.2f}s")
        await asyncio.sleep(COOLDOWN_ENTRE_ONDAS)

    total_duracao = time.perf_counter() - total_inicio
    print("\n📊 RELATÓRIO FINAL GERAL")
    print(f"Tempo total: {total_duracao:.2f}s")
    print(f"Total de requisições: {sum(CARGAS_PROGRESSIVAS)}")
    for chave, valor in erros_total.items():
        print(f"{chave.capitalize()}: {valor}")
    if latencies:
        print(f"Média de latência: {statistics.mean(latencies):.2f}s")
        print(f"Mínima: {min(latencies):.2f}s | Máxima: {max(latencies):.2f}s")
    if VERBOSE:
        print("\n📬 AMOSTRAS DAS RESPOSTAS:")
        for r in respostas[:10]:  # Mostra só as 10 primeiras
            print(f"[{r['id']}] ({r['status']} | {r['tempo']}s): {r['resposta']}\n")
    import csv

    with open("respostas_stress.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id", "status", "tempo", "resposta"])
        writer.writeheader()
        writer.writerows(respostas)


if __name__ == "__main__":
    asyncio.run(main())
