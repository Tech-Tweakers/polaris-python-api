import time
import asyncio
import aiohttp
import statistics

# CONFIGURAÇÕES
URL = "https://538f59746d28.ngrok-free.app/inference/"
TOTAL_REQUESTS = 100
CONCURRENT_REQUESTS = 10
PAYLOAD = {"prompt": "Olá Polaris, me diga uma curiosidade!"}


# MÉTRICAS
latencies = []
errors = {"timeout": 0, "500": 0, "outras": 0, "sucesso": 0}

sem = asyncio.Semaphore(CONCURRENT_REQUESTS)

import random


async def fazer_requisicao(session, i):
    async with sem:
        try:
            # delay aleatório entre 0.1 e 0.5s
            await asyncio.sleep(random.uniform(0.1, 0.5))

            start = time.perf_counter()
            async with session.post(URL, json=PAYLOAD, timeout=10) as resp:
                duracao = time.perf_counter() - start
                status = resp.status

                if status == 200:
                    latencies.append(duracao)
                    errors["sucesso"] += 1
                elif status == 500:
                    errors["500"] += 1
                elif status == 429:
                    errors["outras"] += 1
                    print(f"[{i}] ⛔ Too Many Requests (429)")
                else:
                    errors["outras"] += 1
                    print(f"[{i}] ⚠️ Status inesperado: {status}")

        except asyncio.TimeoutError:
            errors["timeout"] += 1
        except aiohttp.ClientError as e:
            errors["outras"] += 1
            print(f"[{i}] ⚠️ Erro de conexão: {e}")


async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fazer_requisicao(session, i) for i in range(TOTAL_REQUESTS)]
        await asyncio.gather(*tasks)

    print("\n📊 RELATÓRIO FINAL")
    print(f"Total de requisições: {TOTAL_REQUESTS}")
    print(f"Sucesso: {errors['sucesso']}")
    print(f"Timeouts: {errors['timeout']}")
    print(f"Erros 500: {errors['500']}")
    print(f"Outros erros: {errors['outras']}")

    if latencies:
        print(f"Média de latência: {statistics.mean(latencies):.2f}s")
        print(f"Máx: {max(latencies):.2f}s | Mín: {min(latencies):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
