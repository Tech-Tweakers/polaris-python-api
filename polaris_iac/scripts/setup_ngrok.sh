#!/bin/bash

echo "🌐 Iniciando ngrok..."

# Verifica se o ngrok está instalado
if ! command -v ngrok &> /dev/null; then
    echo "❌ Erro: ngrok não está instalado! Instale antes de continuar."
    exit 1
fi

# Inicia o ngrok e roda em background
ngrok http "$TELEGRAM_BOT_PORT" > /dev/null 2>&1 &

# Aguarda alguns segundos para garantir que o ngrok suba
sleep 5

# Verifica se o jq está instalado
if ! command -v jq &> /dev/null; then
    echo "❌ Erro: jq não está instalado! Instale com: sudo apt install jq"
    exit 1
fi

# Obtém a URL pública do ngrok
NGROK_URL=""
for i in {1..5}; do  # Tenta 5 vezes para garantir que a URL foi gerada
    NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | jq -r '.tunnels[] | select(.proto=="https") | .public_url')
    if [[ "$NGROK_URL" != "null" && -n "$NGROK_URL" ]]; then
        break
    fi
    echo "⌛ Aguardando ngrok gerar a URL ($i/10)..."
    sleep 2
done

if [[ -z "$NGROK_URL" || "$NGROK_URL" == "null" ]]; then
    echo "❌ Erro: Não foi possível obter a URL do ngrok!"
    exit 1
fi

echo "🌍 URL do Webhook: $NGROK_URL"

echo "📡 Configurando Webhook no Telegram..."
RESPONSE=$(curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_TOKEN/setWebhook" \
     -d "url=$NGROK_URL/webhook")

if [[ "$RESPONSE" == *"\"ok\":true"* ]]; then
    echo "✅ Webhook configurado com sucesso!"
else
    echo "❌ Erro ao configurar Webhook no Telegram. Resposta da API:"
    echo "$RESPONSE"
fi
