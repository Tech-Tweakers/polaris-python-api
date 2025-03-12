#!/bin/bash

echo "🌐 Iniciando ngrok..."

if ! command -v ngrok &> /dev/null; then
    echo "❌ Erro: ngrok não está instalado! Instale antes de continuar."
    exit 1
fi

ngrok http "$TELEGRAM_BOT_PORT" > /dev/null 2>&1 &

sleep 5

if ! command -v jq &> /dev/null; then
    echo "❌ Erro: jq não está instalado! Instale com: sudo apt install jq"
    exit 1
fi

NGROK_URL=""
for i in {1..5}; do
    NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | jq -r '.tunnels[] | select(.proto=="https") | .public_url')
    if [[ "$NGROK_URL" != "null" && -n "$NGROK_URL" ]]; then
        break
    fi
    echo "⌛ Aguardando ngrok gerar a URL ($i/5)..."
    sleep 2
done

if [[ -z "$NGROK_URL" || "$NGROK_URL" == "null" ]]; then
    echo "❌ Erro: Não foi possível obter a URL do ngrok!"
    exit 1
fi

echo "🌍 URL do Webhook: $NGROK_URL"

echo "📡 Configurando Webhook no Telegram..."
RESPONSE=$(curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_TOKEN/setWebhook" \
     -d "url=$NGROK_URL/telegram-webhook/")

if [[ "$RESPONSE" == *"\"ok\":true"* ]]; then
    echo "✅ Webhook configurado com sucesso!"
else
    echo "❌ Erro ao configurar Webhook no Telegram. Resposta da API:"
    echo "$RESPONSE"
fi
