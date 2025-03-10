#!/bin/bash

# Carregar variáveis do .env
export $(grep -v '^#' .env | xargs)

echo "🌐 Iniciando ngrok..."
ngrok http $TELEGRAM_BOT_PORT > /dev/null &

# Pega a URL pública gerada pelo ngrok
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | jq -r '.tunnels[0].public_url')

echo "🌍 URL do Webhook: $NGROK_URL"

echo "📡 Configurando Webhook no Telegram..."
curl -X POST "https://api.telegram.org/bot$TELEGRAM_TOKEN/setWebhook" \
     -d "url=$NGROK_URL/webhook"

echo "✅ Webhook configurado com sucesso!"
