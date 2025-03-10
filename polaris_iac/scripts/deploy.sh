#!/bin/bash

echo "🚀 Iniciando deploy da Polaris API..."

export $(grep -v '^#' ./.env | xargs)

COMPOSE_FILE="../docker-compose.yml"

echo "🔧 Construindo as imagens da Polaris..."
docker-compose -f $COMPOSE_FILE build || { echo "❌ Erro ao construir as imagens"; exit 1; }

echo "🛑 Parando os containers antigos..."
docker-compose -f $COMPOSE_FILE down || { echo "⚠️ Falha ao remover containers antigos"; }

echo "🚀 Subindo os containers atualizados..."
docker-compose -f $COMPOSE_FILE up -d --build || { echo "❌ Erro ao subir os containers"; exit 1; }

echo "✅ Deploy concluído com sucesso!"
