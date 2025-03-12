#!/bin/bash

# Função para exibir o menu
show_menu() {
    clear
    echo "=================================="
    echo "      🚀 Polaris v2 - Menu        "
    echo "=================================="
    echo "1) 📦 Instalar dependências"
    echo "2) 🤖 Baixar modelo LLaMA 3"
    echo "3) 🐳 Subir MongoDB e Mongo Express"
    echo "4) 🚀 Iniciar API"
    echo "5) 🤖 Iniciar Telegram Bot"
    echo "6) 🌍 Configurar Ngrok + Webhook Telegram"
    echo "7) 📝 Criar .env para API e Bot"
    echo "8) 🔄 Iniciar tudo (API, Bot, DB)"
    echo "9) 🛑 Parar tudo"
    echo "10) 🔄 Reiniciar tudo"
    echo "0) ❌ Sair"
    echo ""
    read -p "Digite a opção desejada: " OPTION
}

# Loop para manter o menu rodando até o usuário sair
while true; do
    show_menu
    case $OPTION in
        1) make install ;;
        2) make download-model ;;
        3) make start-db ;;
        4) make start-api ;;
        5) make start-bot ;;
        6) make setup-ngrok ;;
        7) make create-env-api && make create-env-bot ;;
        8) make start-all ;;
        9) make stop-all ;;
        10) make restart-all ;;
        0) echo "❌ Saindo..." && exit ;;
        *) echo "⚠️ Opção inválida! Tente novamente." && sleep 2 ;;
    esac
done
