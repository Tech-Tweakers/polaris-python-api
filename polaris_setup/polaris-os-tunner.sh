#!/bin/bash

# Polaris-Tuner: Otimizador de Performance para Ubuntu 🚀
# USE POR SUA PROPRIA CONTA E RISCO!!! <3

set -e

green="\e[32m"
yellow="\e[33m"
red="\e[31m"
nc="\e[0m"

echo -e "${green}🔥 Polaris-Tuner: Otimizador de Performance para Ubuntu 🔥${nc}\n"

if [[ $EUID -ne 0 ]]; then
    echo -e "${red}❌ Este script deve ser executado como root!${nc}"
    exit 1
fi

optimize_cpu() {
    echo -e "${yellow}⚙️  Ajustando CPU para máxima performance...${nc}"
    apt install -y cpufrequtils
    echo 'GOVERNOR="performance"' > /etc/default/cpufrequtils
    systemctl restart cpufrequtils
    echo -e "${green}✅ CPU configurada para performance!${nc}"
}

optimize_memory() {
    echo -e "${yellow}🧠 Otimizando uso de memória RAM...${nc}"
    echo 'vm.swappiness=10' >> /etc/sysctl.conf
    echo 'vm.nr_hugepages=512' >> /etc/sysctl.conf
    sysctl -p
    echo -e "${green}✅ Memória otimizada!${nc}"
}

optimize_disk() {
    echo -e "${yellow}💾 Ajustando disco para maior velocidade...${nc}"
    systemctl enable fstrim.timer
    systemctl start fstrim.timer
    echo -e "${yellow}🔄 Ajustando o agendador de I/O para SSD/HDD...${nc}"
    
    AVAILABLE_SCHEDULERS=$(cat /sys/block/sda/queue/scheduler)
    if echo "$AVAILABLE_SCHEDULERS" | grep -q "mq-deadline"; then
        echo "mq-deadline" | sudo tee /sys/block/sda/queue/scheduler
    elif echo "$AVAILABLE_SCHEDULERS" | grep -q "kyber"; then
        echo "kyber" | sudo tee /sys/block/sda/queue/scheduler
    elif echo "$AVAILABLE_SCHEDULERS" | grep -q "bfq"; then
        echo "bfq" | sudo tee /sys/block/sda/queue/scheduler
    else
        echo -e "${red}⚠️ Nenhum agendador compatível encontrado!${nc}"
    fi
    
    echo -e "${green}✅ Disco otimizado!${nc}"
}

optimize_network() {
    echo -e "${yellow}🌐 Melhorando desempenho de rede...${nc}"
    echo 'net.core.rmem_max=16777216' >> /etc/sysctl.conf
    echo 'net.core.wmem_max=16777216' >> /etc/sysctl.conf
    echo 'net.core.default_qdisc=fq' >> /etc/sysctl.conf
    echo 'net.ipv4.tcp_congestion_control=bbr' >> /etc/sysctl.conf
    sysctl -p
    echo -e "${green}✅ Rede otimizada!${nc}"
}

remove_unnecessary_services() {
    echo -e "${yellow}🔧 Desativando serviços desnecessários...${nc}"
    systemctl disable bluetooth || true
    systemctl stop bluetooth || true
    echo -e "${green}✅ Serviços desativados!${nc}"
}

optimize_ui() {
    echo -e "${yellow}🎨 Desativando animações para maior fluidez...${nc}"
    gsettings set org.gnome.desktop.interface enable-animations false
    echo -e "${green}✅ Interface mais responsiva!${nc}"
}

install_essentials() {
    echo -e "${yellow}📦 Instalando pacotes essenciais...${nc}"
    apt install -y preload
    echo -e "${green}✅ Pacotes instalados!${nc}"
}

echo -e "${yellow}🚀 Iniciando otimizações...${nc}"
optimize_cpu
optimize_memory
optimize_disk
optimize_network
remove_unnecessary_services
optimize_ui
install_essentials
echo -e "${green}🎉 Polaris-Tuner aplicado com sucesso! Reinicie para aplicar todas as mudanças.${nc}"
