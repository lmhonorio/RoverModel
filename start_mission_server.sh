#!/bin/bash
# Script para iniciar mission_server com ambiente ROS configurado

echo "🚀 Iniciando Mission Server com ambiente ROS Noetic..."
echo "=" | head -c 80 | tr ' ' '='
echo ""

# Configurar ambiente ROS Noetic
echo "📦 Configurando ambiente ROS Noetic..."
source /opt/ros/noetic/setup.bash

# Verificar se ROS foi configurado
if [ -z "$ROS_DISTRO" ]; then
    echo "❌ Erro: Ambiente ROS não foi configurado!"
    echo "💡 Verifique se ROS Noetic está instalado corretamente"
    exit 1
fi

echo "✅ ROS $ROS_DISTRO configurado"
echo "✅ ROS_PACKAGE_PATH: $ROS_PACKAGE_PATH"
echo ""

# Verificar se Gazebo está rodando
echo "🔍 Verificando se Gazebo está rodando..."
if pgrep -f gzserver > /dev/null; then
    echo "✅ Gazebo detectado rodando"
else
    echo "⚠️  Gazebo não está rodando!"
    echo "💡 Inicie o Gazebo primeiro com: ./world.sh 1"
fi
echo ""

# Ir para diretório do projeto
cd /home/viki/RoverModel

# Iniciar mission_server com ambiente ROS configurado
echo "🌐 Iniciando Mission Server..."
echo "   Porta: 5001"
echo "   Visualização Gazebo: ATIVADA"
echo ""

python3 mission_server.py

# Se mission_server for interrompido
echo ""
echo "🛑 Mission Server parado"


