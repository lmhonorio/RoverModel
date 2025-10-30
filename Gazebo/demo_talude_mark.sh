#!/bin/bash
# Script de demonstração do Talude_mark.py

echo "=== Demonstração do Talude Marker ==="
echo ""

# Verificar se o ROS está rodando
if ! pgrep -f "rosmaster" > /dev/null; then
    echo "❌ ROS não está rodando. Execute 'roscore' primeiro."
    exit 1
fi

# Verificar se o Gazebo está rodando
if ! pgrep -f "gzserver" > /dev/null; then
    echo "❌ Gazebo não está rodando. Execute o Gazebo primeiro."
    exit 1
fi

echo "✅ ROS e Gazebo estão rodando!"
echo ""

# Navegar para o diretório
cd /home/viki/RoverModel/Gazebo

# Executar o programa com diferentes nomes
echo "Criando Talude_1..."
source /opt/ros/noetic/setup.bash
python3 Talude_mark.py Talude_1

echo ""
echo "Tentando criar Talude_1 novamente (deve mostrar que já existe)..."
python3 Talude_mark.py Talude_1

echo ""
echo "Criando Talude_2..."
python3 Talude_mark.py Talude_2

echo ""
echo "=== Resultado ==="
echo "Arquivo criado: Taludes_marker.csv"
echo "Conteúdo:"
cat Taludes_marker.csv

echo ""
echo "✅ Demonstração concluída!"


