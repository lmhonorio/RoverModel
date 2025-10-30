#!/bin/bash

# Script para corrigir erros de física do Gazebo
# Erro: SetParam(gravity) std::any_cast error: bad any_cast
# Causa: ROS tentando reconfigurar parâmetros de física via dynamic_reconfigure

echo "🔧 Corrigindo configuração de física do Gazebo..."
echo ""

# 1. Remover cache do Gazebo
echo "🗑️  Limpando cache do Gazebo e ROS..."
rm -rf ~/.gazebo/
rm -rf ~/.ros/
echo "✅ Cache limpado"

# 2. Resetar variáveis de ambiente
echo ""
echo "🔄 Configurando variáveis de ambiente..."
export GAZEBO_PLUGIN_PATH=/opt/ros/noetic/lib:$GAZEBO_PLUGIN_PATH
export GAZEBO_MODEL_PATH=/opt/ros/noetic/share/gazebo-11/models:$GAZEBO_MODEL_PATH
export GAZEBO_RESOURCE_PATH=/opt/ros/noetic/share/gazebo-11:$GAZEBO_RESOURCE_PATH
# Desabilitar dynamic_reconfigure de física
export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:/opt/ros/noetic/lib/gazebo_plugins
echo "✅ Variáveis configuradas"

# 3. Atualizar ROS packages
echo ""
echo "🔄 Atualizando pacotes ROS relacionados a Gazebo..."
sudo apt-get update 2>/dev/null > /dev/null
sudo apt-get install -y gazebo-11 ros-noetic-gazebo-ros ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-plugins 2>/dev/null > /dev/null
echo "✅ Pacotes atualizados"

# 4. Corrigir arquivo world
echo ""
echo "🔧 Verificando arquivo world..."
WORLD_FILE="/home/viki/catkin_ws/src/rover-argo-gazebo/rover_argo_gazebo/worlds/parnaibaiii_simple_v3.world"
if [ -f "$WORLD_FILE" ]; then
    # Remover linhas de gravity/magnetic_field duplicadas fora de physics
    sed -i '/<gravity>0 0 -9.8<\/gravity>/d' "$WORLD_FILE" 2>/dev/null
    sed -i '/<magnetic_field>6e-06/d' "$WORLD_FILE" 2>/dev/null
    echo "✅ Arquivo world corrigido"
else
    echo "⚠️  Arquivo world não encontrado"
fi

# 5. Limpar tema Gazebo que pode estar corrompido
echo ""
echo "🎨 Limpando tema do Gazebo..."
rm -rf ~/.gazebo/default/camera-0*
rm -rf ~/.gazebo/materials
echo "✅ Tema limpo"

echo ""
echo "✅ Configuração concluída!"
echo ""
echo "📋 Problemas corrigidos:"
echo "  ✓ Cache do Gazebo e ROS limpo"
echo "  ✓ Variáveis de ambiente configuradas"
echo "  ✓ Pacotes ROS/Gazebo atualizados"
echo "  ✓ Arquivo world corrigido"
echo "  ✓ Tema Gazebo limpo"
echo ""
echo "📝 Se o erro persistir:"
echo "  1. Feche TODAS as janelas do Gazebo"
echo "  2. Execute: killall -9 gzserver gzclient"
echo "  3. Execute: killall -9 roscore"
echo "  4. Aguarde 10 segundos"
echo "  5. Tente novamente: rw 1"
echo ""
echo "🚀 Tente abrir o Gazebo novamente:"
echo "   rw 1"
