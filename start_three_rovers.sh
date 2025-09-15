#!/bin/bash

# Script para iniciar três instâncias do simulador de rover ArduPilot
# e um terminal ROS para o Gazebo
# Cada instância roda em uma porta UDP diferente

# Cores para os terminais (opcional)
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Diretório do ArduPilot
ARDUPILOT_DIR="/home/gabrielle/ardupilot/ardupilot"

# Diretório do ROS workspace
ROS_WS_DIR="/home/gabrielle/catkin_ws"

# Verificar se o diretório existe
if [ ! -d "$ARDUPILOT_DIR" ]; then
    echo "Erro: Diretório do ArduPilot não encontrado em $ARDUPILOT_DIR"
    exit 1
fi

# Verificar se o diretório ROS existe
if [ ! -d "$ROS_WS_DIR" ]; then
    echo "Erro: Diretório do ROS workspace não encontrado em $ROS_WS_DIR"
    exit 1
fi

# Verificar se o sim_vehicle.py existe
SIM_VEHICLE_PATH="$ARDUPILOT_DIR/Tools/autotest/sim_vehicle.py"
if [ ! -f "$SIM_VEHICLE_PATH" ]; then
    echo "Erro: sim_vehicle.py não encontrado em $SIM_VEHICLE_PATH"
    exit 1
fi

echo "Iniciando três instâncias do simulador de rover e ROS Gazebo..."
echo "Diretório ArduPilot: $ARDUPILOT_DIR"
echo "Diretório ROS: $ROS_WS_DIR"
echo ""

# Função para abrir terminal com comando
open_terminal() {
    local title="$1"
    local command="$2"
    local color="$3"
    local work_dir="$4"
    
    # Usar diretório padrão se não especificado
    if [ -z "$work_dir" ]; then
        work_dir="$ARDUPILOT_DIR"
    fi
    
    # Detectar o terminal padrão e abrir com o comando
    if command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal --title="$title" -- bash -c "cd '$work_dir' && echo -e '${color}=== $title ===${NC}' && echo 'Executando: $command' && echo '' && $command; exec bash"
    elif command -v xterm >/dev/null 2>&1; then
        xterm -title "$title" -e "cd '$work_dir' && echo -e '${color}=== $title ===${NC}' && echo 'Executando: $command' && echo '' && $command; exec bash" &
    elif command -v konsole >/dev/null 2>&1; then
        konsole --title "$title" -e "cd '$work_dir' && echo -e '${color}=== $title ===${NC}' && echo 'Executando: $command' && echo '' && $command; exec bash" &
    else
        echo "Terminal não suportado. Execute manualmente:"
        echo "cd '$work_dir'"
        echo "$command"
    fi
}

echo "Abrindo Rover 1 (porta 14555)..."
open_terminal "Rover 1 - Porta 14555" \
"python3 $SIM_VEHICLE_PATH -v Rover -f gazebo-rover -L ARGO gzrover.param --out=udp:127.0.0.1:14555 -I0" "$RED"

sleep 2

echo "Abrindo Rover 2 (porta 14565)..."
open_terminal "Rover 2 - Porta 14565" \
"python3 $SIM_VEHICLE_PATH -v Rover -f gazebo-rover -L ARGO gzrover.param --out=udp:127.0.0.1:14565 -I1" "$GREEN"

sleep 2

echo "Abrindo Rover 3 (porta 14575)..."
open_terminal "Rover 3 - Porta 14575" \
"python3 $SIM_VEHICLE_PATH -v Rover -f gazebo-rover -L ARGO gzrover.param --out=udp:127.0.0.1:14575 -I2" "$BLUE"

sleep 2

# Abrir terminal ROS Gazebo
echo "Abrindo ROS Gazebo (3 rovers)..."
open_terminal "ROS Gazebo - 3 Rovers" "source devel/setup.bash && roslaunch rover_argo_gazebo multi_rover_argo.launch N:=3 world_name:=parnaibaiii_simple_v3" "$YELLOW" "$ROS_WS_DIR"

echo ""
echo "Todos os simuladores foram iniciados!"
echo "Portas UDP dos Rovers:"
echo "  - Rover 1: 127.0.0.1:14555"
echo "  - Rover 2: 127.0.0.1:14565"
echo "  - Rover 3: 127.0.0.1:14575"
echo ""
echo "ROS Gazebo:"
echo "  - Terminal ROS com 3 rovers no mundo parnaibaiii_simple_v3"
echo ""
echo "Para parar os simuladores, feche os terminais ou pressione Ctrl+C em cada um."
