#!/bin/bash

# Script para iniciar 1 a 3 instâncias do simulador de rover ArduPilot com Gazebo
# Uso: ./start_three_rovers.sh [numero_de_rovers] [mundo]
# Exemplo: ./start_three_rovers.sh 1  (abre 1 rover)
#          ./start_three_rovers.sh 2  (abre 2 rovers)
#          ./start_three_rovers.sh 3  (abre 3 rovers)
#          ./start_three_rovers.sh    (padrão: 1 rover)

# Diretório do ArduPilot Rover
ARDUPILOT_ROVER_DIR="/home/viki/ardupilot/ardupilot/Rover"
GZROVER_PARAM="$ARDUPILOT_ROVER_DIR/gzrover.param"

# Número de rovers (padrão: 1)
NUM_ROVERS="${1:-1}"

# Selecionar mundo
WORLD_NAME="${2}"

# Se mundo não foi fornecido, perguntar ao usuário
if [ -z "$WORLD_NAME" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "🌍 Escolha o mundo para a simulação:"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "  1 - parnaiba3"
    echo "  2 - parnaibaiii_charlie_delta"
    echo "  3 - parnaibaiii_simple_v3 (padrão)"
    echo ""
    read -p "Digite o número do mundo (1-3) [padrão: 3]: " world_choice
    
    case "${world_choice:-3}" in
        1)
            WORLD_NAME="parnaiba3"
            echo "✅ Mundo selecionado: parnaiba3"
            ;;
        2)
            WORLD_NAME="parnaibaiii_charlie_delta"
            echo "✅ Mundo selecionado: parnaibaiii_charlie_delta"
            ;;
        3|*)
            WORLD_NAME="parnaibaiii_simple_v3"
            echo "✅ Mundo selecionado: parnaibaiii_simple_v3 (padrão)"
            ;;
    esac
    echo ""
fi

# Validar número de rovers
if ! [[ "$NUM_ROVERS" =~ ^[1-3]$ ]]; then
    echo "❌ Erro: Número de rovers deve ser 1, 2 ou 3"
    echo "Uso: $0 [numero_de_rovers] [mundo]"
    echo "Padrão: 1 rover, mundo padrão"
    exit 1
fi

# Função para verificar e fechar processos existentes
kill_existing_processes() {
    echo "🔍 Verificando processos abertos..."
    echo ""
    
    local killed_count=0
    
    # Array de processos a verificar
    local processes=("mavproxy" "gazebo" "gzserver" "gzclient" "roslaunch" "sim_vehicle" "mavros" "ardurover" "rosmaster" "rosnode" "rostopic")
    
    for process in "${processes[@]}"; do
        if pgrep -f "$process" > /dev/null 2>&1; then
            local count=$(pgrep -f "$process" | wc -l)
            echo "⏹️  Encontrados $count processo(s) de '$process'. Fechando..."
            pkill -9 -f "$process" 2>/dev/null
            sleep 0.5
            killed_count=$((killed_count + count))
        fi
    done
    
    # Aguardar um pouco para garantir que tudo foi fechado
    sleep 2
    
    # Verificação final detalhada
    echo ""
    echo "📋 Verificação final:"
    
    local still_running=0
    for process in "${processes[@]}"; do
        local remaining=$(pgrep -f "$process" 2>/dev/null | wc -l)
        if [ "$remaining" -gt 0 ]; then
            echo "  ⚠️  $process: $remaining processo(s) ainda em execução"
            still_running=$((still_running + remaining))
            # Tentar forçar novamente
            killall -9 "$process" 2>/dev/null
        else
            echo "  ✅ $process: nenhum processo"
        fi
    done
    
    echo ""
    
    if [ "$still_running" -eq 0 ]; then
        echo "✅ Todos os processos foram fechados com sucesso!"
        echo "   (Total de $killed_count processo(s) encerrado(s))"
    else
        echo "⚠️  Aviso: $still_running processo(s) ainda estão em execução"
        echo "   Tentando força máxima..."
        sleep 1
        # Última tentativa com força máxima
        pkill -9 -f "mavproxy\|gazebo\|roslaunch\|sim_vehicle\|mavros" 2>/dev/null
        sleep 1
    fi
    
    echo ""
}

# Executar limpeza de processos
kill_existing_processes

echo "🚀 Iniciando $NUM_ROVERS rover(s) com mundo: $WORLD_NAME"
echo ""

# Definir portas base
declare -a ROVER_PORTS=(14551 14552 14553)
declare -a MAVPROXY_OUT1=(14561 14562 14563)
declare -a MAVPROXY_OUT2=(14571 14572 14573)
declare -a SOURCE_SYSTEMS=(250 251 252)

# Abrir rovers conforme o número solicitado
for ((i=1; i<=NUM_ROVERS; i++)); do
    port_idx=$((i-1))
    rover_port=${ROVER_PORTS[$port_idx]}
    mavproxy_out1=${MAVPROXY_OUT1[$port_idx]}
    mavproxy_out2=${MAVPROXY_OUT2[$port_idx]}
    source_sys=${SOURCE_SYSTEMS[$port_idx]}
    
    echo "📡 Abrindo Rover $i (porta $rover_port)..."
    
    # Terminal para Robô
    gnome-terminal --title="Rover $i" -- bash -c "cd '$ARDUPILOT_ROVER_DIR' && sim_vehicle.py -L SEParnaiba -S 5 -v Rover --sysid $i --instance $port_idx -f rover-skid --out=udp:127.0.0.1:$rover_port --add-param-file '$GZROVER_PARAM'; exec bash"
    
    sleep 1
    
    # Terminal para MAVProxy
    gnome-terminal --title="MAVProxy Rover $i" -- bash -c "mavproxy.py \
      --master=udp:127.0.0.1:$rover_port \
      --out=udp:127.0.0.1:$mavproxy_out1 \
      --out=udp:127.0.0.1:$mavproxy_out2 \
      --source-system=$source_sys \
      --console \
      --cmd='map grid; map follow'; exec bash"
    
    sleep 1
done

# Terminal para agregar todos os rovês e enviar para QGC
echo "🔗 Abrindo agregador de MAVProxy..."

if [ "$NUM_ROVERS" -eq 1 ]; then
    master_ports="--master=udp:127.0.0.1:14571"
elif [ "$NUM_ROVERS" -eq 2 ]; then
    master_ports="--master=udp:127.0.0.1:14571 --master=udp:127.0.0.1:14572"
else
    master_ports="--master=udp:127.0.0.1:14571 --master=udp:127.0.0.1:14572 --master=udp:127.0.0.1:14573"
fi

gnome-terminal --title="MAVProxy Agregador" -- bash -c "mavproxy.py \
  $master_ports \
  --out=udp:192.168.0.131:14550 \
  --console \
  --map \
  --cmd='map grid; map follow'; exec bash"

sleep 1

# Terminal para ROS Gazebo
echo "🌍 Abrindo Gazebo com $NUM_ROVERS rover(s)..."

gnome-terminal --title="ROS Gazebo - $NUM_ROVERS Rovers" -- bash -c "cd /home/viki/catkin_ws && source devel/setup.bash && roslaunch rover_argo_gazebo multi_rover_argo.launch N:=$NUM_ROVERS world_name:=$WORLD_NAME; exec bash"

echo ""
echo "✅ Todos os terminais foram abertos!"
echo "   Número de rovers: $NUM_ROVERS"
echo "   Mundo: $WORLD_NAME"
echo "   Verifique os terminais para mais informações"
