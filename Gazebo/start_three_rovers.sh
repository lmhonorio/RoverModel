#!/usr/bin/env bash
set -euo pipefail

# Script para iniciar 1 a 3 instâncias do simulador de rover ArduPilot com Gazebo
# Uso: ./start_three_rovers.sh [numero_de_rovers] [mundo]
# Exemplo: ./start_three_rovers.sh 1  (abre 1 rover)
#          ./start_three_rovers.sh 2  (abre 2 rovers)
#          ./start_three_rovers.sh 3  (abre 3 rovers)
#          ./start_three_rovers.sh    (pergunta quantos rovers)

# ===================== DESATIVAR VENV =====================
echo "🔍 Verificando e desativando ambientes virtuais Python..."
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "  ⚠️  Ambiente virtual detectado: $VIRTUAL_ENV"
    echo "  🔓 Desativando venv..."
    deactivate 2>/dev/null || true
    unset VIRTUAL_ENV
    unset PYTHONHOME
    echo "  ✅ Venv desativado!"
else
    echo "  ✅ Nenhum venv ativo"
fi

# Limpar variáveis de ambiente relacionadas ao venv
unset VIRTUAL_ENV
unset PYTHONHOME
export PATH=$(echo "$PATH" | sed -e 's|[^:]*venv[^:]*:||g' -e 's|:[^:]*venv[^:]*||g')
echo "  ✅ Variáveis de ambiente limpas"
echo ""

# ===================== CONFIGURAÇÃO =====================
# Diretório do ArduPilot
ARDUPILOT_DIR="${HOME}/ardupilot"
ARDUPILOT_ROVER_DIR="${ARDUPILOT_DIR}/Rover"
PARAM_FILE="$ARDUPILOT_ROVER_DIR/gzrover.param"

# Localização padrão
#DEFAULT_LOCATION="SEParnaiba"
DEFAULT_LOCATION="ARGO"

# ===================== PARÂMETROS =====================
QGC_PORT=14550          # QGC recebe de todas as instâncias
MAV_BASE=14551          # porta base p/ MAVROS (instância 0)
STEP=100                # incremento por instância: 14551, 14651, 14751...
SYSID0=1                # SYSID da instância 0
I0=0                    # índice base do --instance (-I)
FRAME="rover-skid"
MODEL="gazebo-rover"
PAUSE=2

# ===================== NÚMERO DE ROVERS =====================
# Se fornecido como argumento, usar; senão perguntar
if [ -n "${1:-}" ]; then
    NUM_ROVERS="$1"
else
    echo ""
    read -p "Quantos robôs (1-3)? [1]: " NUM_ROVERS
    NUM_ROVERS=${NUM_ROVERS:-1}
fi

# Validar número de rovers
if ! [[ "$NUM_ROVERS" =~ ^[1-3]$ ]]; then
    echo "❌ Erro: Número de rovers deve ser 1, 2 ou 3"
    echo "Uso: $0 [numero_de_rovers] [mundo]"
    echo "Exemplo: $0 1 parnaibaiii_simple_v3"
    exit 1
fi

# ===================== SELECIONAR MUNDO =====================
WORLD_NAME="${2:-}"

# Se mundo não foi fornecido, perguntar ao usuário
if [ -z "$WORLD_NAME" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "🌍 Escolha o mundo para a simulação:"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "  1 - parnaibaiii_simple_v2"
    echo "  2 - parnaibaiii_charlie_delta"
    echo "  3 - parnaibaiii_simple_v3 (padrão)"
    echo "  4 - gravel_plane"
    echo ""
    read -p "Digite o número do mundo (1-4) [padrão: 3]: " world_choice
    
    case "${world_choice:-3}" in
        1)
            WORLD_NAME="parnaibaiii_simple_v2"
            echo "✅ Mundo selecionado: parnaibaiii_simple_v2"
            ;;
        2)
            WORLD_NAME="parnaibaiii_charlie_delta"
            echo "✅ Mundo selecionado: parnaibaiii_charlie_delta"
            ;;
        3|*)
            WORLD_NAME="parnaibaiii_simple_v3"
            echo "✅ Mundo selecionado: parnaibaiii_simple_v3 (padrão)"
            ;;
        4)
            WORLD_NAME="gravel_plane"
            echo "✅ Mundo selecionado: gravel_plane"
            ;;
    esac
    echo ""
fi

# ===================== RESOLVE sim_vehicle.py =====================
SIMV_BIN="$(command -v sim_vehicle.py || true)"
if [[ -z "$SIMV_BIN" ]]; then
  CAND="${ARDUPILOT_DIR}/Tools/autotest/sim_vehicle.py"
  if [[ -x "$CAND" ]]; then
    SIMV_BIN="$CAND"
  else
    echo "[ERRO] sim_vehicle.py não encontrado no PATH nem em $CAND"
    echo "       Ajuste ARDUPILOT_DIR ou instale corretamente o ArduPilot."
    exit 1
  fi
fi

# ===================== TERMINAL HELPER =====================
pick_term() {
  if   command -v gnome-terminal >/dev/null 2>&1; then echo "gnome-terminal"
  elif command -v konsole         >/dev/null 2>&1; then echo "konsole"
  elif command -v mate-terminal   >/dev/null 2>&1; then echo "mate-terminal"
  elif command -v kitty           >/dev/null 2>&1; then echo "kitty"
  elif command -v alacritty       >/dev/null 2>&1; then echo "alacritty"
  elif command -v xterm           >/dev/null 2>&1; then echo "xterm"
  else echo ""; fi
}

open_term() {
  local title="$1"; shift
  local remote_cmd="$*"
  local term="$(pick_term)"

  # Sem GUI (DISPLAY vazio) ou sem terminal gráfico → roda no terminal atual
  if [[ -z "${DISPLAY:-}" || -z "$term" ]]; then
    echo "[WARN] Sem terminal gráfico disponível; rodando aqui mesmo."
    bash -lc "$remote_cmd"
    return
  fi

  case "$term" in
    gnome-terminal)
      gnome-terminal --title "$title" -- bash -lc "$remote_cmd" || {
        echo "[WARN] gnome-terminal falhou; rodando aqui."
        bash -lc "$remote_cmd"
      }
      ;;
    konsole)
      konsole --new-tab -p tabtitle="$title" -e bash -lc "$remote_cmd" || bash -lc "$remote_cmd" &
      ;;
    mate-terminal)
      mate-terminal --title "$title" -- bash -lc "$remote_cmd" || bash -lc "$remote_cmd"
      ;;
    kitty)
      kitty --title "$title" bash -lc "$remote_cmd" || bash -lc "$remote_cmd" &
      ;;
    alacritty)
      alacritty -t "$title" -e bash -lc "$remote_cmd" || bash -lc "$remote_cmd" &
      ;;
    xterm)
      xterm -T "$title" -hold -e bash -lc "$remote_cmd" || bash -lc "$remote_cmd" &
      ;;
  esac
}

# ===================== VERIFICAR E FECHAR PROCESSOS =====================
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

# ===================== EXECUTAR LIMPEZA DE PROCESSOS =====================
kill_existing_processes

# ===================== LIMPAR ARQUIVOS DE ESTADO DO SITL =====================
echo "🧹 Limpando arquivos de estado do ArduPilot SITL e Gazebo..."
echo ""

# Limpar eeprom.bin e arquivos de terrain que podem persistir estado antigo
if [ -f "$ARDUPILOT_ROVER_DIR/eeprom.bin" ]; then
    echo "  🗑️  Removendo eeprom.bin (parâmetros salvos)"
    rm -f "$ARDUPILOT_ROVER_DIR/eeprom.bin"
fi

if [ -d "$ARDUPILOT_ROVER_DIR/terrain" ]; then
    echo "  🗑️  Removendo pasta terrain"
    rm -rf "$ARDUPILOT_ROVER_DIR/terrain"
fi

# Limpar arquivos de estado de múltiplas instâncias
for i in {0..2}; do
    if [ -f "$ARDUPILOT_ROVER_DIR/eeprom${i}.bin" ]; then
        echo "  🗑️  Removendo eeprom${i}.bin"
        rm -f "$ARDUPILOT_ROVER_DIR/eeprom${i}.bin"
    fi
done

# Limpar cache e estado do Gazebo
if [ -d "$HOME/.gazebo/log" ]; then
    echo "  🗑️  Limpando logs do Gazebo"
    rm -rf "$HOME/.gazebo/log/"*
fi

if [ -f "$HOME/.gazebo/server-11345/default.log" ]; then
    echo "  🗑️  Removendo logs de servidor do Gazebo"
    rm -f "$HOME/.gazebo/server-"*/default.log 2>/dev/null || true
fi

# Limpar possíveis arquivos de estado de modelos
if [ -d "/tmp/.gazebo" ]; then
    echo "  🗑️  Limpando cache temporário do Gazebo"
    rm -rf /tmp/.gazebo 2>/dev/null || true
fi

echo "✅ Limpeza concluída!"
echo ""

echo "🚀 Iniciando $NUM_ROVERS rover(s) com mundo: $WORLD_NAME"
echo ""

# ═══════════════════════════════════════════════════════════
# PASSO 1: Iniciar Gazebo PRIMEIRO (para estabelecer portas UDP)
# ═══════════════════════════════════════════════════════════
echo "🌍 Iniciando Gazebo com $NUM_ROVERS rover(s)..."
echo "   Aguarde o Gazebo carregar completamente antes de iniciar os rovers..."
echo ""

# CRÍTICO: Adicionar plugin ArduPilot ao GAZEBO_PLUGIN_PATH
export GAZEBO_PLUGIN_PATH=/home/gabrielle/catkin_ws/src/ardupilot_gazebo/build:${GAZEBO_PLUGIN_PATH}
echo "✅ GAZEBO_PLUGIN_PATH configurado com plugin ArduPilot"
echo ""

# Iniciar Gazebo com roslaunch
gazebo_cmd="unset VIRTUAL_ENV; unset PYTHONHOME; export GAZEBO_PLUGIN_PATH=/home/gabrielle/catkin_ws/src/ardupilot_gazebo/build:\${GAZEBO_PLUGIN_PATH} && cd /home/gabrielle/catkin_ws && source devel/setup.bash && roslaunch rover_argo_gazebo multi_rover_argo.launch N:=$NUM_ROVERS world_name:=$WORLD_NAME; echo; echo 'Gazebo finalizado. Pressione ENTER para fechar.'; read"

open_term "ROS Gazebo - $NUM_ROVERS Rovers" "$gazebo_cmd"

# Aguardar Gazebo carregar completamente (crítico!)
echo "⏳ Aguardando Gazebo inicializar (15 segundos)..."
sleep 15

# ═══════════════════════════════════════════════════════════
# PASSO 2: Iniciar ArduPilot SITL rovers
# ═══════════════════════════════════════════════════════════
echo ""
echo "🚁 Iniciando instâncias ArduPilot SITL..."
echo ""

for ((i=0; i<NUM_ROVERS; i++)); do
  inst=$((I0 + i))
  sysid=$((SYSID0 + i))
  mav_port=$((MAV_BASE + STEP*i))
  log="/tmp/sitl_${inst}.log"

  # Comando que será executado DENTRO de cada terminal
  remote_cmd=$(cat <<EOF
# Desativar venv se existir
unset VIRTUAL_ENV;
unset PYTHONHOME;
export PATH=0;
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games";
source ~/.bashrc 2>/dev/null || true;
[ -d "\$HOME/.local/bin" ] && export PATH="\$PATH:\$HOME/.local/bin";
echo "[ENV] VENV desativado (se existia)";
echo "[ENV] PATH=\$PATH";
echo "[ENV] using: $SIMV_BIN";
echo "[INFO] Mudando para diretório: $ARDUPILOT_ROVER_DIR";
cd "$ARDUPILOT_ROVER_DIR" || exit 1;
"$SIMV_BIN" -D \\
  -v Rover \\
  --out=udp:127.0.0.1:${QGC_PORT} \\
  --out=udp:127.0.0.1:${mav_port} \\
  -L "${DEFAULT_LOCATION}" -f ${FRAME} --model ${MODEL} \\
  -I ${inst} --sysid ${sysid} \\
  --console --add-param-file ${PARAM_FILE} \\
  2>&1 | tee -a "$log" ;
echo; echo "Log: $log";
echo "[SITL] Instância ${i}  (I=${inst}, SYSID=${sysid})  QGC:${QGC_PORT}  MAVROS:${mav_port}"
echo "Finalizado. Pressione ENTER para fechar.";
read
EOF
)

  echo "[SITL] Instância ${i}  (I=${inst}, SYSID=${sysid})  QGC:${QGC_PORT}  MAVROS:${mav_port}"
  open_term "SITL I=${inst} SYSID=${sysid} MAV=${mav_port}" "$remote_cmd"
  sleep "$PAUSE"
done

echo ""
echo "✅ Sistema iniciado com sucesso!"
echo "   - Gazebo: $NUM_ROVERS rovers no mundo $WORLD_NAME"
echo "   - ArduPilot SITL: $NUM_ROVERS instâncias rodando"
echo ""
echo "💡 Lembrete: cada MAVROS deve escutar nessas portas (udp://:<porta>@):"
echo "  Instância 0 → ${MAV_BASE} (14551)"
echo "  Instância 1 → $((MAV_BASE+STEP)) (14651)"
echo "  Instância 2 → $((MAV_BASE+STEP*2)) (14751)"
echo "  Se 'não acontecer nada', veja os logs em /tmp/sitl_<I>.log."
echo ""


# ═══════════════════════════════════════════════════════════
# PASSO 3: Perguntar se deseja executar QGroundControl
# ═══════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════════════"
echo "🛰️  Deseja executar o QGroundControl?"
echo "═══════════════════════════════════════════════════════════"
echo ""
read -p "Executar QGroundControl? (s/N): " qgc_choice

if [[ "$qgc_choice" =~ ^[Ss]$ ]]; then
    # Procurar QGroundControl em Downloads
    QGCPATH=""
    
    if [ -f "$HOME/QGroundControl.AppImage" ]; then
        QGCPATH="$HOME/QGroundControl.AppImage"
    elif [ -f "$HOME/qgroundcontrol.AppImage" ]; then
        QGCPATH="$HOME/qgroundcontrol.AppImage"
    elif [ -f "$HOME/QGC.AppImage" ]; then
        QGCPATH="$HOME/QGC.AppImage"
    else
        # Procurar qualquer AppImage que contenha "ground" no nome
        QGCPATH=$(find "$HOME" -maxdepth 1 -iname "*ground*.AppImage" 2>/dev/null | head -1)
    fi
    
    if [ -n "$QGCPATH" ] && [ -f "$QGCPATH" ]; then
        echo "✅ QGroundControl encontrado: $QGCPATH"
        echo "🚀 Executando QGroundControl..."
        
        # Tornar executável se necessário
        chmod +x "$QGCPATH" 2>/dev/null || true
        
        # Executar QGroundControl em background
        "$QGCPATH" &> /dev/null &
        
        echo "✅ QGroundControl iniciado!"
    else
        echo "❌ QGroundControl não encontrado em ~/Downloads"
        echo "   Procure por arquivos .AppImage do QGroundControl"
        echo "   Caminho esperado: ~/Downloads/QGroundControl.AppImage"
    fi
else
    echo "⏭️  Pulando execução do QGroundControl"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Todos os terminais foram abertos!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📊 Resumo da simulação:"
echo "   • Número de rovers: $NUM_ROVERS"
echo "   • Mundo: $WORLD_NAME"
echo "   • Localização: $DEFAULT_LOCATION"
echo "   • Porta QGC: $QGC_PORT"
echo "   • Portas MAVROS: $MAV_BASE (14551), $((MAV_BASE+STEP)) (14651), $((MAV_BASE+STEP*2)) (14751)"
echo ""
echo "🔧 Verifique os terminais para mais informações"
echo "📝 Logs disponíveis em: /tmp/sitl_*.log"
echo ""
