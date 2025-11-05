#!/bin/bash

echo "🛑 Fechando TODOS os processos da simulação..."
echo ""

# Array de processos a verificar
processes=("mavproxy" "gazebo" "gzserver" "gzclient" "roslaunch" "sim_vehicle" "mavros" "ardurover" "rosmaster" "rosnode" "rostopic" "rosbridge")

for process in "${processes[@]}"; do
    if pgrep -f "$process" > /dev/null 2>&1; then
        count=$(pgrep -f "$process" | wc -l)
        echo "⏹️  Fechando $count processo(s) de '$process'..."
        pkill -9 -f "$process" 2>/dev/null
    fi
done

sleep 2

echo ""
echo "✅ Todos os processos foram fechados!"
echo ""
echo "Verificação final:"
still_running=0
for process in "${processes[@]}"; do
    remaining=$(pgrep -f "$process" 2>/dev/null | wc -l)
    if [ "$remaining" -gt 0 ]; then
        echo "  ⚠️  $process: $remaining processo(s) ainda rodando"
        still_running=1
    fi
done

if [ "$still_running" -eq 0 ]; then
    echo "  ✅ Nenhum processo restante"
fi
echo ""

