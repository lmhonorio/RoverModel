#!/bin/bash

# Script para verificar portas em uso após iniciar a simulação
# Use este script para diagnosticar problemas de comunicação

echo "════════════════════════════════════════════════════════"
echo "🔍 DIAGNÓSTICO DE PORTAS - ArduPilot + Gazebo"
echo "════════════════════════════════════════════════════════"
echo ""

echo "📊 Portas MAVLink (14550-14573):"
echo "────────────────────────────────────────────────────────"
ss -ulpn 2>/dev/null | grep -E ":(1455[0-3]|1456[1-3]|1457[1-3]|14550)" && echo "" || echo "   ⚠️  Nenhuma porta MAVLink em LISTEN (UDP pode não aparecer)"
lsof -i UDP:14550,14551,14552,14553,14561,14562,14563,14571,14572,14573 2>/dev/null | grep -v "COMMAND" || echo "   (Use: sudo lsof -i UDP:14550-14573 para ver mais detalhes)"
echo ""

echo "📊 Portas FDM Gazebo (9002, 9003, 9012, 9013, 9022, 9023):"
echo "────────────────────────────────────────────────────────"
ss -ulpn 2>/dev/null | grep -E ":(900[2-3]|901[2-3]|902[2-3])" && echo "" || echo "   ⚠️  Nenhuma porta FDM em LISTEN"
lsof -i UDP:9002,9003,9012,9013,9022,9023 2>/dev/null | grep -v "COMMAND" || echo "   ❌ CRÍTICO: Plugin ArduPilot não está carregado no Gazebo!"
echo ""

echo "🔍 Processos ArduPilot:"
echo "────────────────────────────────────────────────────────"
pgrep -fa "ardurover|sim_vehicle" || echo "   ❌ Nenhum processo ArduPilot encontrado"
echo ""

echo "🔍 Processos Gazebo:"
echo "────────────────────────────────────────────────────────"
pgrep -fa "gzserver|gzclient" || echo "   ❌ Nenhum processo Gazebo encontrado"
echo ""

echo "🔍 Processos MAVProxy:"
echo "────────────────────────────────────────────────────────"
pgrep -fa "mavproxy" || echo "   ❌ Nenhum processo MAVProxy encontrado"
echo ""

echo "🔍 Verificação do Plugin ArduPilot:"
echo "────────────────────────────────────────────────────────"
PLUGIN_PATH="/home/viki/catkin_ws/src/ardupilot_gazebo/build/libArduPilotPlugin.so"
if [ -f "$PLUGIN_PATH" ]; then
    echo "   ✅ Plugin encontrado: $PLUGIN_PATH"
    ls -lh "$PLUGIN_PATH"
else
    echo "   ❌ Plugin NÃO encontrado em: $PLUGIN_PATH"
    echo "   Compile o plugin com: cd $HOME/catkin_ws/src/ardupilot_gazebo/build && cmake .. && make"
fi
echo ""

echo "🔍 GAZEBO_PLUGIN_PATH:"
echo "────────────────────────────────────────────────────────"
if echo "$GAZEBO_PLUGIN_PATH" | grep -q "ardupilot_gazebo"; then
    echo "   ✅ Plugin ArduPilot no PATH"
else
    echo "   ❌ Plugin ArduPilot NÃO está no GAZEBO_PLUGIN_PATH!"
    echo "   Adicione ao script: export GAZEBO_PLUGIN_PATH=/home/viki/catkin_ws/src/ardupilot_gazebo/build:\$GAZEBO_PLUGIN_PATH"
fi
echo ""

echo "════════════════════════════════════════════════════════"
echo "💡 DICAS DE DIAGNÓSTICO:"
echo "════════════════════════════════════════════════════════"
echo ""
echo "1. Verifique se os processos ArduPilot iniciaram corretamente"
echo "2. Portas FDM devem estar LISTENING (UDP) no gzserver"
echo "3. Portas MAVLink devem estar ativas nos processos ardurover"
echo "4. Se as portas FDM não aparecem, o Gazebo pode não estar"
echo "   carregando o plugin ArduPilotPlugin corretamente"
echo ""
echo "Para ver logs do Gazebo:"
echo "  tail -f ~/.gazebo/server-*.log | grep -i ardupilot"
echo ""
echo "Para testar comunicação UDP manualmente:"
echo "  nc -u -l 9002  # escutar na porta 9002"
echo "  nc -u 127.0.0.1 9003  # enviar para porta 9003"
echo ""

