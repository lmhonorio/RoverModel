#!/bin/bash

echo "🔍 Monitorando tráfego UDP nas portas FDM..."
echo "Porta 9002 (Gazebo → ArduPilot)"
echo "Porta 9003 (ArduPilot → Gazebo)"
echo ""
echo "Pressione Ctrl+C para parar"
echo ""

# Monitorar pacotes UDP nas portas 9002 e 9003
sudo tcpdump -i lo -n -X 'udp port 9002 or udp port 9003' 2>/dev/null | grep -E "IP|length|9002|9003" | head -50



