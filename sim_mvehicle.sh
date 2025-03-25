#!/bin/bash

# Terminal para Robô 1
gnome-terminal -- bash -c "sim_vehicle.py -L SEParnaiba -S 5 --mcast -v Rover --sysid 1 --instance 0 -f rover-skid --out=udp:127.0.0.1:14551 --add-param-file nrover.param; exec bash"

# Terminal para Robô 2
gnome-terminal -- bash -c "sim_vehicle.py -L SEParnaiba -S 5 --mcast -v Rover --sysid 2 --instance 1 -f rover-skid --out=udp:127.0.0.1:14552 --add-param-file nrover.param; exec bash"

# Terminal para o MAVProxy - substituir 192.168.0.131 pelo IP da sua máquina
gnome-terminal -- bash -c "mavproxy.py \
  --master=udp:127.0.0.1:14551 \
  --out=udp:192.168.0.131:14551 \
  --master=udp:127.0.0.1:14552 \
  --out=udp:192.168.0.131:14552 \
  --console \
  --map; exec bash"