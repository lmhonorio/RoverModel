#!/usr/bin/env python3

from movns_ains_argo import send_mission_argo

# Tem que colocar ponto inicial, senão não funciona
send_mission_argo.generate_mission([[-3.1231947, -41.7653947],[-3.1232803141546217, -41.76542310046668]], 2)
print(f"✅ Missão enviada para robô {1}")