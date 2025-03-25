from pymavlink import mavutil
from pymavlink.mavwp import MAVWPLoader
import time


def iniciar_missao(master):
    # Arma o robô e inicia missão
    master.arducopter_arm()
    master.motors_armed_wait()
    print("✅ Robô armado.")

    # Verifica e muda para modo AUTO, se necessário
    if master.flightmode != "AUTO":
        master.set_mode_auto()
        print("🚀 Modo AUTO ativado.")
    else:
        print("🚀 Modo AUTO já ativo.")

    # Iniciar missão com os 7 parâmetros obrigatórios
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_MISSION_START,
        0, 0, 0, 0, 0, 0, 0, 0
    )
    print("🧭 Missão iniciada.")

def enviar_missao(udp_channel, mission_points, ALT=2.0):
    # Conecta ao ArduPilot via MAVProxy
    master = mavutil.mavlink_connection(udp_channel)
    master.wait_heartbeat()
    print(f"✅ Conectado ao ArduPilot via {udp_channel}")

    # Limpa missões anteriores
    master.waypoint_clear_all_send()

    wp = MAVWPLoader()

    # Adiciona os waypoints
    for i, point in enumerate(mission_points):
        wp.add(
            mavutil.mavlink.MAVLink_mission_item_message(
                target_system=master.target_system,
                target_component=master.target_component,
                seq=i,
                frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                current=1 if i == 0 else 0,
                autocontinue=1,
                param1=0, param2=0, param3=0, param4=0,
                x=point["lat"], y=point["lon"], z=ALT
            )
        )

    # Adiciona comando RTL no final
    wp.add(
        mavutil.mavlink.MAVLink_mission_item_message(
            target_system=master.target_system,
            target_component=master.target_component,
            seq=wp.count(),
            frame=mavutil.mavlink.MAV_FRAME_MISSION,
            command=mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            current=0,
            autocontinue=1,
            param1=0, param2=0, param3=0, param4=0,
            x=0, y=0, z=0
        )
    )

    # Envia missão com handshake MAVLink
    print(f"📦 Enviando total de {wp.count()} waypoints")
    master.waypoint_count_send(wp.count())
    for i in range(wp.count()):
        msg = master.recv_match(type='MISSION_REQUEST', blocking=True, timeout=15)
        if msg is None:
            print(f"❌ Timeout esperando MISSION_REQUEST para WP {i}.")
            return
        print(f"📥 Solicitação recebida: WP {msg.seq}")
        master.mav.send(wp.wp(i))
        print(f"📤 Waypoint {i} enviado.")
        time.sleep(1)

    # Confirma missão carregada
    ack = master.recv_match(type='MISSION_ACK', blocking=True, timeout=15)
    if ack is None:
        print("❌ Timeout esperando confirmação da missão (MISSION_ACK).")
        return
    print("📦 Missão carregada com sucesso!")

   # 🔍 Conferência dos waypoints armazenados
    print("\n📋 Verificando missão armazenada no robô...\n")
    master.mav.mission_request_list_send(master.target_system, master.target_component)
    msg = master.recv_match(type='MISSION_COUNT', blocking=True, timeout=15)
    if msg is None:
        print("❌ Timeout esperando MISSION_COUNT.")
        return
    wp_count = msg.count
    print(f"➡️  {wp_count} waypoints armazenados.\n")

    waypoints = []
    for i in range(wp_count):
        master.mav.mission_request_send(master.target_system, master.target_component, i)
        msg = master.recv_match(type='MISSION_ITEM', blocking=True, timeout=15)
        if msg is None:
            print(f"❌ Timeout esperando MISSION_ITEM {i}.")
            return
        waypoints.append(msg)
        if msg.command == mavutil.mavlink.MAV_CMD_NAV_WAYPOINT:
            print(f"📍 WP {msg.seq}: Navegar para (lat: {msg.x:.6f}, lon: {msg.y:.6f}, alt: {msg.z:.1f})")
        elif msg.command == mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH:
            print(f"🔙 WP {msg.seq}: RETURN TO LAUNCH")
        else:
            print(f"⚠️ WP {msg.seq}: comando desconhecido ({msg.command})")
        time.sleep(0.2)

    # Confirma RTL como último waypoint
    if waypoints and waypoints[-1].command == mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH:
        print("\n✅ RTL corretamente posicionado como último waypoint.")
    else:
        print("\n⚠️ RTL não encontrado como último waypoint.")

    print("\n✅ Verificação completa.")

    return master



# # Conecta ao ArduPilot via MAVProxy
# master = mavutil.mavlink_connection("udp:0.0.0.0:14550")
# master.wait_heartbeat()
# print("✅ Conectado ao ArduPilot")
#
# # Limpa missões anteriores
# master.waypoint_clear_all_send()

# Coordenadas GPS da missão
mission_points_1 = [
    {"lat": -3.1241450, "lon": -41.7671046},
    {"lat": -3.1188871, "lon": -41.7663323},
    {"lat": -3.1178706, "lon": -41.7630022}
]

mission_points_2 = [
    {"lat": -3.1178706, "lon": -41.7630022},
    {"lat": -3.1188871, "lon": -41.7663323},
    {"lat": -3.1241450, "lon": -41.7671046}
]

ALT = 2.0  # Altitude padrão

robo1 = enviar_missao("udp:0.0.0.0:14551", mission_points_1)
robo2 = enviar_missao("udp:0.0.0.0:14552", mission_points_2)

if robo1:
    iniciar_missao(robo1)
else:
    print("❌ Erro ao configurar missão para robô 1.")
    robo1 = enviar_missao("udp:0.0.0.0:14551", mission_points_1)
    if robo1:
        iniciar_missao(robo1)
    else:
        print("❌ Erro ao configurar missão para robô 1.")

if robo2:
    iniciar_missao(robo2)
else:
    print("❌ Erro ao configurar missão para robô 2.")


#
# wp = MAVWPLoader()
#
# # Adiciona os waypoints
# for i, point in enumerate(mission_points):
#     wp.add(
#         mavutil.mavlink.MAVLink_mission_item_message(
#             target_system=master.target_system,
#             target_component=master.target_component,
#             seq=i,
#             frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
#             command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
#             current=1 if i == 0 else 0,
#             autocontinue=1,
#             param1=0, param2=0, param3=0, param4=0,
#             x=point["lat"], y=point["lon"], z=ALT
#         )
#     )
#
# # Adiciona comando RTL no final
# wp.add(
#     mavutil.mavlink.MAVLink_mission_item_message(
#         target_system=master.target_system,
#         target_component=master.target_component,
#         seq=wp.count(),
#         frame=mavutil.mavlink.MAV_FRAME_MISSION,
#         command=mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
#         current=0,
#         autocontinue=1,
#         param1=0, param2=0, param3=0, param4=0,
#         x=0, y=0, z=0
#     )
# )
#
#
# # Envia missão com handshake MAVLink
# print(f"📦 Enviando total de {wp.count()} waypoints")
# master.waypoint_count_send(wp.count())
# for i in range(wp.count()):
#     msg = master.recv_match(type='MISSION_REQUEST', blocking=True, timeout=15)
#     if msg is None:
#         print(f"❌ Timeout esperando MISSION_REQUEST para WP {i}.")
#         exit(1)
#     print(f"📥 Solicitação recebida: WP {msg.seq}")
#     master.mav.send(wp.wp(i))
#     print(f"📤 Waypoint {i} enviado.")
#     time.sleep(1)
#
# # Confirma missão carregada
# ack = master.recv_match(type='MISSION_ACK', blocking=True, timeout=15)
# if ack is None:
#     print("❌ Timeout esperando confirmação da missão (MISSION_ACK).")
#     exit(1)
# print("📦 Missão carregada com sucesso!")
#
# # Arma o robô e inicia missão
# master.arducopter_arm()
# master.motors_armed_wait()
# print("✅ Robô armado.")
#
# # Verifica e muda para modo AUTO, se necessário
# if master.flightmode != "AUTO":
#     master.set_mode_auto()
#     print("🚀 Modo AUTO ativado.")
# else:
#     print("🚀 Modo AUTO já ativo.")
#
# # Iniciar missão com os 7 parâmetros obrigatórios
# master.mav.command_long_send(
#     master.target_system,
#     master.target_component,
#     mavutil.mavlink.MAV_CMD_MISSION_START,
#     0, 0, 0, 0, 0, 0, 0, 0  # param1 a param7
# )
# print("🧭 Missão iniciada.")
#
# # 🔍 Conferência dos waypoints armazenados
# print("\n📋 Verificando missão armazenada no robô...\n")
# master.mav.mission_request_list_send(master.target_system, master.target_component)
# msg = master.recv_match(type='MISSION_COUNT', blocking=True, timeout=15)
# if msg is None:
#     print("❌ Timeout esperando MISSION_COUNT.")
#     exit(1)
# wp_count = msg.count
# print(f"➡️  {wp_count} waypoints armazenados.\n")
#
# waypoints = []
# for i in range(wp_count):
#     master.mav.mission_request_send(master.target_system, master.target_component, i)
#     msg = master.recv_match(type='MISSION_ITEM', blocking=True, timeout=15)
#     if msg is None:
#         print(f"❌ Timeout esperando MISSION_ITEM {i}.")
#         exit(1)
#     waypoints.append(msg)
#     if msg.command == mavutil.mavlink.MAV_CMD_NAV_WAYPOINT:
#         print(f"📍 WP {msg.seq}: Navegar para (lat: {msg.x:.6f}, lon: {msg.y:.6f}, alt: {msg.z:.1f})")
#     elif msg.command == mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH:
#         print(f"🔙 WP {msg.seq}: RETURN TO LAUNCH")
#     else:
#         print(f"⚠️ WP {msg.seq}: comando desconhecido ({msg.command})")
#     time.sleep(0.2)
#
# # Confirma RTL como último waypoint
# if waypoints and waypoints[-1].command == mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH:
#     print("\n✅ RTL corretamente posicionado como último waypoint.")
# else:
#     print("\n⚠️ RTL não encontrado como último waypoint.")
#
# print("\n✅ Verificação completa.")
