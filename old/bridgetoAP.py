from pymavlink import mavutil
from pymavlink.mavwp import MAVWPLoader
import time





def iniciar_missao(master):
    master.arducopter_arm()
    master.motors_armed_wait()
    print("✅ Robô armado.")

    if master.flightmode != "AUTO":
        master.set_mode_auto()
        print("🚀 Modo AUTO ativado.")
    else:
        print("🚀 Modo AUTO já ativo.")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_MISSION_START,
        0, 0, 0, 0, 0, 0, 0, 0
    )
    print("🧭 Missão iniciada.")


def enviar_missao(udp_channel, mission_points, ALT, source_system):
    master = mavutil.mavlink_connection(udp_channel, source_system=source_system)
    master.wait_heartbeat()
    print(f"✅ Conectado ao ArduPilot via {udp_channel}, source_system={source_system}")

    master.waypoint_clear_all_send()
    wp = MAVWPLoader()
    mission_points_sorted = sorted(mission_points, key=lambda x: x['id'])
    lastpoint = None

    # Adiciona os waypoints
    for i, point in enumerate(mission_points_sorted):
        lastpoint = point
        wp.add(
            mavutil.mavlink.MAVLink_mission_item_int_message(
                target_system=master.target_system,
                target_component=master.target_component,
                seq=i,
                frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                current=1 if i == 0 else 0,
                autocontinue=1,
                param1=0, param2=0, param3=0, param4=0,
                x=int(point["lat"] * 1e7),
                y=int(point["lon"] * 1e7),
                z=ALT
            )
        )

    # Adiciona comando RTL
    wp.add(
        mavutil.mavlink.MAVLink_mission_item_int_message(
            target_system=master.target_system,
            target_component=master.target_component,
            seq=wp.count(),
            frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            command=mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            current=0,
            autocontinue=1,
            param1=0, param2=0, param3=0, param4=0,
            x=int(lastpoint["lat"] * 1e7),
            y=int(lastpoint["lon"] * 1e7),
            z=ALT
        )
    )

    print(f"📦 Enviando total de {wp.count()} waypoints")
    max_attempts = 100
    timeout = 15
    master.waypoint_count_send(wp.count())

    for i in range(wp.count()):
        success = False
        for attempt in range(max_attempts):
            msg = master.recv_match(type=['MISSION_REQUEST_INT', 'MISSION_REQUEST'], blocking=True, timeout=timeout)
            if msg is not None and msg.seq == i:
                print(f"📥 Solicitação recebida: {msg.get_type()} para WP {msg.seq}")
                master.mav.send(wp.wp(i))
                print(f"📤 Waypoint {i} enviado (tentativa {attempt + 1}).")
                success = True
                break
            elif msg:
                print(f"⚠️ Recebido tipo inesperado: {msg.get_type()}")
            else:
                print(f"⏳ Tentativa {attempt + 1} falhou para WP {i}. Reenviando...")
                time.sleep(0.2)
        if not success:
            print(f"❌ Falha após {max_attempts} tentativas para WP {i}. Abortando.")
            return None
        time.sleep(0.2)

    # Espera pelo MISSION_ACK
    for attempt in range(max_attempts):
        msg = master.recv_match(blocking=True, timeout=timeout)
        if msg and msg.get_type() == "MISSION_ACK":
            print("📦 Missão carregada com sucesso!")
            print(f"📨 Recebido ACK: {msg.get_type()}")
            break
        elif msg:
            print(f"📨 Recebido durante espera do ACK: {msg.get_type()}")
        else:
            print(f"⏳ Tentativa {attempt + 1} aguardando MISSION_ACK...")
    else:
        print("❌ Timeout esperando confirmação da missão (MISSION_ACK).")
        return None

    return master

def verificar_missao(master, timeout=5, max_attempts=10):
    print("\n📋 Verificando missão armazenada no robô...\n")
    master.mav.mission_request_list_send(master.target_system, master.target_component)
    msg = master.recv_match(type='MISSION_COUNT', blocking=True, timeout=timeout)
    if msg is None:
        print("❌ Timeout esperando MISSION_COUNT.")
        return
    wp_count = msg.count
    print(f"➡️  {wp_count} waypoints armazenados.\n")


    waypoints = []
    for i in range(wp_count):
        for attempt in range(max_attempts):
            master.mav.mission_request_send(master.target_system, master.target_component, i)
            msg = master.recv_match(type='MISSION_ITEM', blocking=True, timeout=timeout)
            if msg:
                break
            else:
                print(f"⏳ Tentativa {attempt + 1} esperando MISSION_ITEM {i}...")
                time.sleep(0.5)
        else:
            print(f"❌ Timeout esperando MISSION_ITEM {i} após {max_attempts} tentativas.")
            return

        waypoints.append(msg)
        if msg.command == mavutil.mavlink.MAV_CMD_NAV_WAYPOINT:
            print(f"📍 WP {msg.seq}: Navegar para (lat: {msg.x:.6f}, lon: {msg.y:.6f}, alt: {msg.z:.1f})")
        elif msg.command == mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH:
            print(f"🔙 WP {msg.seq}: RETURN TO LAUNCH")
        else:
            print(f"⚠️ WP {msg.seq}: comando desconhecido ({msg.command})")
        time.sleep(0.2)

    if waypoints and waypoints[-1].command == mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH:
        print("\n✅ RTL corretamente posicionado como último waypoint.")
    else:
        print("\n⚠️ RTL não encontrado como último waypoint.")

    print("\n✅ Verificação completa.")
    return master




# ATENCAO - por algum motivo que ainda nao descobri, o primeiro ponto tem que ser enviado de forma duplicada.


# Coordenadas GPS da missão
mission_points_1 =  [
    {"id": 0, "lat": -3.12316325, "lon": -41.76546074},
    {"id": 1, "lat": -3.12316325, "lon": -41.76546074},
    {"id": 2, "lat": -3.12404350, "lon": -41.76555384},
    {"id": 3, "lat": -3.12401445, "lon": -41.76353468},
    {"id": 4, "lat": -3.12205639, "lon": -41.76364815},
    {"id": 5, "lat": -3.12204477, "lon": -41.76397692},
    {"id": 6, "lat": -3.12256770, "lon": -41.76403220},
    {"id": 7, "lat": -3.12255317, "lon": -41.76507669},
    {"id": 8, "lat": -3.12257641, "lon": -41.76543747}
]

mission_points_2 = [
    {"id": 0, "lat": -3.12257641, "lon": -41.76543747},
    {"id": 1, "lat": -3.12257641, "lon": -41.76543747},
    {"id": 2, "lat": -3.12255317, "lon": -41.76507669},
    {"id": 3, "lat": -3.12256770, "lon": -41.76403220},
    {"id": 4, "lat": -3.12204477, "lon": -41.76397692},
    {"id": 5, "lat": -3.12205639, "lon": -41.76364815},
    {"id": 6, "lat": -3.12401445, "lon": -41.76353468},
    {"id": 7, "lat": -3.12404350, "lon": -41.76555384},
    {"id": 8, "lat": -3.12316325, "lon": -41.76546074}
]

ALT = 2.0  # Altitude padrão


robo1 = enviar_missao("udp:0.0.0.0:14551", mission_points_1, ALT,201)
robo2 = enviar_missao("udp:0.0.0.0:14552", mission_points_2, ALT, 202)


if robo1:
    iniciar_missao(robo1)
else:
    print("❌ Erro ao configurar missão para robô 1.")

if robo2:
    iniciar_missao(robo2)
else:
    print("❌ Erro ao configurar missão para robô 2.")
