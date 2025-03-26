from pymavlink import mavutil
from pymavlink.mavwp import MAVWPLoader
import time


class MissionManager:
    def __init__(self, udp_channel, source_system, timeout=2, max_attempts=100):
        self.udp_channel = udp_channel
        self.source_system = source_system
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.master = None
        self.connected = False

    def connect(self):
        for attempt in range(self.max_attempts):
            try:
                self.master = mavutil.mavlink_connection(
                    self.udp_channel,
                    source_system=self.source_system
                )
                self.master.wait_heartbeat()
                self.connected = True
                print(f"✅ Conectado ao veículo via {self.udp_channel}, source_system={self.source_system}")
                return True
            except Exception as e:
                print(f"⚠️ Tentativa {attempt + 1} de conexão falhou: {str(e)}")
                time.sleep(1)
        self.connected = False
        return False

    def clear_mission(self):
        if not self.connected:
            raise ConnectionError("Não conectado ao veículo")
        try:
            self.master.waypoint_clear_all_send()
            print("🧹 Missão atual limpa")
            return True
        except Exception as e:
            print(f"❌ Falha ao limpar missão: {str(e)}")
            return False

    def upload_mission(self, mission_points, altitude=2.0):
        if not self.connected and not self.connect():
            return False
        if not self.clear_mission():
            return False

        mission_points_sorted = sorted(mission_points, key=lambda x: x['id'])
        wp = MAVWPLoader()

        for i, point in enumerate(mission_points_sorted):
            wp.add(mavutil.mavlink.MAVLink_mission_item_int_message(
                target_system=self.master.target_system,
                target_component=self.master.target_component,
                seq=i,
                frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                current=1 if i == 0 else 0,
                autocontinue=1,
                param1=0, param2=0, param3=0, param4=0,
                x=int(point["lat"] * 1e7),
                y=int(point["lon"] * 1e7),
                z=altitude
            ))

        last_point = mission_points_sorted[-1]
        wp.add(mavutil.mavlink.MAVLink_mission_item_int_message(
            target_system=self.master.target_system,
            target_component=self.master.target_component,
            seq=wp.count(),
            frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            command=mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            current=0,
            autocontinue=1,
            param1=0, param2=0, param3=0, param4=0,
            x=int(last_point["lat"] * 1e7),
            y=int(last_point["lon"] * 1e7),
            z=altitude
        ))

        print(f"📦 Preparados {wp.count()} waypoints para envio")
        self.master.waypoint_count_send(wp.count())

        for i in range(wp.count()):
            for attempt in range(self.max_attempts):
                msg = self.master.recv_match(
                    type=['MISSION_REQUEST_INT', 'MISSION_REQUEST'],
                    blocking=True,
                    timeout=self.timeout
                )
                if msg is not None and msg.seq == i:
                    self.master.mav.send(wp.wp(i))
                    print(f"📤 Waypoint {i} enviado (tentativa {attempt + 1})")
                    break
                elif msg:
                    print(f"⚠️ Recebido {msg.get_type()} fora de sequência (esperado seq={i}, recebido seq={getattr(msg, 'seq', -1)})")
                else:
                    print(f"⏳ Reenviando solicitação para WP {i} (tentativa {attempt + 1})")
            else:
                print(f"❌ Falha ao enviar WP {i} após {self.max_attempts} tentativas")
                return False

        for attempt in range(self.max_attempts):
            ack = self.master.recv_match(
                type='MISSION_ACK',
                blocking=True,
                timeout=self.timeout
            )
            if ack:
                if ack.type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                    print("✅ Missão carregada com sucesso!")
                    return True
                else:
                    print(f"❌ Erro no MISSION_ACK: {ack.type}")
                    return False
            else:
                print(f"⏳ Aguardando MISSION_ACK (tentativa {attempt + 1})")
        print("❌ Timeout esperando confirmação da missão")
        return False

    def verify_mission(self):
        if not self.connected and not self.connect():
            return False
        print("\n📋 Verificando missão no veículo...")
        self.master.mav.mission_request_list_send(
            self.master.target_system,
            self.master.target_component
        )
        msg = self.master.recv_match(
            type='MISSION_COUNT',
            blocking=True,
            timeout=self.timeout
        )
        if not msg:
            print("❌ Timeout esperando MISSION_COUNT")
            return False

        wp_count = msg.count
        print(f"➡️ {wp_count} waypoints armazenados")
        waypoints = []
        for i in range(wp_count):
            for attempt in range(self.max_attempts):
                self.master.mav.mission_request_send(
                    self.master.target_system,
                    self.master.target_component,
                    i
                )
                msg = self.master.recv_match(
                    type='MISSION_ITEM_INT',
                    blocking=True,
                    timeout=self.timeout
                )
                if msg:
                    waypoints.append(msg)
                    lat = msg.x / 1e7 if hasattr(msg, 'x') else 0
                    lon = msg.y / 1e7 if hasattr(msg, 'y') else 0
                    if msg.command == mavutil.mavlink.MAV_CMD_NAV_WAYPOINT:
                        print(f"📍 WP {msg.seq}: (lat: {lat:.6f}, lon: {lon:.6f}, alt: {msg.z:.1f})")
                    elif msg.command == mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH:
                        print(f"🔙 WP {msg.seq}: RETURN TO LAUNCH")
                    break
                else:
                    print(f"⏳ Tentativa {attempt + 1} para WP {i}...")
            else:
                print(f"❌ Falha ao verificar WP {i}")
                return False

        if waypoints and waypoints[-1].command == mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH:
            print("✅ RTL configurado corretamente como último waypoint")
            return True
        else:
            print("❌ RTL não encontrado como último waypoint")
            return False

    def arm_and_start(self):
        if not self.connected and not self.connect():
            return False
        try:
            self.master.arducopter_arm()
            for attempt in range(self.max_attempts):
                msg = self.master.recv_match(
                    type='HEARTBEAT',
                    blocking=True,
                    timeout=self.timeout
                )
                if msg and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                    print("✅ Veículo armado")
                    break
                else:
                    print(f"⏳ Aguardando armamento (tentativa {attempt + 1})")
            else:
                print("❌ Timeout ao esperar armamento")
                return False

            if self.master.flightmode != "AUTO":
                self.master.set_mode_auto()
                print("🚀 Modo AUTO ativado")

            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_MISSION_START,
                0, 0, 0, 0, 0, 0, 0, 0
            )
            print("🧭 Missão iniciada")
            return True
        except Exception as e:
            print(f"❌ Falha ao armar/iniciar missão: {str(e)}")
            return False

    def close(self):
        if self.connected:
            try:
                self.master.close()
                self.connected = False
                print("🔌 Conexão fechada")
            except:
                pass