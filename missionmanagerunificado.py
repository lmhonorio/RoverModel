
from typing import Dict, List, Optional, Union
from pymavlink import mavutil
from pymavlink.mavwp import MAVWPLoader
import time
import math
import pandas as pd

"""
MissionManager unificado: uma ÚNICA classe que gerencia
  • Conexão MAVLink (um ou vários robôs)
  • Upload/controle de missão (por robô)
  • Coleta de telemetria (lat, lon, alt, heading, etc.) dos robôs conectados
Assim, você passa a configuração de conexão APENAS UMA VEZ.

Modo de uso (single-robot):
    mm = MissionManager(udp_channel='udp:127.0.0.1:14551', source_system=1)
    mm.connect()                      # conecta ao único robô
    mm.upload_mission([...])
    while True:
        mm.poll_once()
        print(mm.get_latest_positions())

Modo de uso (multi-robot):
    robots = [
        {'name': 'R1', 'channel': 'udp:127.0.0.1:14551', 'source_system': 1},
        {'name': 'R2', 'channel': 'udpin:0.0.0.0:14562', 'source_system': 2},  # somente recepção
    ]
    mm = MissionManager(robots=robots)
    mm.connect_all()
    mm.upload_mission('R1', [...])
    while True:
        mm.poll_once()
        print(mm.get_latest_positions())

Observações:
  • Para ENVIAR comandos/missão, a conexão do robô deve ser 'udp:' (leitura/escrita).
  • Para SOMENTE RECEBER telemetria, use 'udpin:'.
  • Você pode misturar ambos em 'robots'.
"""


def _cdeg_to_deg(x: Optional[int]) -> Optional[float]:
    """Converte centidegrees para degrees (GLOBAL_POSITION_INT.hdg)."""
    if x is None or x == 65535:
        return None
    return x * 0.01


def _yaw_rad_to_deg(yaw: float) -> float:
    """Converte yaw (rad) para [0,360) deg."""
    deg = math.degrees(yaw) % 360.0
    return deg if deg >= 0 else deg + 360.0


class MissionManager:
    """MissionManager unificado (missão + telemetria, single/multi-robô)."""

    def __init__(
        self,
        udp_channel: Optional[str] = None,
        source_system: Optional[int] = None,
        timeout: float = 2.0,
        max_attempts: int = 100,
        robots: Optional[List[Dict[str, Union[str, int]]]] = None,
    ):
        self.timeout = timeout
        self.max_attempts = max_attempts

        self.single_mode = robots is None
        if self.single_mode:
            if udp_channel is None or source_system is None:
                raise ValueError("Para modo single, informe udp_channel e source_system.")
            self.udp_channel = udp_channel
            self.source_system = source_system
            self.robots_cfg: List[Dict] = [{
                'name': 'R1',
                'channel': udp_channel,
                'source_system': source_system
            }]
        else:
            if not robots:
                raise ValueError("Lista 'robots' vazia.")
            self.udp_channel = None
            self.source_system = None
            self.robots_cfg = []
            for i, rb in enumerate(robots):
                name = rb.get('name') or f"R{i+1}"
                chan = rb['channel']
                sid  = rb.get('source_system', 1)
                self.robots_cfg.append({'name': name, 'channel': chan, 'source_system': sid})

        self.masters: Dict[str, mavutil.mavfile] = {}
        self.last_states: Dict[str, Dict] = {}
        self.connected = set()
        self.master: Optional[mavutil.mavfile] = None
    #----------------

    @staticmethod
    def read_parametros_conversao_lat_lon(PARAMS_XLSX ):
        dfp = pd.read_excel(PARAMS_XLSX, sheet_name="ParametrosConversao")
        lat_ref = float(dfp.loc[dfp["Parametro"] == "Latitude Média", "Valor"].iloc[0])
        lon_ref = float(dfp.loc[dfp["Parametro"] == "Longitude Média", "Valor"].iloc[0])
        return {'lat_ref':lat_ref, 'lon_ref':lon_ref}


    @staticmethod
    def posicao_em_metros_lat(lat, lat_ref):
        return (lat - lat_ref) * 111132.0

    @staticmethod
    def posicao_em_metros_lon(lon, lon_ref, lat_ref):
        return (lon - lon_ref) * (111320.0 * math.cos(math.radians(lat_ref)))

    @staticmethod
    def gps_to_xy(lat, lon,  lat_ref, lon_ref):
        x = MissionManager.posicao_em_metros_lon(lon, lon_ref, lat_ref)  # Leste (+)
        y = MissionManager.posicao_em_metros_lat(lat, lat_ref)  # Norte (+)
        return x, y


    # ------------------------------ Helpers ------------------------------
    def _iter_target_names(self, robot: Optional[str] = None):
        if self.single_mode:
            return ['R1']
        if robot is None:
            return list(self.masters.keys())
        if robot not in self.masters:
            raise RuntimeError(f"Robô '{robot}' não está conectado.")
        return [robot]

    def _require_master(self, robot: Optional[str] = None) -> mavutil.mavfile:
        if self.single_mode:
            robot = 'R1'
        if robot is None:
            if len(self.masters) == 1:
                name = next(iter(self.masters))
                return self.masters[name]
            raise ValueError("Informe o nome do robô em modo multi-robô.")
        if robot not in self.masters:
            raise RuntimeError(f"Robô '{robot}' não está conectado.")
        return self.masters[robot]

    # ------------------------------ Conexões ------------------------------
    def connect_all(self) -> bool:
        ok_any = False
        for rb in self.robots_cfg:
            name = rb['name']
            chan = rb['channel']
            sid  = rb['source_system']
            ok = False
            for attempt in range(1, self.max_attempts + 1):
                try:
                    master = mavutil.mavlink_connection(chan, source_system=sid)
                    master.wait_heartbeat(timeout=self.timeout)
                    self.masters[name] = master
                    self.connected.add(name)
                    if self.single_mode and name == 'R1':
                        self.master = master
                    print(f"✅ [MIS][{name}] Conectado em {chan} (source_system={sid})")
                    ok = True
                    ok_any = True
                    break
                except Exception as e:
                    print(f"⚠️  [MIS][{name}] Tentativa {attempt}/{self.max_attempts} falhou: {e}")
                    time.sleep(1.0)
            if not ok:
                print(f"❌ [MIS][{name}] Não conectou em {chan}")
        return ok_any

    def connect(self) -> bool:
        if not self.single_mode:
            return self.connect_all()
        if 'R1' in self.masters:
            return True
        return self.connect_all()

    # ------------------------------ Telemetria ------------------------------
    def _update_from_msg(self, name: str, msg):
        now = time.time()
        state = self.last_states.get(name, {
            'lat': None, 'lon': None, 'alt': None, 'hdg_deg': None,
            'groundspeed': None, 'fix_type': None, 'ts': None, 'system_id': None
        })

        mtype = msg.get_type()

        if mtype == 'GLOBAL_POSITION_INT':
            state['lat'] = msg.lat / 1e7
            state['lon'] = msg.lon / 1e7
            rel_alt = getattr(msg, 'relative_alt', None)
            state['alt'] = (rel_alt / 1000.0) if rel_alt not in (None, 0) else (msg.alt / 1000.0)
            state['hdg_deg'] = _cdeg_to_deg(getattr(msg, 'hdg', None))
            state['ts'] = now
            state['system_id'] = getattr(msg, 'target_system', None) or getattr(msg, 'sysid', None)

        elif mtype == 'GPS_RAW_INT':
            state['lat'] = msg.lat / 1e7
            state['lon'] = msg.lon / 1e7
            state['alt'] = msg.alt / 1000.0
            state['fix_type'] = getattr(msg, 'fix_type', None)
            state['ts'] = now
            state['system_id'] = getattr(msg, 'sysid', None)

        elif mtype == 'VFR_HUD':
            state['groundspeed'] = getattr(msg, 'groundspeed', None)
            alt = getattr(msg, 'alt', None)
            if alt is not None:
                state['alt'] = float(alt)
            state['ts'] = now

        elif mtype == 'ATTITUDE':
            yaw = getattr(msg, 'yaw', None)
            if yaw is not None:
                state['hdg_deg'] = _yaw_rad_to_deg(yaw)
                state['ts'] = now

        elif mtype == 'HEARTBEAT':
            state['ts'] = state.get('ts') or now

        self.last_states[name] = state

    def poll_once(self, per_robot_reads: int = 5):
        types = ['GLOBAL_POSITION_INT', 'GPS_RAW_INT', 'VFR_HUD', 'ATTITUDE', 'HEARTBEAT']
        for name, master in list(self.masters.items()):
            for _ in range(per_robot_reads):
                msg = master.recv_match(type=types, blocking=False)
                if msg is None:
                    break
                self._update_from_msg(name, msg)

    def get_latest_positions(self) -> Dict[str, Dict]:
        return self.last_states

    # ---------- Forçar fluxo de mensagens de posição ----------
    def request_message_interval(self, msg_id: int, rate_hz: float = 5.0, robot: Optional[str] = None):
        interval_us = int(1e6 / rate_hz) if rate_hz > 0 else -1
        for name in self._iter_target_names(robot):
            master = self.masters[name]
            master.mav.command_long_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                msg_id,
                float(interval_us),
                0, 0, 0, 0, 0
            )

    def force_gps_stream(self, rate_hz: float = 5.0, robot: Optional[str] = None):
        ids = (
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
            mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT,
            mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
            mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD,
        )
        for name in self._iter_target_names(robot):
            master = self.masters[name]
            for mid in ids:
                self.request_message_interval(mid, rate_hz, robot=name)
            try:
                master.mav.request_data_stream_send(
                    master.target_system, master.target_component,
                    mavutil.mavlink.MAV_DATA_STREAM_POSITION, int(rate_hz), 1
                )
                master.mav.request_data_stream_send(
                    master.target_system, master.target_component,
                    mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, int(rate_hz), 1
                )
            except Exception:
                pass

    def wait_for_position(self, robot: Optional[str] = None, timeout: float = 5.0, require_all: bool = False):
        t0 = time.time()
        if robot is None and not self.single_mode:
            if require_all:
                needed = set(self.masters.keys())
                have = set()
                while (time.time() - t0) < timeout:
                    self.poll_once(per_robot_reads=20)
                    for name in list(needed - have):
                        st = self.last_states.get(name, {})
                        if st.get('lat') is not None and st.get('lon') is not None:
                            have.add(name)
                    if have == needed:
                        return {name: self.last_states[name] for name in sorted(have)}
                    time.sleep(0.05)
                return None
            else:
                while (time.time() - t0) < timeout:
                    self.poll_once(per_robot_reads=20)
                    for name, st in self.last_states.items():
                        if st.get('lat') is not None and st.get('lon') is not None:
                            return (name, st)
                    time.sleep(0.05)
                return None
        else:
            name = 'R1' if (self.single_mode and robot is None) else robot
            while (time.time() - t0) < timeout:
                self.poll_once(per_robot_reads=20)
                st = self.last_states.get(name or 'R1')
                if st and (st.get('lat') is not None) and (st.get('lon') is not None):
                    return st
                time.sleep(0.05)
            return None

    # -------------------------- Operações de Missão --------------------------
    def clear_mission(self, robot: Optional[str] = None) -> bool:
        master = self._require_master(robot)
        try:
            master.waypoint_clear_all_send()
            t0 = time.time()
            while time.time() - t0 < 5.0:
                msg = master.recv_match(type=['MISSION_ACK'], blocking=False)
                if msg is not None:
                    print(f"🧹 [MIS][{robot or 'R1'}] Missão limpa (ACK)")
                    return True
                time.sleep(0.05)
            print(f"🧹 [MIS][{robot or 'R1'}] Missão limpa (sem ACK explícito)")
            return True
        except Exception as e:
            print(f"❌ [MIS][{robot or 'R1'}] Erro ao limpar missão: {e}")
            return False

    def upload_mission(self, mission_points: List[Dict], altitude: float = 2.0, robot: Optional[str] = None) -> bool:
        if self.single_mode:
            if 'R1' not in self.masters and not self.connect():
                return False
        if not self.clear_mission(robot=robot):
            return False

        mission_points_sorted = sorted(mission_points, key=lambda x: x['id'])
        wp = MAVWPLoader()
        master = self._require_master(robot)
        for i, point in enumerate(mission_points_sorted):
            lat = int(point['lat'] * 1e7)
            lon = int(point['lon'] * 1e7)
            hold = float(point.get("hold", 0.0))
            accept = float(point.get("accept_radius", 0.0))
            passrd = float(point.get("pass_radius", 0.0))
            yaw = float(point.get("yaw_deg", 0.0))

            msg = mavutil.mavlink.MAVLink_mission_item_int_message(
                target_system=master.target_system,
                target_component=master.target_component,
                seq=i,
                frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                current=0,
                autocontinue=1,
                param1=hold, param2=accept, param3=passrd, param4=yaw,  # <<---
                x=lat, y=lon, z=float(altitude)
            )
            wp.add(msg)
            # lat = int(point['lat'] * 1e7)
            # lon = int(point['lon'] * 1e7)
            # msg = mavutil.mavlink.MAVLink_mission_item_int_message(
            #     target_system=master.target_system,
            #     target_component=master.target_component,
            #     seq=i,
            #     frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            #     command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            #     current=0,
            #     autocontinue=1,
            #     param1=0, param2=0, param3=0, param4=0,
            #     x=lat, y=lon, z=float(altitude)
            # )
            # wp.add(msg)

        try:
            master.mav.mission_count_send(master.target_system, master.target_component, wp.count())
            expected = 0
            t_start = time.time()
            while True:
                msg = master.recv_match(
                    type=['MISSION_REQUEST_INT', 'MISSION_REQUEST', 'MISSION_ACK'],
                    blocking=True, timeout=5
                )
                if msg is None:
                    if time.time() - t_start > 30:
                        print(f"❌ [MIS][{robot or 'R1'}] Timeout no envio da missão")
                        return False
                    continue

                mtype = msg.get_type()
                if mtype in ('MISSION_REQUEST_INT', 'MISSION_REQUEST'):
                    seq_req = int(msg.seq)
                    if seq_req != expected:
                        expected = seq_req
                    master.mav.send(wp.wp(seq_req))
                    expected += 1
                elif mtype == 'MISSION_ACK':
                    print(f"📤 [MIS][{robot or 'R1'}] Missão enviada com sucesso (ACK)")
                    return True
        except Exception as e:
            print(f"❌ [MIS][{robot or 'R1'}] Erro no envio da missão: {e}")
            return False

    def set_mode(self, mode: str = 'AUTO', robot: Optional[str] = None) -> bool:
        try:
            self._require_master(robot).set_mode(mode)
            print(f"🎛️  [MIS][{robot or 'R1'}] Modo = {mode}")
            return True
        except Exception as e:
            print(f"❌ [MIS][{robot or 'R1'}] Erro ao setar modo {mode}: {e}")
            return False

    def arm_and_start(self, robot: Optional[str] = None) -> bool:
        try:
            master = self._require_master(robot)
            master.set_mode('AUTO')
            master.arducopter_arm()
            master.motors_armed_wait(timeout=5)
            try:
                master.mav.command_long_send(
                    master.target_system, master.target_component,
                    mavutil.mavlink.MAV_CMD_MISSION_START, 0, 0, 0, 0, 0, 0, 0, 0
                )
            except Exception:
                pass
            print(f"🚀 [MIS][{robot or 'R1'}] Armado e missão iniciada")
            return True
        except Exception as e:
            print(f"❌ [MIS][{robot or 'R1'}] Falha ao armar/iniciar missão: {e}")
            return False

    def disarm(self, robot: Optional[str] = None) -> bool:
        try:
            self._require_master(robot).arducopter_disarm()
            print(f"🛑 [MIS][{robot or 'R1'}] Desarmado")
            return True
        except Exception as e:
            print(f"❌ [MIS][{robot or 'R1'}] Falha ao desarmar: {e}")
            return False

    def close(self):
        for name, master in list(self.masters.items()):
            try:
                master.close()
                print(f"🔌 [MIS][{name}] Conexão fechada")
            except Exception:
                pass
        self.masters.clear()
        self.connected.clear()
        self.master = None

    @staticmethod
    def build_mission(coords: List[tuple]):
        mission = []
        for i, (lat, lon) in enumerate(coords):
            mission.append({'id': i, 'lat': lat, 'lon': lon})
        if len(mission) >= 1:
            mission.insert(1, mission[0])
        return mission