
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

    # ------------ INVERSA ------------
    @staticmethod
    def posicao_em_graus_lat(y_m, lat_ref):
        """Converte deslocamento norte (m) para latitude (graus)."""
        return y_m / 111132.0 + lat_ref

    @staticmethod
    def posicao_em_graus_lon(x_m, lon_ref, lat_ref):
        """Converte deslocamento leste (m) para longitude (graus)."""
        return x_m / (111320.0 * math.cos(math.radians(lat_ref))) + lon_ref

    @staticmethod
    def xy_to_gps(x, y, lat_ref, lon_ref):
        """
        Converte (x,y) em metros (Leste,Norte) para (lat, lon) em graus,
        usando a mesma aproximação e referências da ida.
        """
        lat = MissionManager.posicao_em_graus_lat(y, lat_ref)
        lon = MissionManager.posicao_em_graus_lon(x, lon_ref, lat_ref)
        return lat, lon

    @staticmethod
    def _hav_m(p, q):
        (lat1, lon1), (lat2, lon2) = p, q
        R = 6378137.0
        dphi = math.radians(lat2 - lat1)
        dlmb = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(
            dlmb / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))

    def set_home_to_current(self, robot: str | None = None, timeout: float = 5.0, tol_m: float = 2.0):
        """
        Define o HOME (launch) do autopiloto como a posição atual (GPS) e valida.
        Retorna dict {'lat','lon','alt'} do HOME confirmado, ou None se falhar.
        """
        # 1) força stream
        try:
            self.force_gps_stream(rate_hz=5.0, robot=robot)
        except Exception:
            pass

        # 2) posição fresca
        st = self.wait_for_position(robot=robot, timeout=timeout, require_all=False)  # use sua versão "fresh_only"
        if not st:
            print("❌ Não consegui posição fresca para setar HOME")
            return None

        master = self._require_master(robot)
        # 3) pedir HOME = posição atual
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_HOME, 0,
            1, 0, 0, 0, 0, 0, 0  # param1=1 => usar posição atual
        )

        # Solicita explicitamente HOME_POSITION
        try:
            master.mav.command_long_send(
                master.target_system, master.target_component,
                mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 0,
                mavutil.mavlink.MAVLINK_MSG_ID_HOME_POSITION, 0, 0, 0, 0, 0, 0
            )
        except Exception:
            pass

        t0 = time.time()
        got = None
        while time.time() - t0 < 3.0:
            msg = master.recv_match(type='HOME_POSITION', blocking=True, timeout=0.5)
            if not msg:
                continue
            home_lat = msg.latitude * 1e-7
            home_lon = msg.longitude * 1e-7
            # alt do HOME_POSITION é em milímetros acima do MSL
            home_alt = getattr(msg, 'altitude', 0) / 1000.0
            if MissionManager._hav_m((home_lat, home_lon), (st['lat'], st['lon'])) <= tol_m:
                got = {'lat': home_lat, 'lon': home_lon, 'alt': home_alt}
                break
        if got:
            print(f"🏠 HOME atualizado: {got['lat']:.7f}, {got['lon']:.7f} (±{tol_m} m)")
        else:
            print("⚠️ HOME não confirmou dentro da tolerância/tempo")
        return got

    def get_home_position(self, robot: str | None = None, timeout: float = 2.0):
        """Consulta o HOME atual via REQUEST_MESSAGE -> HOME_POSITION."""
        master = self._require_master(robot)
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 0,
            mavutil.mavlink.MAVLINK_MSG_ID_HOME_POSITION, 0, 0, 0, 0, 0, 0
        )
        msg = master.recv_match(type='HOME_POSITION', blocking=True, timeout=timeout)
        if not msg:
            return None
        return {'lat': msg.latitude * 1e-7, 'lon': msg.longitude * 1e-7, 'alt': getattr(msg, 'altitude', 0) / 1000.0}

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

    def new_wait_for_position(self,
                              robot: Optional[str] = None,
                              timeout: float = 5.0,
                              require_all: bool = False,
                              fresh_only: bool = True):
        """
        fresh_only=True -> só aceita posições com st['ts'] >= t0 (frescas).
        min_move_m     -> se >0, exige que tenha havido deslocamento >= min_move_m
                          em relação ao snapshot do início da chamada.
        """
        t0 = time.time()

        # Garante telemetria fluindo (IDs principais: GLOBAL_POSITION_INT, GPS_RAW_INT, ATTITUDE, VFR_HUD)
        try:
            self.force_gps_stream(rate_hz=5.0, robot=robot)
        except Exception:
            pass

        # snapshot de partida para (opcional) checar movimento mínimo
        baseline = {}
        for name in (self.masters.keys() if (robot is None and not self.single_mode) else [
            ("R1" if (self.single_mode and robot is None) else robot)]):
            st0 = self.last_states.get(name, {}).copy()
            baseline[name] = st0 if st0 else None

        def _is_valid(name: str, st: dict) -> bool:
            if not st or st.get('lat') is None or st.get('lon') is None:
                return False
            if fresh_only:
                ts = st.get('ts')
                if ts is None or ts < t0:
                    return False
            return True

        # loops
        if robot is None and not self.single_mode:
            if require_all:
                needed = set(self.masters.keys())
                have = set()
                while (time.time() - t0) < timeout:
                    self.poll_once(per_robot_reads=60)
                    for name in list(needed - have):
                        st = self.last_states.get(name, {})
                        if _is_valid(name, st):
                            have.add(name)
                    if have == needed:
                        return {name: self.last_states[name] for name in sorted(have)}
                    time.sleep(0.03)
                return None
            else:
                while (time.time() - t0) < timeout:
                    self.poll_once(per_robot_reads=60)
                    for name, st in list(self.last_states.items()):
                        if _is_valid(name, st):
                            return (name, st)
                    time.sleep(0.03)
                return None
        else:
            name = 'R1' if (self.single_mode and robot is None) else robot
            while (time.time() - t0) < timeout:
                self.poll_once(per_robot_reads=60)
                st = self.last_states.get(name or 'R1')
                if _is_valid(name or 'R1', st or {}):
                    return st
                time.sleep(0.03)
            return None

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
            
    def download_mission(self, robot: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Baixa a missão atual do ArduPilot
        
        Args:
            robot: Nome do robô (opcional para modo single)
            
        Returns:
            Lista de waypoints ou None em caso de erro
        """
        try:
            master = self._require_master(robot)
            
            # Solicitar lista de waypoints
            master.mav.mission_request_list_send(
                master.target_system,
                master.target_component
            )
            
            # Aguardar resposta com contagem de waypoints
            msg = master.recv_match(
                type='MISSION_COUNT',
                blocking=True,
                timeout=5
            )
            
            if not msg:
                print(f"❌ [MIS][{robot or 'R1'}] Timeout esperando MISSION_COUNT")
                return None
            
            wp_count = msg.count
            print(f"📋 [MIS][{robot or 'R1'}] Baixando {wp_count} waypoints...")
            
            waypoints = []
            for i in range(wp_count):
                # Solicitar waypoint específico
                master.mav.mission_request_int_send(
                    master.target_system,
                    master.target_component,
                    i
                )
                
                # Aguardar waypoint
                wp_msg = master.recv_match(
                    type=['MISSION_ITEM_INT', 'MISSION_ITEM'],
                    blocking=True,
                    timeout=3
                )
                
                if wp_msg:
                    # Converter para formato padrão
                    if wp_msg.get_type() == 'MISSION_ITEM_INT':
                        lat = wp_msg.x / 1e7
                        lon = wp_msg.y / 1e7
                    else:
                        lat = wp_msg.x
                        lon = wp_msg.y
                    
                    waypoint = {
                        'seq': wp_msg.seq,
                        'lat': lat,
                        'lon': lon,
                        'alt': wp_msg.z,
                        'command': wp_msg.command,
                        'param1': wp_msg.param1,
                        'param2': wp_msg.param2,
                        'param3': wp_msg.param3,
                        'param4': wp_msg.param4,
                        'frame': wp_msg.frame,
                        'current': wp_msg.current,
                        'autocontinue': wp_msg.autocontinue
                    }
                    waypoints.append(waypoint)
                    
                    # Log apenas para waypoints de navegação
                    if wp_msg.command == mavutil.mavlink.MAV_CMD_NAV_WAYPOINT:
                        print(f"   📍 WP {wp_msg.seq}: lat={lat:.6f}, lon={lon:.6f}, alt={wp_msg.z:.1f}")
                else:
                    print(f"❌ [MIS][{robot or 'R1'}] Timeout no waypoint {i}")
                    return None
            
            print(f"✅ [MIS][{robot or 'R1'}] {len(waypoints)} waypoints baixados com sucesso")
            return waypoints
            
        except Exception as e:
            print(f"❌ [MIS][{robot or 'R1'}] Erro ao baixar missão: {e}")
            return None

    def verify_mission_and_return_waypoints(self, robot: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Verifica e retorna waypoints da missão atual (fallback para download_mission)
        
        Args:
            robot: Nome do robô (opcional para modo single)
            
        Returns:
            Lista de waypoints ou None em caso de erro
        """
        return self.download_mission(robot)

    @property
    def is_connected(self) -> bool:
        """Verifica se há pelo menos um robô conectado"""
        return len(self.connected) > 0
        
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