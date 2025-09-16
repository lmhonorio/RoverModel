"""
Servidor de missões baseado no mission_bridgetoap.py
Recebe robôs e equipamentos selecionados via API REST
Retorna waypoints e monitora posições dos robôs via WebSocket
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import threading
import time
import os
from datetime import datetime
import asyncio

# Importar módulos do sistema de planejamento
from missionmanagerunificado import MissionManager
from PlanejadorHeterogeneoIntegrado import (
    run_planner, 
    montar_missoes_por_robo, 
    retorna_pontos_passagem, 
    build_mission_points_from_path_gps, 
    extract_path_gps, 
    load_label2gps, 
    extract_path_gps_from_obp
)
from segmentutils import SegmentUtils

app = Flask(__name__)
CORS(app)
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    logger=False,
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25,
    async_mode='threading',  # Usar threading em vez de eventlet para evitar problemas de compatibilidade
    transports=['websocket', 'polling']  # Permitir fallback para polling
)

# Configurações globais
GRAPH_PATH = "./jsons/graph9_new.json"
OBSERVATION_POINTS_JSON_PATH = "./jsons/obp_6.json"
PARAMETERS_FILE_PATH = "./planilhas/obstaculos_processado6.xlsx"

# Configurações de frequência de atualização (em segundos)
MONITORING_UPDATE_RATE = 2  # Frequência base de monitoramento (500ms - mais controlado)
WEBSOCKET_THROTTLE_STATIONARY = 2  # Emitir a cada 2 ciclos para robôs parados (1 segundo)
WEBSOCKET_THROTTLE_MISSION = 2  # Emitir a cada 2 ciclos durante missões (1 segundo)
MISSION_PROGRESS_UPDATE_RATE = 5  # Progresso de missão a cada 5 segundos

# Estados globais
mission_manager = None
robot_positions = {}
mission_waypoints = {}
mission_status = {"active": False, "completed": False, "robots": {}}
position_monitoring_thread = None
stop_monitoring = False

# Canal isolado para monitoramento contínuo de robôs
robot_monitoring_manager = None
robot_monitoring_thread = None
stop_robot_monitoring = False
pause_robot_monitoring = False  # Flag para pausar temporariamente o monitoramento
all_robot_positions = {}  # Posições de TODOS os robôs conectados
robot_connection_status = {}  # Status de conexão de cada robô

def get_latlon(all_states, robot):
    """Extrai lat/lon dos estados dos robôs"""
    d = all_states.get(robot)
    if d and "lat" in d and "lon" in d:
        return float(d["lat"]), float(d["lon"])
    return None, None

def initialize_robot_monitoring():
    """
    Inicializa o sistema de monitoramento contínuo de robôs
    Conecta a todos os robôs conhecidos e mantém conexão ativa
    """
    global robot_monitoring_manager, robot_connection_status
    
    print(f"\n🔄 INICIALIZANDO CANAL ISOLADO DE MONITORAMENTO DE ROBÔS")
    print(f"="*80)
    
    # Criar lista de todos os robôs conhecidos do mapeamento
    all_known_robots = []
    for i, (original_identifier, mission_id) in enumerate(ROVER_ID_MAPPING.items()):
        robot_config = {
            'name': mission_id,
            'channel': f"udp:0.0.0.0:145{5+i}1",
            'source_system': i + 1,
            'original_identifier': original_identifier
        }
        all_known_robots.append(robot_config)
        # Inicializar status de conexão
        robot_connection_status[mission_id] = {
            'connected': False,
            'last_seen': None,
            'connection_attempts': 0,
            'original_identifier': original_identifier
        }
    
    print(f"🤖 Robôs conhecidos para monitoramento:")
    for robot in all_known_robots:
        print(f"   • {robot['original_identifier']} -> {robot['name']} ({robot['channel']})")
    
    # Criar MissionManager dedicado para monitoramento
    robot_monitoring_manager = MissionManager(robots=all_known_robots)
    
    print(f"🔌 Tentando conectar a todos os robôs conhecidos...")
    connection_success = robot_monitoring_manager.connect_all()
    
    if connection_success:
        print(f"✅ Canal de monitoramento inicializado com {len(robot_monitoring_manager.connected)} robô(s)")
        for robot_name in robot_monitoring_manager.connected:
            robot_connection_status[robot_name]['connected'] = True
            robot_connection_status[robot_name]['last_seen'] = time.time()
            original_id = REVERSE_ROVER_MAPPING[robot_name]
            print(f"   • {robot_name} ({original_id}) conectado")
        
        # Log dos robôs que não conectaram
        all_robot_names = set(robot_config['name'] for robot_config in all_known_robots)
        connected_robots = set(robot_monitoring_manager.connected)
        not_connected = all_robot_names - connected_robots
        
        if not_connected:
            print(f"⚠️ Robôs que não conectaram: {list(not_connected)}")
            print(f"💡 Para conectar os robôs restantes, execute:")
            for robot_name in not_connected:
                index = list(ROVER_ID_MAPPING.values()).index(robot_name)
                port = 14551 + (index * 10)
                original_id = REVERSE_ROVER_MAPPING[robot_name]
                print(f"   • {robot_name} ({original_id}): sim_rover.py -I{index} --console --map (porta {port})")
    else:
        print(f"⚠️ Nenhum robô conectado inicialmente, continuando monitoramento...")
        print(f"💡 Para conectar os robôs, execute os comandos ArduPilot:")
        for robot in all_known_robots:
            index = list(ROVER_ID_MAPPING.values()).index(robot['name'])
            port = 14551 + (index * 10)
            print(f"   • {robot['name']}: sim_rover.py -I{index} --console --map (porta {port})")
    
    return robot_monitoring_manager

def continuous_robot_monitoring():
    """
    Thread dedicada para monitoramento contínuo de TODOS os robôs
    Funciona independentemente das missões ativas
    """
    global stop_robot_monitoring, pause_robot_monitoring, robot_monitoring_manager, all_robot_positions, robot_connection_status
    
    print("🔄 Iniciando monitoramento contínuo de robôs...")
    
    # Inicializar o sistema de monitoramento
    if not robot_monitoring_manager:
        robot_monitoring_manager = initialize_robot_monitoring()
    
    if not robot_monitoring_manager:
        print("❌ Falha ao inicializar sistema de monitoramento")
        return
    
    telemetry_counter = 0
    reconnection_counter = 0
    
    while not stop_robot_monitoring:
        try:
            # Pausar monitoramento se solicitado (durante envio de missão)
            if pause_robot_monitoring:
                print("⏸️ Monitoramento pausado temporariamente...")
                time.sleep(1)
                continue
            
            current_time = time.time()
            
            # Tentar reconectar robôs desconectados a cada 30 segundos
            if reconnection_counter % 15 == 0:  # A cada 30 segundos (2s * 15)
                try_reconnect_robots()
            
            # Fazer polling de telemetria se houver robôs conectados
            if robot_monitoring_manager.is_connected:
                robot_monitoring_manager.poll_once(per_robot_reads=20)
                
                # Tentar obter posições de todos os robôs primeiro
                all_states = robot_monitoring_manager.wait_for_position(timeout=1.0, require_all=True)
                
                # Se não conseguir todos, obter individualmente
                if not all_states:
                    all_states_dict = {}
                    for robot_name in robot_monitoring_manager.connected:
                        individual_state = robot_monitoring_manager.wait_for_position(timeout=0.5, require_all=False, robot=robot_name)
                        if individual_state:
                            if isinstance(individual_state, tuple):
                                _, state = individual_state
                                all_states_dict[robot_name] = state
                            else:
                                all_states_dict[robot_name] = individual_state
                else:
                    # Processar estados recebidos
                    if isinstance(all_states, tuple):
                        robot_name, state = all_states
                        all_states_dict = {robot_name: state}
                    elif isinstance(all_states, dict):
                        all_states_dict = all_states
                    else:
                        all_states_dict = {}
                
                # Obter telemetria adicional
                try:
                    latest_positions = robot_monitoring_manager.get_latest_positions()
                except AttributeError:
                    latest_positions = {}
                
                # Atualizar posições de todos os robôs conectados
                robots_updated = 0
                for robot_name in robot_monitoring_manager.connected:
                    lat, lon = get_latlon(all_states_dict, robot_name)
                    robot_telemetry = latest_positions.get(robot_name, {})
                    
                    if lat is not None and lon is not None:
                            robots_updated += 1
                            
                            # Atualizar status de conexão
                            robot_connection_status[robot_name]['connected'] = True
                            robot_connection_status[robot_name]['last_seen'] = current_time
                            
                            # Dados completos de posição com precisão garantida
                            position_data = {
                                "robot_id": robot_name,
                                "latitude": round(float(lat), 8),  # Garantir precisão de 8 casas decimais
                                "longitude": round(float(lon), 8),  # Garantir precisão de 8 casas decimais
                                "timestamp": current_time,
                                "status": all_states_dict.get(robot_name, {}).get("status", "unknown"),
                                "altitude": robot_telemetry.get("alt", 0),
                                "heading": robot_telemetry.get("hdg", 0),
                                "ground_speed": robot_telemetry.get("groundspeed", 0),
                                "battery_voltage": robot_telemetry.get("voltage_battery", 0),
                                "battery_remaining": robot_telemetry.get("battery_remaining", 0),
                                "mode": robot_telemetry.get("mode", "UNKNOWN"),
                                "armed": robot_telemetry.get("armed", False),
                                "mission_current": robot_telemetry.get("mission_current", 0),
                                "mission_count": robot_telemetry.get("mission_count", 0),
                                "original_identifier": robot_connection_status[robot_name]['original_identifier']
                            }
                            
                            all_robot_positions[robot_name] = position_data
                            
                            # Log de posição removido para reduzir spam
                            
                            # Emitir posição via WebSocket para canal isolado (sempre alta frequência)
                            try:
                                # Emissão otimizada para múltiplos robôs - sempre alta frequência
                                num_robots = len(robot_monitoring_manager.connected)
                                is_multi_robot = num_robots > 1
                                
                                # Sempre usar alta frequência (100ms) para 3 robôs via UDP
                                should_emit = (
                                    telemetry_counter % WEBSOCKET_THROTTLE_MISSION == 0 or  # A cada ciclo (100ms)
                                    position_data.get('ground_speed', 0) > 0.01 or  # Se em movimento
                                    is_multi_robot or  # Para múltiplos robôs, emitir mais frequentemente
                                    position_data.get('mode', 'UNKNOWN') != 'UNKNOWN'  # Se temos dados de telemetria válidos
                                )
                                
                                if should_emit:
                                    # Log reduzido - apenas a cada 10 envios para reduzir spam
                                    if telemetry_counter % 20 == 0:  # Log a cada 10 segundos (20 ciclos * 0.5s)
                                        print(f"📡 Enviando posição {robot_name}: lat={position_data['latitude']:.6f}, lon={position_data['longitude']:.6f}")
                                    socketio.emit('robot_position_continuous', position_data)
                            except Exception as e:
                                if telemetry_counter % 50 == 0:
                                    print(f"⚠️ Erro ao emitir posição contínua via WebSocket: {e}")
                    
                    # Log resumo a cada 120 ciclos (60 segundos) - menos frequente
                    if telemetry_counter % 120 == 0:
                        total_known = len(ROVER_ID_MAPPING)
                        connected_count = len(robot_monitoring_manager.connected)
                        active_count = robots_updated
                        print(f"📊 Monitoramento: {active_count}/{connected_count}/{total_known} robôs (ativo/conectado/total)")
                        
                        # Emitir status geral via WebSocket
                        try:
                            socketio.emit('robot_monitoring_status', {
                                "total_known_robots": total_known,
                                "connected_robots": connected_count,
                                "active_robots": active_count,
                                "robot_positions": all_robot_positions,
                                "connection_status": robot_connection_status,
                                "timestamp": current_time
                            })
                        except Exception as e:
                            print(f"⚠️ Erro ao emitir status de monitoramento: {e}")
            
            else:
                # Nenhum robô conectado
                if telemetry_counter % 100 == 0:  # A cada 20 segundos
                    print(f"⚠️ Nenhum robô conectado no canal de monitoramento")
            
            telemetry_counter += 1
            reconnection_counter += 1
            time.sleep(MONITORING_UPDATE_RATE)  # Monitoramento configurável (0.5s)
            
        except Exception as e:
            print(f"⚠️ Erro no monitoramento contínuo de robôs: {e}")
            time.sleep(5)  # Aguardar mais em caso de erro
    
    print("🔄 Monitoramento contínuo de robôs finalizado")

def pause_continuous_monitoring():
    """
    Para completamente o monitoramento contínuo (para envio de missões)
    Desconecta o canal de monitoramento para evitar conflitos MAVLink
    """
    global pause_robot_monitoring, robot_monitoring_manager
    print("⏸️ Pausando monitoramento contínuo para envio de missão...")
    pause_robot_monitoring = True
    
    # Desconectar completamente o canal de monitoramento para evitar conflitos
    if robot_monitoring_manager:
        print("🔌 Desconectando canal de monitoramento temporariamente...")
        try:
            # Fechar todas as conexões do canal de monitoramento
            for robot_name, master in robot_monitoring_manager.masters.items():
                try:
                    master.close()
                    print(f"   🔌 Conexão fechada para {robot_name}")
                except Exception as e:
                    print(f"   ⚠️ Erro ao fechar conexão para {robot_name}: {e}")
            
            # Limpar masters para forçar reconexão depois
            robot_monitoring_manager.masters.clear()
            robot_monitoring_manager.connected.clear()
            print("✅ Canal de monitoramento desconectado temporariamente")
        except Exception as e:
            print(f"⚠️ Erro ao desconectar canal de monitoramento: {e}")
    
    time.sleep(3)  # Aguardar desconexão completa

def resume_continuous_monitoring():
    """
    Retoma o monitoramento contínuo
    Reconecta o canal de monitoramento após envio de missões
    """
    global pause_robot_monitoring, robot_monitoring_manager
    print("▶️ Retomando monitoramento contínuo...")
    
    # Aguardar um pouco para garantir que o canal de missão terminou
    time.sleep(2)
    
    # Reconectar o canal de monitoramento
    if robot_monitoring_manager:
        print("🔄 Reconectando canal de monitoramento...")
        try:
            # Tentar reconectar todos os robôs
            connection_success = robot_monitoring_manager.connect_all()
            if connection_success:
                print(f"✅ Canal de monitoramento reconectado com {len(robot_monitoring_manager.connected)} robô(s)")
            else:
                print("⚠️ Falha na reconexão do canal de monitoramento")
        except Exception as e:
            print(f"⚠️ Erro ao reconectar canal de monitoramento: {e}")
    
    pause_robot_monitoring = False

def try_reconnect_robots():
    """
    Tenta reconectar robôs que perderam conexão
    """
    global robot_monitoring_manager, robot_connection_status
    
    if not robot_monitoring_manager:
        return
    
    current_time = time.time()
    robots_to_reconnect = []
    
    # Identificar robôs desconectados que já estiveram conectados antes
    for robot_id, status in robot_connection_status.items():
        # Só considerar desconectado se já esteve conectado antes (last_seen não é None)
        if (status['last_seen'] is not None and 
            not status['connected'] and 
            (current_time - status['last_seen']) > 30):
            robots_to_reconnect.append(robot_id)
            status['connection_attempts'] += 1
    
    if robots_to_reconnect:
        print(f"🔄 Tentando reconectar {len(robots_to_reconnect)} robô(s): {robots_to_reconnect}")
        
        # Tentar reconectar apenas robôs que já estiveram conectados
        for robot_id in robots_to_reconnect:
            try:
                # Lógica de reconexão específica pode ser implementada aqui
                print(f"   🔄 Tentativa de reconexão para {robot_id} (tentativa #{robot_connection_status[robot_id]['connection_attempts']})")
            except Exception as e:
                print(f"⚠️ Erro ao tentar reconectar {robot_id}: {e}")
    
    # Log informativo sobre robôs que nunca se conectaram
    never_connected = [robot_id for robot_id, status in robot_connection_status.items() 
                      if status['last_seen'] is None and not status['connected']]
    
    if never_connected and len(never_connected) > 0:
        # Log apenas a cada 10 tentativas de reconexão para não spammar
        if all(robot_connection_status[rid]['connection_attempts'] % 10 == 0 for rid in never_connected):
            print(f"💡 Robôs que nunca se conectaram: {never_connected}")
            print(f"   Verifique se estão rodando no ArduPilot:")
            for robot_id in never_connected:
                index = list(ROVER_ID_MAPPING.values()).index(robot_id)
                port = 14551 + (index * 10)
                original_id = REVERSE_ROVER_MAPPING[robot_id]
                print(f"   • {robot_id} ({original_id}): porta {port}")
                print(f"     Comando: sim_rover.py -I{index} --console --map")

def start_continuous_robot_monitoring():
    """
    Inicia o thread de monitoramento contínuo de robôs
    """
    global robot_monitoring_thread, stop_robot_monitoring
    
    if robot_monitoring_thread and robot_monitoring_thread.is_alive():
        print("🔄 Monitoramento contínuo já está ativo")
        return True
    
    stop_robot_monitoring = False
    robot_monitoring_thread = threading.Thread(target=continuous_robot_monitoring)
    robot_monitoring_thread.daemon = True
    robot_monitoring_thread.start()
    
    print("✅ Thread de monitoramento contínuo iniciado")
    return True

def stop_continuous_robot_monitoring():
    """
    Para o thread de monitoramento contínuo de robôs
    """
    global stop_robot_monitoring, robot_monitoring_manager
    
    stop_robot_monitoring = True
    
    if robot_monitoring_manager:
        try:
            robot_monitoring_manager.close()
            print("✅ Conexões do canal de monitoramento fechadas")
        except Exception as e:
            print(f"⚠️ Erro ao fechar conexões do canal de monitoramento: {e}")
        robot_monitoring_manager = None
    
    print("🔄 Monitoramento contínuo parado")

# Mapeamento de rovers do banco Django para identificadores do mission_server
ROVER_ID_MAPPING = {
    # Mapear identifiers do banco para IDs simples do mission_server
    "Rover-Beta": "R1",
    "Rover-Charlie": "R2", 
    "Rover-Delta": "R3"
}

# Mapeamento reverso para logs e debug
REVERSE_ROVER_MAPPING = {v: k for k, v in ROVER_ID_MAPPING.items()}

def parse_robots_string(robots_str):
    """
    Converte string de robôs em lista de dicionários
    Ex: "R1,R2,R3" -> [{'name': 'R1', 'channel': '...', 'source_system': 1}, ...]
    """
    robot_names = [name.strip() for name in robots_str.split(',') if name.strip()]
    robots = []
    
    for i, name in enumerate(robot_names):
        robots.append({
            'name': name,
            'channel': f"udp:0.0.0.0:145{5+i}1",  # 14551, 14561, 14571, etc.
            'source_system': i + 1
        })
    
    return robots

def map_rover_identifier_to_mission_id(rover_identifier):
    """
    Mapeia identifier do banco Django para ID do mission_server
    
    Args:
        rover_identifier: Identifier do banco (ex: "Rover-Beta")
    
    Returns:
        str: ID mapeado (ex: "R1") ou identifier original se não encontrado
    """
    mapped_id = ROVER_ID_MAPPING.get(rover_identifier, rover_identifier)
    print(f"🔄 Mapeamento rover: {rover_identifier} -> {mapped_id}")
    return mapped_id

def create_rover_config_from_frontend_data(rover_data, index):
    """
    Cria configuração de rover para o mission_server a partir dos dados do frontend
    
    Args:
        rover_data: Dados do rover do frontend
        index: Índice do rover na lista
    
    Returns:
        dict: Configuração do rover para mission_server
    """
    # Obter identifier original do banco
    original_identifier = rover_data.get('identifier', f'R{index+1}')
    
    # Mapear para ID do mission_server
    mission_id = map_rover_identifier_to_mission_id(original_identifier)
    
    # Configuração para mission_server
    config = {
        'name': mission_id,  # ID mapeado (R1, R2, R3, etc.)
        'channel': f"udp:0.0.0.0:145{5+index}1",  # 14551, 14561, 14571, etc.
        'source_system': index + 1,
        # Dados originais para referência
        'original_identifier': original_identifier,
        'db_id': rover_data.get('id'),
        'display_name': rover_data.get('name', mission_id),
        'model': rover_data.get('model', 'Unknown')
    }
    
    print(f"🤖 Rover configurado:")
    print(f"   • DB Identifier: {original_identifier}")
    print(f"   • Mission ID: {mission_id}")
    print(f"   • Display Name: {config['display_name']}")
    print(f"   • Channel: {config['channel']}")
    
    return config

def parse_equipments_string(equipments_str):
    """
    Converte string de equipamentos em lista
    Ex: "b_busip4,ef_reator1,ls_pr4" -> ['b_busip4', 'ef_reator1', 'ls_pr7']
    """
    return [eq.strip() for eq in equipments_str.split(',') if eq.strip()]

def extract_mission_id_from_equipment(equipment_data):
    """
    Extrai o ID da missão diretamente do campo equipmentId do equipamento
    
    Args:
        equipment_data: Dados do equipamento do frontend
        
    Returns:
        str: Identificador da missão (ex: 'ef_reator1', 'b_busip4', etc.)
    """
    # O equipmentId já contém o identificador correto da missão (última coluna do CSV)
    mission_id = equipment_data.get('equipmentId', '')
    
    if not mission_id:
        # Fallback: tentar extrair do ID principal se equipmentId não estiver disponível
        equipment_id = equipment_data.get('id', '')
        equipment_name = equipment_data.get('name', '')
        print(f"⚠️ equipmentId não encontrado para {equipment_name}, usando fallback")
        return None
    
    print(f"✅ Equipamento {equipment_data.get('name', 'N/A')} -> missão: {mission_id}")
    return mission_id

def get_available_missions_from_obp():
    """
    Carrega e retorna lista de missões disponíveis do arquivo obp_6.json
    
    Returns:
        list: Lista de identificadores de missões disponíveis
    """
    try:
        with open(OBSERVATION_POINTS_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        available_missions = list(data.keys())
        print(f"📋 Missões disponíveis no obp_6.json: {available_missions}")
        return available_missions
    except Exception as e:
        print(f"❌ Erro ao carregar missões do obp_6.json: {e}")
        return []

def extract_waypoints_from_missions(missoes_por_robo):
    """
    Extrai waypoints das missões geradas internamente
    
    Args:
        missoes_por_robo: dict com missões por robô
        
    Returns:
        dict: waypoints organizados por robô
    """
    waypoints = {}
    for robot, mission_points in missoes_por_robo.items():
        waypoints[robot] = []
        print(f"\n📍 Extraindo waypoints para {robot}:")
        print(f"   Total de pontos na missão: {len(mission_points)}")
        
        for point in mission_points:
            wp_data = {
                'id': point['id'],
                'lat': point['lat'],
                'lon': point['lon'],
                'hold': point.get('hold', 0.0),
                'accept_radius': point.get('accept_radius', 0.0),
                'pass_radius': point.get('pass_radius', 0.0),
                'yaw_deg': point.get('yaw_deg', 0.0)
            }
            waypoints[robot].append(wp_data)
            
            # Print detalhado de cada waypoint
            print(f"   WP {wp_data['id']:2d}: lat={wp_data['lat']:10.6f}, lon={wp_data['lon']:10.6f}, "
                  f"hold={wp_data['hold']:4.1f}s, yaw={wp_data['yaw_deg']:6.1f}°")
    
    return waypoints

def verify_and_download_mission_from_ardupilot(mission_manager, robot_name):
    """
    Verifica e baixa a missão atual do ArduPilot para sincronização (OPCIONAL)
    
    Args:
        mission_manager: Instância do MissionManager
        robot_name: Nome do robô
        
    Returns:
        list: Lista de waypoints baixados do ArduPilot ou None em caso de erro
    """
    try:
        print(f"🔍 Verificando missão no ArduPilot para {robot_name}...")
        
        # Usar o método download_mission com timeout menor
        if hasattr(mission_manager, 'download_mission'):
            downloaded_waypoints = mission_manager.download_mission(robot=robot_name)
        else:
            # Fallback para método de verificação existente
            downloaded_waypoints = mission_manager.verify_mission_and_return_waypoints(robot=robot_name)
        
        if downloaded_waypoints:
            print(f"✅ {len(downloaded_waypoints)} waypoints verificados no ArduPilot para {robot_name}")
            return downloaded_waypoints
        else:
            print(f"ℹ️ Verificação opcional falhou para {robot_name} - continuando normalmente")
            return None
            
    except Exception as e:
        print(f"ℹ️ Verificação opcional com erro para {robot_name}: {e} - continuando normalmente")
        return None

def format_ardupilot_waypoints_for_frontend(ardupilot_waypoints, robot_name):
    """
    Formata waypoints baixados do ArduPilot para o formato esperado pelo frontend
    
    Args:
        ardupilot_waypoints: Lista de waypoints do ArduPilot
        robot_name: Nome do robô
        
    Returns:
        list: Waypoints formatados para o frontend
    """
    formatted_waypoints = []
    
    for i, wp in enumerate(ardupilot_waypoints):
        try:
            # Converter formato MAVLink para formato do frontend
            if hasattr(wp, 'x') and hasattr(wp, 'y'):
                # MISSION_ITEM_INT format
                lat = wp.x / 1e7
                lon = wp.y / 1e7
                alt = wp.z
            else:
                # Formato já convertido
                lat = wp.get('lat', 0)
                lon = wp.get('lon', 0)
                alt = wp.get('alt', 0)
            
            wp_data = {
                'id': i,
                'lat': lat,
                'lon': lon,
                'alt': alt,
                'hold': wp.get('param1', 0.0),
                'accept_radius': wp.get('param2', 0.0),
                'pass_radius': wp.get('param3', 0.0),
                'yaw_deg': wp.get('param4', 0.0),
                'robot': robot_name,
                'source': 'ardupilot'  # Marcar como vindo do ArduPilot
            }
            formatted_waypoints.append(wp_data)
            
        except Exception as e:
            print(f"⚠️ Erro ao formatar waypoint {i} para {robot_name}: {e}")
            continue
    
    return formatted_waypoints

def monitor_robot_positions():
    """Thread para monitorar posições dos robôs continuamente com telemetria aprimorada"""
    global stop_monitoring, mission_manager, robot_positions, mission_status
    
    print("🔄 Iniciando monitoramento de posições dos robôs...")
    print(f"🤖 Robôs conectados: {list(mission_manager.connected) if mission_manager else 'Nenhum'}")
    
    # Contador para polling menos frequente de telemetria detalhada
    telemetry_counter = 0
    
    # Verificar se mission_manager existe e tem conexões
    if not mission_manager:
        print("❌ MissionManager não inicializado")
        return
    
    if not mission_manager.is_connected:
        print("❌ Nenhum robô conectado para monitoramento")
        return
    
    # Forçar stream GPS inicial para todos os robôs
    for robot_name in mission_manager.connected:
        try:
            mission_manager.force_gps_stream(rate_hz=5.0, robot=robot_name)
            print(f"   📡 Stream GPS ativado para monitoramento de {robot_name}")
        except Exception as e:
            print(f"   ⚠️ Erro ao ativar stream GPS para {robot_name}: {e}")
    
    while not stop_monitoring and mission_manager and mission_manager.is_connected:
        try:
            # Fazer polling de telemetria mais intensivo para garantir dados de todos os robôs
            for polling_round in range(3):  # Múltiplas rodadas de polling
                mission_manager.poll_once(per_robot_reads=20)
                time.sleep(0.05)  # Pequena pausa entre rodadas
            
            # Tentar obter posições de TODOS os robôs primeiro
            all_states = mission_manager.wait_for_position(timeout=2.0, require_all=True)
            
            current_time = time.time()
            
            # SEMPRE obter dados individualmente para garantir cobertura de todos os robôs
            all_states_dict = {}
            
            # Se conseguiu dados de todos, usar esses dados
            if all_states and isinstance(all_states, dict):
                all_states_dict.update(all_states)
                if telemetry_counter % 20 == 0:
                    print(f"🔍 Debug - Estados obtidos de todos: {list(all_states.keys())}")
            
            # SEMPRE complementar com dados individuais para robôs que não tiveram dados
            for robot_name in mission_manager.connected:
                if robot_name not in all_states_dict:
                    # Fazer polling adicional específico para este robô
                    mission_manager.poll_once(per_robot_reads=15)
                    time.sleep(0.1)
                    
                    individual_state = mission_manager.wait_for_position(timeout=1.0, require_all=False, robot=robot_name)
                    if individual_state:
                        if isinstance(individual_state, tuple):
                            _, state = individual_state
                            all_states_dict[robot_name] = state
                        else:
                            all_states_dict[robot_name] = individual_state
                        
                        # Debug: dados individuais obtidos
                        if telemetry_counter % 10 == 0:
                            print(f"🔍 Debug - Estado individual obtido para {robot_name}: {all_states_dict[robot_name]}")
                    else:
                        # Debug: falha ao obter dados individuais
                        if telemetry_counter % 10 == 0:
                            print(f"⚠️ Debug - Falha ao obter estado individual para {robot_name}")
            
            # Debug: resumo final dos estados disponíveis
            if telemetry_counter % 20 == 0:
                print(f"🔍 Debug - Estados finais disponíveis: {list(all_states_dict.keys())}")
                
                # Obter telemetria adicional dos robôs conectados
                try:
                    latest_positions = mission_manager.get_latest_positions()
                except AttributeError:
                    # Fallback se método não existir
                    latest_positions = {}
                
                # Processar TODOS os robôs conectados individualmente para garantir cobertura completa
                robots_with_position = 0
                
                # Debug: mostrar estado atual dos robôs conectados
                if telemetry_counter % 20 == 0:
                    print(f"🔍 Robôs conectados para monitoramento: {list(mission_manager.connected)}")
                    print(f"🔍 Estados disponíveis: {list(all_states_dict.keys())}")
                
                for robot_name in mission_manager.connected:
                    # Combinar dados de posição e telemetria
                    lat, lon = get_latlon(all_states_dict, robot_name)
                    robot_telemetry = latest_positions.get(robot_name, {})
                    
                    # Debug removido para reduzir spam de logs
                    
                    if lat is not None and lon is not None:
                        robots_with_position += 1
                        
                        # Dados básicos de posição com precisão garantida
                        position_data = {
                            "latitude": round(float(lat), 8),  # Garantir precisão de 8 casas decimais
                            "longitude": round(float(lon), 8),  # Garantir precisão de 8 casas decimais
                            "timestamp": current_time,
                            "status": all_states_dict.get(robot_name, {}).get("status", "unknown")
                        }
                        
                        # Adicionar telemetria detalhada se disponível
                        if robot_telemetry:
                            position_data.update({
                                "altitude": robot_telemetry.get("alt", 0),
                                "heading": robot_telemetry.get("hdg", 0),
                                "ground_speed": robot_telemetry.get("groundspeed", 0),
                                "battery_voltage": robot_telemetry.get("voltage_battery", 0),
                                "battery_remaining": robot_telemetry.get("battery_remaining", 0),
                                "mode": robot_telemetry.get("mode", "UNKNOWN"),
                                "armed": robot_telemetry.get("armed", False),
                                "mission_current": robot_telemetry.get("mission_current", 0),
                                "mission_count": robot_telemetry.get("mission_count", 0)
                            })
                        
                        robot_positions[robot_name] = position_data
                        
                        # Log de posição removido para reduzir spam
                        
                        # Emitir posição via WebSocket com throttling otimizado para múltiplos robôs
                        try:
                            # Para múltiplos robôs, ser mais agressivo na emissão
                            num_robots = len(mission_manager.connected)
                            is_multi_robot = num_robots > 1
                            
                            # Sempre usar alta frequência (100ms) para 3 robôs via UDP
                            should_emit = (
                                telemetry_counter % WEBSOCKET_THROTTLE_MISSION == 0 or  # A cada ciclo (100ms)
                                position_data.get('ground_speed', 0) > 0.01 or  # Se em movimento
                                is_multi_robot or  # Para múltiplos robôs, emitir mais frequentemente
                                position_data.get('mode', 'UNKNOWN') != 'UNKNOWN'  # Se temos dados de telemetria válidos
                            )
                            
                            if should_emit:
                                # Log reduzido - apenas a cada 10 envios para reduzir spam
                                if telemetry_counter % 20 == 0:  # Log a cada 10 segundos (20 ciclos * 0.5s)
                                    print(f"📡 Enviando posição missão {robot_name}: lat={position_data['latitude']:.6f}, lon={position_data['longitude']:.6f}")
                                socketio.emit('robot_position_update', {
                                    "robot_id": robot_name,
                                    **position_data
                                })
                        except Exception as e:
                            if telemetry_counter % 50 == 0:  # Log erro menos frequente
                                print(f"⚠️ Erro ao emitir posição via WebSocket: {e}")
                    else:
                        # Debug: robô sem posição válida (log reduzido)
                        if telemetry_counter % 100 == 0:  # A cada 100 ciclos (20 segundos)
                            print(f"⚠️ {robot_name}: Sem posição GPS válida")
                
                # Log resumo detalhado a cada 120 ciclos (60 segundos) - menos frequente
                if telemetry_counter % 120 == 0:
                    print(f"📊 Monitoramento missão: {robots_with_position}/{len(mission_manager.connected)} robôs com posição")
                
                # Verificar progresso da missão com base no waypoint atual
                if mission_status["active"] and not mission_status["completed"]:
                    mission_progress = check_mission_progress(robot_positions, mission_waypoints)
                    
                    if mission_progress["completed"]:
                        mission_status["completed"] = True
                        mission_status["active"] = False
                        try:
                            print(f"📡 [WEBSOCKET] Enviando conclusão de missão:")
                            print(f"   ✅ Missão concluída com sucesso!")
                            print(f"   🤖 Robôs: {list(mission_status['robots'].keys())}")
                            print(f"   📊 Progresso final: {mission_progress}")
                            
                            socketio.emit('mission_completed', {
                                "message": "Missão concluída com sucesso!",
                                "timestamp": current_time,
                                "robots": list(mission_status["robots"].keys()),
                                "progress": mission_progress
                            })
                        except Exception as e:
                            print(f"⚠️ Erro ao emitir conclusão de missão via WebSocket: {e}")
                        print("✅ Missão concluída!")
                    else:
                        # Emitir progresso da missão periodicamente
                        if telemetry_counter % MISSION_PROGRESS_UPDATE_RATE == 0:  # Frequência configurável
                            try:
                                print(f"📡 [WEBSOCKET] Enviando atualização de progresso da missão:")
                                print(f"   📊 Progresso: {mission_progress.get('overall_progress', 0):.1f}%")
                                print(f"   🎯 Waypoints completados: {mission_progress.get('completed_waypoints', 0)}")
                                
                                socketio.emit('mission_progress_update', {
                                    "progress": mission_progress,
                                    "timestamp": current_time
                                })
                            except Exception as e:
                                print(f"⚠️ Erro ao emitir progresso via WebSocket: {e}")
            
            else:
                # Nenhum estado recebido
                if telemetry_counter % 20 == 0:  # Debug menos frequente
                    print(f"⚠️ Nenhum estado recebido dos robôs conectados: {list(mission_manager.connected)}")
            
            telemetry_counter += 1
            time.sleep(MONITORING_UPDATE_RATE)  # Usar frequência configurável (0.5s)
            
        except Exception as e:
            print(f"⚠️ Erro no monitoramento de posições: {e}")
            # Debug: mostrar status de conexão
            if mission_manager:
                print(f"🔍 Debug - Robôs conectados: {list(mission_manager.connected)}")
                print(f"🔍 Debug - Masters disponíveis: {list(mission_manager.masters.keys()) if hasattr(mission_manager, 'masters') else 'N/A'}")
                print(f"🔍 Debug - is_connected: {mission_manager.is_connected}")
            
            # Tentar reativar streams GPS em caso de erro
            if mission_manager and mission_manager.is_connected:
                for robot_name in mission_manager.connected:
                    try:
                        mission_manager.force_gps_stream(rate_hz=5.0, robot=robot_name)
                        print(f"   🔄 Stream GPS reativado para {robot_name}")
                    except Exception as stream_error:
                        print(f"   ❌ Erro ao reativar stream GPS para {robot_name}: {stream_error}")
            
            time.sleep(5)  # Aguardar mais tempo em caso de erro
    
    print("🔄 Monitoramento de posições finalizado")

def check_mission_progress(robot_positions, mission_waypoints):
    """
    Verifica o progresso das missões baseado nas posições dos robôs e waypoints
    
    Args:
        robot_positions: Dicionário com posições atuais dos robôs
        mission_waypoints: Dicionário com waypoints das missões por robô
        
    Returns:
        dict: Informações sobre o progresso da missão
    """
    progress = {
        "completed": False,
        "overall_progress": 0.0,
        "robot_progress": {},
        "total_robots": len(mission_waypoints),
        "active_robots": 0
    }
    
    if not mission_waypoints:
        return progress
    
    total_progress = 0
    active_robots = 0
    
    for robot_name, waypoints in mission_waypoints.items():
        if not waypoints:
            continue
            
        robot_pos = robot_positions.get(robot_name)
        if not robot_pos:
            continue
            
        active_robots += 1
        
        # Calcular progresso baseado no waypoint atual da telemetria
        current_wp = robot_pos.get("mission_current", 0)
        total_wp = len(waypoints)
        
        robot_progress = min(current_wp / max(total_wp, 1), 1.0) if total_wp > 0 else 0
        
        progress["robot_progress"][robot_name] = {
            "current_waypoint": current_wp,
            "total_waypoints": total_wp,
            "progress_percent": robot_progress * 100,
            "completed": robot_progress >= 0.95  # Considerar completo quando > 95%
        }
        
        total_progress += robot_progress
    
    progress["active_robots"] = active_robots
    progress["overall_progress"] = (total_progress / max(active_robots, 1)) * 100 if active_robots > 0 else 0
    
    # Missão completa quando todos os robôs ativos completaram suas rotas
    completed_robots = sum(1 for rp in progress["robot_progress"].values() if rp["completed"])
    progress["completed"] = completed_robots == active_robots and active_robots > 0
    
    return progress

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar se o servidor está funcionando"""
    return jsonify({
        "status": "healthy",
        "message": "Servidor de missões ativo",
        "version": "1.0.0",
        "mission_active": mission_status["active"],
        "connected_robots": len(robot_positions),
        "continuous_monitoring_active": robot_monitoring_thread is not None and robot_monitoring_thread.is_alive(),
        "total_robots_monitored": len(all_robot_positions)
    })

@app.route('/robot-monitoring/status', methods=['GET'])
def get_robot_monitoring_status():
    """Endpoint para obter status do canal de monitoramento contínuo"""
    global robot_monitoring_thread, all_robot_positions, robot_connection_status
    
    monitoring_active = robot_monitoring_thread is not None and robot_monitoring_thread.is_alive()
    
    return jsonify({
        "success": True,
        "data": {
            "monitoring_active": monitoring_active,
            "total_known_robots": len(ROVER_ID_MAPPING),
            "connected_robots": sum(1 for status in robot_connection_status.values() if status['connected']),
            "active_robots": len(all_robot_positions),
            "robot_positions": all_robot_positions,
            "connection_status": robot_connection_status,
            "timestamp": time.time()
        }
    })

@app.route('/robot-monitoring/start', methods=['POST'])
def start_robot_monitoring_endpoint():
    """Endpoint para iniciar o canal de monitoramento contínuo"""
    try:
        success = start_continuous_robot_monitoring()
        if success:
            return jsonify({
                "success": True,
                "message": "Canal de monitoramento contínuo iniciado com sucesso"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Falha ao iniciar canal de monitoramento contínuo"
            }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erro ao iniciar monitoramento: {str(e)}"
        }), 500

@app.route('/robot-monitoring/stop', methods=['POST'])
def stop_robot_monitoring_endpoint():
    """Endpoint para parar o canal de monitoramento contínuo"""
    try:
        stop_continuous_robot_monitoring()
        return jsonify({
            "success": True,
            "message": "Canal de monitoramento contínuo parado com sucesso"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erro ao parar monitoramento: {str(e)}"
        }), 500

@app.route('/robot-positions', methods=['GET'])
def get_all_robot_positions():
    """Endpoint para obter posições de todos os robôs monitorados"""
    return jsonify({
        "success": True,
        "data": {
            "robot_positions": all_robot_positions,
            "connection_status": robot_connection_status,
            "total_robots": len(all_robot_positions),
            "timestamp": time.time(),
            "update_frequency": {
                "monitoring_rate": MONITORING_UPDATE_RATE,
                "websocket_throttle_stationary": WEBSOCKET_THROTTLE_STATIONARY,
                "websocket_throttle_mission": WEBSOCKET_THROTTLE_MISSION,
                "mission_progress_rate": MISSION_PROGRESS_UPDATE_RATE
            }
        }
    })

@app.route('/optimize-frequencies', methods=['POST'])
def optimize_monitoring_frequencies():
    """Endpoint para aplicar configurações otimizadas de frequência"""
    global MONITORING_UPDATE_RATE, WEBSOCKET_THROTTLE_STATIONARY, WEBSOCKET_THROTTLE_MISSION, MISSION_PROGRESS_UPDATE_RATE
    
    try:
        # Configurações otimizadas para responsividade controlada
        MONITORING_UPDATE_RATE = 0.5  # 500ms - responsivo mas controlado
        WEBSOCKET_THROTTLE_STATIONARY = 2  # Emitir a cada 2 ciclos (1 segundo)
        WEBSOCKET_THROTTLE_MISSION = 2  # Emitir a cada 2 ciclos (1 segundo)
        MISSION_PROGRESS_UPDATE_RATE = 5  # Progresso a cada 5 segundos
        
        print(f"🚀 Frequências otimizadas aplicadas:")
        print(f"   • Taxa de monitoramento: {MONITORING_UPDATE_RATE}s (500ms)")
        print(f"   • Throttle robôs parados: {WEBSOCKET_THROTTLE_STATIONARY} ciclo (1 segundo)")
        print(f"   • Throttle durante missões: {WEBSOCKET_THROTTLE_MISSION} ciclo (1 segundo)")
        print(f"   • Progresso de missão: {MISSION_PROGRESS_UPDATE_RATE}s")
        
        return jsonify({
            "success": True,
            "message": "Frequências otimizadas aplicadas com sucesso",
            "data": {
                "monitoring_rate": MONITORING_UPDATE_RATE,
                "websocket_throttle_stationary": WEBSOCKET_THROTTLE_STATIONARY,
                "websocket_throttle_mission": WEBSOCKET_THROTTLE_MISSION,
                "mission_progress_rate": MISSION_PROGRESS_UPDATE_RATE,
                "expected_websocket_frequency": "~1 segundo (1 Hz)",
                "performance_impact": "CPU moderado, Responsividade controlada"
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erro ao otimizar frequências: {str(e)}"
        }), 500

@app.route('/update-frequency', methods=['POST'])
def update_monitoring_frequency():
    """Endpoint para ajustar frequência de monitoramento em tempo real"""
    global MONITORING_UPDATE_RATE, WEBSOCKET_THROTTLE_STATIONARY, WEBSOCKET_THROTTLE_MISSION, MISSION_PROGRESS_UPDATE_RATE
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "message": "Nenhum dado recebido"
            }), 400
        
        # Validar e atualizar configurações
        updated_settings = {}
        
        if 'monitoring_rate' in data:
            rate = float(data['monitoring_rate'])
            if 0.1 <= rate <= 5.0:  # Entre 100ms e 5 segundos
                MONITORING_UPDATE_RATE = rate
                updated_settings['monitoring_rate'] = rate
            else:
                return jsonify({
                    "success": False,
                    "message": "monitoring_rate deve estar entre 0.1 e 5.0 segundos"
                }), 400
        
        if 'websocket_throttle_stationary' in data:
            throttle = int(data['websocket_throttle_stationary'])
            if 1 <= throttle <= 10:
                WEBSOCKET_THROTTLE_STATIONARY = throttle
                updated_settings['websocket_throttle_stationary'] = throttle
            else:
                return jsonify({
                    "success": False,
                    "message": "websocket_throttle_stationary deve estar entre 1 e 10"
                }), 400
        
        if 'websocket_throttle_mission' in data:
            throttle = int(data['websocket_throttle_mission'])
            if 1 <= throttle <= 10:
                WEBSOCKET_THROTTLE_MISSION = throttle
                updated_settings['websocket_throttle_mission'] = throttle
            else:
                return jsonify({
                    "success": False,
                    "message": "websocket_throttle_mission deve estar entre 1 e 10"
                }), 400
        
        if 'mission_progress_rate' in data:
            rate = int(data['mission_progress_rate'])
            if 1 <= rate <= 10:
                MISSION_PROGRESS_UPDATE_RATE = rate
                updated_settings['mission_progress_rate'] = rate
            else:
                return jsonify({
                    "success": False,
                    "message": "mission_progress_rate deve estar entre 1 e 10"
                }), 400
        
        print(f"🔧 Frequências atualizadas: {updated_settings}")
        
        return jsonify({
            "success": True,
            "message": "Frequências atualizadas com sucesso",
            "data": {
                "updated_settings": updated_settings,
                "current_settings": {
                    "monitoring_rate": MONITORING_UPDATE_RATE,
                    "websocket_throttle_stationary": WEBSOCKET_THROTTLE_STATIONARY,
                    "websocket_throttle_mission": WEBSOCKET_THROTTLE_MISSION,
                    "mission_progress_rate": MISSION_PROGRESS_UPDATE_RATE
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erro ao atualizar frequências: {str(e)}"
        }), 500

@app.route('/rover-mapping', methods=['GET'])
def get_rover_mapping():
    """Endpoint para verificar o mapeamento de rovers"""
    return jsonify({
        "success": True,
        "message": "Mapeamento de rovers do banco Django para mission_server",
        "data": {
            "mapping": ROVER_ID_MAPPING,
            "reverse_mapping": REVERSE_ROVER_MAPPING,
            "total_rovers": len(ROVER_ID_MAPPING),
            "mapping_info": {
                "description": "Mapeia identifiers do banco Django para IDs simples do mission_server",
                "format": "banco_identifier -> mission_id",
                "examples": [
                    {"banco": "Rover-Beta", "mission": "R1"},
                    {"banco": "Rover-Charlie", "mission": "R2"},
                    {"banco": "Rover-Delta", "mission": "R3"}
                ]
            }
        }
    })

@app.route('/performance-stats', methods=['GET'])
def get_performance_stats():
    """Endpoint para obter estatísticas de performance do monitoramento"""
    global mission_manager, robot_monitoring_manager, robot_positions, all_robot_positions
    
    current_time = time.time()
    
    # Estatísticas do canal de missão
    mission_stats = {
        "active": mission_manager is not None and mission_manager.is_connected,
        "connected_robots": list(mission_manager.connected) if mission_manager else [],
        "robot_count": len(mission_manager.connected) if mission_manager else 0,
        "positions_available": len(robot_positions),
        "last_updates": {}
    }
    
    for robot_id, pos_data in robot_positions.items():
        if pos_data.get('timestamp'):
            age = current_time - pos_data['timestamp']
            mission_stats["last_updates"][robot_id] = {
                "age_seconds": round(age, 2),
                "is_fresh": age < 5.0,
                "ground_speed": pos_data.get('ground_speed', 0),
                "mode": pos_data.get('mode', 'UNKNOWN')
            }
    
    # Estatísticas do canal contínuo
    continuous_stats = {
        "active": robot_monitoring_manager is not None,
        "connected_robots": list(robot_monitoring_manager.connected) if robot_monitoring_manager else [],
        "robot_count": len(robot_monitoring_manager.connected) if robot_monitoring_manager else 0,
        "positions_available": len(all_robot_positions),
        "last_updates": {}
    }
    
    for robot_id, pos_data in all_robot_positions.items():
        if pos_data.get('timestamp'):
            age = current_time - pos_data['timestamp']
            continuous_stats["last_updates"][robot_id] = {
                "age_seconds": round(age, 2),
                "is_fresh": age < 5.0,
                "ground_speed": pos_data.get('ground_speed', 0),
                "mode": pos_data.get('mode', 'UNKNOWN')
            }
    
    return jsonify({
        "success": True,
        "data": {
            "mission_channel": mission_stats,
            "continuous_channel": continuous_stats,
            "config": {
                "monitoring_rate": MONITORING_UPDATE_RATE,
                "websocket_throttle_mission": WEBSOCKET_THROTTLE_MISSION,
                "websocket_throttle_stationary": WEBSOCKET_THROTTLE_STATIONARY,
                "mission_progress_rate": MISSION_PROGRESS_UPDATE_RATE
            },
            "timestamp": current_time
        }
    })

@app.route('/check-qgc-missions', methods=['GET'])
def check_qgc_missions():
    """Endpoint para verificar se as missões chegaram no QGroundControl"""
    global mission_manager
    
    if not mission_manager or not mission_manager.is_connected:
        return jsonify({
            "success": False,
            "message": "Nenhum canal de missão ativo"
        }), 400
    
    try:
        qgc_missions = {}
        
        for robot_name in mission_manager.connected:
            print(f"🔍 Verificando missão no QGroundControl para {robot_name}...")
            
            try:
                # Tentar baixar missão atual do ArduPilot
                downloaded_waypoints = mission_manager.download_mission(robot=robot_name)
                
                if downloaded_waypoints:
                    qgc_missions[robot_name] = {
                        "status": "success",
                        "waypoint_count": len(downloaded_waypoints),
                        "first_wp": {
                            "lat": downloaded_waypoints[0].get('lat', 0),
                            "lon": downloaded_waypoints[0].get('lon', 0)
                        } if downloaded_waypoints else None,
                        "last_wp": {
                            "lat": downloaded_waypoints[-1].get('lat', 0),
                            "lon": downloaded_waypoints[-1].get('lon', 0)
                        } if downloaded_waypoints else None
                    }
                    print(f"   ✅ {robot_name}: {len(downloaded_waypoints)} waypoints encontrados no QGC")
                else:
                    qgc_missions[robot_name] = {
                        "status": "no_mission",
                        "waypoint_count": 0
                    }
                    print(f"   ❌ {robot_name}: Nenhuma missão encontrada no QGC")
                    
            except Exception as e:
                qgc_missions[robot_name] = {
                    "status": "error",
                    "error": str(e)
                }
                print(f"   ❌ {robot_name}: Erro ao verificar - {e}")
        
        return jsonify({
            "success": True,
            "message": "Verificação de missões no QGroundControl concluída",
            "data": {
                "qgc_missions": qgc_missions,
                "connected_robots": list(mission_manager.connected),
                "timestamp": time.time()
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erro ao verificar missões no QGC: {str(e)}"
        }), 500

@app.route('/validate-rovers', methods=['POST'])
def validate_rovers():
    """
    Endpoint para validar e mapear rovers antes da execução da missão
    
    Recebe lista de rovers do Django e retorna mapeamento para mission_server
    """
    try:
        data = request.get_json()
        
        if not data or 'rovers' not in data:
            return jsonify({
                "success": False,
                "message": "Lista de rovers não fornecida"
            }), 400
        
        rovers_list = data['rovers']
        
        if not isinstance(rovers_list, list) or len(rovers_list) == 0:
            return jsonify({
                "success": False,
                "message": "Lista de rovers deve ser um array não vazio"
            }), 400
        
        # Mapear rovers e validar
        mapped_rovers = []
        validation_errors = []
        mapping_log = []
        
        for i, rover_data in enumerate(rovers_list):
            try:
                rover_config = create_rover_config_from_frontend_data(rover_data, i)
                mapped_rovers.append(rover_config)
                
                # Log do mapeamento
                mapping_log.append({
                    "index": i,
                    "original_identifier": rover_config['original_identifier'],
                    "mission_id": rover_config['name'],
                    "display_name": rover_config['display_name'],
                    "channel": rover_config['channel'],
                    "mapped": rover_config['original_identifier'] in ROVER_ID_MAPPING
                })
                
                # Validações básicas
                if not rover_config['name']:
                    validation_errors.append(f"Rover {i+1}: Nome/identifier não pode estar vazio")
                
                if not rover_config.get('original_identifier'):
                    validation_errors.append(f"Rover {i+1}: Identifier não encontrado")
                    
            except Exception as e:
                validation_errors.append(f"Rover {i+1}: Erro no mapeamento - {str(e)}")
        
        if validation_errors:
            return jsonify({
                "success": False,
                "message": "Erros de validação encontrados",
                "errors": validation_errors,
                "mapped_rovers": mapped_rovers,
                "mapping_log": mapping_log
            }), 400
        
        return jsonify({
            "success": True,
            "message": f"{len(mapped_rovers)} rovers validados e mapeados com sucesso",
            "data": {
                "mapped_rovers": mapped_rovers,
                "mapping_log": mapping_log,
                "mapping_info": {
                    "total_rovers": len(mapped_rovers),
                    "channels_used": [r['channel'] for r in mapped_rovers],
                    "source_systems": [r['source_system'] for r in mapped_rovers],
                    "mission_ids": [r['name'] for r in mapped_rovers],
                    "original_identifiers": [r['original_identifier'] for r in mapped_rovers]
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erro interno na validação: {str(e)}"
        }), 500

@app.route('/execute-mission', methods=['POST'])
def execute_mission():
    """
    Endpoint principal para executar planejamento de missão
    
    Espera receber formato do MissionPlanningView.js:
    {
        "name": "Nome da Missão",
        "description": "Descrição da missão",
        "rovers": [{"identifier": "R1", "name": "Rover 1", "id": 1}, ...],
        "equipments": [{"id": "eq1", "name": "Equipment 1", "type": "type1"}, ...],
        "substation": "substation_id"
    }
    
    OU formato simplificado:
    {
        "robots": "R1,R2,R3",
        "equipments": "b_busip4,ef_reator1,ls_pr4"
    }
    """
    global mission_manager, mission_waypoints, mission_status, position_monitoring_thread, stop_monitoring
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "message": "Nenhum dado recebido no POST"
            }), 400
        
        print(f"\n" + "="*80)
        print(f"🎯 NOVA MISSÃO RECEBIDA")
        print(f"="*80)
        print(f"📋 Dados recebidos: {data}")
        
        # Detectar formato dos dados (novo formato do MissionPlanningView vs antigo formato)
        if 'rovers' in data and isinstance(data['rovers'], list):
            # Novo formato do MissionPlanningView.js
            mission_name = data.get('name', f'Missão {datetime.now().strftime("%Y-%m-%d %H:%M")}')
            mission_description = data.get('description', 'Missão de inspeção automática')
            rovers_list = data.get('rovers', [])
            equipments_list = data.get('equipments', [])
            substation = data.get('substation', 'default')
            
            print(f"📝 Nome: {mission_name}")
            print(f"📝 Descrição: {mission_description}")
            print(f"🏭 Subestação: {substation}")
            print(f"🤖 Rovers: {[r.get('identifier', r.get('name', 'N/A')) for r in rovers_list]}")
            print(f"⚡ Equipamentos: {[e.get('name', e.get('id', 'N/A')) for e in equipments_list]}")
            
            # Converter para formato interno usando mapeamento
            robots = []
            for i, rover_data in enumerate(rovers_list):
                rover_config = create_rover_config_from_frontend_data(rover_data, i)
                robots.append(rover_config)
            
            # Converter equipamentos para lista de missões usando o equipmentId do CSV
            missions = []
            print(f"\n📋 PROCESSANDO EQUIPAMENTOS SELECIONADOS:")
            for equipment in equipments_list:
                mission_id = extract_mission_id_from_equipment(equipment)
                if mission_id:
                    missions.append(mission_id)
                    print(f"   ✅ {equipment.get('name', 'N/A')} -> {mission_id}")
                else:
                    print(f"   ❌ Falha ao extrair mission_id para {equipment.get('name', 'N/A')}")
            
            # Remover duplicatas mantendo ordem
            missions = list(dict.fromkeys(missions))
            
            print(f"📋 MISSÕES FINAIS EXTRAÍDAS:")
            print(f"   • Total de equipamentos selecionados: {len(equipments_list)}")
            print(f"   • Total de missões únicas: {len(missions)}")
            print(f"   • Missões: {missions}")
            
        else:
            # Formato antigo (compatibilidade)
            robots_str = data.get('robots', '')
            equipments_str = data.get('equipments', '')
            
            print(f"🤖 Robôs (formato antigo): {robots_str}")
            print(f"⚡ Equipamentos (formato antigo): {equipments_str}")
            
            # Validações para formato antigo
            if not robots_str:
                return jsonify({
                    "success": False,
                    "message": "String de robôs não pode estar vazia"
                }), 400
                
            if not equipments_str:
                return jsonify({
                    "success": False,
                    "message": "String de equipamentos não pode estar vazia"
                }), 400
            
            # Parsear strings
            robots = parse_robots_string(robots_str)
            missions = parse_equipments_string(equipments_str)
        
        # Validações finais
        if not robots:
            return jsonify({
                "success": False,
                "message": "Lista de robôs não pode estar vazia"
            }), 400
            
        if not missions:
            return jsonify({
                "success": False,
                "message": "Lista de equipamentos não pode estar vazia"
            }), 400
        
        print(f"\n📋 CONFIGURAÇÃO PROCESSADA:")
        print(f"   • Missões: {', '.join(missions)}")
        print(f"   • Robôs: {', '.join([r['name'] for r in robots])}")
        
        # Configurar mission_execution_config
        robot_names = [r["name"] for r in robots]
        mission_execution_config = {m: robot_names[:] for m in missions}
        
        # Carregar grafo
        G_mapa = SegmentUtils.load_graph_json(GRAPH_PATH)
        
        # Obter referência de conversão
        ref = MissionManager.read_parametros_conversao_lat_lon(PARAMETERS_FILE_PATH)
        lat_ref, lon_ref = ref["lat_ref"], ref["lon_ref"]
        
        # Limpar conexões antigas se existir mission_manager global
        global mission_manager
        if mission_manager:
            print(f"🧹 Limpando conexões antigas do canal de missão...")
            try:
                mission_manager.close()
                print("✅ Conexões antigas do canal de missão fechadas")
            except Exception as e:
                print(f"⚠️ Erro ao fechar conexões antigas: {e}")
            mission_manager = None
        
        # Aguardar um pouco para garantir que as portas estão livres
        time.sleep(1)
        
        # Criar novo MissionManager
        mission_manager = MissionManager(robots=robots)
        
        # Conectar aos robôs
        print(f"\n🔌 CONECTANDO AOS ROBÔS...")
        connection_success = mission_manager.connect_all()
        
        if not connection_success:
            print("❌ Falha ao conectar em qualquer robô")
            return jsonify({
                "success": False,
                "message": "Falha ao conectar em qualquer robô"
            }), 500
        
        print(f"✅ Conectado a {len(mission_manager.connected)} robô(s): {list(mission_manager.connected)}")
        
        # Limpar estados antigos dos robôs antes de obter novas posições
        print(f"🧹 Limpando estados antigos dos robôs...")
        if hasattr(mission_manager, 'last_states'):
            mission_manager.last_states.clear()
            print("✅ Estados antigos limpos")
        
        # Usar posições já capturadas pelo canal de monitoramento contínuo
        print(f"📍 Obtendo posições dos robôs do canal de monitoramento...")
        
        # Obter posições dos robôs (do canal de monitoramento ou simuladas)
        robot_positions_xy = {}
        
        # Verificar se temos posições do canal de monitoramento
        positions_from_monitoring = False
        if all_robot_positions:
            print(f"✅ Usando posições do canal de monitoramento contínuo")
            for robot in robots:
                robot_name = robot['name']
                
                # Verificar se temos posição deste robô no canal de monitoramento
                if robot_name in all_robot_positions:
                    position_data = all_robot_positions[robot_name]
                    lat = position_data.get('latitude')
                    lon = position_data.get('longitude')
                    
                    if lat is not None and lon is not None:
                        tx, ty = MissionManager.gps_to_xy(lat, lon, lat_ref, lon_ref)
                        robot_positions_xy[robot_name] = (tx, ty)
                        print(f"   ✅ {robot_name}: (x={tx:.2f}, y={ty:.2f}) - GPS: ({lat:.6f}, {lon:.6f}) [do canal]")
                        positions_from_monitoring = True
                        continue
                
                # Se não temos posição no canal, tentar obter diretamente
                print(f"   ⚠️ {robot_name}: Posição não disponível no canal de monitoramento")
        
        # Se não conseguiu obter todas as posições do canal, tentar método tradicional
        if not positions_from_monitoring or len(robot_positions_xy) < len(robots):
            print(f"🔄 Fallback: Tentando obter posições diretamente dos robôs...")
            
            # Forçar stream GPS para todos os robôs conectados
            for robot_name in mission_manager.connected:
                try:
                    mission_manager.force_gps_stream(rate_hz=5.0, robot=robot_name)
                    print(f"   📡 Stream GPS ativado para {robot_name}")
                except Exception as e:
                    print(f"   ⚠️ Falha ao ativar stream GPS para {robot_name}: {e}")
            
            # Fazer polling inicial para limpar buffer antigo
            print(f"🔄 Fazendo polling inicial para limpar dados antigos...")
            for _ in range(5):
                mission_manager.poll_once(per_robot_reads=10)
                time.sleep(0.2)
            
            time.sleep(2)  # Aguardar stream GPS estabilizar
            
            # Tentar obter posições com múltiplas tentativas se necessário
            print(f"📡 Obtendo posições atuais dos robôs...")
            all_states = None
            for attempt in range(3):  # Até 3 tentativas
                print(f"   Tentativa {attempt + 1}/3...")
                all_states = mission_manager.wait_for_position(timeout=10.0, require_all=True)
                if all_states:
                    break
                else:
                    print(f"   ⚠️ Tentativa {attempt + 1} falhou, fazendo polling adicional...")
                    # Polling adicional entre tentativas
                    for _ in range(10):
                        mission_manager.poll_once(per_robot_reads=20)
                        time.sleep(0.1)
                    time.sleep(1)
            
            if all_states is None:
                print("❌ Falha ao obter posições de todos os robôs")
                print("   Tentando obter posições individuais...")
                
                # Tentar obter posições individualmente
                for robot in robots:
                    robot_name = robot['name']
                    if robot_name in mission_manager.connected and robot_name not in robot_positions_xy:
                        # Tentar obter posição individual
                        individual_state = mission_manager.wait_for_position(timeout=5.0, require_all=False, robot=robot_name)
                        if individual_state:
                            if isinstance(individual_state, tuple):
                                # Se retornou tupla (nome, estado)
                                _, state = individual_state
                                lat, lon = get_latlon({robot_name: state}, robot_name)
                            else:
                                # Se retornou estado diretamente
                                lat, lon = state.get('lat'), state.get('lon')
                            
                            if lat is not None and lon is not None:
                                tx, ty = MissionManager.gps_to_xy(lat, lon, lat_ref, lon_ref)
                                robot_positions_xy[robot_name] = (tx, ty)
                                print(f"   ✅ {robot_name}: (x={tx:.2f}, y={ty:.2f}) - GPS: ({lat:.6f}, {lon:.6f})")
                                continue
                        
                        # Se ainda não conseguiu, erro
                        if robot_name not in robot_positions_xy:
                            print(f"   ⚠️ {robot_name}: Não foi possível obter posição GPS real")
                            return jsonify({
                                "success": False,
                                "message": f"Não foi possível obter posição GPS do robô {robot_name}. Verifique se o robô está conectado e enviando telemetria GPS."
                            }), 500
            else:
                print("✅ Posições obtidas com sucesso!")
                # Mostrar estados dos robôs (como no mission_bridgetoap.py)
                for robot in robots:
                    robot_name = robot['name']
                    if robot_name in all_states:
                        print(f"   • {robot_name}: {all_states[robot_name]}")
                    else:
                        print(f"   ⚠️ {robot_name}: Estado não disponível")
                
                # Processar posições obtidas pelo método tradicional
                for robot in robots:
                    robot_name = robot['name']
                    if robot_name not in robot_positions_xy:  # Só processar se ainda não temos a posição
                        lat, lon = get_latlon(all_states, robot_name)
                        
                        if lat is None or lon is None:
                            print(f"   ⚠️ {robot_name}: Posição GPS não disponível")
                            return jsonify({
                                "success": False,
                                "message": f"Posição GPS não disponível para o robô {robot_name}"
                            }), 500
                        
                        tx, ty = MissionManager.gps_to_xy(lat, lon, lat_ref, lon_ref)
                        robot_positions_xy[robot_name] = (tx, ty)
                        print(f"   • {robot_name}: (x={tx:.2f}, y={ty:.2f}) - GPS: ({lat:.6f}, {lon:.6f})")
        
        # Se conseguimos posições do canal de monitoramento, mostrar sucesso
        if positions_from_monitoring:
            print("✅ Posições obtidas com sucesso!")
        
        # Verificar se conseguimos posições para todos os robôs necessários
        if len(robot_positions_xy) < len(robots):
            missing_robots = [robot['name'] for robot in robots if robot['name'] not in robot_positions_xy]
            print(f"❌ Não foi possível obter posições para todos os robôs necessários")
            print(f"   Robôs sem posição: {missing_robots}")
            return jsonify({
                "success": False,
                "message": f"Não foi possível obter posições para os robôs: {', '.join(missing_robots)}"
            }), 500
        
        # Executar planejador
        print(f"\n🧠 EXECUTANDO PLANEJADOR HETEROGÊNEO...")
        saida = run_planner(
            GRAPH_PATH,
            OBSERVATION_POINTS_JSON_PATH,
            PARAMETERS_FILE_PATH,
            missions,
            mission_execution_config,
            robot_positions_xy=robot_positions_xy,
            do_plots=False
        )
        print("✅ Planejamento concluído!")
        
        # Processar rotas
        pontos_vistoria = []
        for robo, missions in saida["missoes_completas"].items():
            for m in missions:
                for t in m["tasks"]:
                    pontos_vistoria.append(t["point"])
        
        # Obter pontos de passagem
        missoes_completas = list(retorna_pontos_passagem(
            G_mapa,
            saida["rotas_otimas_por_robo"],
            pontos_vistoria
        ))
        
        # Montar missões por robô
        missoes_por_robo = montar_missoes_por_robo(
            missoes_completas=missoes_completas,
            G_mapa=G_mapa,
            observation_points_json_path=OBSERVATION_POINTS_JSON_PATH,
            lat_ref=lat_ref, lon_ref=lon_ref,
            duplicate_first=True,
            hold_vistoria=5.0,
            hold_passagem=0.0,
            MissionManager=MissionManager
        )
        
        # Extrair waypoints
        waypoints = extract_waypoints_from_missions(missoes_por_robo)
        mission_waypoints = waypoints
        
        # Enviar missões para os robôs
        print(f"\n🚀 ENVIANDO MISSÕES PARA OS ROBÔS...")
        
        # PARAR COMPLETAMENTE o monitoramento contínuo para evitar conflito de comunicação MAVLink
        print(f"🛑 Parando canal de monitoramento COMPLETAMENTE para evitar conflitos MAVLink...")
        stop_continuous_robot_monitoring()  # Parar completamente em vez de apenas pausar
        time.sleep(3)  # Aguardar desconexão completa
        
        try:
            upload_results = {}
            verified_waypoints = {}  # Waypoints verificados do ArduPilot
            
            for i, robot in enumerate(robots):
                robot_name = robot['name']
                
                if robot_name not in missoes_por_robo:
                    print(f"   ⚠️ Nenhuma missão planejada para {robot_name}")
                    continue
                
                # Aguardar entre robôs para evitar conflitos de comunicação
                if i > 0:
                    print(f"   ⏱️ Aguardando 3 segundos entre envios de robôs...")
                    time.sleep(3)
                
                # Verificar se o robô está conectado no canal de missão
                if robot_name in mission_manager.connected:
                    # 1. Enviar missão para o ArduPilot
                    print(f"📤 Enviando missão para {robot_name}...")
                    
                    # Fazer polling mais agressivo para limpar buffer completamente
                    print(f"   🔄 Limpando buffer de comunicação para {robot_name}...")
                    for _ in range(10):  # Mais ciclos de limpeza
                        mission_manager.poll_once(per_robot_reads=20)
                        time.sleep(0.05)  # Polling mais rápido
                    
                    # Aguardar estabilização da comunicação
                    time.sleep(1)
                    
                    # Enviar missão (como no código original - simples e direto)
                    print(f"   📤 Enviando missão para {robot_name}...")
                    success = mission_manager.upload_mission(missoes_por_robo[robot_name], robot=robot_name)
                    
                    if success:
                        print(f"   ✅ Missão enviada com sucesso para {robot_name}")
                    else:
                        print(f"   ❌ Falha ao enviar missão para {robot_name}")
                        # Tentar uma segunda vez se falhar
                        print(f"   🔄 Tentando novamente...")
                        time.sleep(1)
                        success = mission_manager.upload_mission(missoes_por_robo[robot_name], robot=robot_name)
                        if success:
                            print(f"   ✅ Missão enviada com sucesso para {robot_name} na segunda tentativa")
                        else:
                            print(f"   ❌ Falha definitiva ao enviar missão para {robot_name}")
                    
                    upload_results[robot_name] = success
                    
                    if success:
                        print(f"   ✅ Missão enviada com sucesso para {robot_name}")
                        
                        # Usar waypoints originais diretamente (como no código original)
                        print(f"   ✅ Missão enviada - usando waypoints originais para {robot_name}")
                        verified_waypoints[robot_name] = waypoints[robot_name]
                            
                    else:
                        print(f"   ❌ Falha ao enviar missão para {robot_name}")
                        verified_waypoints[robot_name] = waypoints[robot_name]  # Usar waypoints originais
                        
                else:
                    print(f"   ❌ {robot_name} não está conectado no canal de missão")
                    upload_results[robot_name] = False
                    verified_waypoints[robot_name] = waypoints[robot_name]  # Usar waypoints originais
            
            # Usar waypoints verificados quando disponíveis, senão usar originais
            if verified_waypoints:
                print(f"\n🔄 Usando waypoints processados (mix de sincronizados + originais)")
                waypoints = verified_waypoints
                mission_waypoints = waypoints
            else:
                print(f"\n🔄 Usando waypoints originais do planejador")
                mission_waypoints = waypoints
                
        finally:
            # Aguardar mais tempo para garantir que todas as missões foram processadas
            print(f"\n🔄 Finalizando envio de missões...")
            time.sleep(5)  # Aguardar mais tempo para garantir que tudo foi processado
        
        # Atualizar status da missão
        mission_status = {
            "active": True,
            "completed": False,
            "robots": {r['name']: {"waypoints": len(waypoints.get(r['name'], []))} for r in robots},
            "start_time": time.time(),
            "total_waypoints": sum(len(wps) for wps in waypoints.values())
        }
        
        # Emitir waypoints via WebSocket
        try:
            # Preparar waypoints em formato compatível com frontend
            waypoints_for_frontend = []
            for robot, robot_waypoints in waypoints.items():
                for wp in robot_waypoints:
                    waypoints_for_frontend.append({
                        **wp,
                        'robot': robot  # Adicionar identificação do robô
                    })
            
            print(f"📡 [WEBSOCKET] Enviando waypoints da missão:")
            print(f"   🎯 Total de waypoints: {len(waypoints_for_frontend)}")
            print(f"   🤖 Robôs: {list(waypoints.keys())}")
            for robot, robot_waypoints in waypoints.items():
                print(f"   📍 {robot}: {len(robot_waypoints)} waypoints")
            
            socketio.emit('mission_waypoints_update', {
                "waypoints": waypoints_for_frontend,  # Array de waypoints
                "waypoints_by_robot": waypoints,  # Formato original por robô
                "mission_status": mission_status,
                "timestamp": time.time()
            })
        except Exception as e:
            print(f"⚠️ Erro ao emitir waypoints via WebSocket: {e}")
        
        # Reiniciar monitoramento contínuo APÓS todo o processo estar completo
        print(f"\n🔄 Reiniciando canal de monitoramento contínuo...")
        start_continuous_robot_monitoring()
        
        # Iniciar monitoramento de posições usando o canal de missão
        print(f"\n🔄 Iniciando monitoramento de posições dos robôs...")
        stop_monitoring = False
        position_monitoring_thread = threading.Thread(target=monitor_robot_positions)
        position_monitoring_thread.daemon = True
        position_monitoring_thread.start()
        
        print(f"\n✅ MISSÃO CONFIGURADA COM SUCESSO!")
        print(f"🤖 Robôs conectados: {list(mission_manager.connected) if mission_manager else 'Nenhum'}")
        for robot_name in mission_manager.connected:
            try:
                mission_manager.force_gps_stream(rate_hz=5.0, robot=robot_name)
                print(f"   📡 Stream GPS ativado para monitoramento de {robot_name}")
            except Exception as e:
                print(f"   ⚠️ Erro ao ativar stream GPS para {robot_name}: {e}")
        print(f"   • {len(robots)} robôs configurados")
        print(f"   • {len(missions)} missões planejadas")
        print(f"   • {sum(len(wps) for wps in waypoints.values())} waypoints totais gerados")
        
        # Preparar waypoints em formato compatível com frontend
        waypoints_for_frontend = []
        for robot, robot_waypoints in waypoints.items():
            for wp in robot_waypoints:
                waypoints_for_frontend.append({
                    **wp,
                    'robot': robot  # Adicionar identificação do robô
                })

        return jsonify({
            "success": True,
            "message": "Missão configurada e enviada com sucesso",
            "data": {
                "robots_used": len(robots),
                "robots": [r['name'] for r in robots],
                "total_missions": len(missions),
                "missions": missions,
                "waypoints": waypoints_for_frontend,  # Array de waypoints para frontend
                "waypoints_by_robot": waypoints,  # Formato original por robô
                "upload_results": upload_results,
                "mission_status": mission_status,
                "total_waypoints": sum(len(wps) for wps in waypoints.values()),
                "equipments_count": len(missions)
            }
        }), 200
        
    except Exception as e:
        print(f"\n❌ Erro durante execução: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "success": False,
            "message": f"Erro durante planejamento: {str(e)}"
        }), 500

@app.route('/mission-status', methods=['GET'])
def get_mission_status():
    """Endpoint para obter status atual das missões"""
    return jsonify({
        "success": True,
        "data": {
            "mission_status": mission_status,
            "robot_positions": robot_positions,
            "mission_waypoints": mission_waypoints,
            "active_robots": len(robot_positions),
            "timestamp": time.time()
        }
    })

@app.route('/stop-mission', methods=['POST'])
def stop_mission():
    """Endpoint para parar missão atual"""
    global stop_monitoring, mission_status, mission_manager
    
    stop_monitoring = True
    mission_status["active"] = False
    mission_status["completed"] = False
    
    # Fechar conexões do mission_manager se existir
    if mission_manager:
        print(f"🧹 Fechando conexões do canal de missão...")
        try:
            mission_manager.close()
            print("✅ Conexões do canal de missão fechadas com sucesso")
        except Exception as e:
            print(f"⚠️ Erro ao fechar conexões do canal de missão: {e}")
        mission_manager = None
    
    # Emitir via WebSocket
    try:
        print(f"📡 [WEBSOCKET] Enviando parada de missão:")
        print(f"   🛑 Missão interrompida pelo usuário")
        print(f"   ⏰ Timestamp: {time.time()}")
        
        socketio.emit('mission_stopped', {
            "message": "Missão interrompida",
            "timestamp": time.time()
        })
    except Exception as e:
        print(f"⚠️ Erro ao emitir parada de missão via WebSocket: {e}")
    
    return jsonify({
        "success": True,
        "message": "Missão interrompida com sucesso"
    })

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Cliente conectado ao WebSocket"""
    print(f"🔌 Cliente conectado: {request.sid}")
    # Enviar dados atuais para o cliente recém-conectado
    emit('initial_data', {
        "mission_status": mission_status,
        "robot_positions": robot_positions,
        "mission_waypoints": mission_waypoints,
        "timestamp": time.time()
    })
    
    # Enviar dados do canal de monitoramento contínuo
    emit('robot_monitoring_initial', {
        "monitoring_active": robot_monitoring_thread is not None and robot_monitoring_thread.is_alive(),
        "all_robot_positions": all_robot_positions,
        "robot_connection_status": robot_connection_status,
        "total_known_robots": len(ROVER_ID_MAPPING),
        "timestamp": time.time()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Cliente desconectado do WebSocket"""
    print(f"🔌 Cliente desconectado: {request.sid}")

@socketio.on('request_robot_monitoring_status')
def handle_robot_monitoring_status_request():
    """Cliente solicitou status do monitoramento de robôs"""
    emit('robot_monitoring_status', {
        "monitoring_active": robot_monitoring_thread is not None and robot_monitoring_thread.is_alive(),
        "total_known_robots": len(ROVER_ID_MAPPING),
        "connected_robots": sum(1 for status in robot_connection_status.values() if status['connected']),
        "active_robots": len(all_robot_positions),
        "robot_positions": all_robot_positions,
        "connection_status": robot_connection_status,
        "timestamp": time.time()
    })

@socketio.on('start_robot_monitoring')
def handle_start_robot_monitoring():
    """Cliente solicitou iniciar monitoramento contínuo"""
    try:
        success = start_continuous_robot_monitoring()
        emit('robot_monitoring_started', {
            "success": success,
            "message": "Canal de monitoramento contínuo iniciado" if success else "Falha ao iniciar monitoramento",
            "timestamp": time.time()
        })
    except Exception as e:
        emit('robot_monitoring_error', {
            "error": str(e),
            "timestamp": time.time()
        })

@socketio.on('stop_robot_monitoring')
def handle_stop_robot_monitoring():
    """Cliente solicitou parar monitoramento contínuo"""
    try:
        stop_continuous_robot_monitoring()
        emit('robot_monitoring_stopped', {
            "success": True,
            "message": "Canal de monitoramento contínuo parado",
            "timestamp": time.time()
        })
    except Exception as e:
        emit('robot_monitoring_error', {
            "error": str(e),
            "timestamp": time.time()
        })

if __name__ == '__main__':
    print(f"\n🚀 SERVIDOR DE MISSÕES BASEADO EM mission_bridgetoap.py")
    print(f"="*80)
    print(f"📁 Diretório de trabalho: {os.getcwd()}")
    print(f"📋 Arquivos de configuração:")
    print(f"   • Grafo: {GRAPH_PATH}")
    print(f"   • Pontos de observação: {OBSERVATION_POINTS_JSON_PATH}")
    print(f"   • Parâmetros: {PARAMETERS_FILE_PATH}")
    print(f"\n🔗 Endpoints disponíveis:")
    print(f"   GET  /health - Verificar status do servidor")
    print(f"   GET  /rover-mapping - Ver mapeamento de rovers")
    print(f"   POST /validate-rovers - Validar e mapear rovers do Django")
    print(f"   POST /execute-mission - Executar planejamento")
    print(f"   GET  /mission-status - Obter status das missões")
    print(f"   POST /stop-mission - Parar missão atual")
    print(f"   GET  /robot-monitoring/status - Status do canal de monitoramento")
    print(f"   POST /robot-monitoring/start - Iniciar monitoramento contínuo")
    print(f"   POST /robot-monitoring/stop - Parar monitoramento contínuo")
    print(f"   GET  /robot-positions - Posições de todos os robôs")
    print(f"   POST /optimize-frequencies - Aplicar configurações otimizadas (1 segundo)")
    print(f"   POST /update-frequency - Ajustar frequência personalizada")
    print(f"\n🌐 Servidor rodando em: http://localhost:5001")
    print(f"🔄 CORS habilitado para requisições da interface web")
    print(f"🔌 WebSocket habilitado para comunicação em tempo real")
    print(f"🤖 Canal isolado de monitoramento de robôs disponível")
    print(f"="*80)
    
    # Iniciar monitoramento contínuo de robôs automaticamente
    print(f"\n🚀 Iniciando canal isolado de monitoramento de robôs...")
    try:
        start_continuous_robot_monitoring()
        print(f"✅ Canal de monitoramento contínuo iniciado automaticamente")
    except Exception as e:
        print(f"⚠️ Erro ao iniciar monitoramento contínuo: {e}")
        print(f"💡 O monitoramento pode ser iniciado manualmente via WebSocket ou endpoint")
    
    # Rodar servidor com SocketIO
    try:
        socketio.run(
            app,
            host='127.0.0.1',  # Usar localhost em vez de 0.0.0.0 para evitar problemas de permissão
            port=5001,
            debug=True,
            allow_unsafe_werkzeug=True,
            use_reloader=False  # Desabilitar reloader automático para evitar conflitos de porta
        )
    except OSError as e:
        if "permissões de acesso" in str(e) or "permission denied" in str(e).lower() or "address already in use" in str(e).lower():
            print(f"\n❌ Erro na porta 5001. Tentando porta alternativa...")
            try:
                socketio.run(
                    app,
                    host='127.0.0.1',
                    port=5002,  # Porta alternativa
                    debug=True,
                    allow_unsafe_werkzeug=True,
                    use_reloader=False  # Desabilitar reloader automático para evitar conflitos de porta
                )
            except OSError as e2:
                print(f"❌ Erro também na porta 5002: {e2}")
                print("💡 Sugestões:")
                print("   1. Feche outros programas que possam estar usando as portas 5001/5002")
                print("   2. Execute como administrador")
                print("   3. Altere a porta no serverConfig.js para uma porta livre (ex: 8000, 3000)")
        else:
            print(f"❌ Erro inesperado: {e}")