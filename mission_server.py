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

# Importar a função principal do mission_bridgetoap
from mission_bridgetoap import preparar_e_enviar_missoes, get_latlon

app = Flask(__name__)
CORS(app)
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    logger=False,
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25,
    async_mode='threading',
    transports=['websocket', 'polling']
)

# Configurações globais
graph_path = "./jsons/graph9d_new.json"
observation_points_json_path = "./jsons/obpc_7.json"
PARAMETERS_FILE_PATH = "./planilhas/obstaculos_processado6.xlsx"
deltax_m = -2
deltay_m = -12.0

# Estados globais
mission_manager = None
mission_active = False
monitoring_thread = None
stop_monitoring = False

# Mapeamento de rovers do banco Django para identificadores do mission_server
ROVER_ID_MAPPING = {
    # Mapear identifiers do banco para IDs simples do mission_server
    "Rover-Beta": "R1",
    "Rover-Charlie": "R2", 
    "Rover-Delta": "R3"
}

# Mapeamento reverso para logs e debug
REVERSE_ROVER_MAPPING = {v: k for k, v in ROVER_ID_MAPPING.items()}

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

def monitor_robot_positions():
    """
    Thread para monitorar posições dos robôs de 2 em 2 segundos
    """
    global stop_monitoring, mission_manager, mission_active
    
    print("🔄 Iniciando monitoramento de posições dos robôs...")
    
    while not stop_monitoring and mission_manager:
        try:
            if mission_manager.is_connected:
                # Obter estados de todos os robôs
                all_states = mission_manager.new_wait_for_position(timeout=2.0, require_all=False)
                
                if all_states:
                    current_time = time.time()
                    positions_data = []
                    
                    # Processar cada robô conectado
                    for robot_name in mission_manager.connected:
                        lat, lon = get_latlon(all_states, robot_name)
                        
                        if lat is not None and lon is not None:
                            # Obter identifier original do mapeamento reverso
                            original_identifier = REVERSE_ROVER_MAPPING.get(robot_name, robot_name)
                            
                            position_data = {
                                "robot_id": robot_name,
                                "original_identifier": original_identifier,
                                "latitude": round(float(lat), 8),
                                "longitude": round(float(lon), 8),
                                "timestamp": current_time,
                                "status": all_states.get(robot_name, {}).get("status", "unknown")
                            }
                            
                            positions_data.append(position_data)
                            print(f"📍 {robot_name} ({original_identifier}): lat={lat:.6f}, lon={lon:.6f}")
                    
                    # Enviar via WebSocket
                    if positions_data:
                        try:
                            socketio.emit('robot_positions_update', {
                                "positions": positions_data,
                                "timestamp": current_time,
                                "mission_active": mission_active
                            })
                        except Exception as e:
                            print(f"⚠️ Erro ao enviar posições via WebSocket: {e}")
                
            time.sleep(2)  # Aguardar 2 segundos
            
        except Exception as e:
            print(f"⚠️ Erro no monitoramento de posições: {e}")
            time.sleep(5)
    
    print("🔄 Monitoramento de posições finalizado")

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar se o servidor está funcionando"""
    return jsonify({
        "status": "healthy",
        "message": "Servidor de missões ativo",
        "version": "1.0.0",
        "mission_active": mission_active,
        "connected_robots": len(mission_manager.connected) if mission_manager and mission_manager.is_connected else 0
    })

@app.route('/rover-mapping', methods=['GET'])
def get_rover_mapping():
    """Endpoint para verificar o mapeamento de rovers"""
    return jsonify({
        "success": True,
        "message": "Mapeamento de rovers do banco Django para mission_server",
        "data": {
            "mapping": ROVER_ID_MAPPING,
            "reverse_mapping": REVERSE_ROVER_MAPPING,
            "total_rovers": len(ROVER_ID_MAPPING)
        }
    })

@app.route('/execute-mission', methods=['POST'])
def execute_mission():
    """
    Endpoint principal para executar planejamento de missão usando preparar_e_enviar_missoes
    """
    global mission_manager, mission_active, monitoring_thread, stop_monitoring
    
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
        
        # Processar dados do frontend (formato do MissionPlanningView.js)
        if 'rovers' in data and isinstance(data['rovers'], list):
            mission_name = data.get('name', f'Missão {datetime.now().strftime("%Y-%m-%d %H:%M")}')
            rovers_list = data.get('rovers', [])
            equipments_list = data.get('equipments', [])
            
            print(f"📝 Nome: {mission_name}")
            print(f"🤖 Rovers: {[r.get('identifier', r.get('name', 'N/A')) for r in rovers_list]}")
            print(f"⚡ Equipamentos: {[e.get('name', e.get('id', 'N/A')) for e in equipments_list]}")
            
            # Converter rovers usando mapeamento
            robots = []
            for i, rover_data in enumerate(rovers_list):
                rover_config = create_rover_config_from_frontend_data(rover_data, i)
                robots.append(rover_config)
            
            # Converter equipamentos para lista de missões
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
            
        else:
            return jsonify({
                "success": False,
                "message": "Formato de dados inválido. Esperado: {rovers: [...], equipments: [...]}"
            }), 400
        
        # Validações
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
        
        # Executar a função principal do mission_bridgetoap.py
        print(f"\n🚀 EXECUTANDO preparar_e_enviar_missoes...")
        
        resultado = preparar_e_enviar_missoes(
            robots=robots,
            missions=missions,
            graph_path=graph_path,
            observation_points_json_path=observation_points_json_path,
            file_path_parametros=PARAMETERS_FILE_PATH,
            deltax_m=deltax_m,
            deltay_m=deltay_m,
            gps_rate_hz=5.0,
            wait_timeout_s=25.0,
            wait_each_timeout_s=5.0,
            duplicate_first=True,
            hold_vistoria_s=2.0,
            hold_passagem_s=0.0,
            tol_ct_m=0.10,
            preserve_loop_closure=True,
            renumber_ids=True,
            do_plots=False
        )
        
        print("✅ Missão executada com sucesso!")
        
        # Armazenar mission_manager para monitoramento
        mission_manager = resultado["mission_manager"]
        mission_active = True
        
        # Iniciar monitoramento de posições
        stop_monitoring = False
        monitoring_thread = threading.Thread(target=monitor_robot_positions)
        monitoring_thread.daemon = True
        monitoring_thread.start()
        
        # Extrair waypoints das missões otimizadas para o frontend
        waypoints_for_frontend = []
        total_waypoints = 0
        
        for robot_name, mission_points in resultado["missoes_otimizadas"].items():
            for point in mission_points:
                waypoint = {
                    'id': point['id'],
                    'lat': point['lat'],
                    'lon': point['lon'],
                    'hold': point.get('hold', 0.0),
                    'robot': robot_name,
                    'original_identifier': REVERSE_ROVER_MAPPING.get(robot_name, robot_name)
                }
                waypoints_for_frontend.append(waypoint)
                total_waypoints += 1
        
        # Emitir via WebSocket
        try:
            print(f"📡 [WEBSOCKET] Enviando waypoints da missão:")
            print(f"   🎯 Total de waypoints: {total_waypoints}")
            print(f"   🤖 Robôs: {list(resultado['missoes_otimizadas'].keys())}")
            
            socketio.emit('mission_waypoints_update', {
                "waypoints": waypoints_for_frontend,
                "mission_active": True,
                "robots": [r['name'] for r in robots],
                "timestamp": time.time()
            })
        except Exception as e:
            print(f"⚠️ Erro ao emitir waypoints via WebSocket: {e}")
        
        print(f"\n✅ MISSÃO CONFIGURADA COM SUCESSO!")
        print(f"   • {len(robots)} robôs configurados")
        print(f"   • {len(missions)} missões planejadas")
        print(f"   • {total_waypoints} waypoints totais gerados")
        
        return jsonify({
            "success": True,
            "message": "Missão executada e enviada com sucesso",
            "data": {
                "robots_used": len(robots),
                "robots": [r['name'] for r in robots],
                "total_missions": len(missions),
                "missions": missions,
                "waypoints": waypoints_for_frontend,
                "total_waypoints": total_waypoints,
                "mission_active": True
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

@app.route('/stop-mission', methods=['POST'])
def stop_mission():
    """Endpoint para parar missão atual"""
    global stop_monitoring, mission_active, mission_manager
    
    stop_monitoring = True
    mission_active = False
    
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
        print(f"📡 [WEBSOCKET] Enviando parada de missão")
        socketio.emit('mission_stopped', {
            "message": "Missão interrompida",
            "mission_active": False,
            "timestamp": time.time()
        })
    except Exception as e:
        print(f"⚠️ Erro ao emitir parada de missão via WebSocket: {e}")
    
    return jsonify({
        "success": True,
        "message": "Missão interrompida com sucesso"
    })

@app.route('/mission-status', methods=['GET'])
def get_mission_status():
    """Endpoint para obter status atual da missão"""
    return jsonify({
        "success": True,
        "data": {
            "mission_active": mission_active,
            "connected_robots": len(mission_manager.connected) if mission_manager and mission_manager.is_connected else 0,
            "robots": list(mission_manager.connected) if mission_manager and mission_manager.is_connected else [],
            "timestamp": time.time()
        }
    })

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Cliente conectado ao WebSocket"""
    print(f"🔌 Cliente conectado: {request.sid}")
    # Enviar dados atuais para o cliente recém-conectado
    emit('initial_data', {
        "mission_active": mission_active,
        "connected_robots": len(mission_manager.connected) if mission_manager and mission_manager.is_connected else 0,
        "timestamp": time.time()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Cliente desconectado do WebSocket"""
    print(f"🔌 Cliente desconectado: {request.sid}")

if __name__ == '__main__':
    print(f"\n🚀 SERVIDOR DE MISSÕES SIMPLES")
    print(f"="*80)
    print(f"📁 Diretório de trabalho: {os.getcwd()}")
    print(f"📋 Arquivos de configuração:")
    print(f"   • Grafo: {graph_path}")
    print(f"   • Pontos de observação: {observation_points_json_path}")
    print(f"   • Parâmetros: {PARAMETERS_FILE_PATH}")
    print(f"\n🔗 Endpoints disponíveis:")
    print(f"   GET  /health - Verificar status do servidor")
    print(f"   GET  /rover-mapping - Ver mapeamento de rovers")
    print(f"   POST /execute-mission - Executar missão usando preparar_e_enviar_missoes")
    print(f"   GET  /mission-status - Obter status da missão")
    print(f"   POST /stop-mission - Parar missão atual")
    print(f"\n🌐 Servidor rodando em: http://localhost:5001")
    print(f"🔄 CORS habilitado para requisições da interface web")
    print(f"🔌 WebSocket habilitado para monitoramento de posições (2 em 2 segundos)")
    print(f"🤖 Usando mapeamento: {ROVER_ID_MAPPING}")
    print(f"="*80)
    
    # Rodar servidor com SocketIO
    try:
        socketio.run(
            app,
            host='127.0.0.1',
            port=5001,
            debug=True,
            allow_unsafe_werkzeug=True,
            use_reloader=False
        )
    except OSError as e:
        print(f"❌ Erro na porta 5001: {e}")
        print("💡 Verifique se a porta está livre ou execute como administrador")
