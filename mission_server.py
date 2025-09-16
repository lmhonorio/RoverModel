"""
Servidor de missões modularizado
Recebe robôs e equipamentos selecionados via API REST
Retorna waypoints e monitora posições dos robôs via WebSocket
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import time

# Importar módulos do servidor
from server_modules import (
    ROVER_ID_MAPPING,
    REVERSE_ROVER_MAPPING,
    create_position_monitoring_manager,
    MonitoringService,
    MissionService
)

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

# Inicializar serviços
monitoring_service = MonitoringService(socketio)
mission_service = MissionService(socketio, monitoring_service)

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar se o servidor está funcionando"""
    # Contar robôs conectados (missão ou monitoramento)
    mission_robots = len(mission_service.mission_manager.connected) if mission_service.mission_manager and mission_service.mission_manager.is_connected else 0
    monitoring_robots = len(monitoring_service.position_monitoring_manager.connected) if monitoring_service.position_monitoring_manager and monitoring_service.position_monitoring_manager.is_connected else 0
    
    return jsonify({
        "status": "healthy",
        "message": "Servidor de missões ativo",
        "version": "1.0.0",
        "mission_active": mission_service.mission_active,
        "connected_robots_mission": mission_robots,
        "connected_robots_monitoring": monitoring_robots,
        "total_connected_robots": max(mission_robots, monitoring_robots),
        "monitoring_active": monitoring_service.position_monitoring_manager is not None
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
    """Endpoint principal para executar planejamento de missão"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "message": "Nenhum dado recebido no POST"
            }), 400
        
        # Executar missão usando o serviço
        result = mission_service.execute_mission(data)
        return jsonify(result), result.get("status_code", 200)
        
    except Exception as e:
        print(f"\n❌ Erro no endpoint execute-mission: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Erro no servidor: {str(e)}"
        }), 500

@app.route('/stop-mission', methods=['POST'])
def stop_mission():
    """Endpoint para parar missão atual"""
    result = mission_service.stop_mission()
    return jsonify(result)

@app.route('/mission-status', methods=['GET'])
def get_mission_status():
    """Endpoint para obter status atual da missão"""
    result = mission_service.get_mission_status()
    return jsonify(result)

@app.route('/connect-robots', methods=['POST'])
def connect_robots():
    """Endpoint para conectar aos robôs para monitoramento contínuo"""
    try:
        if monitoring_service.position_monitoring_manager is None:
            monitoring_service.position_monitoring_manager = create_position_monitoring_manager()
        else:
            # Tentar reconectar
            connected = monitoring_service.position_monitoring_manager.connect_all()
            if connected:
                print(f"✅ Reconectado a {len(monitoring_service.position_monitoring_manager.connected)} robô(s)")
            else:
                print("⚠️ Nenhum robô conectado")
        
        return jsonify({
            "success": True,
            "message": "Tentativa de conexão com robôs realizada",
            "data": {
                "connected_robots": len(monitoring_service.position_monitoring_manager.connected) if monitoring_service.position_monitoring_manager else 0,
                "robots": list(monitoring_service.position_monitoring_manager.connected) if monitoring_service.position_monitoring_manager else [],
                "timestamp": time.time()
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erro ao conectar robôs: {str(e)}"
        }), 500

@app.route('/disconnect-robots', methods=['POST'])
def disconnect_robots():
    """Endpoint para desconectar dos robôs"""
    try:
        if monitoring_service.position_monitoring_manager:
            monitoring_service.position_monitoring_manager.close()
            monitoring_service.position_monitoring_manager = None
            print("🔌 Robôs desconectados do monitoramento")
        
        return jsonify({
            "success": True,
            "message": "Robôs desconectados com sucesso",
            "timestamp": time.time()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erro ao desconectar robôs: {str(e)}"
        }), 500

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Cliente conectado ao WebSocket"""
    print(f"🔌 Cliente conectado: {request.sid}")
    # Enviar dados atuais para o cliente recém-conectado
    emit('initial_data', {
        "mission_active": mission_service.mission_active,
        "connected_robots": len(mission_service.mission_manager.connected) if mission_service.mission_manager and mission_service.mission_manager.is_connected else 0,
        "timestamp": time.time()
    })
    
    # Enviar dados do monitoramento contínuo se disponível
    if monitoring_service.position_monitoring_manager:
        emit('robot_monitoring_initial', {
            "monitoring_active": True,
            "connected_robots": len(monitoring_service.position_monitoring_manager.connected) if monitoring_service.position_monitoring_manager else 0,
            "timestamp": time.time()
        })
    
    # Debug: verificar se o cliente está recebendo eventos
    print(f"🔍 Cliente {request.sid} conectado - eventos WebSocket disponíveis")

@socketio.on('disconnect')
def handle_disconnect():
    """Cliente desconectado do WebSocket"""
    print(f"🔌 Cliente desconectado: {request.sid}")

@socketio.on('request_robot_positions')
def handle_robot_positions_request():
    """Cliente solicitou posições atuais dos robôs"""
    try:
        if monitoring_service.position_monitoring_manager and monitoring_service.position_monitoring_manager.is_connected:
            # Obter posições atuais dos robôs
            positions_data = []
            for robot_name in monitoring_service.position_monitoring_manager.connected:
                try:
                    robot_state = monitoring_service.position_monitoring_manager.new_wait_for_position(robot=robot_name, timeout=1.0, require_all=False)
                    if robot_state:
                        from mission_bridgetoap import get_latlon
                        lat, lon = get_latlon({robot_name: robot_state}, robot_name)
                        if lat is not None and lon is not None:
                            position_data = {
                                "robot_id": robot_name,
                                "original_identifier": REVERSE_ROVER_MAPPING.get(robot_name, robot_name),
                                "latitude": round(float(lat), 8),
                                "longitude": round(float(lon), 8),
                                "timestamp": time.time(),
                                "status": robot_state.get("status", "unknown") if isinstance(robot_state, dict) else "unknown",
                                "mission_active": mission_service.mission_active
                            }
                            positions_data.append(position_data)
                except Exception as e:
                    print(f"⚠️ Erro ao obter posição do {robot_name}: {e}")
            
            # Enviar cada posição individualmente no formato que o frontend espera
            for position_data in positions_data:
                emit('robot_position_continuous', position_data)
        else:
            emit('robot_positions_update', {
                "positions": [],
                "timestamp": time.time(),
                "mission_active": mission_service.mission_active,
                "monitoring_mode": "none",
                "message": "Nenhum robô conectado para monitoramento"
            })
    except Exception as e:
        print(f"⚠️ Erro ao processar solicitação de posições: {e}")
        emit('error', {"message": f"Erro ao obter posições: {str(e)}"})

@socketio.on('request_mission_status')
def handle_mission_status_request():
    """Cliente solicitou status da missão"""
    try:
        status = mission_service.get_mission_status()
        emit('mission_status_update', status["data"])
    except Exception as e:
        print(f"⚠️ Erro ao processar solicitação de status: {e}")
        emit('error', {"message": f"Erro ao obter status: {str(e)}"})

@socketio.on('request_robot_monitoring_status')
def handle_robot_monitoring_status_request():
    """Cliente solicitou status do monitoramento de robôs"""
    try:
        if monitoring_service.position_monitoring_manager:
            emit('robot_monitoring_status', {
                "monitoring_active": True,
                "total_known_robots": len(ROVER_ID_MAPPING),
                "connected_robots": len(monitoring_service.position_monitoring_manager.connected) if monitoring_service.position_monitoring_manager else 0,
                "active_robots": len(monitoring_service.position_monitoring_manager.connected) if monitoring_service.position_monitoring_manager else 0,
                "robot_positions": {},  # Será preenchido pelo monitoramento contínuo
                "connection_status": {},
                "timestamp": time.time()
            })
        else:
            emit('robot_monitoring_status', {
                "monitoring_active": False,
                "total_known_robots": len(ROVER_ID_MAPPING),
                "connected_robots": 0,
                "active_robots": 0,
                "robot_positions": {},
                "connection_status": {},
                "timestamp": time.time()
            })
    except Exception as e:
        print(f"⚠️ Erro ao processar solicitação de status de monitoramento: {e}")
        emit('error', {"message": f"Erro ao obter status de monitoramento: {str(e)}"})

@socketio.on('request_mission_waypoints')
def handle_mission_waypoints_request():
    """Cliente solicitou waypoints da missão atual"""
    try:
        print(f"🔍 Cliente solicitou waypoints da missão")
        # Emitir dados atuais da missão se disponível
        if mission_service.mission_active and mission_service.mission_manager:
            emit('mission_waypoints_update', {
                "waypoints": [],  # Será preenchido pelo MissionService
                "mission_active": mission_service.mission_active,
                "robots": list(mission_service.mission_manager.connected) if mission_service.mission_manager else [],
                "timestamp": time.time(),
                "message": "Solicite via endpoint /execute-mission para obter waypoints"
            })
        else:
            emit('mission_waypoints_update', {
                "waypoints": [],
                "mission_active": False,
                "robots": [],
                "timestamp": time.time(),
                "message": "Nenhuma missão ativa"
            })
    except Exception as e:
        print(f"⚠️ Erro ao processar solicitação de waypoints: {e}")
        emit('error', {"message": f"Erro ao obter waypoints: {str(e)}"})

@socketio.on('mission_waypoints_update')
def handle_mission_waypoints_update(data):
    """Debug: Handler para receber waypoints (para debug)"""
    print(f"🔍 DEBUG: Cliente enviou waypoints via WebSocket: {type(data)}")
    if isinstance(data, dict):
        print(f"   • Chaves: {list(data.keys())}")
        if 'waypoints' in data:
            print(f"   • Waypoints: {len(data['waypoints'])}")
    else:
        print(f"   • Dados: {data}")

@socketio.on('start_robot_monitoring')
def handle_start_robot_monitoring():
    """Cliente solicitou iniciar monitoramento contínuo"""
    try:
        if monitoring_service.position_monitoring_manager is None:
            monitoring_service.position_monitoring_manager = create_position_monitoring_manager()
            if monitoring_service.position_monitoring_manager:
                monitoring_service.start_monitoring(monitoring_service.position_monitoring_manager)
        
        emit('robot_monitoring_started', {
            "success": True,
            "message": "Canal de monitoramento contínuo iniciado",
            "timestamp": time.time()
        })
    except Exception as e:
        print(f"⚠️ Erro ao iniciar monitoramento: {e}")
        emit('robot_monitoring_error', {
            "error": str(e),
            "timestamp": time.time()
        })

@socketio.on('stop_robot_monitoring')
def handle_stop_robot_monitoring():
    """Cliente solicitou parar monitoramento contínuo"""
    try:
        monitoring_service.stop_monitoring_service()
        if monitoring_service.position_monitoring_manager:
            monitoring_service.position_monitoring_manager.close()
            monitoring_service.position_monitoring_manager = None
        
        emit('robot_monitoring_stopped', {
            "success": True,
            "message": "Canal de monitoramento contínuo parado",
            "timestamp": time.time()
        })
    except Exception as e:
        print(f"⚠️ Erro ao parar monitoramento: {e}")
        emit('robot_monitoring_error', {
            "error": str(e),
            "timestamp": time.time()
        })

if __name__ == '__main__':
    print(f"\n🚀 SERVIDOR DE MISSÕES MODULARIZADO")
    print(f"="*80)
    print(f"📁 Diretório de trabalho: {os.getcwd()}")
    print(f"📋 Arquivos de configuração:")
    print(f"   • Grafo: {mission_service.graph_path}")
    print(f"   • Pontos de observação: {mission_service.observation_points_json_path}")
    print(f"   • Parâmetros: {mission_service.parameters_file_path}")
    print(f"\n🔗 Endpoints disponíveis:")
    print(f"   GET  /health - Verificar status do servidor")
    print(f"   GET  /rover-mapping - Ver mapeamento de rovers")
    print(f"   POST /execute-mission - Executar missão usando preparar_e_enviar_missoes")
    print(f"   GET  /mission-status - Obter status da missão")
    print(f"   POST /stop-mission - Parar missão atual")
    print(f"   POST /connect-robots - Conectar aos robôs para monitoramento")
    print(f"   POST /disconnect-robots - Desconectar dos robôs")
    print(f"\n🌐 Servidor rodando em: http://localhost:5001")
    print(f"🔄 CORS habilitado para requisições da interface web")
    print(f"🔌 WebSocket habilitado para monitoramento de posições (2 em 2 segundos)")
    print(f"🤖 Usando mapeamento: {ROVER_ID_MAPPING}")
    print(f"\n📡 Eventos WebSocket disponíveis:")
    print(f"   • mission_waypoints_update - Waypoints da missão")
    print(f"   • robot_position_continuous - Posições dos robôs")
    print(f"   • mission_status_update - Status da missão")
    print(f"   • robot_monitoring_status - Status do monitoramento")
    print(f"="*80)
    
    # Inicializar monitoramento contínuo de posições
    print(f"\n🔗 INICIALIZANDO MONITORAMENTO CONTÍNUO...")
    try:
        position_monitoring_manager = create_position_monitoring_manager()
        if position_monitoring_manager:
            monitoring_service.start_monitoring(position_monitoring_manager)
            print(f"✅ Monitoramento contínuo iniciado com sucesso")
        else:
            print(f"⚠️ Falha ao criar position_monitoring_manager")
    except Exception as e:
        print(f"⚠️ Erro ao inicializar monitoramento contínuo: {e}")
        print(f"💡 O monitoramento pode ser iniciado manualmente via endpoint /connect-robots")
    
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
    except KeyboardInterrupt:
        print(f"\n🛑 Parando servidor...")
        monitoring_service.stop_monitoring_service()
        if monitoring_service.position_monitoring_manager:
            monitoring_service.position_monitoring_manager.close()
        print(f"✅ Servidor parado com sucesso")
