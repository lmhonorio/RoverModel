#!/usr/bin/env python3
"""
Servidor Flask leve para receber dados de missões da interface web
e executar o planejamento de rotas dos rovers.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import sys
import os
from collections import defaultdict
import threading
import time

# Importar módulos do sistema de planejamento
try:
    from PlanejadorHeterogeneo import mrta, MISSION_PRESETS
    print("✅ PlanejadorHeterogeneo importado com sucesso")
except ImportError as e:
    print(f"❌ Erro ao importar PlanejadorHeterogeneo: {e}")
    print("Certifique-se de que PlanejadorHeterogeneo.py está disponível no diretório RoverModel/")
    sys.exit(1)

app = Flask(__name__)
CORS(app)  # Permitir requisições da interface web
# Configurações otimizadas para WebSocket - reduzir overhead
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    logger=False,  # Reduzir logs para melhor performance
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25
)

# Armazenamento em memória para posições dos robôs e waypoints
robot_positions = {}
mission_waypoints = {}
mission_status = {}

# Throttling para emissões WebSocket - evitar spam de atualizações
last_position_emission = {}
POSITION_EMISSION_INTERVAL = 3.0  # 3 segundos entre emissões por robô

# Configurações agora são gerenciadas pelo PlanejadorHeterogeneo.py
# - Arquivos JSON: GRAPH_JSON, OBS_POINTS_JSON, etc.
# - Missões: MISSION_PRESETS com diferentes presets
# - Posições dos robôs: ALL_ROBOT_COORDS obtidas dinamicamente via GPS
# - Configurações de execução: RUN_MOVNS, RUN_BASELINE_CLUSTER, SEND_MISSIONS, etc.

def execute_mission_planning(rovers_data, equipments_data, substation_id):
    """
    Executa o planejamento de missão EXATAMENTE como o PlanejadorHeterogeneo.py
    
    Args:
        rovers_data: Lista de rovers selecionados (será forçado para R1, R2, R3)
        equipments_data: Lista de equipamentos selecionados (será forçado para example_22)
        substation_id: ID da subestação selecionada
        
    Returns:
        dict: Resultado do planejamento com rotas otimizadas
    """
    
    print(f"\n🚀 Executando EXATAMENTE como PlanejadorHeterogeneo.py")
    print(f"📍 Subestação recebida: {substation_id}")
    print(f"🤖 Rovers recebidos: {[r.get('name', r.get('identifier', 'Unknown')) for r in rovers_data]}")
    print(f"⚡ Equipamentos recebidos: {len(equipments_data)} selecionados")
    
    try:
        # FORÇAR DADOS PARA SEREM IGUAIS AO HARDCODED DO PlanejadorHeterogeneo.py
        print(f"\n🔄 FORÇANDO COMPATIBILIDADE TOTAL:")
        
        # FORÇAR MISSÕES: Sempre usar example_22
        forced_missions = MISSION_PRESETS["default"]
        print(f"⚡ MISSÕES FORÇADAS: example_22 ({len(forced_missions)} missões)")
        
        # FORÇAR ROVERS: Sempre usar R1, R2, R3  
        forced_rovers = ["R1", "R2", "R3"]
        print(f"🤖 ROVERS FORÇADOS: {forced_rovers}")
        
        print(f"="*60)
        print(f"🎯 CHAMANDO FUNÇÃO mrta() DO PlanejadorHeterogeneo.py")
        print(f"="*60)
        
        # EXECUTAR EXATAMENTE COMO O PlanejadorHeterogeneo.py
        # Chama a função mrta diretamente com os parâmetros forçados
        mrta(forced_missions, forced_rovers)
        
        print(f"="*60)
        print(f"✅ EXECUÇÃO CONCLUÍDA - Verifique os logs acima")
        print(f"="*60)
        
        # Preparar resultado básico (já que mrta() não retorna dados estruturados)
        result = {
            "success": True,
            "message": "Planejamento executado com sucesso usando PlanejadorHeterogeneo.py",
            "data": {
                "substation": substation_id,
                "robots_used": len(forced_rovers),
                "total_missions": len(forced_missions),
                "missions_executed": forced_missions[:10],  # Primeiras 10 para visualização
                "rovers_used": forced_rovers,
                "note": "Verifique o console do servidor para logs completos do planejamento"
            }
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Erro durante execução do PlanejadorHeterogeneo.py: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Erro durante planejamento: {str(e)}",
            "data": None
        }

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar se o servidor está funcionando."""
    return jsonify({
        "status": "healthy",
        "message": "Servidor de planejamento de missões ativo",
        "version": "1.0.0"
    })

@app.route('/execute-mission', methods=['POST'])
def execute_mission():
    """
    Endpoint principal para executar planejamento de missão.
    
    Espera receber:
    {
        "rovers": [...],
        "equipments": [...], 
        "substation": "string"
    }
    """
    
    try:
        # Obter dados do POST
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "message": "Nenhum dado recebido no POST"
            }), 400
        
        # Extrair dados
        rovers_data = data.get('rovers', [])
        equipments_data = data.get('equipments', [])
        substation_id = data.get('substation', '')
        
        # Log dos dados recebidos
        print(f"\n" + "="*60)
        print(f"📨 DADOS RECEBIDOS VIA POST:")
        print(f"="*60)
        print(f"🏭 Subestação: {substation_id}")
        print(f"🤖 Rovers ({len(rovers_data)}):")
        for i, rover in enumerate(rovers_data):
            print(f"   {i+1}. {rover.get('name', 'N/A')} (ID: {rover.get('identifier', 'N/A')})")
        
        print(f"⚡ Equipamentos ({len(equipments_data)}):")
        for i, equipment in enumerate(equipments_data):
            print(f"   {i+1}. {equipment.get('name', 'N/A')} - {equipment.get('type', 'N/A')} (ID: {equipment.get('equipmentId', 'N/A')})")
        
        print(f"="*60)
        
        # Validações básicas
        if not rovers_data:
            return jsonify({
                "success": False,
                "message": "Nenhum rover selecionado"
            }), 400
            
        if not equipments_data:
            return jsonify({
                "success": False,
                "message": "Nenhum equipamento selecionado"
            }), 400
            
        if not substation_id:
            return jsonify({
                "success": False,
                "message": "Nenhuma subestação selecionada"
            }), 400
        
        # Executar planejamento
        result = execute_mission_planning(rovers_data, equipments_data, substation_id)
        
        if result["success"]:
            print(f"\n✅ Planejamento concluído com sucesso!")
            return jsonify(result), 200
        else:
            print(f"\n❌ Falha no planejamento: {result['message']}")
            return jsonify(result), 500
            
    except Exception as e:
        error_msg = f"Erro interno do servidor: {str(e)}"
        print(f"\n💥 {error_msg}")
        return jsonify({
            "success": False,
            "message": error_msg
        }), 500

# Endpoint /config removido - configurações agora são gerenciadas pelo PlanejadorHeterogeneo.py

@app.route('/send_gps', methods=['POST'])
def receive_robot_position():
    """
    Endpoint para receber posições dos robôs do MonitoraRobo.py
    
    Espera receber:
    {
        "robo": "1",
        "latitude": -3.123456,
        "longitude": -41.765432
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "Nenhum dado recebido"}), 400
        
        robot_id = str(data.get('robo'))
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        if not all([robot_id, latitude is not None, longitude is not None]):
            return jsonify({
                "success": False, 
                "message": "Dados incompletos: robo, latitude e longitude são obrigatórios"
            }), 400
        
        # Atualizar posição do robô (sempre atualiza internamente)
        current_time = time.time()
        robot_positions[robot_id] = {
            "latitude": latitude,
            "longitude": longitude,
            "timestamp": current_time
        }
        
        # Throttling para emissão WebSocket - apenas a cada 3 segundos
        last_emission = last_position_emission.get(robot_id, 0)
        should_emit = (current_time - last_emission) >= POSITION_EMISSION_INTERVAL
        
        if should_emit:
            # Emitir atualização via WebSocket para todos os clientes conectados
            socketio.emit('robot_position_update', {
                "robot_id": robot_id,
                "latitude": latitude,
                "longitude": longitude,
                "timestamp": current_time
            })
            
            last_position_emission[robot_id] = current_time
            print(f"📍 Posição atualizada (emitida) - Robô {robot_id}: [{latitude:.6f}, {longitude:.6f}]")
        else:
            # Log silencioso - posição atualizada mas não emitida
            time_since_last = current_time - last_emission
            # print(f"⏱️ Posição do robô {robot_id} throttled ({time_since_last:.1f}s desde última emissão)")
            pass
        
        return jsonify({
            "success": True,
            "message": f"Posição do robô {robot_id} atualizada com sucesso"
        })
        
    except Exception as e:
        print(f"❌ Erro ao processar posição do robô: {e}")
        return jsonify({
            "success": False,
            "message": f"Erro interno: {str(e)}"
        }), 500

@app.route('/waypoints', methods=['POST'])
def receive_mission_waypoints():
    """
    Endpoint para receber waypoints das missões do PlanejadorHeterogeneo.py
    
    Espera receber:
    {
        "robo": "1",
        "trajetoria": [
            {"latitude": -3.123456, "longitude": -41.765432},
            {"latitude": -3.123457, "longitude": -41.765433}
        ]
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "Nenhum dado recebido"}), 400
        
        robot_id = str(data.get('robo'))
        trajetoria = data.get('trajetoria', [])
        
        if not robot_id:
            return jsonify({
                "success": False, 
                "message": "ID do robô é obrigatório"
            }), 400
        
        if not trajetoria or not isinstance(trajetoria, list):
            return jsonify({
                "success": False, 
                "message": "Trajetória deve ser uma lista não vazia de waypoints"
            }), 400
        
        # Validar estrutura dos waypoints
        for i, waypoint in enumerate(trajetoria):
            if not isinstance(waypoint, dict) or 'latitude' not in waypoint or 'longitude' not in waypoint:
                return jsonify({
                    "success": False,
                    "message": f"Waypoint {i+1} deve conter 'latitude' e 'longitude'"
                }), 400
        
        # Armazenar waypoints da missão
        mission_waypoints[robot_id] = {
            "waypoints": trajetoria,
            "timestamp": time.time(),
            "total_points": len(trajetoria)
        }
        
        # Emitir waypoints via WebSocket para todos os clientes conectados
        socketio.emit('mission_waypoints_update', {
            "robot_id": robot_id,
            "waypoints": trajetoria,
            "total_points": len(trajetoria),
            "timestamp": time.time()
        })
        
        print(f"🗺️ Waypoints recebidos - Robô {robot_id}: {len(trajetoria)} pontos")
        
        return jsonify({
            "success": True,
            "message": f"Waypoints do robô {robot_id} recebidos com sucesso",
            "data": {
                "robot_id": robot_id,
                "total_waypoints": len(trajetoria)
            }
        })
        
    except Exception as e:
        print(f"❌ Erro ao processar waypoints: {e}")
        return jsonify({
            "success": False,
            "message": f"Erro interno: {str(e)}"
        }), 500

@app.route('/mission-status', methods=['GET'])
def get_mission_status():
    """Endpoint para obter status atual das missões, posições dos robôs e waypoints."""
    return jsonify({
        "success": True,
        "data": {
            "robot_positions": robot_positions,
            "mission_waypoints": mission_waypoints,
            "mission_status": mission_status,
            "active_robots": len(robot_positions),
            "active_missions": len(mission_waypoints)
        }
    })

# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Cliente conectado ao WebSocket."""
    print(f"🔌 Cliente conectado: {request.sid}")
    # Enviar dados atuais para o cliente recém-conectado
    emit('initial_data', {
        "robot_positions": robot_positions,
        "mission_waypoints": mission_waypoints,
        "mission_status": mission_status
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Cliente desconectado do WebSocket."""
    print(f"🔌 Cliente desconectado: {request.sid}")

if __name__ == '__main__':
    print(f"\n🚀 Iniciando servidor de planejamento de missões...")
    print(f"📁 Diretório de trabalho: {os.getcwd()}")
    print(f"🔗 Endpoints disponíveis:")
    print(f"   GET  /health - Verificar status do servidor")
    print(f"   POST /execute-mission - Executar planejamento")
    print(f"   POST /send_gps - Receber posições dos robôs")
    print(f"   POST /waypoints - Receber waypoints das missões")
    print(f"   GET  /mission-status - Obter status das missões")
    print(f"\n🌐 Servidor rodando em: http://localhost:5000")
    print(f"🔄 CORS habilitado para requisições da interface web")
    print(f"🔌 WebSocket habilitado para comunicação em tempo real")
    print(f"="*60)
    
    # Rodar servidor com SocketIO
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=True,
        allow_unsafe_werkzeug=True
    )
