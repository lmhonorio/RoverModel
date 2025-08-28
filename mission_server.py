#!/usr/bin/env python3
"""
Servidor Flask leve para receber dados de missões da interface web
e executar o planejamento de rotas dos rovers.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import sys
import os
from collections import defaultdict

# Importar módulos do sistema de planejamento
try:
    from old.multigraphplanner import MultiGraphPlanner
    from segmentutils import SegmentUtils
    from plotutils import PlotUtils
    from tspOptimization import FixedTaskPlanner
    from aabbutils import AABBUtils
except ImportError as e:
    print(f"❌ Erro ao importar módulos: {e}")
    print("Certifique-se de que todos os módulos estão disponíveis no diretório RoverModel/")
    sys.exit(1)

app = Flask(__name__)
CORS(app)  # Permitir requisições da interface web

# Configurações padrão
DEFAULT_CONFIG = {
    "graph_file": "./jsons/graph9_new.json",
    "observation_points_file": "./jsons/obp_6.json",
    "missions": ['b_busip4', 'ef_reator1', 'ls_pr4'],
    "mission_execution_time": 15,  # segundos
    "robot_positions": {
        "R1": {"x": -165.9766, "y": -77.6645},
        "R2": {"x": 87.9766, "y": 30.6645}
    }
}

def execute_mission_planning(rovers_data, equipments_data, substation_id):
    """
    Executa o planejamento de missão com os dados recebidos da interface.
    
    Args:
        rovers_data: Lista de rovers selecionados
        equipments_data: Lista de equipamentos selecionados  
        substation_id: ID da subestação selecionada
        
    Returns:
        dict: Resultado do planejamento com rotas otimizadas
    """
    
    print(f"\n🚀 Iniciando planejamento de missão...")
    print(f"📍 Subestação: {substation_id}")
    print(f"🤖 Rovers: {[r.get('name', r.get('identifier', 'Unknown')) for r in rovers_data]}")
    print(f"⚡ Equipamentos: {len(equipments_data)} selecionados")
    
    try:
        # Carregar configurações padrão
        file_path = DEFAULT_CONFIG["graph_file"]
        observation_points_json_path = DEFAULT_CONFIG["observation_points_file"]
        
        # Verificar se arquivos existem
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo de grafo não encontrado: {file_path}")
        if not os.path.exists(observation_points_json_path):
            raise FileNotFoundError(f"Arquivo de pontos de observação não encontrado: {observation_points_json_path}")
        
        # Carregar pontos de observação por obstáculo
        observacao_por_obstaculo = SegmentUtils.load_observation_points_from_json(observation_points_json_path)
        
        # Usar missões padrão por enquanto (pode ser adaptado para usar equipments_data)
        missions = DEFAULT_CONFIG["missions"]
        mission_positions = MultiGraphPlanner.gerar_mission_positions_from_json(observacao_por_obstaculo, missions)
        
        # Transformar pontos de observação em missões individuais
        point_mission_positions = {}
        for mission, points in mission_positions.items():
            for point in points:
                point_mission_positions[point] = point
        
        print(f"\n🔍 {len(point_mission_positions)} missões individuais identificadas")
        
        # Carrega o grafo do ambiente
        G_mapa = SegmentUtils.load_graph_json(file_path)
        
        # Mapear rovers recebidos para posições padrão
        robots_positions = {}
        robot_coords = list(DEFAULT_CONFIG["robot_positions"].items())
        
        for i, rover in enumerate(rovers_data[:len(robot_coords)]):
            robot_id = rover.get('identifier', f'R{i+1}')
            coord_key = list(DEFAULT_CONFIG["robot_positions"].keys())[i]
            coord_data = DEFAULT_CONFIG["robot_positions"][coord_key]
            
            # Encontrar nó mais próximo no grafo
            label_pos, _, _ = MultiGraphPlanner.find_nearest_node(G_mapa, coord_data["x"], coord_data["y"])
            robots_positions[robot_id] = label_pos
            
            print(f"🤖 {rover.get('name', robot_id)} -> Posição: {label_pos}")
        
        # Construir grafo reduzido para inspeção
        initial_positions = list(robots_positions.values())
        Greduced_map = MultiGraphPlanner.build_inspection_graph(initial_positions, point_mission_positions, G_mapa)
        
        # Configuração de execução por missão (todos os rovers podem executar todas as missões)
        mission_execution = {}
        robot_ids = list(robots_positions.keys())
        
        for point in point_mission_positions:
            mission_execution[point] = robot_ids
        
        # Executar clusterização balanceada
        pontos_por_robo = FixedTaskPlanner.clusterizar_pontos_balanceado(
            Greduced_map, point_mission_positions, robots_positions, mission_execution
        )
        
        print(f"\n📌 Clusterização balanceada:")
        for robo, pontos in pontos_por_robo.items():
            print(f"  {robo}: {len(pontos)} pontos -> {pontos}")
        
        # Calcular rotas ótimas por robô
        rotas_otimas_por_robo = {}
        
        for robo, pontos in pontos_por_robo.items():
            ponto_inicial = robots_positions[robo]
            rota_otima = FixedTaskPlanner.tsp_nearest_neighbor(Greduced_map, ponto_inicial, pontos)
            rotas_otimas_por_robo[robo] = rota_otima
            
            print(f"\n🚗 Rota ótima para {robo}:")
            print(f"   {' -> '.join(rota_otima)}")
        
        # Preparar resultado
        result = {
            "success": True,
            "message": "Planejamento executado com sucesso",
            "data": {
                "substation": substation_id,
                "robots_used": len(robots_positions),
                "total_missions": len(point_mission_positions),
                "routes": {}
            }
        }
        
        # Adicionar detalhes das rotas
        for robot_id, route in rotas_otimas_por_robo.items():
            # Encontrar rover correspondente
            rover_info = next((r for r in rovers_data if r.get('identifier') == robot_id), {})
            
            result["data"]["routes"][robot_id] = {
                "rover_name": rover_info.get('name', robot_id),
                "rover_id": robot_id,
                "route": route,
                "total_points": len(route) - 1,  # -1 porque inclui posição inicial
                "estimated_time": (len(route) - 1) * DEFAULT_CONFIG["mission_execution_time"]
            }
        
        return result
        
    except Exception as e:
        print(f"❌ Erro durante planejamento: {str(e)}")
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

@app.route('/config', methods=['GET'])
def get_config():
    """Endpoint para obter configurações atuais do servidor."""
    return jsonify({
        "success": True,
        "data": DEFAULT_CONFIG
    })

if __name__ == '__main__':
    print(f"\n🚀 Iniciando servidor de planejamento de missões...")
    print(f"📁 Diretório de trabalho: {os.getcwd()}")
    print(f"🔗 Endpoints disponíveis:")
    print(f"   GET  /health - Verificar status do servidor")
    print(f"   POST /execute-mission - Executar planejamento")
    print(f"   GET  /config - Obter configurações")
    print(f"\n🌐 Servidor rodando em: http://localhost:5000")
    print(f"🔄 CORS habilitado para requisições da interface web")
    print(f"="*60)
    
    # Rodar servidor
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
