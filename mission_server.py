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
    from PlanejadorHeterogeneo import mrta, MISSION_PRESETS
    print("✅ PlanejadorHeterogeneo importado com sucesso")
except ImportError as e:
    print(f"❌ Erro ao importar PlanejadorHeterogeneo: {e}")
    print("Certifique-se de que PlanejadorHeterogeneo.py está disponível no diretório RoverModel/")
    sys.exit(1)

app = Flask(__name__)
CORS(app)  # Permitir requisições da interface web

# Configurações padrão
DEFAULT_CONFIG = {
    "graph_file": "./jsons/graph8_new_funcionando.json",  # Usando arquivo do PlanejadorHeterogeneo.py
    "observation_points_file": "./jsons/obp_6_funcionando.json",  # Usando arquivo do PlanejadorHeterogeneo.py
    # FORÇANDO example_22 (independente dos equipamentos selecionados na interface)
    "missions": [
        'ef_reator1', 'ef_reator2', 'ef_reator3', 'ef_reator4', 'ef_reator5', 'ef_reator6', 'ef_reator7', 'ef_reator8', 'ef_reator9', 'ef_reator10',
        'ef_pr2', 'ef_pr3', 'ef_pr4', 'ef_pr5', 'ef_pr6', 'ef_pr7', 'ef_pr8', 'ef_pr9', 'ef_pr10', 'ef_pr11', 'ef_pr12', 'ef_pr13',
        'ls_tpc1', 'ls_tpc2', 'ls_tpc3', 'ls_tpc4', 'ls_tpc5', 'ls_tpc6',
        'r_pr1', 'r_reator1', 'r_reator2', 'r_pr2'
    ],
    "mission_execution_time": 15,  # segundos
    "robot_positions": {
        "R1": {"x": -165.9766, "y": -77.6645},
        "R2": {"x": 87.9766, "y": 30.6645},
        "R3": {"x": 87.9766, "y": 30.6645}  # Adicionando R3 para compatibilidade
    }
}

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
        forced_missions = MISSION_PRESETS["example_22"]
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
