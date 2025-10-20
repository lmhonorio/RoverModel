#!/usr/bin/env python3
"""
Simulador do Dashboard de Objetos de Inspeção
Simula comandos que seriam enviados pelo frontend para o mission_server.py
"""

import requests
import json
import time
import threading
from typing import Dict, List, Optional

class DashboardSimulator:
    def __init__(self, server_url: str = "http://127.0.0.1:5001"):
        self.server_url = server_url
        self.session = requests.Session()
        
    def test_connection(self) -> bool:
        """Testa se o servidor está rodando"""
        try:
            response = self.session.get(f"{self.server_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Servidor conectado: {data['message']}")
                print(f"   📊 Robôs conectados: {data['total_connected_robots']}")
                return True
            else:
                print(f"❌ Servidor não respondeu: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Erro ao conectar: {e}")
            return False
    
    def execute_inspection_mission(self, robots: List[str], equipment: List[str], 
                                 inspection_points: List[Dict]) -> Dict:
        """
        Simula execução de missão de inspeção
        
        Args:
            robots: Lista de IDs dos robôs
            equipment: Lista de equipamentos de inspeção
            inspection_points: Pontos de inspeção com coordenadas
        """
        mission_data = {
            "robots": robots,
            "equipment": equipment,
            "inspection_points": inspection_points,
            "mission_type": "inspection",
            "priority": "high",
            "timestamp": time.time()
        }
        
        try:
            print(f"🚀 Executando missão de inspeção...")
            print(f"   🤖 Robôs: {robots}")
            print(f"   🔧 Equipamentos: {equipment}")
            print(f"   📍 Pontos de inspeção: {len(inspection_points)}")
            
            response = self.session.post(
                f"{self.server_url}/execute-mission",
                json=mission_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Missão iniciada com sucesso!")
                print(f"   📋 ID da missão: {result.get('mission_id', 'N/A')}")
                print(f"   🎯 Status: {result.get('status', 'N/A')}")
                return result
            else:
                print(f"❌ Erro ao executar missão: {response.status_code}")
                print(f"   📝 Resposta: {response.text}")
                return {"success": False, "error": response.text}
                
        except Exception as e:
            print(f"❌ Erro na execução: {e}")
            return {"success": False, "error": str(e)}
    
    def get_mission_status(self) -> Dict:
        """Obtém status atual da missão"""
        try:
            response = self.session.get(f"{self.server_url}/mission-status")
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def stop_mission(self) -> Dict:
        """Para a missão atual"""
        try:
            response = self.session.post(f"{self.server_url}/stop-mission")
            if response.status_code == 200:
                result = response.json()
                print(f"🛑 Missão parada: {result.get('message', 'N/A')}")
                return result
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def connect_robots(self) -> Dict:
        """Conecta aos robôs para monitoramento"""
        try:
            response = self.session.post(f"{self.server_url}/connect-robots")
            if response.status_code == 200:
                result = response.json()
                print(f"🔌 Robôs conectados: {result.get('data', {}).get('connected_robots', 0)}")
                return result
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def disconnect_robots(self) -> Dict:
        """Desconecta dos robôs"""
        try:
            response = self.session.post(f"{self.server_url}/disconnect-robots")
            if response.status_code == 200:
                result = response.json()
                print(f"🔌 Robôs desconectados: {result.get('message', 'N/A')}")
                return result
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run_inspection_simulation(self):
        """Executa simulação completa de inspeção"""
        print("🎬 INICIANDO SIMULAÇÃO DE INSPEÇÃO")
        print("=" * 50)
        
        # 1. Testar conexão
        if not self.test_connection():
            return
        
        # 2. Conectar robôs
        print("\n🔌 Conectando robôs...")
        self.connect_robots()
        time.sleep(2)
        
        # 3. Executar missão de inspeção
        print("\n🚀 Executando missão de inspeção...")
        robots = ["rover_argo_1", "rover_argo_2"]
        equipment = ["camera", "sensor_temperature", "sensor_pressure"]
        inspection_points = [
            {"id": "point_1", "lat": -3.123199, "lon": -41.764537, "priority": "high"},
            {"id": "point_2", "lat": -3.123200, "lon": -41.764538, "priority": "medium"},
            {"id": "point_3", "lat": -3.123201, "lon": -41.764539, "priority": "low"}
        ]
        
        mission_result = self.execute_inspection_mission(robots, equipment, inspection_points)
        
        if mission_result.get("success", False):
            # 4. Monitorar progresso
            print("\n📊 Monitorando progresso da missão...")
            for i in range(10):  # Monitorar por 20 segundos
                status = self.get_mission_status()
                if status.get("success", False):
                    print(f"   Status: {status.get('data', {}).get('status', 'N/A')}")
                time.sleep(2)
            
            # 5. Parar missão
            print("\n🛑 Parando missão...")
            self.stop_mission()
        
        # 6. Desconectar robôs
        print("\n🔌 Desconectando robôs...")
        self.disconnect_robots()
        
        print("\n✅ Simulação concluída!")

def main():
    """Função principal para executar a simulação"""
    print("🎯 SIMULADOR DO DASHBOARD DE INSPEÇÃO")
    print("=" * 50)
    
    # Criar simulador
    simulator = DashboardSimulator()
    
    # Executar simulação
    simulator.run_inspection_simulation()

if __name__ == "__main__":
    main()