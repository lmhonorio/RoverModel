#!/usr/bin/env python3
"""
Exemplo Simples de Comandos do Dashboard
Testa endpoints específicos do mission_server.py
"""

import requests
import json
import time

def test_dashboard_commands():
    """Testa comandos básicos do dashboard"""
    server_url = "http://127.0.0.1:5001"
    
    print("🎯 TESTANDO COMANDOS DO DASHBOARD")
    print("=" * 50)
    
    # 1. Verificar se servidor está rodando
    print("1️⃣ Testando conexão...")
    try:
        response = requests.get(f"{server_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Servidor ativo: {data['message']}")
            print(f"   📊 Robôs conectados: {data['total_connected_robots']}")
        else:
            print(f"❌ Servidor não respondeu: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Erro ao conectar: {e}")
        return
    
    # 2. Conectar robôs
    print("\n2️⃣ Conectando robôs...")
    try:
        response = requests.post(f"{server_url}/connect-robots")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Robôs conectados: {data['data']['connected_robots']}")
        else:
            print(f"❌ Erro ao conectar robôs: {response.status_code}")
    except Exception as e:
        print(f"❌ Erro: {e}")
    
    # 3. Executar missão de inspeção
    print("\n3️⃣ Executando missão de inspeção...")
    mission_data = {
        "robots": ["rover_argo_1", "rover_argo_2"],
        "equipment": ["camera", "sensor_temperature"],
        "inspection_points": [
            {
                "id": "equipment_1",
                "lat": -3.123199,
                "lon": -41.764537,
                "priority": "high",
                "equipment_type": "reactor"
            },
            {
                "id": "equipment_2", 
                "lat": -3.123200,
                "lon": -41.764538,
                "priority": "medium",
                "equipment_type": "pump"
            }
        ],
        "mission_type": "inspection",
        "priority": "high"
    }
    
    try:
        response = requests.post(
            f"{server_url}/execute-mission",
            json=mission_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Missão iniciada!")
            print(f"   📋 Status: {result.get('status', 'N/A')}")
            print(f"   🎯 Sucesso: {result.get('success', False)}")
        else:
            print(f"❌ Erro ao executar missão: {response.status_code}")
            print(f"   📝 Resposta: {response.text}")
    except Exception as e:
        print(f"❌ Erro: {e}")
    
    # 4. Verificar status da missão
    print("\n4️⃣ Verificando status da missão...")
    try:
        response = requests.get(f"{server_url}/mission-status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status obtido: {data.get('success', False)}")
            if data.get('data'):
                print(f"   📊 Dados: {data['data']}")
        else:
            print(f"❌ Erro ao obter status: {response.status_code}")
    except Exception as e:
        print(f"❌ Erro: {e}")
    
    # 5. Parar missão
    print("\n5️⃣ Parando missão...")
    try:
        response = requests.post(f"{server_url}/stop-mission")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Missão parada: {data.get('message', 'N/A')}")
        else:
            print(f"❌ Erro ao parar missão: {response.status_code}")
    except Exception as e:
        print(f"❌ Erro: {e}")
    
    # 6. Desconectar robôs
    print("\n6️⃣ Desconectando robôs...")
    try:
        response = requests.post(f"{server_url}/disconnect-robots")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Robôs desconectados: {data.get('message', 'N/A')}")
        else:
            print(f"❌ Erro ao desconectar: {response.status_code}")
    except Exception as e:
        print(f"❌ Erro: {e}")
    
    print("\n✅ Teste concluído!")

def test_specific_command(command: str):
    """Testa um comando específico"""
    server_url = "http://127.0.0.1:5001"
    
    if command == "health":
        response = requests.get(f"{server_url}/health")
        print(f"Health Check: {response.json()}")
    
    elif command == "connect":
        response = requests.post(f"{server_url}/connect-robots")
        print(f"Connect Robots: {response.json()}")
    
    elif command == "disconnect":
        response = requests.post(f"{server_url}/disconnect-robots")
        print(f"Disconnect Robots: {response.json()}")
    
    elif command == "status":
        response = requests.get(f"{server_url}/mission-status")
        print(f"Mission Status: {response.json()}")
    
    elif command == "stop":
        response = requests.post(f"{server_url}/stop-mission")
        print(f"Stop Mission: {response.json()}")
    
    else:
        print(f"❌ Comando desconhecido: {command}")
        print("Comandos disponíveis: health, connect, disconnect, status, stop")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Testar comando específico
        command = sys.argv[1]
        test_specific_command(command)
    else:
        # Executar teste completo
        test_dashboard_commands()
