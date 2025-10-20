# 🎯 Simulador do Dashboard de Objetos de Inspeção

Este documento explica como simular comandos do dashboard de objetos de inspeção usando Python, testando o `mission_server.py`.

## 📁 Arquivos Criados

### 1. `simulador_dashboard.py`
**Simulador completo** com WebSocket e monitoramento em tempo real.

**Funcionalidades:**
- ✅ Teste de conexão com servidor
- ✅ Execução de missões de inspeção
- ✅ Monitoramento via WebSocket
- ✅ Controle de robôs (conectar/desconectar)
- ✅ Status de missão em tempo real

### 2. `teste_dashboard_simples.py`
**Teste básico** para comandos específicos.

**Funcionalidades:**
- ✅ Teste de endpoints individuais
- ✅ Execução de comandos específicos
- ✅ Verificação de respostas
- ✅ Teste rápido e simples

## 🚀 Como Usar

### **Pré-requisitos:**
```bash
# Instalar dependências
pip install requests websocket-client

# Certificar que o mission_server.py está rodando
python mission_server.py
```

### **Teste Simples:**
```bash
# Teste completo
python teste_dashboard_simples.py

# Teste de comando específico
python teste_dashboard_simples.py health
python teste_dashboard_simples.py connect
python teste_dashboard_simples.py status
```

### **Simulador Completo:**
```bash
# Executar simulação completa
python simulador_dashboard.py
```

## 📊 Comandos Disponíveis

### **1. Health Check**
```python
GET /health
```
- Verifica se servidor está ativo
- Retorna status dos robôs conectados

### **2. Conectar Robôs**
```python
POST /connect-robots
```
- Conecta aos robôs para monitoramento
- Retorna número de robôs conectados

### **3. Executar Missão**
```python
POST /execute-mission
```
**Dados de exemplo:**
```json
{
    "robots": ["rover_argo_1", "rover_argo_2"],
    "equipment": ["camera", "sensor_temperature"],
    "inspection_points": [
        {
            "id": "equipment_1",
            "lat": -3.123199,
            "lon": -41.764537,
            "priority": "high",
            "equipment_type": "reactor"
        }
    ],
    "mission_type": "inspection",
    "priority": "high"
}
```

### **4. Status da Missão**
```python
GET /mission-status
```
- Retorna status atual da missão
- Progresso e informações dos robôs

### **5. Parar Missão**
```python
POST /stop-mission
```
- Para a missão atual
- Retorna confirmação

### **6. Desconectar Robôs**
```python
POST /disconnect-robots
```
- Desconecta dos robôs
- Limpa conexões

## 🔌 WebSocket Events

### **Eventos Recebidos:**
- `robot_position_continuous` - Posições dos robôs
- `mission_status_update` - Status da missão
- `mission_waypoints_update` - Waypoints atualizados

### **Eventos Enviados:**
- `request_robot_positions` - Solicitar posições
- `request_mission_status` - Solicitar status
- `start_robot_monitoring` - Iniciar monitoramento

## 📝 Exemplo de Uso Programático

```python
import requests

# Conectar ao servidor
server_url = "http://127.0.0.1:5001"

# 1. Verificar saúde
response = requests.get(f"{server_url}/health")
print(f"Status: {response.json()}")

# 2. Conectar robôs
response = requests.post(f"{server_url}/connect-robots")
print(f"Robôs conectados: {response.json()}")

# 3. Executar missão
mission_data = {
    "robots": ["rover_argo_1"],
    "equipment": ["camera"],
    "inspection_points": [
        {
            "id": "equipment_1",
            "lat": -3.123199,
            "lon": -41.764537,
            "priority": "high"
        }
    ]
}

response = requests.post(
    f"{server_url}/execute-mission",
    json=mission_data
)
print(f"Missão: {response.json()}")

# 4. Verificar status
response = requests.get(f"{server_url}/mission-status")
print(f"Status: {response.json()}")

# 5. Parar missão
response = requests.post(f"{server_url}/stop-mission")
print(f"Parada: {response.json()}")
```

## 🎯 Casos de Uso

### **1. Teste de Conectividade**
```bash
python teste_dashboard_simples.py health
```

### **2. Teste de Missão Completa**
```bash
python teste_dashboard_simples.py
```

### **3. Monitoramento em Tempo Real**
```bash
python simulador_dashboard.py
```

### **4. Teste de Comando Específico**
```bash
python teste_dashboard_simples.py connect
python teste_dashboard_simples.py status
python teste_dashboard_simples.py stop
```

## 🔧 Personalização

### **Modificar Dados da Missão:**
Edite o arquivo `teste_dashboard_simples.py` na seção `mission_data`:

```python
mission_data = {
    "robots": ["rover_argo_1", "rover_argo_2"],  # Seus robôs
    "equipment": ["camera", "sensor_temperature"],  # Seus equipamentos
    "inspection_points": [
        {
            "id": "equipment_1",
            "lat": -3.123199,  # Suas coordenadas
            "lon": -41.764537,
            "priority": "high",
            "equipment_type": "reactor"
        }
    ]
}
```

### **Modificar URL do Servidor:**
```python
# No início dos arquivos
server_url = "http://127.0.0.1:5001"  # Sua URL
```

## 🐛 Troubleshooting

### **Erro de Conexão:**
```
❌ Erro ao conectar: [Errno 111] Connection refused
```
**Solução:** Certifique que o `mission_server.py` está rodando na porta 5001.

### **Erro de Robôs:**
```
❌ Erro ao conectar robôs: HTTP 500
```
**Solução:** Verifique se os robôs estão disponíveis no ArduPilot.

### **Erro de WebSocket:**
```
❌ Erro WebSocket: [Errno 111] Connection refused
```
**Solução:** Certifique que o SocketIO está habilitado no servidor.

## 📈 Monitoramento

### **Logs do Servidor:**
O `mission_server.py` mostra logs detalhados:
```
🚀 SERVIDOR DE MISSÕES MODULARIZADO
📁 Diretório de trabalho: /home/viki/RoverModel
🔗 Endpoints disponíveis:
   GET  /health - Verificar status do servidor
   POST /execute-mission - Executar missão
   GET  /mission-status - Obter status da missão
```

### **Logs do Simulador:**
```
🎯 TESTANDO COMANDOS DO DASHBOARD
1️⃣ Testando conexão...
✅ Servidor ativo: Servidor de missões ativo
   📊 Robôs conectados: 0
```

## 🎉 Conclusão

Com estes simuladores, você pode:
- ✅ Testar todos os endpoints do dashboard
- ✅ Simular missões de inspeção completas
- ✅ Monitorar robôs em tempo real
- ✅ Verificar funcionamento do sistema
- ✅ Desenvolver e testar integrações

Os arquivos são **modulares** e **fáceis de personalizar** para suas necessidades específicas! 🚀
