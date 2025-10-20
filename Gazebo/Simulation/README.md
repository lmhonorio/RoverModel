# 🎯 Simulador do Dashboard - Pasta Simuation

Esta pasta contém todos os arquivos para simular comandos do dashboard de objetos de inspeção.

## 📁 Arquivos Disponíveis

### 1. `simulador_dashboard.py`
**Simulador completo** com todas as funcionalidades:
- ✅ Teste de conexão com servidor
- ✅ Execução de missões de inspeção
- ✅ Monitoramento via WebSocket
- ✅ Controle de robôs (conectar/desconectar)
- ✅ Status de missão em tempo real

**Uso:**
```bash
cd Gazebo/Kml/Simuation/
python simulador_dashboard.py
```

### 2. `teste_dashboard_simples.py`
**Teste básico** para comandos específicos:
- ✅ Teste de endpoints individuais
- ✅ Execução de comandos específicos
- ✅ Verificação de respostas
- ✅ Teste rápido e simples

**Uso:**
```bash
cd Gazebo/Kml/Simuation/
python teste_dashboard_simples.py
python teste_dashboard_simples.py health
python teste_dashboard_simples.py connect
python teste_dashboard_simples.py status
```

### 3. `README_simulador_dashboard.md`
**Documentação completa** de como usar:
- ✅ Instruções detalhadas
- ✅ Exemplos práticos
- ✅ Troubleshooting
- ✅ Casos de uso e personalização

## 🚀 Como Usar

### **1. Pré-requisitos:**
```bash
# Instalar dependências
pip install requests websocket-client

# Certificar que o mission_server.py está rodando
cd /home/viki/RoverModel
python mission_server.py
```

### **2. Executar Simuladores:**
```bash
# Navegar para a pasta
cd Gazebo/Kml/Simuation/

# Teste simples
python teste_dashboard_simples.py

# Simulador completo
python simulador_dashboard.py
```

## 📊 Comandos Disponíveis

- ✅ **Health Check** - Verificar se servidor está ativo
- ✅ **Conectar Robôs** - Conectar aos robôs para monitoramento
- ✅ **Executar Missão** - Simular missão de inspeção
- ✅ **Status da Missão** - Verificar progresso
- ✅ **Parar Missão** - Parar missão atual
- ✅ **Desconectar Robôs** - Desconectar dos robôs

## 🎯 Exemplo de Uso

```python
# Executar missão de inspeção
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
        }
    ],
    "mission_type": "inspection",
    "priority": "high"
}
```

## 🔧 Personalização

Os arquivos são **modulares** e **fáceis de personalizar**:
- **URL do servidor** (porta, IP)
- **Dados da missão** (robôs, equipamentos, pontos)
- **Comportamento** (timeouts, retries)
- **Logs** (nível de detalhamento)

## 📈 Monitoramento

### **Logs do Servidor:**
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

**Todos os arquivos estão organizados nesta pasta para facilitar o uso!** 🚀
