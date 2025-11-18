# 🔄 Como Iniciar o Mission Server com Visualização Gazebo

## ⚠️ IMPORTANTE: Use o script de inicialização!

As correções para visualização de waypoints no Gazebo foram aplicadas:

✅ **Coordenadas GPS corrigidas** (LAT_REF=-3.123, LON_REF=-41.764)  
✅ **Código simplificado** (mesma abordagem do RealTime_CSV2World.py)  
✅ **Script de inicialização com ambiente ROS** (start_mission_server.sh)

## 🚀 Como Iniciar CORRETAMENTE

### ⭐ MÉTODO RECOMENDADO: Script com ambiente ROS

```bash
cd /home/viki/RoverModel
./start_mission_server.sh
```

Este script:
- ✅ Configura automaticamente o ambiente ROS Noetic
- ✅ Verifica se Gazebo está rodando
- ✅ Inicia o mission_server com ambiente correto

### Método Alternativo: Configuração Manual

Se preferir iniciar manualmente:

```bash
# 1. Configurar ambiente ROS
source /opt/ros/noetic/setup.bash

# 2. Verificar configuração
echo $ROS_DISTRO  # Deve mostrar: noetic

# 3. Iniciar mission_server
cd /home/viki/RoverModel
python3 mission_server.py
```

### 3. Verificar Inicialização

Você deve ver as mensagens:
```
✅ Nó ROS inicializado para visualização Gazebo
🚀 SERVIDOR DE MISSÕES MODULARIZADO
...
```

### 4. Executar Nova Missão

Após reiniciar, execute uma missão e observe os logs:

**Logs esperados (sucesso):**
```
🎯 VISUALIZADOR DE WAYPOINTS NO GAZEBO
============================================================
✅ ROS Gazebo detectado
🔧 Usando API Python do ROS (rospy)    ← DEVE APARECER ISTO!

🔍 MODO DEBUG: Testando primeiro waypoint...
   GPS: lat=-3.123179, lon=-41.764336
   Gazebo: x=-37.30, y=-19.89, z=3.00   ← Coordenadas razoáveis!
   Robô: R2, Cor: azul
   Nome do modelo: waypoint_test_R2_0
   
   Tentando adicionar modelo de teste...
   ✅ Teste bem-sucedido! Prosseguindo...   ← DEVE VER ISTO!
   
📊 RESULTADOS:
✅ Waypoints adicionados: 38
✅ Linhas adicionadas: 37
```

**Se ainda aparecer erro:**
```
🔧 Usando subprocess (rosrun spawn_model)    ← PROBLEMA!
```

Significa que o ambiente ROS não está configurado. Execute:

```bash
source /opt/ros/noetic/setup.bash    # ou seu ROS workspace
cd /home/viki/RoverModel
python mission_server.py
```

## 🎨 Cores dos Waypoints

Após reiniciar e executar missão, você verá no Gazebo:

- 🟢 **Rover 1 (R1, Rover_Beta)**: Bolas verdes
- 🔵 **Rover 2 (R2, Rover_Charlie)**: Bolas azuis
- 🔴 **Rover 3 (R3)**: Bolas vermelhas (se existir)

## 🐛 Troubleshooting

### Problema: "rospy não disponível"

**Solução:**
```bash
source /opt/ros/noetic/setup.bash
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH
python mission_server.py
```

### Problema: Waypoints ainda não aparecem

**Verificar:**

1. **Gazebo está rodando?**
   ```bash
   pgrep -f gzserver    # Deve retornar um PID
   ```

2. **Serviço spawn está disponível?**
   ```bash
   rosservice list | grep spawn
   # Deve mostrar: /gazebo/spawn_sdf_model
   ```

3. **Coordenadas corretas?**
   - Waypoints devem estar próximos: x ~ -50 a 50, y ~ -50 a 50
   - Se x > 1000 ou y > 1000, ajuste LAT_REF/LON_REF em `server_modules/gazebo_visualizer.py`

### Problema: Erro "model already exists"

**Solução:**
```bash
curl -X POST http://localhost:5001/gazebo/clear-waypoints
```

## 📊 Teste Rápido

Após reiniciar:

```bash
# Terminal 1: Verificar se serviço está disponível
rosservice list | grep gazebo

# Terminal 2: Executar missão via API ou frontend
# Observe os logs do mission_server para ver as bolas sendo adicionadas
```

## ✅ Checklist de Sucesso

- [ ] Mission server reiniciado
- [ ] Viu "✅ Nó ROS inicializado para visualização Gazebo"
- [ ] Viu "🔧 Usando API Python do ROS (rospy)"
- [ ] Viu "✅ Teste bem-sucedido!"
- [ ] Viu "✅ Waypoints adicionados: XX"
- [ ] **Bolas aparecem no Gazebo!** 🎉

---

**Qualquer dúvida, verifique:** `README_VISUALIZACAO_GAZEBO.md`

