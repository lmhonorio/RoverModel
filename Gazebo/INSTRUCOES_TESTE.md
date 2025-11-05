# 🚀 INSTRUÇÕES PARA TESTAR A CORREÇÃO

## ⚠️ IMPORTANTE
Você precisa fechar TODOS os processos antigos e iniciar novamente com o script corrigido!

## 📋 PASSO A PASSO

### 1. Fechar todos os processos antigos

```bash
cd ~/RoverModel/Gazebo
./kill_all.sh
```

**Resultado esperado:**
- Todos os processos devem ser fechados
- Mensagem: ✅ Nenhum processo restante

---

### 2. Verificar que tudo está limpo

```bash
./check_ports.sh
```

**Resultado esperado:**
```
❌ Nenhuma porta MAVLink encontrada
❌ CRÍTICO: Plugin ArduPilot não está carregado no Gazebo!
❌ Nenhum processo ArduPilot encontrado
❌ Nenhum processo Gazebo encontrado
❌ Nenhum processo MAVProxy encontrado
✅ Plugin encontrado: /home/viki/catkin_ws/src/ardupilot_gazebo/build/libArduPilotPlugin.so
```

---

### 3. Iniciar a simulação com o script corrigido

```bash
./start_three_rovers.sh 1 parnaibaiii_charlie_delta
```

**Aguarde:**
- 15 segundos para o Gazebo inicializar
- Mais 3-5 segundos para o ArduPilot conectar
- **Total: ~20-25 segundos**

---

### 4. Executar diagnóstico completo

```bash
./check_ports.sh
```

**Resultado esperado (CORRETO):**
```
✅ Plugin encontrado: /home/viki/catkin_ws/src/ardupilot_gazebo/build/libArduPilotPlugin.so
✅ Plugin ArduPilot no PATH

📊 Portas FDM Gazebo:
gzserver [PID] viki UDP *:9002
gzserver [PID] viki UDP *:9003

🔍 Processos ArduPilot: [mostrará processos]
🔍 Processos Gazebo: [mostrará processos]
🔍 Processos MAVProxy: [mostrará processos]
```

---

### 5. Se as portas FDM AINDA não aparecerem

#### 5.1. Verificar logs do Gazebo:

```bash
tail -f ~/.gazebo/server-*.log | grep -i "ardupilot\|plugin\|bind"
```

**Procure por:**
- ✅ `Loading plugin libArduPilotPlugin.so`
- ✅ `ArduPilotPlugin: Bind to 127.0.0.1:9002`
- ✅ `ArduPilotPlugin: Connected to ArduPilot`

**OU erros:**
- ❌ `Failed to load plugin`
- ❌ `Cannot bind to port`

#### 5.2. Verificar se o plugin está no URDF do rover:

```bash
grep -i "ArduPilotPlugin" ~/catkin_ws/src/rover-argo-gazebo/rover_argo_description/urdf/rover_argo_N1.urdf.xacro
```

**Deve mostrar:**
```xml
<plugin name="ArduPilotPlugin" filename="libArduPilotPlugin.so">
  <fdm_addr>$(arg fdm_addr)</fdm_addr>
  <fdm_port_in>$(arg fdm_port_in)</fdm_port_in>
  <fdm_port_out>$(arg fdm_port_out)</fdm_port_out>
```

---

### 6. Testar no QGroundControl

Se tudo estiver OK:

1. Execute o QGroundControl
2. Conecte na porta 14550
3. Arme o rover
4. Envie comandos de movimento

**Resultado esperado:**
- ✅ Rover se move no Gazebo
- ✅ Dados de telemetria aparecem no QGC
- ✅ Posição GPS atualiza

---

## 🐛 TROUBLESHOOTING

### Problema: Plugin não carrega

**Solução:** Recompilar o plugin

```bash
cd ~/catkin_ws/src/ardupilot_gazebo/build
cmake ..
make -j4
```

### Problema: Gazebo não encontra o modelo do rover

**Solução:** Verificar GAZEBO_MODEL_PATH

```bash
echo $GAZEBO_MODEL_PATH
```

Deve conter: `/home/viki/catkin_ws/src/rover-argo-gazebo/rover_argo_gazebo/models`

### Problema: ArduPilot não conecta no Gazebo

**Verificar:** No terminal do ArduPilot, procure por:
- `SIM_Gazebo: bind port 9002` 
- `Waiting for connection`

**Se aparecer "Connection refused":** O Gazebo não está escutando na porta 9002.

---

## 📞 PRÓXIMOS PASSOS

Depois de seguir TODOS os passos acima, me envie:

1. A saída completa do `./check_ports.sh`
2. Se houver erros, a saída de `tail -100 ~/.gazebo/server-*.log | grep -i ardupilot`
3. Screenshot do terminal do ArduPilot mostrando se conectou

---

**Última atualização:** Script corrigido com GAZEBO_PLUGIN_PATH configurado


