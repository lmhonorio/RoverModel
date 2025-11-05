# Diagnóstico de Conexões - Sistema de Simulação Rover

## 📊 Fluxo de Comunicação

```
┌─────────────────────┐
│   ArduPilot SITL    │  Portas FDM (Flight Dynamics Model)
│   (sim_vehicle.py)  │  ↕ 9002 IN  / 9003 OUT
└──────────┬──────────┘
           │ UDP
           ├──────────► 14551 → MAVProxy Rover 1
           │
           ▼
  ┌────────────────────┐
  │  Gazebo Plugin     │  Porta 9002/9003 (conectado!)
  │  ArduPilotPlugin   │  ✅ "ArduPilot ready to fly"
  └────────────────────┘
           │
           ▼
  ┌────────────────────┐
  │   Modelo Gazebo    │  Movimenta o robô visualmente
  │   rover_argo_N1    │
  └────────────────────┘

┌─────────────────────┐
│  MAVProxy Rover 1   │
│  14551 (input)      │
└──────────┬──────────┘
           │
           ├──────────► 14571 (output 1)
           └──────────► 14561 (output 2)
           
┌─────────────────────┐
│  MAVProxy Agregador │
│  14571 (input)      │
└──────────┬──────────┘
           │
           └──────────► 127.0.0.1:14550 → QGroundControl
                        (CORRIGIDO!)

┌─────────────────────┐
│  QGroundControl     │  Recebe telemetria e envia comandos
│  127.0.0.1:14550    │  ✅ Conectado!
└─────────────────────┘

┌─────────────────────┐
│  MAVROS (ROS)       │  Alternativa ao MAVProxy
│  127.0.0.1:14550    │  Para integração com ROS
└─────────────────────┘
```

## ✅ Status Atual

### Conexões Funcionando:
- ✅ ArduPilot SITL → Gazebo Plugin (portas 9002/9003)
- ✅ ArduPilot SITL → MAVProxy (porta 14551)
- ✅ MAVProxy → MAVProxy Agregador (porta 14571)
- ✅ MAVProxy Agregador → QGroundControl (porta 14550)
- ✅ Plugin ArduPilot carregado no Gazebo

### ⚠️ Problema Identificado:

**O robô NÃO se move no Gazebo porque:**

1. **Modo de voo incorreto**: Robô pode estar em modo MANUAL, HOLD ou ACRO
2. **Robô não armado**: Precisa armar o robô antes de mover
3. **Sem missão/comandos**: Precisa enviar waypoints ou comandos de navegação

## 🔧 Como Resolver

### Opção 1: Via QGroundControl

1. Conecte no QGroundControl (já conectado!)
2. **Arme o veículo:**
   - Clique no botão de STATUS no topo
   - Clique em "ARM" ou pressione o botão de armar
   
3. **Mude para modo GUIDED ou AUTO:**
   - Menu: Vehicle → Flight Modes
   - Selecione "GUIDED" ou "AUTO"
   
4. **Envie comandos:**
   - Em modo GUIDED: Clique no mapa e "Go to location"
   - Em modo AUTO: Carregue uma missão (waypoints)

### Opção 2: Via MAVProxy (Terminal)

No terminal do MAVProxy Rover 1, digite:

```bash
# Armar o robô
arm throttle

# Mudar para modo GUIDED
mode GUIDED

# Enviar comandos de velocidade (exemplo)
rc 3 1700  # Acelerador
rc 1 1500  # Direção (centro)

# OU enviar waypoint
wp set 1
```

### Opção 3: Via MAVROS (ROS)

```bash
# Armar
rosservice call /rover_argo_N1/Instance1/mavros/cmd/arming "value: true"

# Mudar modo
rosservice call /rover_argo_N1/Instance1/mavros/set_mode "custom_mode: 'GUIDED'"

# Enviar waypoint
rostopic pub /rover_argo_N1/Instance1/mavros/setpoint_position/local geometry_msgs/PoseStamped "..."
```

## 📋 Comandos de Diagnóstico

### Verificar portas em uso:
```bash
netstat -tulpn | grep -E "9002|9003|14550|14551"
```

### Verificar processos:
```bash
ps aux | grep -E "sim_vehicle|ardurover|mavproxy|gazebo"
```

### Verificar conexão do plugin:
```bash
tail -f ~/.gazebo/server-*/default.log | grep -i ardupilot
```

### Verificar tópicos ROS:
```bash
rostopic list | grep rover
rostopic echo /rover_argo_N1/Instance1/mavros/state
```

## 🎯 Checklist de Troubleshooting

- [x] Plugin ArduPilot instalado
- [x] Plugin ArduPilot carregado no Gazebo
- [x] ArduPilot SITL rodando
- [x] MAVProxy conectado
- [x] QGroundControl conectado
- [x] Modelo aparece no Gazebo
- [ ] Robô ARMADO
- [ ] Modo GUIDED ou AUTO ativo
- [ ] Comandos/missão enviados

## 💡 Próximos Passos

1. **Abra o QGroundControl** (já deve estar aberto)
2. **Verifique o status no canto superior esquerdo:**
   - Deve mostrar "DISARMED" ou "ARMED"
   - Deve mostrar o modo atual (MANUAL, HOLD, GUIDED, AUTO)

3. **Arme o robô:**
   - Clique no botão de armar
   - Aguarde confirmação

4. **Mude para modo GUIDED**

5. **Clique em um ponto no mapa** para o robô se mover

Agora o robô deve se mover tanto no Gazebo quanto no QGroundControl!

---

**Data:** $(date '+%d/%m/%Y %H:%M')
**Sistema:** ArduPilot Rover + Gazebo 11 + ROS Noetic

