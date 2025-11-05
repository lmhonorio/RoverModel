# 🔧 TESTE DE MOVIMENTO DO ROVER - DIAGNÓSTICO COMPLETO

## Status Atual
- ✅ Portas FDM abertas (9002/9003)
- ✅ Plugin ArduPilotPlugin carregado no modelo
- ✅ Joints do rover detectados (4 rodas)
- ❌ Joints não estão se movendo (position = 0.0)

## 🎯 TESTE MANUAL - Terminal do ArduPilot

### 1. Abra o terminal do ArduPilot (xterm "Rover 1")

Nesse terminal, digite os seguintes comandos:

```
# Ver parâmetros dos servos
param show SERVO1_FUNCTION
param show SERVO3_FUNCTION

# Devem mostrar:
# SERVO1_FUNCTION = 73 (Throttle Left)
# SERVO3_FUNCTION = 74 (Throttle Right)
```

### 2. Teste Manual dos Servos

No terminal do ArduPilot, digite:

```
# Armar o rover
arm throttle

# Testar servo 1 (rodas esquerdas) manualmente
rc 3 1600

# Aguardar 2 segundos e parar
rc 3 1500
```

**O QUE DEVE ACONTECER:**
- As rodas ESQUERDAS devem girar no Gazebo
- Se não girarem, há problema no mapeamento servo → joint

### 3. Teste com QGroundControl

Se o teste manual funcionar:
1. Mode: MANUAL
2. Arm
3. Use controle para mover

Se o teste manual NÃO funcionar:
- O problema está na configuração do plugin no URDF

## 🔍 VERIFICAÇÕES ADICIONAIS

### A. Verificar se ArduPilot detectou o modelo Gazebo

No terminal do ArduPilot, procure por:
```
SIM_Gazebo: Bind to 9002
```

### B. Verificar configuração dos canais

Execute no terminal do ArduPilot:
```
param show SERVO*_FUNCTION
```

Deve mostrar:
- SERVO1_FUNCTION = 73 (Throttle Left)
- SERVO3_FUNCTION = 74 (Throttle Right)

### C. Verificar se o plugin está recebendo comandos

Execute em outro terminal:
```bash
rostopic echo /rover_argo_N1/Instance1/joint_states
```

Depois, no ArduPilot, envie:
```
rc 3 1600
```

**Se joint_states mudar:** ✅ Plugin está funcionando!
**Se joint_states NÃO mudar:** ❌ Plugin não está recebendo/aplicando comandos

## 🚨 POSSÍVEIS PROBLEMAS

### Problema 1: Canais Errados no URDF

O URDF define:
- Canal 0 (channel="0") → Throttle Left (SERVO1)
- Canal 2 (channel="2") → Throttle Right (SERVO3)

Mas pode estar usando os canais errados.

**Solução:** Verificar o arquivo:
`~/catkin_ws/src/rover-argo-gazebo/rover_argo_description/urdf/rover_argo_N1.urdf.xacro`

Procure por:
```xml
<control channel="0">
  <jointName>front_left_wheel_joint</jointName>
```

### Problema 2: Multiplier Negativo

No URDF tem:
```xml
<multiplier>-50</multiplier>
```

Isso pode estar invertendo os comandos.

**Teste:** Mudar para `<multiplier>50</multiplier>` (sem negativo)

### Problema 3: Lock Step

O plugin usa:
```xml
<lock_step>1</lock_step>
```

Isso significa que o Gazebo espera o ArduPilot enviar comandos sincronizados.

**Teste:** No gzrover.param, verificar se tem `SIM_SPEEDUP`

## 📊 EXECUTE ESTE TESTE AGORA:

1. Vá no terminal "Rover 1" (xterm)
2. Digite: `arm throttle`
3. Digite: `rc 3 1600`
4. Olhe o Gazebo
5. Me diga o que acontece!



