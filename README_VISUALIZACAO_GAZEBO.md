# 🎨 Visualização de Waypoints no Gazebo

Sistema para visualizar automaticamente os waypoints de missões MAVLink no Gazebo com cores diferentes por robô.

## 📋 Descrição

Quando uma missão é executada via `mission_server.py`, os waypoints enviados aos robôs são automaticamente visualizados no Gazebo como:
- **Bolas coloridas** representando cada waypoint
- **Linhas conectando** waypoints consecutivos
- **Cores diferentes** para cada robô:
  - 🟢 **R1 (rover1)**: Verde
  - 🔵 **R2 (rover2)**: Azul  
  - 🔴 **R3 (rover3)**: Vermelho
  - 🟡 **R4 (rover4)**: Amarelo
  - 🟣 **R5 (rover5)**: Magenta
  - 🔷 **R6 (rover6)**: Ciano

## 🚀 Funcionalidades

### Automático
- ✅ Waypoints adicionados automaticamente ao executar missão
- ✅ Waypoints removidos automaticamente ao parar missão
- ✅ Processamento paralelo (8 threads) para velocidade
- ✅ Modo debug integrado para diagnóstico

### Manual
- 🗑️ Limpar waypoints via endpoint: `POST /gazebo/clear-waypoints`

## 🔧 Como Usar

### 1. Visualização Automática

Simplesmente execute uma missão normalmente:

```bash
# No frontend web ou via API
POST http://localhost:5001/execute-mission
{
  "rovers": [...],
  "equipments": [...]
}
```

Os waypoints aparecerão automaticamente no Gazebo! 🎉

### 2. Limpar Waypoints Manualmente

```bash
curl -X POST http://localhost:5001/gazebo/clear-waypoints
```

### 3. Parar Missão

```bash
curl -X POST http://localhost:5001/stop-mission
```

Os waypoints serão automaticamente removidos.

## 📐 Configuração

### Altura das Bolas

Padrão: **3.0 metros** acima do terreno

Para alterar, edite em `server_modules/mission_service.py`:

```python
visualizacao_resultado = adicionar_waypoints_missao(
    waypoints_by_robot,
    z_altura=3.0,  # ← Altere aqui
    adicionar_linhas=True,
    max_workers=8
)
```

### Cores dos Robôs

Para personalizar cores, edite `server_modules/gazebo_visualizer.py`:

```python
ROBOT_COLORS = {
    'R1': {'name': 'verde', 'ambient': '0 0.8 0', ...},
    'R2': {'name': 'azul', 'ambient': '0 0 0.8', ...},
    # ... adicione mais robôs
}
```

Cores em formato RGB normalizado (0.0 a 1.0).

### Número de Threads Paralelas

Padrão: **8 threads**

Para ajustar performance:

```python
visualizacao_resultado = adicionar_waypoints_missao(
    waypoints_by_robot,
    max_workers=8,  # ← Aumente para mais velocidade
)
```

**Recomendações:**
- 4-8 threads: Seguro, velocidade moderada
- 8-12 threads: Rápido, uso moderado de CPU
- 16+ threads: Muito rápido, mas pode sobrecarregar Gazebo

## 🔍 Modo Debug

O modo debug testa o primeiro waypoint antes de processar todos:

```python
visualizacao_resultado = adicionar_waypoints_missao(
    waypoints_by_robot,
    debug_mode=True  # ← Ativa debug detalhado
)
```

Quando ativado, você verá:
```
🔍 MODO DEBUG: Testando primeiro waypoint...
   GPS: lat=-3.123195, lon=-41.764336
   Gazebo: x=123.45, y=67.89, z=3.00
   Robô: R2, Cor: azul
   Nome do modelo: waypoint_test_R2_0
   SDF gerado (456 bytes)
   
   Tentando adicionar modelo de teste...
   ✅ Teste bem-sucedido! Prosseguindo...
```

Se houver erro, verá mensagens detalhadas com stdout/stderr do comando.

## 📊 Estatísticas

Após adicionar waypoints, você verá:

```
📊 RESULTADOS:
✅ Waypoints adicionados: 36
✅ Linhas adicionadas: 34
❌ Erros: 0

🎉 Waypoints visualizados no Gazebo!
```

## 🐛 Troubleshooting

### Problema: Nenhum waypoint aparece

**✅ CORRIGIDO! Duas soluções implementadas:**

#### 1. Coordenadas GPS de Referência Corrigidas

As coordenadas de referência foram ajustadas para corresponder à área dos waypoints:

```python
LAT_REF = -3.123  # Centro aproximado dos waypoints
LON_REF = -41.764 # Centro aproximado dos waypoints
```

**Como Verificar:**
Se os waypoints aparecerem muito longe (exemplo: x=2653, y=-28197), as coordenadas de referência estão incorretas. Eles devem aparecer próximo aos robôs (exemplo: x=5, y=10).

**Como Ajustar:**
1. Pegue as coordenadas GPS de um waypoint típico
2. Atualize LAT_REF e LON_REF em `server_modules/gazebo_visualizer.py`
3. Reinicie o mission_server

#### 2. Método de Spawn Corrigido

Agora usa **API Python do ROS** ao invés de subprocess, resolvendo o erro:
```
File "/opt/ros/noetic/lib/gazebo_ros/spawn_model", line 20, in <module>
    import rospy
```

**Métodos Disponíveis:**
1. **API Python (Primário)**: Usa `rospy.ServiceProxy('/gazebo/spawn_sdf_model')`
2. **Subprocess (Fallback)**: Usa `rosrun gazebo_ros spawn_model`

O sistema tenta automaticamente o método que funciona!

**Verificar qual método está sendo usado:**
```
🔧 Usando API Python do ROS (rospy)    ← Melhor opção
```
ou
```
🔧 Usando subprocess (rosrun spawn_model)    ← Fallback
```

### Outras Soluções

**Verifique se Gazebo está rodando:**
```bash
pgrep -f gzserver  # Deve retornar um PID
```

**Limpe modelos antigos:**
```bash
curl -X POST http://localhost:5001/gazebo/clear-waypoints
```

**Verifique logs detalhados:**
O modo debug mostra exatamente o que está acontecendo:
```
🔍 MODO DEBUG: Testando primeiro waypoint...
   GPS: lat=-3.123197, lon=-41.764390
   Gazebo: x=5.12, y=8.88, z=3.00    ← Coordenadas razoáveis!
   ✅ Teste bem-sucedido!
```

### Problema: Waypoints muito distantes

Ajuste as coordenadas GPS de referência (LAT_REF, LON_REF) para corresponder ao centro do seu mundo Gazebo.

### Problema: Erros de timeout

Aumente o timeout em `gazebo_visualizer.py`:

```python
result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)  # ← Aumente
```

### Problema: Cores erradas

Verifique se o nome do robô corresponde ao dicionário `ROBOT_COLORS`. 

Robôs suportados: `R1`, `R2`, `R3`, `R4`, `R5`, `R6` ou `rover1-6`.

## 🔧 Arquivos Principais

- **`server_modules/gazebo_visualizer.py`**: Lógica de visualização
- **`server_modules/mission_service.py`**: Integração com missões
- **`mission_server.py`**: Endpoints REST e WebSocket
- **`RealTime_CSV2World.py`**: Sistema base (paralelo otimizado)

## 📝 API Endpoints

### GET /health
Verifica status do servidor, incluindo Gazebo.

### POST /execute-mission
Executa missão e adiciona waypoints automaticamente.

### POST /stop-mission
Para missão e remove waypoints automaticamente.

### POST /gazebo/clear-waypoints
Remove todos os waypoints do Gazebo manualmente.

**Resposta:**
```json
{
  "success": true,
  "message": "15 waypoints removidos do Gazebo",
  "data": {
    "waypoints_removed": 15
  }
}
```

## 🎯 Exemplo Completo

```python
# 1. Iniciar mission_server
python mission_server.py

# 2. Executar missão (via API ou frontend)
# → Waypoints aparecem automaticamente no Gazebo

# 3. Observar waypoints coloridos:
#    - R1: Bolas verdes conectadas
#    - R2: Bolas azuis conectadas

# 4. Parar missão
# → Waypoints desaparecem automaticamente
```

## ⚡ Performance

Com paralelização (8 threads):
- **36 waypoints**: ~2-3 segundos
- **100 waypoints**: ~5-7 segundos
- **1000 waypoints**: ~30-40 segundos

Sem paralelização seria **6-8x mais lento**! 🚀

## 📜 Licença

Mesmo do projeto RoverModel.

## 👨‍💻 Autor

Integração criada para o projeto OLHE 5G.

