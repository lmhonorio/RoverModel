# 📁 Pasta Gazebo - Documentação Completa

Esta pasta contém scripts e ferramentas para trabalhar com simulações Gazebo do projeto Rover ARGO, incluindo extração de dados, conversão de coordenadas, marcação de obstáculos e gerenciamento de simulações.

---

## 📋 Índice

1. [Scripts Principais](#scripts-principais)
2. [Scripts de Marcação](#scripts-de-marcação)
3. [Scripts de Inicialização](#scripts-de-inicialização)
4. [Arquivos de Dados](#arquivos-de-dados)
5. [Fluxo de Trabalho Recomendado](#fluxo-de-trabalho-recomendado)

---

## 🎯 Scripts Principais

### 1. **Gazebo2CSV.py** 🔄
**Função:** Extrai posições de objetos do mundo Gazebo via ROS e converte para coordenadas GPS.

**Entradas:**
- Tópicos ROS:
  - `/gazebo/link_states` - Estado de todos os links dos modelos
  - `/gazebo/model_states` - Estado de todos os modelos
  - `/gazebo/default/geo_coordinates` - Coordenadas geográficas de referência
- Serviços ROS:
  - `/gazebo/get_model_state` - Obtém posição de modelos individuais

**Saídas:**
- `todos_pontos_gps.csv` - Arquivo CSV com todos os objetos
- `todos_pontos_gps.xlsx` - Arquivo Excel com os mesmos dados

**Formato de Saída:**
| Coluna | Descrição |
|--------|-----------|
| Model Name | Nome do modelo/link no Gazebo |
| Latitude | Coordenada GPS - Latitude |
| Longitude | Coordenada GPS - Longitude |
| Altitude | Altitude (Z + 77) |
| Vx | Coordenadas dos vértices em X |
| Vy | Coordenadas dos vértices em Y |
| Py | Coordenada Y cartesiana (Gazebo) |
| Px | Coordenada X cartesiana (Gazebo) |
| Vx_largura | Largura do objeto |
| Vy_altura | Altura do objeto |
| ID | Identificador único (ef_nome_do_objeto) |

**Como Usar:**
```bash
# Mundo completo (padrão)
python3 Gazebo2CSV.py

# Mundo charlie_delta
python3 Gazebo2CSV.py --mundo charlie_delta

# Com timeout customizado
python3 Gazebo2CSV.py --timeout 60
```

**Observações:**
- O Gazebo **deve estar rodando** antes de executar este script
- As coordenadas GPS são obtidas automaticamente do tópico `/gazebo/default/geo_coordinates`
- Caso o tópico não esteja disponível, usa coordenadas padrão: lat=-3.123199, lon=-41.764537
- O script filtra automaticamente objetos do sistema (ground_plane, sun, rovers)

---

### 2. **CSV2World.py** 🌍
**Função:** Cria um arquivo de mundo Gazebo modificado adicionando bolas coloridas baseadas em dados CSV e JSON.

**Entradas:**
- `todos_pontos_gps.csv` - Dados dos equipamentos (gerado pelo Gazebo2CSV.py)
- `../jsons/graph_equipment.json` - Pontos do grafo de navegação
- `parnaibaiii_simple_v3.world` - Arquivo de mundo original

**Saídas:**
- `parnaibaiii_simple_v3_modificado.world` - Mundo com bolas adicionadas

**Tipos de Bolas:**
- 🟢 **Bolas Verdes** (raio 0.5m): Pontos do JSON (graph_equipment.json)
- 🔵 **Bolas Azuis** (raio 0.8m): Equipamentos do CSV (todos_pontos_gps.csv)

**Como Usar:**
```bash
python3 CSV2World.py
```

**Interface Interativa:**
1. Escolha o tipo de bolas:
   - Bolas Verdes
   - Bolas Azuis
   - Ambas (Verdes e Azuis)
   - Cancelar
2. O script processa os dados e cria o arquivo modificado

**Observações:**
- O arquivo original **NÃO é alterado** - uma cópia modificada é criada
- As coordenadas são transformadas para corresponder ao sistema de coordenadas do Gazebo
- Bolas verdes: X_verde = Y_json, Y_verde = -X_json
- Bolas azuis: Aplica rotação de -90° às coordenadas do CSV

---

### 3. **RealTime_CSV2World.py** ⚡
**Função:** Adiciona ou remove bolas em um mundo Gazebo **já aberto** em tempo real.

**Entradas:**
- `todos_pontos_gps.csv` - Dados dos equipamentos
- `../jsons/graph_equipment.json` - Pontos do grafo de navegação
- Gazebo rodando (ROS ou standalone)

**Funcionalidades:**
1. **Adicionar Bolas:**
   - 🟢 Bolas Verdes + 🔗 Linhas de conexão (do JSON)
   - 🔵 Bolas Azuis (do CSV)
   - Ambas
   - Filtro por grupos de equipamentos (REATOR, TPC, TC, etc.)

2. **Deletar Bolas:**
   - Remove todas as bolas e linhas do mundo

**Como Usar:**
```bash
# Com ROS Gazebo rodando
python3 RealTime_CSV2World.py

# Com Gazebo standalone rodando
python3 RealTime_CSV2World.py
```

**Menu Interativo:**
```
1. ➕ Adicionar Bolas
2. 🗑️  Deletar Todas as Bolas
3. ❌ Cancelar
```

**Filtro de Grupos (para Bolas Azuis):**
- REATOR - Reatores
- SVC - Seccionadores
- TPC - Transformadores de Potência
- TC - Transformadores de Corrente
- SECV - Seccionadores Verticais
- SECH - Seccionadores Horizontais
- IP - Interruptores de Potência
- DISJUNTOR - Disjuntores
- BUSIP - Barramentos de Potência
- BUSCSB - Barramentos de Controle
- PR - Protetores

**Observações:**
- O Gazebo **DEVE estar rodando** antes de executar
- Detecta automaticamente se é ROS Gazebo ou Gazebo standalone
- Modelos leves sem colisão para melhor performance
- Adiciona objetos via `rosrun gazebo_ros spawn_model` (ROS) ou `gz model` (standalone)

---

## 🏷️ Scripts de Marcação

### 4. **marcador_vao.py** 🖱️
**Função:** Marca regiões/vãos em pontos GPS usando interface gráfica interativa.

**Entrada:**
- `todos_pontos_gps.csv` (ou arquivo especificado)

**Saída:**
- `todos_pontos_gps_modificado.csv` - CSV com nomes de regiões adicionados aos modelos

**Como Usar:**
```bash
# Usar arquivo padrão
python3 marcador_vao.py

# Usar arquivo específico
python3 marcador_vao.py meu_arquivo.csv
```

**Fluxo de Trabalho:**
1. Selecione múltiplas áreas no mapa (clique e arraste)
2. Pressione `F` para finalizar seleções
3. Digite o nome de cada região (ex: ECHO, DELTA, BRAVO)
4. O script adiciona o nome da região ao Model Name de cada objeto

**Formato de Nome Modificado:**
```
Antes:  ARGO_PARNAIBAIII_V3::TPC1
Depois: ARGO_PARNAIBAIII_V3::ECHO::TPC1
```

**Observações:**
- Use matplotlib interativo para selecionar áreas
- Suporta múltiplas seleções em uma única sessão
- Verifica duplicatas automaticamente

---

### 5. **Talude_mark.py** 🏔️
**Função:** Cria marcadores de taludes baseados em 4 torres que formam um quadrilátero.

**Entrada:**
- 4 torres no Gazebo: `Torre`, `Torre_0`, `Torre_1`, `Torre_2`
- Serviço ROS: `world_obstacles_service/get` (coordenadas GPS das torres)
- Serviço ROS: `/gazebo/get_model_state` (posições locais)

**Saída:**
- `Taludes_marker.csv` - Arquivo CSV com dados do talude

**Como Usar:**
```bash
# Modo interativo
python3 Talude_mark.py

# Com nome do modelo como argumento
python3 Talude_mark.py Talude_1
```

**Formato de Saída:**
- Calcula o centro geométrico das 4 torres
- Determina Vx_largura e Vy_altura baseado nos extremos
- Gera coordenadas da caixa delimitadora
- Salva com ID único: `ef_talude_1`

**Observações:**
- Requer ROS e Gazebo rodando
- As 4 torres devem estar presentes no mundo
- Coordenadas GPS são obtidas do serviço ROS
- Verifica duplicatas antes de salvar

---

### 6. **Unir_csv.py** 🔗
**Função:** Une dados de `Taludes_marker.csv` em `todos_pontos_gps`.

**Entradas:**
- `Taludes_marker.csv` - Dados dos taludes
- `todos_pontos_gps.csv` - Dados principais
- `todos_pontos_gps.xlsx` - Dados principais em Excel

**Saídas:**
- Atualiza `todos_pontos_gps.csv` com novos taludes
- Atualiza `todos_pontos_gps.xlsx` com novos taludes

**Como Usar:**
```bash
python3 Unir_csv.py
```

**Observações:**
- Verifica duplicatas por ID antes de adicionar
- Mantém dados existentes intactos
- Atualiza tanto CSV quanto Excel
- Preserva aba ParametrosConversao (se existir)

---

## 🚀 Scripts de Inicialização

### 7. **start_three_rovers.sh** 🤖
**Função:** Inicializa 1 a 3 rovers ArduPilot com Gazebo e ROS.

**Funcionalidades:**
- Inicia Gazebo com ROS Launch
- Inicia instâncias ArduPilot SITL
- Limpa processos e estados antigos
- Opção de iniciar QGroundControl
- Suporta múltiplos mundos

**Como Usar:**
```bash
# Modo interativo (pergunta quantos rovers e qual mundo)
./start_three_rovers.sh

# Com argumentos
./start_three_rovers.sh 1 parnaibaiii_simple_v3
./start_three_rovers.sh 2 parnaibaiii_charlie_delta
./start_three_rovers.sh 3 gravel_plane
```

**Mundos Disponíveis:**
1. `parnaibaiii_simple_v2`
2. `parnaibaiii_charlie_delta`
3. `parnaibaiii_simple_v3` (padrão)
4. `gravel_plane`

**Portas de Comunicação:**
- QGC (QGroundControl): 14550
- MAVROS Rover 0: 14551
- MAVROS Rover 1: 14651
- MAVROS Rover 2: 14751

**Processos Encerrados:**
- mavproxy
- gazebo/gzserver/gzclient
- roslaunch/rosmaster
- sim_vehicle
- mavros
- ardurover

**Limpeza Realizada:**
- `eeprom.bin` (parâmetros salvos)
- Pasta `terrain`
- Logs do Gazebo
- Cache temporário

**Observações:**
- Requer ArduPilot instalado em `~/ardupilot/ardupilot`
- Requer plugin ArduPilot Gazebo em `/home/viki/catkin_ws/src/ardupilot_gazebo/build`
- Aguarda 15 segundos para Gazebo carregar antes de iniciar rovers
- Logs salvos em `/tmp/sitl_*.log`

---

### 8. **Install_All.sh** ⚙️
**Função:** Script de instalação completa do ambiente ROS Noetic + Gazebo + ArduPilot.

**O que Instala:**
1. ROS Noetic Desktop Full
2. Dependências do ROS
3. Gazebo 11
4. Plugins do Gazebo
5. ArduPilot SITL
6. MAVProxy
7. Catkin workspace
8. Pacotes ROS customizados

**Como Usar:**
```bash
./Install_All.sh
```

**Observações:**
- Requer Ubuntu 20.04 LTS
- Requer sudo/root para instalação
- Processo pode demorar 30-60 minutos
- Configura automaticamente .bashrc

---

## 📊 Arquivos de Dados

### 9. **todos_pontos_gps.csv** 📄
**Formato:** CSV com todos os objetos extraídos do Gazebo
**Gerado por:** `Gazebo2CSV.py`
**Usado por:** `CSV2World.py`, `RealTime_CSV2World.py`, `marcador_vao.py`

### 10. **todos_pontos_gps.xlsx** 📊
**Formato:** Excel com os mesmos dados do CSV
**Gerado por:** `Gazebo2CSV.py`
**Usado por:** `Unir_csv.py`

### 11. **todos_pontos_gps_modificado.csv** 📝
**Formato:** CSV com nomes de regiões/vãos adicionados
**Gerado por:** `marcador_vao.py`

### 12. **Taludes_marker.csv** 🏔️
**Formato:** CSV com dados de taludes
**Gerado por:** `Talude_mark.py`
**Usado por:** `Unir_csv.py`

---

## 🔄 Fluxo de Trabalho Recomendado

### Cenário 1: Extração e Visualização de Dados

```bash
# 1. Iniciar simulação Gazebo
./start_three_rovers.sh 1 parnaibaiii_simple_v3

# 2. Extrair dados dos objetos do Gazebo
python3 Gazebo2CSV.py

# 3. Criar mundo modificado com bolas para visualização
python3 CSV2World.py
```

### Cenário 2: Adicionar Bolas em Tempo Real

```bash
# 1. Iniciar simulação
./start_three_rovers.sh 1 parnaibaiii_simple_v3

# 2. Adicionar bolas interativamente
python3 RealTime_CSV2World.py
```

### Cenário 3: Marcar Regiões/Vãos

```bash
# 1. Extrair dados (se ainda não extraiu)
python3 Gazebo2CSV.py

# 2. Marcar regiões com interface gráfica
python3 marcador_vao.py

# 3. Usar o arquivo modificado
# todos_pontos_gps_modificado.csv agora tem os nomes das regiões
```

### Cenário 4: Adicionar Taludes

```bash
# 1. Iniciar simulação com torres
./start_three_rovers.sh 1 parnaibaiii_simple_v3

# 2. Marcar talude baseado nas 4 torres
python3 Talude_mark.py Talude_1

# 3. Unir taludes com dados principais
python3 Unir_csv.py

# 4. Extrair novamente para atualizar
python3 Gazebo2CSV.py
```

---

## 🛠️ Requisitos

### Software
- Ubuntu 20.04 LTS
- ROS Noetic
- Gazebo 11
- Python 3.8+
- ArduPilot SITL

### Bibliotecas Python
```bash
pip install pandas openpyxl matplotlib rospy
```

### Pacotes ROS
```bash
sudo apt install ros-noetic-gazebo-ros-pkgs
sudo apt install ros-noetic-gazebo-plugins
sudo apt install ros-noetic-mavros
```

---

## 📝 Notas Importantes

### Coordenadas de Referência
- **Latitude padrão:** -3.123199
- **Longitude padrão:** -41.764537
- Obtidas automaticamente do tópico `/gazebo/default/geo_coordinates` quando disponível

### Transformações de Coordenadas

**CSV → Gazebo (Gazebo2CSV.py):**
- X_gazebo = Px (sem rotação)
- Y_gazebo = Py (sem rotação)
- GPS calculado usando raio da Terra: 6378100.0 m

**JSON → Bolas Verdes (CSV2World.py):**
- X_verde = Y_json
- Y_verde = -X_json

**CSV → Bolas Azuis (CSV2World.py):**
- Aplica rotação de -90° (ângulo = -1.570796 rad)
- X_azul = X_csv * cos(ângulo) - Y_csv * sin(ângulo)
- Y_azul = X_csv * sin(ângulo) + Y_csv * sin(ângulo)

### Dimensões dos Objetos
Os scripts usam dimensões predefinidas para cada tipo de equipamento:
- **REATOR/TRANSFORMADOR:** 3.069 × 6.109 m
- **TPC:** 1.729 × 1.556 m
- **TC:** 1.713 × 1.408 m
- **IP:** 1.448 × 1.047 m
- **SECH:** 6.607 × 1.060 m
- **SECV:** 2.178 × 1.100 m
- **DISJUNTOR:** 4.917 × 1.476 m
- **BUSCSB/BUSIP:** 1.561 × 1.100 m
- **PR:** 1.374 × 1.236 m
- **Outros:** 1.0 × 1.0 m (padrão)

---

## 🐛 Troubleshooting

### "Gazebo não está rodando!"
**Solução:** Execute `./start_three_rovers.sh` antes dos scripts Python

### "Arquivo CSV não encontrado!"
**Solução:** Execute `python3 Gazebo2CSV.py` primeiro para gerar o CSV

### "Coordenadas de referência não disponíveis"
**Solução:** O script usará coordenadas padrão automaticamente

### "Erro ao adicionar modelo via ROS"
**Solução:** Verifique se `roscore` está rodando e se o Gazebo está conectado ao ROS

### Processos não fecham corretamente
**Solução:** Use `killall -9 gazebo gzserver gzclient roslaunch` manualmente

---

## 📞 Suporte

Para dúvidas ou problemas:
1. Verifique os logs em `/tmp/sitl_*.log`
2. Verifique se todos os requisitos estão instalados
3. Execute `./Install_All.sh` se necessário

---

**Última atualização:** Novembro 2025
**Versão:** 1.0

