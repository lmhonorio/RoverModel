import json
import networkx as nx

# Se estiver em outro módulo, ajuste os imports conforme sua estrutura de pastas
from segmentutils import SegmentUtils  # -> onde está load_graph_json
from ajusteplanilha import AjustePlanilha  # -> hipotético, para fazer a conversão metros -> geocoord
from missionmanager import MissionManager  # -> hipotético, para conectar e enviar missão

# Arquivos de entrada
file_path_graph = "./jsons/graph7.json"
file_path_missao = "./jsons/missao_6.json"
file_path_parametros = "./planilhas/obstaculos_processado6.xlsx"  # Ajuste se precisar

# 1) Carrega o grafo como NetworkX
graph_nx = SegmentUtils.load_graph_json(file_path_graph)

# 2) Função auxiliar para converter labels -> coords no grafo
def get_path_from_label(g: nx.Graph, path_labels: list[str]):
    """
    Dado um grafo NX cujos nós têm atributo "label",
    encontra as coordenadas (x,y) que correspondem a cada label da lista.
    """
    coords = []
    for label in path_labels:
        # Percorre todos os nós do grafo procurando o que tenha data["label"] == label
        found = False
        for node, data in g.nodes(data=True):
            if data.get("label") == label:
                coords.append(node)  # node é a tupla (x, y)
                found = True
                break
        if not found:
            print(f"⚠️  Label não encontrado no grafo: {label}")
    return coords

# 3) Carrega o JSON de missão e extrai todos os "path"
with open(file_path_missao, "r") as f:
    mission_data = json.load(f)

# Se a estrutura for "R1": [...], "R2": [...]
# cada item contendo "mission": "...", "tasks": [ { "path": [...], ... }, ... ]

path_r1_labels = []
if "R1" in mission_data:
    for mission_block in mission_data["R1"]:
        for task in mission_block["tasks"]:
            path_r1_labels.extend(task["path"])

path_r2_labels = []
if "R2" in mission_data:
    for mission_block in mission_data["R2"]:
        for task in mission_block["tasks"]:
            path_r2_labels.extend(task["path"])

print(f"[DEBUG] R1 path (labels): {path_r1_labels}")
print(f"[DEBUG] R2 path (labels): {path_r2_labels}")

# Converte labels -> coordenadas (tuplas x,y)
path_r1_coords = get_path_from_label(graph_nx, path_r1_labels)
path_r2_coords = get_path_from_label(graph_nx, path_r2_labels)

# 4) Converte coords (x,y) -> (lat, lon) usando arquivo de parâmetros
mission_points_1 = AjustePlanilha.metros_para_geocoordenadas(path_r1_coords, file_path_parametros)
mission_points_2 = AjustePlanilha.metros_para_geocoordenadas(path_r2_coords, file_path_parametros)

# 5) Constrói as missões no formato exigido pelo MissionManager
def build_mission(coords):
    """
    coords: lista de (lat, lon)
    retorna lista de dicts [{"id":0, "lat":..., "lon":...}, ...]
    duplicando o primeiro ponto na segunda posição (conforme snippet original).
    """
    mission = []
    for i, (lat, lon) in enumerate(coords):
        mission.append({"id": i, "lat": lat, "lon": lon})
    if len(mission) >= 1:
        mission.insert(1, mission[0])  # duplicar o primeiro ponto
    return mission

mission_dict_1 = build_mission(mission_points_1)
mission_dict_2 = build_mission(mission_points_2)

# 6) Configura e envia para os robôs
robots = [
    {"channel": "udp:0.0.0.0:14551", "mission": mission_dict_1, "source_system": 201},
    {"channel": "udp:0.0.0.0:14552", "mission": mission_dict_2, "source_system": 202}
]

managers = []

for robot in robots:
    print(f"\n🛠 Conectando com robô em {robot['channel']}")
    manager = MissionManager(
        udp_channel=robot["channel"],
        source_system=robot["source_system"],
        timeout=15,
        max_attempts=50
    )

    if manager.connect():
        if manager.upload_mission(robot["mission"]):
            managers.append(manager)
            continue

    print(f"❌ Falha ao configurar robô em {robot['channel']}")

# (Opcional) Armar e iniciar a missão
# for manager in managers:
#     manager.arm_and_start()

print("\n✅ Missões planejadas e enviadas com sucesso.")