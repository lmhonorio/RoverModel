import math
from roverclass import ObstacleLoader
from segmentutils import SegmentUtils
from aabbutils import AABBUtils
from multigraphplanner import MultiGraphPlanner
from bridgetoAP2 import MissionManager
from ajusteplanilha import AjustePlanilha

##################################################
# 1) CARREGAR O GRAFO E CONVERTER
##################################################
file_path_grafo = "./jsons/graph6.json"
file_path_parametros = "./planilhas/obstaculos_processado6.xlsx"  # Excel com aba 'ParametrosConversao'

# Carregar o grafo
graph_nx = SegmentUtils.load_graph_json(file_path_grafo)
grafo_mapa = AABBUtils.convert_graph_to_dict(graph_nx)

##################################################
# 2) DEFINIR LABELS DE PARTIDA/DESTINO
##################################################
robots_positions = {
    "R1": "b_busip4_3",
    "R2": "ls_pr1_1"
}
destinations = {
    "R1": "ef_pr11_4",
    "R2": "b_busip40_6"
}

planner = MultiGraphPlanner(grafo_mapa, None, None, None, None, None)

def get_path_from_label(nx_graph, path_labels):
    path_coord = []
    for label in path_labels:
        for node, data in nx_graph.nodes(data=True):
            if data.get("label", "") == label:
                path_coord.append(node)
    return path_coord

##################################################
# 3) PLANEJAR CAMINHOS (labels -> (x, y))
##################################################
path_r1_labels = planner.a_star(robots_positions["R1"], destinations["R1"])[0]
path_r2_labels = planner.a_star(robots_positions["R2"], destinations["R2"])[0]

print(f"[DEBUG] R1 path (labels): {path_r1_labels}")
print(f"[DEBUG] R2 path (labels): {path_r2_labels}")

path_r1_coords = get_path_from_label(graph_nx, path_r1_labels)
path_r2_coords = get_path_from_label(graph_nx, path_r2_labels)

##################################################
# 4) CONVERTER PARA LAT/LON USANDO ARQUIVO DE PARÂMETROS
##################################################
mission_points_1 = AjustePlanilha.metros_para_geocoordenadas(path_r1_coords, file_path_parametros)
mission_points_2 = AjustePlanilha.metros_para_geocoordenadas(path_r2_coords, file_path_parametros)

# Montar missões no formato exigido
def build_mission(coords):
    mission = []
    for i, (lat, lon) in enumerate(coords):
        mission.append({"id": i, "lat": lat, "lon": lon})
    if len(mission) >= 1:
        mission.insert(1, mission[0])  # Duplicar o primeiro ponto
    return mission

mission_dict_1 = build_mission(mission_points_1)
mission_dict_2 = build_mission(mission_points_2)

##################################################
# 5) ENVIAR PARA OS DRONES VIA MISSIONMANAGER
##################################################
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

# 🚀 (Opcional) Iniciar missão
# for manager in managers:
#     manager.arm_and_start()

print("\n✅ Missões planejadas e enviadas com sucesso.")
