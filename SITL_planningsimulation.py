import json
from multigraphplanner import MultiGraphPlanner

# Se estiver em outro módulo, ajuste os imports conforme sua estrutura de pastas
from segmentutils import SegmentUtils  # -> onde está load_graph_json
from ajusteplanilha import AjustePlanilha  # -> hipotético, para fazer a conversão metros -> geocoord
from missionmanager import MissionManager  # -> hipotético, para conectar e enviar missão

# Arquivos de entrada
file_path_graph = "./jsons/graph9_new.json"
file_path_missao = "./jsons/missao_6.json"
file_path_parametros = "./planilhas/obstaculos_processado6.xlsx"  # Ajuste se precisar

# 1) Carrega o grafo como NetworkX
graph_nx = SegmentUtils.load_graph_json(file_path_graph)


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
path_r1_coords = MultiGraphPlanner.get_path_from_label(graph_nx, path_r1_labels)
path_r2_coords = MultiGraphPlanner.get_path_from_label(graph_nx, path_r2_labels)

# 4) Converte coords (x,y) -> (lat, lon) usando arquivo de parâmetros
mission_points_1 = AjustePlanilha.metros_para_geocoordenadas(path_r1_coords, file_path_parametros)
mission_points_2 = AjustePlanilha.metros_para_geocoordenadas(path_r2_coords, file_path_parametros)



mission_dict_1 = MissionManager.build_mission(mission_points_1)
mission_dict_2 = MissionManager.build_mission(mission_points_2)

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

# (Opcional) Armar e iniciar a missão - tirei para comecar pelo qground control
# for manager in managers:
#     manager.arm_and_start()

print("\n✅ Missões planejadas e enviadas com sucesso.")