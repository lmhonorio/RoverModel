import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import requests

from old.multigraphplanner import MultiGraphPlanner
from persistent_astar_cache import PersistentAStarCache
from segmentutils import SegmentUtils
from aabbutils import AABBUtils
from plotutils import PlotUtils
from movns_ains_argo import task_priority_argo
from movns_ains_argo.robot import Robot
from movns_ains_argo import solution_priority_argo
from movns_ains_argo import movns_ains_argo
from tspOptimization import FixedTaskPlanner
from movns_ains_argo import send_mission_argo


# =====================================================================================
#                                  CONFIG / TOGGLES
# =====================================================================================

# --- Arquivos ---
GRAPH_JSON = "./jsons/graph8_new_funcionando.json"
OBS_POINTS_JSON = "./jsons/obp_6_funcionando.json"
AABB_POR_EQUIPAMENTO_JSON = "./jsons/aabb_por_equipamento.json"
EQUIPAMENTOS_FILTRADOS_JSON = "./jsons/equipamentos_filtrados.json"
AABB_INFO_JSON = "./jsons/aabb_info.json"
CACHE_ASTAR_PATH = "./cache_astar.json"

# --- Presets de missões (mantenho as tuas, com um seletor simples) ---
MISSION_PRESETS: Dict[str, List[str]] = {
    "mini": ['b_busip4', 'ef_reator1', 'ls_pr4', 'ef_reator2', 'ef_reator3', 'ef_reator4', 'ef_disjuntor1', 'ef_sech1', 'b_buscsb5'],

    ################ 22 TASKS ###############################
    "example_22": [
        'ef_reator1', 'ef_reator2', 'ef_reator3', 'ef_reator4', 'ef_reator5', 'ef_reator6', 'ef_reator7', 'ef_reator8', 'ef_reator9', 'ef_reator10',
        'ef_pr2', 'ef_pr3', 'ef_pr4', 'ef_pr5', 'ef_pr6', 'ef_pr7', 'ef_pr8', 'ef_pr9', 'ef_pr10', 'ef_pr11', 'ef_pr12', 'ef_pr13',
        'ls_tpc1', 'ls_tpc2', 'ls_tpc3', 'ls_tpc4', 'ls_tpc5', 'ls_tpc6',
        'r_pr1', 'r_reator1', 'r_reator2', 'r_pr2'],

    ################ 52 TASKS ###############################
    "example_52": [
        'ef_reator1', 'ef_reator2', 'ef_reator3', 'ef_reator4', 'ef_reator5', 'ef_reator6', 'ef_reator7', 'ef_reator8', 'ef_reator9', 'ef_reator10',
        'ef_pr2', 'ef_pr3', 'ef_pr4', 'ef_pr5', 'ef_pr6', 'ef_pr7', 'ef_pr8', 'ef_pr9', 'ef_pr10', 'ef_pr11', 'ef_pr12', 'ef_pr13',
        
        'ef_disjuntor1', 'ef_disjuntor2', 'ef_disjuntor3', 'ef_disjuntor4', 'ef_disjuntor5', 'ef_disjuntor6', 'ef_disjuntor7', 'ef_disjuntor8', 'ef_disjuntor9', 'ef_disjuntor10',

        'ef_buscsb21', 'ef_buscsb22',
        'ef_busip1', 'ef_busip2', 'ef_busip3', 'ef_busip4', 'ef_busip5', 'ef_busip6', 'ef_busip7', 'ef_busip8', 'ef_busip9', 'ef_busip10',

        'cd_busip21', 'cd_busip20', 'cd_busip18', 'cd_busip14', 'cd_busip13', 'cd_busip4',
        'cd_buscsb10', 'cd_buscsb9', 'cd_buscsb8', 'cd_buscsb7', 'cd_buscsb6', 'cd_buscsb5', 'cd_buscsb4', 'cd_buscsb25', 'cd_buscsb24', 'cd_buscsb23',
    
        'b_tc3', 'b_tc2', 'b_tc1', 'b_tc6', 'b_tc5', 'b_tc4',
        'b_secv3', 'b_secv2', 'b_secv1', 'b_secv6', 'b_secv5', 'b_secv4'
    ],

    #################################### ALL TASKS ###################################
    "example_complete":[
        'ef_reator1', 'ef_reator2', 'ef_reator3', 'ef_reator4', 'ef_reator5', 'ef_reator6', 'ef_reator7', 'ef_reator8', 'ef_reator9', 'ef_reator10',
        'ef_pr2', 'ef_pr3', 'ef_pr4', 'ef_pr5', 'ef_pr6', 'ef_pr7', 'ef_pr8', 'ef_pr9', 'ef_pr10', 'ef_pr11', 'ef_pr12', 'ef_pr13',
        'ef_tpc1', 'ef_tpc2', 'ef_tpc3',
        'ef_ip1', 'ef_ip2', 'ef_ip3', 'ef_ip4', 'ef_ip5', 'ef_ip6', 'ef_ip7', 'ef_ip8', 'ef_ip9', 'ef_ip10', 'ef_ip11', 'ef_ip12', 'ef_ip13', 'ef_ip14', 'ef_ip15', 'ef_ip16', 'ef_ip17', 'ef_ip18',
        'ef_sech1', 'ef_sech2', 'ef_sech3', 'ef_sech4', 'ef_sech5', 'ef_sech6', 'ef_sech7', 'ef_sech8', 'ef_sech9', 'ef_sech10',
        'ef_sech11', 'ef_sech12', 'ef_sech13', 'ef_sech14', 'ef_sech15', 'ef_sech16', 'ef_sech17', 'ef_sech18', 'ef_sech19', 'ef_sech20', 'ef_sech21', 'ef_sech22', 'ef_sech23', 'ef_sech24',
        'ef_tc1', 'ef_tc2', 'ef_tc3', 'ef_tc4', 'ef_tc5', 'ef_tc6', 'ef_tc7', 'ef_tc8', 'ef_tc9', 'ef_tc10', 'ef_tc11', 'ef_tc12', 'ef_tc13', 'ef_tc14', 'ef_tc15',
        'ef_secv1', 'ef_secv2', 'ef_secv3', 'ef_secv4', 'ef_secv5', 'ef_secv6', 'ef_secv7', 'ef_secv8', 'ef_secv9', 'ef_secv10', 'ef_secv11', 'ef_secv12', 'ef_secv13', 'ef_secv14', 'ef_secv15',
        'ef_disjuntor1', 'ef_disjuntor2', 'ef_disjuntor3', 'ef_disjuntor4', 'ef_disjuntor5', 'ef_disjuntor6', 'ef_disjuntor7', 'ef_disjuntor8', 'ef_disjuntor9', 'ef_disjuntor10',
        'ef_buscsb1', 'ef_buscsb2', 'ef_buscsb3', 'ef_buscsb4', 'ef_buscsb5', 'ef_buscsb6', 'ef_buscsb7', 'ef_buscsb8', 'ef_buscsb9', 'ef_buscsb10',
        'ef_buscsb11', 'ef_buscsb12', 'ef_buscsb13', 'ef_buscsb14', 'ef_buscsb15', 'ef_buscsb16', 'ef_buscsb17', 'ef_buscsb18', 'ef_buscsb19', 'ef_buscsb20',
        'ef_buscsb21', 'ef_buscsb22',
        'ef_busip1', 'ef_busip2', 'ef_busip3', 'ef_busip4', 'ef_busip5', 'ef_busip6', 'ef_busip7', 'ef_busip8', 'ef_busip9', 'ef_busip10',
        'ef_busip11', 'ef_busip12', 'ef_busip13', 'ef_busip14', 'ef_busip15', 'ef_busip16', 'ef_busip17', 'ef_busip18', 'ef_busip19', 'ef_busip20',
        'ef_busip21', 'ef_busip22', 'ef_busip23', 'ef_busip24', 'ef_busip25', 'ef_busip26', 'ef_busip27', 'ef_busip28', 'ef_busip29', 'ef_busip30',
        'ef_busip31', 'ef_busip32', 'ef_busip33', 'ef_busip34', 'ef_busip35', 'ef_busip36', 'ef_busip37', 'ef_busip38', 'ef_busip39', 'ef_busip40',
        'ef_busip41', 'ef_busip42', 'ef_busip43', 'ef_busip44', 'ef_busip45', 'ef_busip46', 'ef_busip47',
        'ef_pr1', 'cd_tc9', 'cd_tc8', 'cd_tc7', 'cd_tc6', 'cd_tc5', 'cd_tc4', 'cd_tc3', 'cd_tc2', 'cd_tc12', 'cd_tc11', 'cd_tc10', 'cd_tc1',
        'cd_secv9', 'cd_secv8', 'cd_secv7', 'cd_secv6', 'cd_secv5', 'cd_secv4', 'cd_secv3', 'cd_secv2', 'cd_secv12', 'cd_secv11', 'cd_secv10', 'cd_secv1',
        'cd_sech3', 'cd_sech2', 'cd_sech1', 'cd_sech12', 'cd_sech11', 'cd_sech10', 'cd_sech9', 'cd_sech8', 'cd_sech7', 'cd_sech6', 'cd_sech5', 'cd_sech4',
        'cd_ip9', 'cd_ip8', 'cd_ip7', 'cd_ip12', 'cd_ip11', 'cd_ip10',
        'cd_disjuntor9', 'cd_disjuntor8', 'cd_disjuntor7', 'cd_disjuntor6', 'cd_disjuntor5', 'cd_disjuntor4', 'cd_disjuntor3', 'cd_disjuntor2', 'cd_disjuntor12', 'cd_disjuntor11', 'cd_disjuntor10', 'cd_disjuntor1',
        'cd_busip12', 'cd_busip11', 'cd_busip10', 'cd_busip9', 'cd_busip8', 'cd_busip59', 'cd_busip58', 'cd_busip57', 'cd_busip56', 'cd_busip55',
        'cd_busip54', 'cd_busip53', 'cd_busip52', 'cd_busip7', 'cd_busip51', 'cd_busip50', 'cd_busip49', 'cd_busip46', 'cd_busip45', 'cd_busip44',
        'cd_busip43', 'cd_busip42', 'cd_busip41', 'cd_busip6', 'cd_busip40', 'cd_busip39', 'cd_busip38', 'cd_busip37', 'cd_busip36', 'cd_busip35',
        'cd_busip30', 'cd_busip29', 'cd_busip28', 'cd_busip27', 'cd_busip5', 'cd_busip26', 'cd_busip25', 'cd_busip24', 'cd_busip23', 'cd_busip22',
        'cd_busip21', 'cd_busip20', 'cd_busip18', 'cd_busip14', 'cd_busip13', 'cd_busip4',
        'cd_buscsb10', 'cd_buscsb9', 'cd_buscsb8', 'cd_buscsb7', 'cd_buscsb6', 'cd_buscsb5', 'cd_buscsb4', 'cd_buscsb25', 'cd_buscsb24', 'cd_buscsb23',
        'cd_buscsb3', 'cd_buscsb22', 'cd_buscsb21', 'cd_buscsb20', 'cd_buscsb19', 'cd_buscsb18', 'cd_buscsb17', 'cd_buscsb16', 'cd_buscsb15', 'cd_buscsb14',
        'cd_buscsb11', 'cd_buscsb2',
        'cd_reator1', 'cd_reator2', 'cd_reator3', 'cd_reator4', 'cd_reator5', 'cd_reator6',
        'cd_busip3', 'cd_busip1', 'cd_busip2', 'cd_busip16', 'cd_busip17', 'cd_busip19', 'cd_busip31', 'cd_busip15', 'cd_buscsb12', 'cd_busip48', 'cd_busip34', 'cd_busip33', 'cd_busip32', 'cd_busip47', 'cd_buscsb13', 'cd_buscsb1',
        'cd_pr1', 'cd_pr2', 'cd_pr3', 'cd_pr4', 'cd_pr5', 'cd_pr6',
        'cd_ip1', 'cd_ip2', 'cd_ip3', 'cd_ip4', 'cd_ip5', 'cd_ip6',
        'b_tc3', 'b_tc2', 'b_tc1', 'b_tc6', 'b_tc5', 'b_tc4',
        'b_secv3', 'b_secv2', 'b_secv1', 'b_secv6', 'b_secv5', 'b_secv4',
        'b_sech6', 'b_sech5', 'b_sech4', 'b_sech3', 'b_sech2', 'b_sech1',
        'b_ip3', 'b_ip2', 'b_ip1',
        'b_disjuntor3', 'b_disjuntor2', 'b_disjuntor1', 'b_disjuntor6', 'b_disjuntor5', 'b_disjuntor4',
        'b_busip23', 'b_busip22', 'b_busip21', 'b_busip20', 'b_busip19', 'b_busip31', 'b_busip30', 'b_busip29', 'b_busip28', 'b_busip27',
        'b_busip25', 'b_busip24',
        'b_buscsb5', 'b_buscsb4', 'b_buscsb3', 'b_buscsb2', 'b_buscsb1',
        'b_busip36', 'b_busip32', 'b_buscsb6', 'b_busip37', 'b_busip33', 'b_buscsb7', 'b_busip38', 'b_busip34', 'b_buscsb8', 'b_busip39',
        'b_busip35', 'b_busip40',
        'b_busip16', 'b_busip17', 'b_busip18', 'b_busip15', 'b_busip14', 'b_busip13', 'b_busip9', 'b_busip5', 'b_busip1', 'b_busip2',
        'b_busip3', 'b_busip4', 'b_busip8', 'b_busip7', 'b_busip6', 'b_busip10', 'b_busip11', 'b_busip12', 'b_busip26',
        'ls_pr1', 'ls_pr2', 'ls_pr3', 'ls_pr4', 'ls_pr5', 'ls_pr6',
        'ls_tpc1', 'ls_tpc2', 'ls_tpc3', 'ls_tpc4', 'ls_tpc5', 'ls_tpc6',
        'r_pr1', 'r_reator1', 'r_reator2', 'r_pr2'
    ],


    ################ 45 TASKS ###############################
    "example_45": [
        'ef_reator1', 'ef_reator2', 'ef_reator3', 'ef_reator4', 'ef_reator5', 'ef_reator6', 'ef_reator7', 'ef_reator8', 'ef_reator9', 'ef_reator10',
        'ef_pr2', 'ef_pr3', 'ef_pr4', 'ef_pr5', 'ef_pr6', 'ef_pr7','ef_pr11', 'ef_pr12', 'ef_pr13',
        
        'ef_disjuntor1', 'ef_disjuntor2', 'ef_disjuntor3', 'ef_disjuntor7', 'ef_disjuntor8', 'ef_disjuntor9', 'ef_disjuntor10',

        'ef_buscsb21', 'ef_buscsb22',
        'ef_busip1', 'ef_busip2', 'ef_busip3', 'ef_busip4', 'ef_busip5', 'ef_busip10',

        'cd_busip21', 'cd_busip20', 'cd_busip18', 'cd_busip14', 'cd_busip13', 'cd_busip4',
        'cd_buscsb10', 'cd_buscsb9', 'cd_buscsb8', 'cd_buscsb7', 'cd_buscsb4', 'cd_buscsb25', 'cd_buscsb24', 'cd_buscsb23',
    
        'b_tc3', 'b_tc2', 'b_tc1', 'b_tc6', 'b_tc5', 'b_tc4',
        'b_secv3', 'b_secv2', 'b_secv1', 'b_secv6', 'b_secv5', 'b_secv4',
    
    ],

    ################ 42 TASKS ###############################
    "example_42": [
        'ef_reator1','ef_pr2','ef_tpc1','ef_ip1','ef_sech1','ef_sech11','ef_tc1','ef_secv1','ef_disjuntor1','ef_buscsb1','ef_buscsb11','ef_buscsb21','ef_busip1','ef_busip11',
        'ef_busip21','ef_busip31','ef_busip41','ef_pr1','cd_secv9','cd_sech3','cd_ip9','cd_disjuntor9','cd_busip12','cd_busip54','cd_busip43','cd_busip30','cd_busip21','cd_buscsb10',
        'cd_buscsb3','cd_buscsb11','cd_reator1','cd_busip3','cd_pr1','cd_ip1','b_tc3','b_secv3','b_sech6','b_ip3','b_disjuntor3','b_busip23','b_busip25','b_buscsb5',
        'b_busip36','b_busip35','b_busip16','b_busip3','ls_pr1','ls_tpc1','r_pr1',
    ],



    "example_1": [
        'ef_reator1', 'ef_reator2', 'ef_reator3', 'ef_reator4', 'ef_reator5', 'ef_reator6', 'ef_reator7', 'ef_reator8', 'ef_reator9', 'ef_reator10',
        'ef_pr2', 'ef_pr3', 'ef_pr4', 'ef_pr5', 'ef_pr6', 'ef_pr7', 'ef_pr8', 'ef_pr9', 'ef_pr10', 'ef_pr11', 'ef_pr12', 'ef_pr13',
        'ef_tpc1', 'ef_tpc2', 'ef_tpc3',
        'ef_ip1', 'ef_ip2', 'ef_ip3', 'ef_ip4', 'ef_ip5', 'ef_ip6', 'ef_ip7', 'ef_ip8', 'ef_ip9', 'ef_ip10', 'ef_ip11', 'ef_ip12', 'ef_ip13', 'ef_ip14', 'ef_ip15', 'ef_ip16', 'ef_ip17', 'ef_ip18',
        'ef_sech1', 'ef_sech2', 'ef_sech3', 'ef_sech4', 'ef_sech5', 'ef_sech6', 'ef_sech7', 'ef_sech8', 'ef_sech9', 'ef_sech10',
        'ef_sech11', 'ef_sech12', 'ef_sech13', 'ef_sech14', 'ef_sech15', 'ef_sech16', 'ef_sech17', 'ef_sech18', 'ef_sech19', 'ef_sech20', 'ef_sech21', 'ef_sech22', 'ef_sech23', 'ef_sech24',
        'ef_tc1', 'ef_tc2', 'ef_tc3', 'ef_tc4', 'ef_tc5', 'ef_tc6', 'ef_tc7', 'ef_tc8', 'ef_tc9', 'ef_tc10', 'ef_tc11', 'ef_tc12', 'ef_tc13', 'ef_tc14', 'ef_tc15',
        'ef_secv1', 'ef_secv2', 'ef_secv3', 'ef_secv4', 'ef_secv5', 'ef_secv6', 'ef_secv7', 'ef_secv8', 'ef_secv9', 'ef_secv10', 'ef_secv11', 'ef_secv12', 'ef_secv13', 'ef_secv14', 'ef_secv15',
        'ef_disjuntor1', 'ef_disjuntor2', 'ef_disjuntor3', 'ef_disjuntor4', 'ef_disjuntor5', 'ef_disjuntor6', 'ef_disjuntor7', 'ef_disjuntor8', 'ef_disjuntor9', 'ef_disjuntor10']

}
MISSION_PRESET_KEY = "example_22"  # escolha aqui

# --- Execução ---
RUN_MOVNS = True
RUN_BASELINE_CLUSTER = True
SEND_MISSIONS = True
DO_PLOTS = True
SEND_TO_SERVER = True

# --- MOVNS ---
MOVNS_TIME_LIMIT = 5  # segundos

# --- Parâmetros genéricos ---
DEFAULT_BATTERY_TIME = 1_000_000
EXEC_TIME_PER_POINT = 15  # usado no baseline/cluster
CLUSTER_TITLE = "Abordagem Professor Leonardo"
MOVNS_TITLE = "Abordagem MOVNS"

# --- Posições base dos robôs (coordenadas brutas; serão mapeadas para nós do grafo) ---
ALL_ROBOT_COORDS = [
    (-165.9766, -77.6645),  # R1
    (87.9766, 30.6645),     # R2
    (87.9766, 30.6645),     # R3
    (-165.9766, -77.6645),  # R4
    (-165.9766, -77.6645),  # R5
]

SELECTED_ROBOTS = ["R1", "R2", "R3"]


# =====================================================================================
#                              FUNÇÕES DE SUPORTE
# =====================================================================================

def ensure_file(path: str, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo '{label}' não encontrado em: {path}")


def load_data_and_graphs():
    """Carrega JSONs e grafos necessários."""
    print("📥 Carregando grafos e JSONs...")
    ensure_file(GRAPH_JSON, "GRAPH_JSON")
    ensure_file(OBS_POINTS_JSON, "OBS_POINTS_JSON")
    ensure_file(AABB_POR_EQUIPAMENTO_JSON, "AABB_POR_EQUIPAMENTO_JSON")
    ensure_file(EQUIPAMENTOS_FILTRADOS_JSON, "EQUIPAMENTOS_FILTRADOS_JSON")
    ensure_file(AABB_INFO_JSON, "AABB_INFO_JSON")

    observacao_por_obstaculo = SegmentUtils.load_observation_points_from_json(OBS_POINTS_JSON)
    G_mapa = SegmentUtils.load_graph_json(GRAPH_JSON)
    if G_mapa is None or len(G_mapa) == 0:
        raise RuntimeError("G_mapa não carregado ou vazio.")

    G_p = AABBUtils.convert_graph_to_dict(G_mapa)

    with open(AABB_POR_EQUIPAMENTO_JSON, "r") as f:
        equipamento_aabb = json.load(f)
    with open(EQUIPAMENTOS_FILTRADOS_JSON, "r") as f:
        equipamento_data = json.load(f)
    with open(AABB_INFO_JSON, "r") as f:
        aabb_info_data = json.load(f)

    planner_grafo = MultiGraphPlanner(G_p, mission_graph=None, robots_graphs=None,
                                      mission_positions=None, mission_times=None, mission_execution=None)
    cache_astar = PersistentAStarCache(planner_grafo, CACHE_ASTAR_PATH)

    print("✅ Dados carregados.")
    return observacao_por_obstaculo, G_mapa, G_p, equipamento_aabb, equipamento_data, aabb_info_data, planner_grafo, cache_astar


def build_robots(G_mapa: nx.Graph, selected_ids: List[str]) -> Tuple[List[Robot], Dict[str, str], List[int]]:
    """
    Instancia robôs, encontra o nó inicial mais próximo e retorna:
    - robots (objetos Robot)
    - robots_positions: {robot_id: start_node_label}
    - battery_times: [..]
    """
    print("🤖 Instanciando robôs...")
    all_robots_data = {}
    for i, (tx, ty) in enumerate(ALL_ROBOT_COORDS, 1):
        robot_id = f"R{i}"
        label_pos, _, _ = MultiGraphPlanner.find_nearest_node(G_mapa, tx, ty)
        all_robots_data[robot_id] = {
            "position": label_pos,
            "coords": (tx, ty),
            "battery_time": DEFAULT_BATTERY_TIME
        }

    robots: List[Robot] = []
    robots_positions: Dict[str, str] = {}
    battery_times: List[int] = []

    for robot_id in selected_ids:
        data = all_robots_data[robot_id]
        robots_positions[robot_id] = data["position"]
        battery_times.append(data["battery_time"])
        robots.append(Robot(
            id=robot_id,
            battery_time=data["battery_time"],
            start_node_label=data["position"],
            graph=G_mapa
        ))

    print(f"✅ Robôs: {list(robots_positions.keys())}")
    return robots, robots_positions, battery_times


def generate_tasks(missions: List[str],
                   observacao_por_obstaculo: dict,
                   equipamento_aabb: dict,
                   equipamento_data: dict,
                   aabb_info_data: dict):
    """Gera tasks a partir das missões."""
    print(f"🧩 Gerando tasks de {len(missions)} missões...")
    tasks = task_priority_argo.gerar_tasks(
        missions,
        observacao_por_obstaculo,
        equipamento_aabb,
        equipamento_data,
        aabb_info_data
    )
    print(f"✅ {len(tasks)} tasks geradas.")
    return tasks


def build_label_to_coord_map(observacao_por_obstaculo: dict) -> Dict[str, Tuple[float, float]]:
    """Cria índice label -> coord_gps (com checagem)."""
    label_to_coord = {}
    for _, items in observacao_por_obstaculo.items():
        for item in items:
            label = item.get("label")
            coord = item.get("coord_gps")
            if label and coord:
                label_to_coord[label] = coord
    return label_to_coord


def run_movns_pipeline(robots: List[Robot],
                       tasks,
                       G_p: dict,
                       cache_astar: PersistentAStarCache,
                       observacao_por_obstaculo: dict,
                       time_limit: int):
    """Executa MOVNS, escolhe melhor por tempo e converte rotas a GPS."""
    print("🚀 Rodando MOVNS...")
    population = movns_ains_argo.run_movns(robots, tasks, G_p, cache_astar, time_limit=time_limit)
    best_by_time = min(population, key=lambda s: s.time)
    best_by_time.print_solution_metrics()

    rotas_por_robo_movns = solution_priority_argo.calcula_metricas(best_by_time)

    # Converter rotas (labels) para GPS
    label_to_coord = build_label_to_coord_map(observacao_por_obstaculo)
    coords_by_robot: Dict[str, List[Tuple[float, float]]] = {}
    for robot_id, labels in rotas_por_robo_movns.items():
        coords = []
        for label in labels:
            if label in label_to_coord:
                coords.append(label_to_coord[label])
            else:
                print(f"⚠️  Label sem coord_gps: {label}")
        coords_by_robot[robot_id] = coords

    print("✅ MOVNS concluído.")
    return best_by_time, rotas_por_robo_movns, coords_by_robot

# AQUI ESTÁ A TRAJETÓRIA A SER ENVIADA PARA CADA ROBÔ
def send_gps_routes_to_vehicles(coords_by_robot: Dict[str, List[Tuple[float, float]]]):
    """Gera e envia missões MAVLink com ids simples ('1','2',...)."""
    print("📡 Enviando missões (GPS) para os veículos...")
    for robot_key, trajetoria_gps in coords_by_robot.items():
        # robot_key esperado tipo "robot_1" ou "R1" — normalizo para números
        # Casos aceitos: "robot_1" -> "1"; "R3" -> "3"; "3" -> "3"
        key = str(robot_key)
        if "_" in key:
            rid = key.split("_")[-1]
        elif key.upper().startswith("R") and len(key) > 1:
            rid = key[1:]
        else:
            rid = key
        try:
            send_mission_argo.generate_mission(trajetoria_gps, rid)
            print(f"   ✅ Missão enviada para robô {rid} ({robot_key}) com {len(trajetoria_gps)} pontos.")
        except Exception as e:
            print(f"   ❌ Falha ao enviar missão para {robot_key}: {e}")

def send_gps_routes_to_server(coords_by_robot: dict, servidor_url: str, timeout: float = 2.0):
    """
    Envia a trajetória completa de cada robô para o servidor Flask.
    
    Args:
        coords_by_robot: dict {robot_id: [(lat, lon), ...]}
        servidor_url: URL base do servidor Flask (ex: "http://127.0.0.1:5000")
        timeout: tempo máximo de espera por resposta HTTP
    """
    for robot_key, trajetoria_gps in coords_by_robot.items():
        # Normaliza o ID do robô
        if "_" in robot_key:
            rid = robot_key.split("_")[-1]
        elif robot_key.upper().startswith("R") and len(robot_key) > 1:
            rid = robot_key[1:]
        else:
            rid = robot_key

        payload = {
            "robo": rid,
            "trajetoria": [{"latitude": lat, "longitude": lon} for lat, lon in trajetoria_gps]
        }

        try:
            response = requests.post(f"{servidor_url}/waypoints", json=payload, timeout=timeout)
            if response.status_code == 200:
                print(f"✅ Trajetória enviada para robô {rid} ({robot_key})")
            else:
                print(f"❌ Falha ao enviar para robô {rid}: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro de conexão ao enviar trajetória do robô {rid}: {e}")

def prepare_cluster_baseline_and_routes(G_mapa: nx.Graph,
                                        missions: List[str],
                                        robots_positions: Dict[str, str],
                                        rotas_por_robo_movns: Dict[str, List[str]]):
    """Baseline (cluster + TSP vizinho mais próximo) e rotas reais."""
    print("🧭 Gerando baseline (cluster + TSP NN)...")
    mission_positions = MultiGraphPlanner.gerar_mission_positions_from_json(
        SegmentUtils.load_observation_points_from_json(OBS_POINTS_JSON),
        missions
    )

    # Mapear cada ponto como missão individual
    point_mission_positions = {}
    for mission, points in mission_positions.items():
        for point in points:
            point_mission_positions[point] = [point]

    # Grafo reduzido para inspeção (pode falhar se algum start não conectável; tratamos)
    start_nodes = list(robots_positions.values())
    Greduced_map = MultiGraphPlanner.build_inspection_graph(start_nodes, point_mission_positions, G_mapa)
    if Greduced_map is None or len(Greduced_map) == 0:
        print("⚠️ Greduced_map vazio. Verifique conectividade entre nós iniciais e pontos.")
        return None, None, None, None

    # Quem pode executar cada missão/ponto (aqui: todos os robôs)
    robot_ids = list(robots_positions.keys())
    mission_execution = {mission: robot_ids for mission in missions}
    possible_robots_per_execution_point = {}
    for mission, robots in mission_execution.items():
        for point in mission_positions.get(mission, []):
            possible_robots_per_execution_point[point] = robots

    # Clusterização balanceada
    pontos_por_robo = FixedTaskPlanner.clusterizar_pontos_balanceado(
        Greduced_map, point_mission_positions, robots_positions, possible_robots_per_execution_point
    )

    # Rotas ótimas por vizinho mais próximo
    rotas_otimas_por_robo = {}
    for robo, pontos in pontos_por_robo.items():
        start = robots_positions.get(robo)
        if not start:
            print(f"⚠️ Sem posição inicial para {robo}; ignorando no TSP.")
            continue
        rota = FixedTaskPlanner.tsp_nearest_neighbor(Greduced_map, start, pontos)
        rotas_otimas_por_robo[robo] = rota

    print("✅ Baseline gerada.")
    return Greduced_map, point_mission_positions, rotas_otimas_por_robo, mission_positions


def plot_all(G_mapa: nx.Graph,
             Greduced_map: nx.Graph,
             rotas_otimas_por_robo: Dict[str, List[str]],
             rotas_por_robo_movns: Dict[str, List[str]],
             point_mission_positions: Dict[str, List[str]]):
    """Centraliza plots; pode ser desligado pelos toggles."""
    if not DO_PLOTS:
        return

    if Greduced_map and rotas_otimas_por_robo:
        try:
            PlotUtils.plot_rotas_grafo(Greduced_map, rotas_otimas_por_robo)
        except Exception as e:
            print(f"⚠️ plot_rotas_grafo falhou: {e}")

        try:
            PlotUtils.plot_rotas_reais(G_mapa, rotas_otimas_por_robo, point_mission_positions, CLUSTER_TITLE)
        except Exception as e:
            print(f"⚠️ plot_rotas_reais (baseline) falhou: {e}")

    if rotas_por_robo_movns:
        try:
            PlotUtils.plot_rotas_reais(G_mapa, rotas_por_robo_movns, point_mission_positions, MOVNS_TITLE)
        except Exception as e:
            print(f"⚠️ plot_rotas_reais (MOVNS) falhou: {e}")


# =====================================================================================
#                                      MAIN
# =====================================================================================

def mrta(missions, selected_robots):
    # 1) Carregar dados
    (observacao_por_obstaculo, G_mapa, G_p,
     equipamento_aabb, equipamento_data, aabb_info_data,
     planner_grafo, cache_astar) = load_data_and_graphs()

    # 2) Missões
    # missions = MISSION_PRESETS[MISSION_PRESET_KEY]
    print(f"📌 Usando preset de missões '{MISSION_PRESET_KEY}' ({len(missions)} itens).")

    # 3) Geração de tasks
    tasks = generate_tasks(missions, observacao_por_obstaculo, equipamento_aabb, equipamento_data, aabb_info_data)

    # 4) Robôs
    # robots, robots_positions, _battery_times = build_robots(G_mapa, SELECTED_ROBOTS)
    robots, robots_positions, _battery_times = build_robots(G_mapa, selected_robots)

    # 5) MOVNS (opcional)
    best_by_time = None
    rotas_por_robo_movns = {}
    coords_by_robot = {}

    if RUN_MOVNS:
        best_by_time, rotas_por_robo_movns, coords_by_robot = run_movns_pipeline(
            robots, tasks, G_p, cache_astar, observacao_por_obstaculo, MOVNS_TIME_LIMIT
        )

        # Executar plano no planner se desejar
        try:
            planner_grafo.execute_plan_from_solution(best_by_time)
        except Exception as e:
            print(f"⚠️ execute_plan_from_solution falhou: {e}")

        if SEND_MISSIONS and coords_by_robot:
            send_gps_routes_to_vehicles(coords_by_robot)
        
        # --- Envio para o servidor Flask ---
        if SEND_TO_SERVER and coords_by_robot:
            SERVER_URL = "http://127.0.0.1:5000"  # ajuste para o seu servidor
            send_gps_routes_to_server(coords_by_robot, SERVER_URL)

    # 6) Baseline cluster + TSP (opcional)
    Greduced_map = None
    point_mission_positions = None
    rotas_otimas_por_robo = None
    mission_positions = None

    if RUN_BASELINE_CLUSTER:
        (Greduced_map,
         point_mission_positions,
         rotas_otimas_por_robo,
         mission_positions) = prepare_cluster_baseline_and_routes(
            G_mapa, missions, robots_positions, rotas_por_robo_movns
        )

        # Métricas do baseline (se existir)
        if rotas_otimas_por_robo:
            try:
                tempo, distancia, balances, qtde_pontos = solution_priority_argo.calcular_custos_totais_solucao(
                    rotas_otimas_por_robo, cache_astar
                )
                print(f"📊 Baseline — Tempo: {tempo:.2f}, Distância: {distancia:.2f}, Balance: {balances}, Pontos: {qtde_pontos}")
            except Exception as e:
                print(f"⚠️ calcular_custos_totais_solucao falhou: {e}")

    # 7) Plots
    plot_all(G_mapa, Greduced_map, rotas_otimas_por_robo, rotas_por_robo_movns, point_mission_positions or {})

    print("🏁 Pipeline finalizado.")


if __name__ == "__main__":
    mrta(MISSION_PRESETS[MISSION_PRESET_KEY], SELECTED_ROBOTS)









    # # Inicializa o planejador
    # method = "nearest" # "permutation"   #  "branch_bound" #  ou  or
    # planner = FixedTaskPlanner(grafo_mapa_dict, point_mission_positions, mission_times, mission_execution)
    #
    #
    # # Executa o planejamento de tarefas
    # plan, total_time, schedule = planner.distribute_and_schedule_tasks(
    #     robots_positions,
    #     fixed_tasks_per_robot,
    #     distributed_tasks,
    #     method="nearest" #"branch_bound"
    # )
    #
    # # Mostra o plano resultante
    # print(f"\n🔧 Tempo total de execução (makespan): {total_time:.2f} s")
    # for robot, tasks in plan.items():
    #     print(f"\n📦 Plano para {robot}:")
    #     for task in tasks:
    #         print(f"  ▶ Missão: {task['mission']}, Caminho: {task['path']}, "
    #               f"Deslocamento: {task['travel_time']}s, Execução: {task['execution_time']}s, "
    #               f"Início: {task['start_time']:.2f}, Fim: {task['end_time']:.2f}")
    #
    # # Geração de dicionário com os pontos atribuídos por robô
    # execution_points_per_robot = defaultdict(list)
    # for robot, tasks in plan.items():
    #     for t in tasks:
    #         execution_points_per_robot[robot].append(t["mission"])
    #
    # print("\n🗺️ Pontos de execução por robô:")
    # for robot, points in execution_points_per_robot.items():
    #     print(f"  {robot}: {points}")











    # # Inicializa o planejador
    # method = "nearest" # "permutation"   #  "branch_bound" #  ou  or
    # planner = FixedTaskPlanner(grafo_mapa_dict, point_mission_positions, mission_times, mission_execution)
    #
    #
    # # Executa o planejamento de tarefas
    # plan, total_time, schedule = planner.distribute_and_schedule_tasks(
    #     robots_positions,
    #     fixed_tasks_per_robot,
    #     distributed_tasks,
    #     method="nearest" #"branch_bound"
    # )
    #
    # # Mostra o plano resultante
    # print(f"\n🔧 Tempo total de execução (makespan): {total_time:.2f} s")
    # for robot, tasks in plan.items():
    #     print(f"\n📦 Plano para {robot}:")
    #     for task in tasks:
    #         print(f"  ▶ Missão: {task['mission']}, Caminho: {task['path']}, "
    #               f"Deslocamento: {task['travel_time']}s, Execução: {task['execution_time']}s, "
    #               f"Início: {task['start_time']:.2f}, Fim: {task['end_time']:.2f}")
    #
    # # Geração de dicionário com os pontos atribuídos por robô
    # execution_points_per_robot = defaultdict(list)
    # for robot, tasks in plan.items():
    #     for t in tasks:
    #         execution_points_per_robot[robot].append(t["mission"])
    #
    # print("\n🗺️ Pontos de execução por robô:")
    # for robot, points in execution_points_per_robot.items():
    #     print(f"  {robot}: {points}")
