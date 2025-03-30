from multigraphplanner import MultiGraphPlanner
from segmentutils import SegmentUtils
import networkx as nx
import heapq
from itertools import permutations
import matplotlib.pyplot as plt
from aabbutils import AABBUtils
from plotutils import PlotUtils
import time




# G_p = nx.Graph()
G_m = nx.DiGraph()
G_r = {"R1": nx.Graph(), "R2": nx.Graph()}

file_path = "./jsons/graph6.json"
json_path = "./jsons/obp_6.json"
json_mission = "./jsons/missao_6.json"
observacao_por_obstaculo  = SegmentUtils.load_observation_points_from_json(json_path)

G_mapa = SegmentUtils.load_graph_json(file_path)

G_p = AABBUtils.convert_graph_to_dict(G_mapa)

missions = ['b_busip4','ls_pr4','ef_reator1']
mission_positions  = MultiGraphPlanner.gerar_mission_positions_from_json(observacao_por_obstaculo,missions)
mission_execution = {"b_busip4": ["R1", "R2"], "ef_reator1": ["R1", "R2"], "ls_pr4": ["R1"]}
robots_positions = {"R1": "cd_ip10.92.5.1398", "R2": "ef_buscsb14.58.2.845"}


#transforma as missoes em pontos de observacao individuais

point_mission_possitions = {ponto:[ponto] for pontos in mission_positions.values() for ponto in pontos}
G_m.add_nodes_from([ponto for pontos in mission_positions.values() for ponto in pontos])  # Sem arestas!

# # 📌 Tempo de execução de cada missão (em segundos)
point_mission_times = {ponto: 15 for pontos in mission_positions.values() for ponto in pontos}

possible_robots_per_execution_point = {
    ponto: mission_execution[missao]
    for missao, pontos in mission_positions.items()
    for ponto in pontos
}
#

start = time.time()
planner = MultiGraphPlanner(G_p, G_m, G_r, point_mission_possitions, point_mission_times, possible_robots_per_execution_point)
optimal_plan, min_time, schedule = planner.find_minimum_mission_time_plan_par(robots_positions)
end = time.time()
print(f"⏱ Tempo de execução: {end - start:.4f} segundos")

plano_missoes = planner.convert_plan_to_dict(optimal_plan)

planner.save_plan_dict_to_json(plano_missoes,json_mission)

# planner.execute_plan(optimal_plan, min_time, schedule)
