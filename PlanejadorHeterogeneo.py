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

#leitura do grafo
file_path = "./jsons/graph9_new.json"

#pontos de observacao em relacao a distancia dos objetos
observation_points_json_path = "./jsons/obp_6.json"

json_mission = "./jsons/missao_6.json"

observacao_por_obstaculo  = SegmentUtils.load_observation_points_from_json(observation_points_json_path)

G_mapa = SegmentUtils.load_graph_json(file_path)

G_p = AABBUtils.convert_graph_to_dict(G_mapa)

missions = ['b_busip4','ef_reator1','ls_pr4']
mission_positions  = MultiGraphPlanner.gerar_mission_positions_from_json(observacao_por_obstaculo,missions)
mission_execution = {"b_busip4": ["R1", "R2"], "ef_reator1": ["R1", "R2"], "ls_pr4": ["R2"]}


tx, ty = -165.9766, -77.6645
tx2, ty2 = 87.9766, 30.6645

label_posr1, nearest, dist = MultiGraphPlanner.find_nearest_node(G_mapa, tx, ty)
label_posr2, nearest2, dist2 = MultiGraphPlanner.find_nearest_node(G_mapa, tx2, ty2)

robots_positions = {"R1": label_posr1, "R2": label_posr2}

print(robots_positions)

print(mission_positions)




#transforma as missoes em pontos de observacao individuais

# Use:
point_mission_positions = {}
for mission, points in mission_positions.items():
    for point in points:
        point_mission_positions[point] = [point]  # Cada ponto é sua própria posição alvo

G_m.add_nodes_from([ponto for pontos in mission_positions.values() for ponto in pontos])

point_mission_times = {ponto: 15 for pontos in mission_positions.values() for ponto in pontos}

possible_robots_per_execution_point = {}
for mission, robots in mission_execution.items():
    for point in mission_positions[mission]:
        possible_robots_per_execution_point[point] = robots
#





start = time.time()
planner = MultiGraphPlanner(G_p, G_m, G_r, point_mission_positions, point_mission_times, possible_robots_per_execution_point)
optimal_plan, min_time, schedule = planner.find_minimum_mission_time_plan(robots_positions)
end = time.time()
print(f"⏱ Tempo de execução: {end - start:.4f} segundos")



planner.save_optimal_plan_to_json(optimal_plan,json_mission)

planner.execute_plan(optimal_plan, min_time, schedule)
