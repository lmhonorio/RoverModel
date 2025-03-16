from multigraphplanner import MultiGraphPlanner
from segmentutils import SegmentUtils
import networkx as nx
import heapq
from itertools import permutations
import matplotlib.pyplot as plt
from aabbutils import AABBUtils


# G_p = nx.Graph()
G_m = nx.DiGraph()
G_r = {"R1": nx.Graph(), "R2": nx.Graph()}

file_path = "./jsons/graph8.json"
G_mapa = SegmentUtils.load_graph_json(file_path)

G_p = AABBUtils.convert_graph_to_dict(G_mapa)


#
# # 📌 Adicionando nós ao grafo das posições
# positions = {
#     "P1": (0, 0), "P2": (1, 0), "P3": (2, 0),
#     "P4": (0, 1), "P5": (1, 1), "P6": (2, 1)
# }
#
# for pos, coords in positions.items():
#     G_p.add_node(pos, pos=coords)
#
# # 📌 Adicionando arestas ao grafo de posições com tempos de deslocamento
# edges = [
#     ("P1", "P2", 4), ("P2", "P3", 2),
#     ("P1", "P4", 3), ("P2", "P5", 6), ("P3", "P6", 3),
#     ("P4", "P5", 2), ("P5", "P6", 4), ("P1", "P7", 2)
# ]
#
# for u, v, w in edges:
#     G_p.add_edge(u, v, weight=w)
#
# # 📌 Criando o grafo de missões (dependências)
G_m.add_edges_from([("M_A", "M_B"), ("M_B", "M_C"), ("M_C", "M_D")])
#
# # 📌 Ligando missões às posições no mapa
mission_positions = {"M_A": "PR13_7", "M_B": "TPC2_4", "M_C": "TPC3_0", "M_D": "PR12_4"}
#
# # 📌 Tempo de execução de cada missão (em segundos)
mission_times = {"M_A": 2, "M_B": 3, "M_C": 3, "M_D": 2}
#
# # 📌 Restrições de execução de missões
mission_execution = {"M_A": ["R1", "R2"], "M_B": ["R1", "R2"], "M_C": ["R1", "R2"], "M_D": ["R1","R2"]}
#
robots_positions = {"R1": "PR11_0", "R2": "PR11_2"}

planner = MultiGraphPlanner(G_p, G_m, G_r, mission_positions, mission_times, mission_execution)
optimal_plan, min_time, schedule = planner.find_minimum_mission_time_plan(robots_positions)
planner.execute_plan(optimal_plan, min_time, schedule)
