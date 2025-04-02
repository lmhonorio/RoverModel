from multigraphplanner import MultiGraphPlanner
from segmentutils import SegmentUtils
import networkx as nx
import heapq
from itertools import permutations
import matplotlib.pyplot as plt
from aabbutils import AABBUtils
from plotutils import PlotUtils
import time
from collections import defaultdict
# Visualiza rapidamente
import math
import re

#
# def dict_to_nx_graph(grafo_mapa):
#     """
#     Converte um dicionário 'grafo_mapa' (states + transitions)
#     em um grafo NetworkX não-direcionado, com os pesos em 'weight'.
#     """
#     G = nx.Graph()
#     for state in grafo_mapa["states"]:
#         G.add_node(state)
#     for (orig, dst), (dist, _) in grafo_mapa["transitions"].items():
#         G.add_edge(orig, dst, weight=dist)
#     return G
#
# def find_nearest_node(G_loaded, tx, ty):
#     """
#     Encontra o nó mais próximo de (tx, ty) com base em G.nodes[n]['pos'] = (x, y).
#     Retorna (nó, distancia).
#     Se nenhum nó tiver 'pos', retornará (None, float('inf')).
#     """
#     # G_loaded pode ser (a) um dicionário { 'states', 'transitions'} ou (b) um nx.Graph
#     if isinstance(G_loaded, dict) and "states" in G_loaded and "transitions" in G_loaded:
#         # então converter para nx.Graph
#         G = dict_to_nx_graph(G_loaded)
#     elif isinstance(G_loaded, nx.Graph):
#         # já é grafo Nx
#         G = G_loaded
#     else:
#         raise ValueError("Formato de graph7.json inesperado. Verifique seu pipeline.")
#
#     # 2) Para cada nó, parsear o label (que é o nome do nó)
#     #    Exemplo: se for "(-165.9766, -77.6645)" iremos extrair x=-165.9766, y=-77.6645
#     for node in G.nodes():
#         coords = parse_label_to_xy(str(node))  # 'node' em string
#         if coords is not None:
#             G.nodes[node]["pos"] = coords
#
#
#     nearest = None
#     min_dist = float('inf')
#     for node in G.nodes:
#         if "pos" not in G.nodes[node]:
#             continue  # Ignora nós sem atributo pos
#
#         x_node, y_node = G.nodes[node]["pos"]
#         dist = math.hypot(x_node - tx, y_node - ty)
#         if dist < min_dist:
#             min_dist = dist
#             nearest = node
#
#     label = G.nodes[nearest].get("label", str(nearest))
#     return label, nearest, min_dist
#
# def parse_label_to_xy(label):
#     """
#     Tenta parsear o label do nó no formato:
#       "(-165.9766, -77.6645)"
#     e retornar (x, y) como floats. Retorna None se não conseguir.
#     """
#     pattern = r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)"
#     match = re.match(pattern, label)
#     if match:
#         x_str = match.group(1)
#         y_str = match.group(2)
#         return float(x_str), float(y_str)
#     else:
#         return None
#
#
# def build_inspection_graph(start_point, inspection_points, grafo_mapa):
#     """
#     Cria um grafo NetworkX (G_robot) com nós = {start_point} U {inspection_points}.
#     Para cada par de nós (u, v), calcula o menor caminho no grafo_mapa.
#     Se esse caminho não tiver nenhum outro nó de {start_point} + inspection_points
#     no meio (ou seja, excluindo u e v), então cria aresta (u, v) em G_robot,
#     com peso igual à soma das distâncias do menor caminho.
#     """
#     # 1) Converter o grafo_mapa (dicionário) para um grafo NetworkX
#     G_env = grafo_mapa
#
#     # 2) Conjunto de todos os "estados" que nos interessam
#     states_of_interest = set(inspection_points)
#     states_of_interest.add(start_point)
#
#     # 3) Criar um grafo vazio para retornar
#     G_robot = nx.Graph()
#     # Adicionar nós
#     for st in states_of_interest:
#         G_robot.add_node(st)
#
#     # 4) Vamos testar pares (u, v) usando combinações
#     from itertools import combinations
#     for u, v in combinations(states_of_interest, 2):
#         # Tenta achar caminho mais curto no G_env
#         try:
#             path = nx.shortest_path(G_env, source=u, target=v, weight="weight")
#             dist = nx.shortest_path_length(G_env, source=u, target=v, weight="weight")
#         except nx.NetworkXNoPath:
#             # Não existe caminho
#             continue
#
#         # Verifica se existe algum outro estado de interesse no meio do caminho
#         # path[1:-1] = nós intermediários
#         intermediarios = set(path[1:-1])
#         if intermediarios.intersection(states_of_interest):
#             # Se tiver intersecção, significa que passaria por outro estado
#             # que também nos interessa --> não criamos essa aresta
#             continue
#
#         # Caso não tenha nenhum estado de interesse no meio,
#         # adicionamos a aresta com o peso (dist)
#         G_robot.add_edge(u, v, weight=dist)
#
#         # (Opcional) se quiser guardar o caminho completo no atributo:
#         # G_robot[u][v]['path'] = path
#
#     return G_robot
#
#



if __name__ == "__main__":

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




    #transforma as missoes em pontos de observacao individuais

    # Use:
    point_mission_positions = {}
    for mission, points in mission_positions.items():
        for point in points:
            point_mission_positions[point] = [point]  # Cada ponto é sua própria posição alvo

    G_m.add_nodes_from([ponto for pontos in mission_positions.values() for ponto in pontos])

    point_mission_times = {ponto: 15 for pontos in mission_positions.values() for ponto in pontos}

    execution_points_per_robot = defaultdict(list)

    for mission, robots in mission_execution.items():
        pontos = mission_positions[mission]
        for robo in robots:
            execution_points_per_robot[robo].extend(pontos)

    print(execution_points_per_robot)

    G_robot = {}
    G_robot["R1"] = MultiGraphPlanner.build_inspection_graph(robots_positions['R1'], execution_points_per_robot['R1'], G_mapa)
    G_robot["R2"] = MultiGraphPlanner.build_inspection_graph(robots_positions['R2'], execution_points_per_robot['R2'], G_mapa)

    PlotUtils.plot_grafo_distance(G_robot["R1"])
    PlotUtils.plot_grafo_distance(G_robot["R2"])

    dfaR1 = MultiGraphPlanner.graph_to_dfa_bidirectional(G_robot["R1"],"R1",robots_positions['R1'], execution_points_per_robot['R1'])
    dfaR2 = MultiGraphPlanner.graph_to_dfa_bidirectional(G_robot["R2"],"R2",robots_positions['R2'], execution_points_per_robot['R2'])

    # MultiGraphPlanner.imprimir_multidigraph(G_robot["R1"])
    #
    # print(dfaR1)

    R1 = MultiGraphPlanner.xml_to_graph(dfaR1)
    R2 = MultiGraphPlanner.xml_to_graph(dfaR2)
    # PlotUtils.plot_grafo_distance(R1)
    # PlotUtils.plot_grafo_distance(R2)

    G = MultiGraphPlanner.parallel_composition(R1, R2)

    print("grafo processado")
    PlotUtils.plot_mission_graph(G)




