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

def RetornaVizinhos(automatonG, nodePai, CaminhoAtual, pesoAtual):
  vizinhos = []
  neighb = nx.neighbors(automatonG, nodePai)

  edges = nx.edges(automatonG, nodePai)
  for edge in edges:
    edge_data = automatonG.get_edge_data(edge[0], edge[1])
    keys = edge_data.keys()
    for key in keys:
      weight = edge_data[key]['weight'] + pesoAtual
      caminho = f'{CaminhoAtual};{edge[1]}'
      item = {caminho: (edge[1],weight)}
      vizinhos.append(item)

  return vizinhos

def iteracaoAstar(Ga,No, CaminhoAtual, PesoAtual, Estados_Abertos):
  A = RetornaVizinhos(Ga,No, CaminhoAtual, PesoAtual)
  for item in A:
    Estados_Abertos.append(item)
  lista_ordenada = sorted(Estados_Abertos, key=lambda Estados_Abertos: list(Estados_Abertos.values())[0][1])
  Estado_corrente = lista_ordenada.pop(0)
  Estados_Abertos.remove(Estado_corrente)
  CaminhoAtual = list(Estado_corrente.keys())[0]
  No, PesoAtual = list(Estado_corrente.values())[0]
  return (CaminhoAtual, No, PesoAtual)


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

    missions = ['b_busip4','ef_reator1'] #,'ls_pr4']
    mission_positions  = MultiGraphPlanner.gerar_mission_positions_from_json(observacao_por_obstaculo,missions)
    mission_execution = {"b_busip4": ["R1","R2"], "ef_reator1": ["R1","R2"]} #, "ls_pr4": ["R2"]}

    print(mission_positions)

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




    G_missoes = MultiGraphPlanner.gerar_grafos_execucao(execution_points_per_robot)
    Gt = MultiGraphPlanner.parallel_composition_multiple_fast(G_missoes)


    print("Grafo de missoes pronto")

    # PlotUtils.plot_grafo_distance(Gt)



    G_robot = {}
    G_robot["R1"] = MultiGraphPlanner.build_inspection_graph(robots_positions['R1'], execution_points_per_robot['R1'], G_mapa)
    G_robot["R2"] = MultiGraphPlanner.build_inspection_graph(robots_positions['R2'], execution_points_per_robot['R2'], G_mapa)


    dfaR1 = MultiGraphPlanner.graph_to_dfa_bidirectional("R1",G_robot,robots_positions, execution_points_per_robot)
    dfaR2 = MultiGraphPlanner.graph_to_dfa_bidirectional("R2",G_robot,robots_positions, execution_points_per_robot)


    R1 = MultiGraphPlanner.xml_to_graph(dfaR1)
    R2 = MultiGraphPlanner.xml_to_graph(dfaR2)

    G_robot = MultiGraphPlanner.parallel_composition(R1, R2)

    print("Grafo robo pronto")

    Gt = MultiGraphPlanner.parallel_composition(G_robot, Gt)


    print("Composicao final pronta")

    start_state, accepting_states = MultiGraphPlanner.get_start_and_accepting_states(Gt)

    print(f"grafo processado: start={start_state}, accepting={accepting_states}")

    nome = 'automatosinal'
    tipo = '.png'
    CaminhoAtual = 'S'
    No = 'S'
    EstadoFinal = 'Go'
    PesoAtual = 0
    Estados_Abertos = []
    rodar = True
    iteracao = 0

    while rodar:
        iteracao = iteracao + 1
        (CaminhoAtual, No, PesoAtual) = iteracaoAstar(Ga, No, CaminhoAtual, PesoAtual, Estados_Abertos)
        Gc = Ga.copy()
        listacaminhos = CaminhoAtual.split(';')
        nomefinal = nome + str(iteracao) + tipo

        nlista = len(listacaminhos)
        for i in range(0, nlista - 1):
            for edge in Gc.edges:
                if edge[0] == listacaminhos[i] and edge[1] == listacaminhos[i + 1]:
                    Gc.edges[edge]['color'] = 'red'

        for no in listacaminhos:
            Gc.nodes[no]['color'] = 'blue'
            Gc.nodes[no]['fillcolor'] = 'lightblue'

        for dicionario in Estados_Abertos:
            # Use o método values() para obter os valores do dicionário e adicione-os à lista
            no = list(dicionario.values())[0][0]
            Gc.nodes[no]['color'] = 'blue'
            Gc.nodes[no]['fillcolor'] = 'lightgreen'

        # for no in Estados_Abertos:
        #   Gc.nodes[no]['color'] = 'green'
        #   Gc.nodes[no]['fillcolor'] = 'lighgreen'

        plotgraf(Gc, nomefinal)

        # print(Estados_Abertos)
        print(iteracao, CaminhoAtual, PesoAtual)

        if No == EstadoFinal:
            print('menor caminho encontrado')
            print(CaminhoAtual)
            print('Custo', PesoAtual)

        if len(Estados_Abertos) == 0 or No == EstadoFinal or iteracao >= 100:
            rodar = False





