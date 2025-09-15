from multigraphplanner import MultiGraphPlanner
from segmentutils import SegmentUtils
from plotutils import PlotUtils
from collections import defaultdict
from ajusteplanilha import AjustePlanilha
from tspOptimization import FixedTaskPlanner
from aabbutils import AABBUtils
from collections import OrderedDict, defaultdict
import math
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from networkx.drawing.nx_agraph import to_agraph
from pygraphviz import AGraph
import networkx as nx
import matplotlib.image as mpimg
import os
import subprocess


R = 6378137.0  # raio WGS84

def _to_xy_m(lat: float, lon: float, lat0: float, lon0: float):
    """Converte (lat,lon) para coords locais (x,y) em metros, referenciadas em (lat0,lon0)."""
    x = (lon - lon0) * (math.pi/180.0) * R * math.cos(math.radians(lat0))
    y = (lat - lat0) * (math.pi/180.0) * R
    return x, y

def _get_hold_value(wp: dict) -> float:
    """Descobre o campo de 'hold' mais provável no waypoint."""
    for k in ('hold', 'holding', 'hold_time', 'delay', 'loiter', 'hold_vistoria'):
        if k in wp and wp[k] is not None:
            try:
                return float(wp[k])
            except Exception:
                pass
    return 0.0

def _eq_m(p: dict, q: dict, thr_m: float = 0.05) -> bool:
    """Compara se p e q são praticamente o mesmo ponto (em metros)."""
    x, y = _to_xy_m(q['lat'], q['lon'], p['lat'], p['lon'])
    return math.hypot(x, y) <= thr_m

def _is_collinear_and_between(a: dict, b: dict, c: dict, tol_ct_m: float) -> bool:
    """
    Retorna True se B está colinear com A-C e ENTRE A e C.
    Usa distância ortogonal de B à reta AC (cross-track).
    """
    # A é a origem
    bx, by = _to_xy_m(b['lat'], b['lon'], a['lat'], a['lon'])
    cx, cy = _to_xy_m(c['lat'], c['lon'], a['lat'], a['lon'])
    ACx, ACy = cx, cy
    ABx, ABy = bx, by
    ACn2 = ACx*ACx + ACy*ACy
    ACn = math.sqrt(ACn2)
    if ACn < 1e-9:
        # A e C coincidem: não removemos para não “colar” pontos
        return False

    # Distância ortogonal de B até a linha de A->C
    cross = abs(ABx*ACy - ABy*ACx)
    d = cross / ACn
    if d > tol_ct_m:
        return False

    # Verifica se B projeta ENTRE A e C (0 <= t <= 1)
    t = (ABx*ACx + ABy*ACy) / ACn2
    return 0.0 <= t <= 1.0

def otimizarpontos(missoes_por_robo: Dict[str, List[Dict[str, Any]]],
                   tol_ct_m: float = 0.10,
                   preserve_loop_closure: bool = True,
                   renumber_ids: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """
    Remove waypoints de passagem (hold==0) que estejam em linha reta entre vizinhos.
    - tol_ct_m: tolerância da distância ortogonal para considerar "em linha" (m).
    - preserve_loop_closure: mantém o último se ele repete o primeiro (fecho).
    - renumber_ids: se True, reenumera 'id' sequencialmente após a limpeza.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}

    for robo, wps in missoes_por_robo.items():
        if len(wps) <= 2:
            out[robo] = list(wps)
            continue

        # Detecta fecho (último ~= primeiro) para não quebrar o circuito
        has_loop = preserve_loop_closure and _eq_m(wps[0], wps[-1], thr_m=0.05)
        last_index = (len(wps) - 2) if has_loop else (len(wps) - 1)

        kept: List[Dict[str, Any]] = []
        for i, wp in enumerate(wps):
            # Sempre mantém primeiro e último "útil"
            if i == 0 or i == last_index or not (0 < i < last_index):
                kept.append(wp)
                continue

            # Mantém pontos com hold > 0 (vistoria)
            if _get_hold_value(wp) > 0.0:
                kept.append(wp)
                continue

            a, b, c = wps[i-1], wp, wps[i+1]
            if _is_collinear_and_between(a, b, c, tol_ct_m=tol_ct_m):
                # descarta b (ponto de passagem em reta)
                continue
            else:
                kept.append(wp)

        # Reanexa o fecho, se existia
        if has_loop:
            kept.append(wps[-1])

        # Renumera IDs, se desejado
        if renumber_ids and any('id' in k for k in kept):
            for j in range(len(kept)):
                if 'id' in kept[j]:
                    kept[j] = dict(kept[j])
                    kept[j]['id'] = j

        out[robo] = kept

    return out

@staticmethod
def retorna_rotas_completas(G_robot, rotas_por_robo, pontos_vistoria, xlsx_param_path):
    """
    Retorna no formato:
    {
      "R1": [
        {
          "mission": "b_busip4",
          "tasks": [
            {
              "task": 0,
              "point": "b_busip4.2493",
              "path": ["n0","n1",...],
              "path_gps": [{"lat": -3.12, "lon": -41.76}, ...],
              "distance": 12.34
            }, ...
          ]
        }, ...
      ],
      "R2": [ ... ]
    }
    """
    def euclid(u, v):
        xa, ya = G_robot.nodes[u]['pos']
        xb, yb = G_robot.nodes[v]['pos']
        return math.hypot(xb - xa, yb - ya)

    def path_distance(nodes):
        d = 0.0
        for a, b in zip(nodes, nodes[1:]):
            if G_robot.has_edge(a, b) and 'weight' in G_robot[a][b]:
                d += float(G_robot[a][b]['weight'])
            else:
                d += euclid(a, b)
        return float(d)

    def nodes_to_xy(nodes):
        return [tuple(G_robot.nodes[n]['pos']) for n in nodes]

    def xy_to_gps(xy_list):
        # AjustePlanilha espera uma lista de pares (x,y)
        gps_list = AjustePlanilha.metros_para_geocoordenadas(xy_list, xlsx_param_path)
        return [{"lat": float(lat), "lon": float(lon)} for (lat, lon) in gps_list]

    resultados_por_robo = {}

    for robo, rota in rotas_por_robo.items():
        # 1) caminho completo concatenando caminhos mínimos (ponderados por 'weight')
        caminho_completo = []
        for j in range(len(rota) - 1):
            u, v = rota[j], rota[j + 1]
            subpath = nx.shortest_path(G_robot, source=u, target=v, weight="weight")
            if caminho_completo and subpath[0] == caminho_completo[-1]:
                caminho_completo.extend(subpath[1:])
            else:
                caminho_completo.extend(subpath)

        if not caminho_completo:
            resultados_por_robo[robo] = []
            continue

        # 2) índices dos pontos de vistoria na ordem do caminho
        pontos_vistoria_set = set(pontos_vistoria)
        idx_vistoria = [i for i, n in enumerate(caminho_completo) if n in pontos_vistoria_set]
        if not idx_vistoria:
            resultados_por_robo[robo] = []
            continue

        # 3) agrupa “tasks” por missão (prefixo antes do ".")
        tasks_by_mission = OrderedDict()
        task_counters = defaultdict(int)

        start_idx = 0
        for i in idx_vistoria:
            ponto = caminho_completo[i]
            mission_name = ponto.split('.')[0] if isinstance(ponto, str) and '.' in ponto else "mission"

            # subcaminho desde o último marco até este ponto (inclusive)
            subnodes = caminho_completo[start_idx:i+1]
            dist = path_distance(subnodes)

            # paths
            sub_xy = nodes_to_xy(subnodes)
            sub_gps = xy_to_gps(sub_xy)

            task_id = task_counters[mission_name]
            task_counters[mission_name] += 1

            tasks_by_mission.setdefault(mission_name, []).append({
                "task": int(task_id),
                "point": ponto,
                "path": list(subnodes),            # nós do grafo
                "path_gps": sub_gps,               # lista de {lat, lon}
                "distance": float(dist)            # em metros
            })

            start_idx = i  # próximo trecho começa neste ponto

        # 4) monta a lista de missions
        missions = [{"mission": mname, "tasks": tlist} for mname, tlist in tasks_by_mission.items()]
        resultados_por_robo[robo] = missions

    return resultados_por_robo



if __name__ == "__main__":
    # Leitura do grafo do ambiente
    # file_path = "./jsons/graph9_new.json"
    # observation_points_json_path = "./jsons/obp_6.json"

    graph_file_path = "./jsons/graph9d_new.json"
    observation_points_json_path = "./jsons/obpc_7.json"


    file_path_parametros = "./planilhas/obstaculos_processado6.xlsx"  # Ajuste se precisar


    # Carregar pontos de observação por obstáculo
    observacao_por_obstaculo = SegmentUtils.load_observation_points_from_json(observation_points_json_path)

    # Missões por obstáculo
    missions = ['b_busip4', 'ef_reator1', 'ls_pr4']
    mission_positions = MultiGraphPlanner.gerar_mission_positions_from_json(observacao_por_obstaculo, missions)

    # Transformar pontos de observação em missões individuais
    point_mission_positions = {}
    for mission, points in mission_positions.items():
        for point in points:
            point_mission_positions[point] = point  # Cada ponto é sua própria posição alvo

    print("\n🔍 Missões individuais:")
    for k, v in point_mission_positions.items():
        print(f"{k} -> {v}")

    # Carrega o grafo do ambiente e reduzido para inspeção
    G_mapa = SegmentUtils.load_graph_json(graph_file_path)


    # Define posições iniciais reais dos robôs com base em coordenadas (x, y)
    # tx, ty = -165.9766, -77.6645
    # tx2, ty2 = 87.9766, 30.6645

    tx, ty = -144.66934069048625, -76.90938526639975
    tx2, ty2 = -120.60435639037568, 103.09111513360044

    # tx, ty = -0.9766, -0.6645
    # tx2, ty2 = 0.9766, 0.6645

    label_posr1, _, _ = MultiGraphPlanner.find_nearest_node(G_mapa, tx, ty)
    label_posr2, _, _ = MultiGraphPlanner.find_nearest_node(G_mapa, tx2, ty2)

    robots_positions = {"R1": label_posr1, "R2": label_posr2}

    Greduced_map = MultiGraphPlanner.build_inspection_graph([label_posr1,label_posr2], point_mission_positions, G_mapa)
    # PlotUtils.plot_grafo_distance(Greduced_map)


    # 🔄 Definição de execução permitida por missão principal
    mission_execution_config = {
        "b_busip4": ["R1", "R2"],
        "ef_reator1": ["R1", "R2"],
        "ls_pr4": ["R1","R2"]
    }

    # Inicializar
    mission_execution = {}
    fixed_tasks_per_robot = defaultdict(list)
    distributed_tasks = []

    # Processar missões e pontos
    for mission, points in mission_positions.items():
        robots = mission_execution_config.get(mission, [])
        for point in points:
            mission_execution[point] = robots
            if len(robots) == 1:
                # Tarefa fixa
                fixed_tasks_per_robot[robots[0]].append(point)
            else:
                # Tarefa distribuível
                distributed_tasks.append(point)

    # Tempo de execução fixo por ponto
    mission_times = {point: 15 for point in point_mission_positions}

    grafo_mapa_dict = AABBUtils.convert_graph_to_dict(Greduced_map)

    # Executar clusterização
    pontos_por_robo = FixedTaskPlanner.clusterizar_pontos_balanceado(Greduced_map, point_mission_positions, robots_positions, mission_execution)

    print("\n📌 Clusterização balanceada dos pontos por robô:")
    for robo, pontos in pontos_por_robo.items():
        print(f"  {robo}: {pontos}")


    rotas_otimas_por_robo = {}

    for robo, pontos in pontos_por_robo.items():
        ponto_inicial = robots_positions[robo]
        rota_otima = FixedTaskPlanner.tsp_nearest_neighbor(Greduced_map, ponto_inicial, pontos)
        rotas_otimas_por_robo[robo] = rota_otima


    # PlotUtils.plot_rotas_grafo(Greduced_map,rotas_otimas_por_robo)
    PlotUtils.plot_rotas_reais(G_mapa,rotas_otimas_por_robo,point_mission_positions)

    missoes_completas = retorna_rotas_completas(G_mapa,rotas_otimas_por_robo,point_mission_positions,file_path_parametros)



    # Exibir resultados
    for robo, rota in rotas_otimas_por_robo.items():
        print(f"\n🚗 Rota ótima para {robo}:")
        print(" -> ".join(rota))







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
