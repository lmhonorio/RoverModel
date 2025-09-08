from typing import Dict, List, Tuple, Optional, Any, Union
from multigraphplanner import MultiGraphPlanner
from segmentutils import SegmentUtils
from plotutils import PlotUtils
from ajusteplanilha import AjustePlanilha
from tspOptimization import FixedTaskPlanner
from collections import OrderedDict, defaultdict
import math
import networkx as nx
import json
import numpy as np



def _load_label_to_gps_map(observation_points_json_path):
    """Flatten do obp_6.json: label -> (lat, lon)"""
    with open(observation_points_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    label2gps = {}
    for _, arr in data.items():
        if isinstance(arr, list):
            for obj in arr:
                label = obj.get("label")
                coord = obj.get("coord")
                if label and coord and len(coord) == 2:
                    # coord = [lat, lon]
                    label2gps[label] = (float(coord[0]), float(coord[1]))
    return label2gps

def montar_missoes_por_robo(
    missoes_completas,         # list(retorna_pontos_passagem(...))
    G_mapa,                    # grafo completo com nós e atributo 'pos' = (x,y)
    observation_points_json_path,
    lat_ref, lon_ref,          # ref. para conversão se faltar GPS no JSON
    duplicate_first=True,      # repete o 1º ponto
    hold_vistoria=5.0,
    hold_passagem=0.0,
    MissionManager=None        # se tiver xy_to_gps; senão cai no conversor local
):
    """
    Retorna: dict robô -> [ {id,lat,lon,hold}, ... ] pronto para upload_mission
    """
    # 1) Mapa label->(lat,lon) vindo do JSON de observação
    label2gps = _load_label_to_gps_map(observation_points_json_path)

    # 2) Conversor local caso algum label não esteja no JSON
    import math
    def xy_to_gps_local(x, y):
        R = 6378137.0
        dlat = y / R
        dlon = x / (R * math.cos(math.radians(lat_ref)))
        return (lat_ref + math.degrees(dlat), lon_ref + math.degrees(dlon))

    missoes_por_robo = {}

    for item in missoes_completas:
        info = item[0]  # retorna_pontos_passagem yielda [ { ... } ]
        robo = info["robo"]
        caminho_completo, pts_vistoria, pts_passagem = info["rotas_detalhadas"]

        path_gps = []
        holds = []

        for label in caminho_completo:
            # 3) Pega lat/lon do JSON; se não tiver, converte do (x,y) do grafo
            if label in label2gps:
                lat, lon = label2gps[label]
            else:
                x, y = G_mapa.nodes[label]['pos']
                if MissionManager and hasattr(MissionManager, "xy_to_gps"):
                    lat, lon = MissionManager.xy_to_gps(x, y, lat_ref, lon_ref)
                else:
                    lat, lon = xy_to_gps_local(x, y)

            path_gps.append((lat, lon))
            holds.append(hold_vistoria if label in pts_vistoria else hold_passagem)

        # 4) Constrói mission_points (IDs sequenciais, com opção de duplicar o primeiro)
        mission_points = build_mission_points_from_path_gps(
            path_gps,
            holds=holds,             # <- importante: passa uma lista, não float!
            default_hold=0.0,
            duplicate_first=duplicate_first,
            start_id=0
        )

        missoes_por_robo[robo] = mission_points

    return missoes_por_robo



def retorna_rotas_reais(G_robot, rotas_por_robo, pontos_vistoria):

    for i, (robo, rota) in enumerate(rotas_por_robo.items()):
        caminho_completo = []

        for j in range(len(rota) - 1):
            u, v = rota[j], rota[j + 1]

            subpath = nx.shortest_path(G_robot, source=u, target=v, weight="weight")

            if caminho_completo and subpath[0] == caminho_completo[-1]:
                caminho_completo.extend(subpath[1:])
            else:
                caminho_completo.extend(subpath)

        coords_caminho = np.array([G_robot.nodes[n]['pos'] for n in caminho_completo])


        # Diferencia pontos de vistoria dos pontos intermediários
        for ponto in caminho_completo:
            x, y = G_robot.nodes[ponto]['pos']
            if ponto in pontos_vistoria:
                i=0
                # plt.plot(x, y, marker='o', markersize=10, color='yellow', markeredgecolor='black', zorder=5)
            else:
                i=1
                # plt.plot(x, y, marker='.', markersize=5, color='gray', zorder=4)



def load_label2gps(observation_points_json_path: str) -> dict[str, tuple[float, float]]:
    """Lê o JSON de pontos de observação e retorna {label: (lat, lon)}."""
    with open(observation_points_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # dict: {obstaculo: [{label, coord:[lat,lon]}, ...], ...}
    label2gps = {}
    for _group, arr in data.items():
        for item in arr:
            lab = item.get("label")
            coord = item.get("coord")
            if isinstance(lab, str) and isinstance(coord, list) and len(coord) == 2:
                label2gps[lab] = (float(coord[0]), float(coord[1]))
    return label2gps

def extract_path_gps_from_obp(
    tarefa: dict,
    label2gps: dict[str, tuple[float, float]],
    rota_labels_fallback: list[str] | None = None
) -> list[tuple[float, float]]:
    """
    1) Se já existir 'path_gps' na tarefa, usa-o diretamente.
    2) Senão, procura labels presentes em tarefa['path'] que existam no label2gps.
    3) Senão, usa a lista de labels passada em rota_labels_fallback.
    """
    # 1) já pronto
    if "path_gps" in tarefa and tarefa["path_gps"]:
        gps_seq = []
        for p in tarefa["path_gps"]:
            if isinstance(p, dict) and "lat" in p and "lon" in p:
                gps_seq.append((float(p["lat"]), float(p["lon"])))
            elif isinstance(p, (list, tuple)) and len(p) == 2:
                gps_seq.append((float(p[0]), float(p[1])))
        return gps_seq

    # 2) extrair dos labels dentro da própria tarefa
    labels = []
    for node in tarefa.get("path", []):
        if isinstance(node, str) and node in label2gps:
            labels.append(node)
        elif isinstance(node, dict) and "label" in node and node["label"] in label2gps:
            labels.append(node["label"])
    if labels:
        return [label2gps[l] for l in labels]

    # 3) fallback: rota ótima do robô (se for passar isso de fora)
    if rota_labels_fallback:
        return [label2gps[l] for l in rota_labels_fallback if l in label2gps]

    return []

def build_mission_points_from_path_gps(
    path_gps: list[tuple[float, float]],
    holds: float | None = None,
    default_hold: float = 0.0,
    duplicate_first: bool = True,
    start_id: int = 0,
):
    """
    Converte [(lat,lon), ...] -> [{"id":..., "lat":..., "lon":..., "hold":...}, ...].
    """
    pts = list(path_gps)
    if duplicate_first and pts:
        pts = [pts[0]] + pts
    mp = []
    for i, (lat, lon) in enumerate(pts):
        mp.append({
            "id": start_id + i,
            "lat": float(lat),
            "lon": float(lon),
            "hold": float(holds if holds is not None else default_hold),
            "param2":float(1.0),
            "param3": float(1.0)
        })
    return mp


def extract_path_gps(
    tarefa: Dict[str, Any],
    G_mapa,                        # grafo com nós contendo "pos"=(x,y)
    lat_ref: float,
    lon_ref: float,
    MissionManager=None            # se sua classe tiver xy_to_gps, usamos ela
) -> List[Tuple[float, float]]:
    # 1) Já em GPS
    if "path_gps" in tarefa:
        return list(tarefa["path_gps"])

    # 2) Pares (x,y)
    if "path_xy" in tarefa:
        out = []
        for x, y in tarefa["path_xy"]:
            if MissionManager and hasattr(MissionManager, "xy_to_gps"):
                lat, lon = MissionManager.xy_to_gps(x, y, lat_ref, lon_ref)
            else:
                lat, lon = xy_to_gps_local(x, y, lat_ref, lon_ref)
            out.append((lat, lon))
        return out

    # 3) Labels do grafo OU lista de pares (x,y)
    if "path" in tarefa:
        seq = list(tarefa["path"])
        if not seq:
            return []
        first = seq[0]
        # 3a) Labels (strings) -> buscar pos no grafo
        if isinstance(first, str):
            out = []
            for lbl in seq:
                pos = G_mapa.nodes[lbl].get("pos")
                if pos is None:
                    # fallback: alguns grafos salvam x,y separados
                    x = G_mapa.nodes[lbl].get("x")
                    y = G_mapa.nodes[lbl].get("y")
                    if x is None or y is None:
                        raise KeyError(f"Nó '{lbl}' sem 'pos' nem 'x/y' no grafo.")
                    pos = (x, y)
                x, y = pos
                if MissionManager and hasattr(MissionManager, "xy_to_gps"):
                    lat, lon = MissionManager.xy_to_gps(x, y, lat_ref, lon_ref)
                else:
                    lat, lon = xy_to_gps_local(x, y, lat_ref, lon_ref)
                out.append((lat, lon))
            return out
        # 3b) Já são pares (x,y)
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            out = []
            for x, y in seq:
                if MissionManager and hasattr(MissionManager, "xy_to_gps"):
                    lat, lon = MissionManager.xy_to_gps(x, y, lat_ref, lon_ref)
                else:
                    lat, lon = xy_to_gps_local(x, y, lat_ref, lon_ref)
                out.append((lat, lon))
            return out

    raise KeyError("Estrutura da tarefa não contém 'path_gps', 'path_xy' ou 'path' utilizável.")



# --- inversa simples (x,y)->(lat,lon) caso sua classe já não tenha ---
def xy_to_gps_local(x: float, y: float, lat_ref: float, lon_ref: float) -> Tuple[float, float]:
    R = 6378137.0
    dlat = y / R
    dlon = x / (R * math.cos(math.radians(lat_ref)))
    return (lat_ref + math.degrees(dlat), lon_ref + math.degrees(dlon))


def build_mission_points_from_path_gps(
    path_gps: List[Tuple[float, float]],
    holds: Optional[Union[float, int, List[float], Tuple[float, ...], Dict[int, float]]] = None,
    default_hold: float = 3.0,
    duplicate_first: bool = False,
    start_id: int = 0,
) -> List[Dict[str, Any]]:
    """
    Converte [(lat,lon), ...] -> [{"id":..., "lat":..., "lon":..., "hold":...}, ...].
    Suporta holds como:
      - float/int: mesmo hold para todos
      - list/tuple: por waypoint (trunca/pad até o tamanho do path)
      - dict[int->float]: por índice; demais usam default_hold
      - None: todos default_hold
    """
    if not path_gps:
        return []

    n = len(path_gps)

    # Normaliza holds para uma lista de tamanho n
    if isinstance(holds, (int, float)):
        holds_list = [float(holds)] * n
    elif isinstance(holds, (list, tuple)):
        holds_list = [float(h) for h in holds]
        if len(holds_list) < n:
            holds_list += [float(default_hold)] * (n - len(holds_list))
        else:
            holds_list = holds_list[:n]
    elif isinstance(holds, dict):
        holds_list = [float(default_hold)] * n
        for i, h in holds.items():
            try:
                idx = int(i)
            except Exception:
                continue
            if 0 <= idx < n:
                holds_list[idx] = float(h)
    else:
        holds_list = [float(default_hold)] * n

    pts = list(path_gps)
    if duplicate_first and pts:
        pts = [pts[0]] + pts
        holds_list = [holds_list[0]] + holds_list  # duplica o hold do 1º também

    out = []
    for i, (lat, lon) in enumerate(pts):
        out.append({
            "id": start_id + i,
            "lat": float(lat),
            "lon": float(lon),
            "hold": float(holds_list[i]),
        })
    return out


def retorna_pontos_passagem(G_robot, rotas_por_robo, pontos_vistoria):

    for i, (robo, rota) in enumerate(rotas_por_robo.items()):
        pts_caminho = []
        pts_vistoria = []
        caminho_completo = []

        for j in range(len(rota) - 1):
            u, v = rota[j], rota[j + 1]

            subpath = nx.shortest_path(G_robot, source=u, target=v, weight="weight")

            if caminho_completo and subpath[0] == caminho_completo[-1]:
                caminho_completo.extend(subpath[1:])
            else:
                caminho_completo.extend(subpath)

        coords_caminho = np.array([G_robot.nodes[n]['pos'] for n in caminho_completo])

        # Diferencia pontos de vistoria dos pontos intermediários
        for ponto in caminho_completo:
            x, y = G_robot.nodes[ponto]['pos']
            if ponto in pontos_vistoria:
                pts_vistoria.append(ponto)
            else:
                pts_caminho.append(ponto)

        yield [{"robo":robo, "rotas_detalhadas":[caminho_completo, pts_vistoria, pts_caminho]}]



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


def run_planner(
    file_path: str,
    observation_points_json_path: str,
    file_path_parametros: str,
    missions: List[str],
    mission_execution_config: Dict[str, List[str]],
    robot_positions_xy: Optional[Dict[str, Tuple[float, float]]] = None,
    do_plots: bool = False,
) -> Dict:
    """
    Executa o pipeline de planejamento completo a partir de arquivos.

    Parâmetros
    ----------
    file_path : str
        Caminho do grafo (.json) do ambiente (em formato SegmentUtils.save_graph_json).
    observation_points_json_path : str
        Caminho do JSON com pontos de observação (agrupados por obstáculo).
    file_path_parametros : str
        Planilha de parâmetros p/ converter (x,y)->(lat,lon).
    missions : List[str]
        Lista de missões principais (ex.: ['b_busip4','ef_reator1','ls_pr4']).
    mission_execution_config : Dict[str, List[str]]
        Quais robôs podem executar cada missão principal.
        Ex.: {"b_busip4": ["R1","R2"], ...}
    robot_positions_xy : Optional[Dict[str, Tuple[float,float]]]
        Posições iniciais em metros (x,y) por robô. Se None, usa os defaults do script.
    do_plots : bool
        Se True, plota o grafo reduzido e as rotas.

    Retorno
    -------
    Dict com:
      - "G_reduced": nx.Graph
      - "robots_positions": Dict[robo -> label nó inicial]
      - "pontos_por_robo": Dict[robo -> [labels atribuídos]]
      - "rotas_otimas_por_robo": Dict[robo -> [labels na ordem de visita]]
      - "missoes_completas": estrutura por robô/missão/tarefas com path em nós, path_gps e distância
    """
    # 1) Carregar pontos de observação e missão → pontos unitários
    observacao_por_obstaculo = SegmentUtils.load_observation_points_from_json(observation_points_json_path)  # noqa
    mission_positions = MultiGraphPlanner.gerar_mission_positions_from_json(observacao_por_obstaculo, missions)  # noqa

    point_mission_positions = {}
    for mission, points in mission_positions.items():
        for p in points:
            point_mission_positions[p] = p  # missão unitária por ponto

    # 2) Carregar grafo e localizar posições iniciais
    G_mapa = SegmentUtils.load_graph_json(file_path)

    robots_positions = {}
    for rname, (x, y) in robot_positions_xy.items():
        lbl, _, _ = MultiGraphPlanner.find_nearest_node(G_mapa, x, y)
        robots_positions[rname] = lbl

    # 3) Grafo reduzido de inspeção
    Greduced_map = MultiGraphPlanner.build_inspection_graph(
        list(robots_positions.values()), point_mission_positions, G_mapa
    )

    # 4) Expandir config de execução (missão → pontos) respeitando restrições
    mission_execution = {}
    fixed_tasks_per_robot = defaultdict(list)   # mantido por compatibilidade
    distributed_tasks = []                      # mantido por compatibilidade
    for mission, points in mission_positions.items():
        robots = mission_execution_config.get(mission, [])
        for pt in points:
            mission_execution[pt] = robots
            if len(robots) == 1:
                fixed_tasks_per_robot[robots[0]].append(pt)
            else:
                distributed_tasks.append(pt)

    # 5) Clusterização balanceada dos pontos por robô
    pontos_por_robo = FixedTaskPlanner.clusterizar_pontos_balanceado(
        Greduced_map, point_mission_positions, robots_positions, mission_execution
    )

    # 6) TSP (vizinho mais próximo) por robô
    rotas_otimas_por_robo = {}
    for robo, pontos in pontos_por_robo.items():
        start = robots_positions[robo]
        rota = FixedTaskPlanner.tsp_nearest_neighbor(Greduced_map, start, pontos)
        rotas_otimas_por_robo[robo] = rota


    # 7) Gera estrutura completa (paths nós, XY->GPS por tarefa, distâncias)
    missoes_completas = retorna_rotas_completas(
        G_mapa, rotas_otimas_por_robo, list(point_mission_positions.keys()), file_path_parametros
    )

    return {
        "G_reduced": Greduced_map,
        "robots_positions": robots_positions,
        "pontos_por_robo": pontos_por_robo,
        "rotas_otimas_por_robo": rotas_otimas_por_robo,
        "missoes_completas": missoes_completas,
    }
