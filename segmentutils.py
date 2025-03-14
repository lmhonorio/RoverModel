###############################################################################
# CLASSE: SegmentUtils
###############################################################################
import math
from roverclass import ObstacleLoader
from plotutils import PlotUtils
from aabbutils import AABBUtils
import pickle
import json
import networkx as nx
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial import distance


class SegmentUtils:



    @staticmethod
    def save_graph_json(G, filename):
        graph_data = {
            "nodes": {str(node): G.nodes[node] for node in G.nodes()},
            "edges": [(str(u), str(v), G.edges[u, v]["weight"]) for u, v in G.edges()]
        }
        with open(filename, "w") as f:
            json.dump(graph_data, f, indent=4)

    @staticmethod
    def load_graph_json(filename):
        with open(filename, "r") as f:
            graph_data = json.load(f)

        G = nx.Graph()

        # Restaurar nós com labels
        for node, attr in graph_data["nodes"].items():
            G.add_node(eval(node), **attr)  # Converte string para tupla novamente

        # Restaurar arestas com peso
        for u, v, weight in graph_data["edges"]:
            G.add_edge(eval(u), eval(v), weight=weight)

        return G



    @staticmethod
    def index_graph_labels(graph):
        label_counts = {}
        indexed_labels = {}

        for node, data in graph.nodes(data=True):
            label = data.get("label", "unknown")  # Obtém o label original


            indexed_label = f"{label}"
            indexed_labels[node] = indexed_label  # Associa o novo label ao nó

            # Atualiza o label do nó no grafo
            graph.nodes[node]["indexed_label"] = indexed_label

        # Retorna o mapeamento original -> indexado
        return indexed_labels

    @staticmethod
    def create_graph(final_segments, obstacles):
        G = nx.Graph()

        # Criar um dicionário para armazenar os pontos mais próximos de cada obstáculo
        obstacle_node_dict = defaultdict(list)

        # Normalizar todos os pontos e associar ao obstáculo mais próximo
        unique_nodes = set()
        segment_map = {}  # Para mapear cada segmento aos seus nós

        for segment in final_segments:
            x1, y1, x2, y2 = segment

            # Arredonda as coordenadas para uma casa decimal
            x1, y1 = round(x1, 1), round(y1, 1)
            x2, y2 = round(x2, 1), round(y2, 1)

            # Adiciona os nós únicos ao conjunto
            unique_nodes.add((x1, y1))
            unique_nodes.add((x2, y2))

            # Mapeia os segmentos para seus nós
            segment_map[(x1, y1, x2, y2)] = [(x1, y1), (x2, y2)]

        # Atribuir um índice a cada nó baseado no obstáculo mais próximo
        node_labels = {}
        for node in sorted(unique_nodes):  # Ordenação para garantir índices consistentes
            nearest_label = SegmentUtils.get_nearest_obstacle_label(node, obstacles)
            node_index = len(obstacle_node_dict[nearest_label])  # Índice do nó dentro do obstáculo
            full_label = f"{nearest_label}_{node_index}"  # Exemplo: "obstaculo_3"
            obstacle_node_dict[nearest_label].append(node)  # Adiciona ao dicionário
            node_labels[node] = full_label  # Salva o rótulo do nó

        # Criar o grafo com os novos labels
        for segment in final_segments:
            x1, y1, x2, y2 = segment
            x1, y1 = round(x1, 1), round(y1, 1)
            x2, y2 = round(x2, 1), round(y2, 1)

            distance = math.dist((x1, y1), (x2, y2))

            G.add_node((x1, y1), label=node_labels[(x1, y1)])
            G.add_node((x2, y2), label=node_labels[(x2, y2)])
            G.add_edge((x1, y1), (x2, y2), weight=distance)

        # Conectar nós que ficaram isolados dentro do cluster
        for node in unique_nodes:
            connected_neighbors = list(G.neighbors(node))

            if len(connected_neighbors) == 0:  # Nó isolado
                # Encontrar o segmento ao qual esse nó pertence
                for (sx1, sy1, sx2, sy2), seg_nodes in segment_map.items():
                    if node in seg_nodes:
                        # Conectar o nó isolado aos extremos do segmento original
                        G.add_edge(node, (sx1, sy1), weight=math.dist(node, (sx1, sy1)))
                        G.add_edge(node, (sx2, sy2), weight=math.dist(node, (sx2, sy2)))
                        break

        # Conectar subgrafos desconectados
        G, new_connections = SegmentUtils.connect_disconnected_subgraphs(G)
        return G

    @staticmethod
    def fix_missing_connections(G):
        """
        Identifica nós que pertencem a uma mesma classe (base do label, ex: 'TPC1')
        e verifica se cada nó está conectado a pelo menos 1 outro nó da mesma classe.
        Se não estiver, adiciona esse nó à lista de 'isolados', busca os dois nós mais próximos
        da mesma classe e adiciona conexões a eles no grafo G.
        """
        # Agrupar nós por classe => base_name
        groups = defaultdict(list)
        for node in G.nodes():
            label = G.nodes[node].get("label", "unknown")
            base_name = label.rsplit("_", 1)[0]  # ex: TPC1_3 => TPC1
            groups[base_name].append(node)

        # Lista para armazenar nós isolados com os dois nós mais próximos
        isolated_nodes_with_neighbors = []

        for base_name, nodes in groups.items():
            for n in nodes:
                neighbors = list(G.neighbors(n))

                # Verifica se pelo menos um vizinho tem o mesmo base_name
                has_cluster_neighbor = any(
                    G.nodes[neighbor].get("label", "").rsplit("_", 1)[0] == base_name
                    for neighbor in neighbors
                )

                # Se o nó está isolado dentro do cluster, buscar os dois nós mais próximos
                if not has_cluster_neighbor:
                    # Calcular distâncias para os outros nós do mesmo cluster
                    distances = [
                        (other, distance.euclidean(n, other))
                        for other in nodes if other != n
                    ]

                    # Ordenar pela menor distância e pegar os dois primeiros
                    closest_nodes = sorted(distances, key=lambda x: x[1])[:2]

                    # Criar lista com as coordenadas e labels dos nós mais próximos
                    closest_nodes_info = [
                        {"coordinate": node[0], "label": G.nodes[node[0]].get("label", "")}
                        for node in closest_nodes
                    ]

                    # Adicionar à lista de nós isolados
                    isolated_nodes_with_neighbors.append({
                        "label": G.nodes[n].get("label", ""),  # Nome do nó
                        "coordinate": n,  # Coordenada do nó
                        "closest_nodes": closest_nodes_info  # Lista com os dois nós mais próximos (coordenada + label)
                    })

                    # **Adicionar as conexões no grafo**
                    for node_data in closest_nodes_info:
                        neighbor_coord = node_data["coordinate"]
                        distance_value = distance.euclidean(n, neighbor_coord)

                        # Adiciona aresta no grafo com peso igual à distância
                        G.add_edge(n, neighbor_coord, weight=distance_value)

        return G  # Retorna para testes



    @staticmethod
    def get_nearest_obstacle_label(point, obstacles):
        min_distance = float('inf')
        nearest_label = None
        for obs in obstacles:
            obs_x, obs_y = obs["pos"]
            distance = math.dist(point, (obs_x, obs_y))
            if distance < min_distance:
                min_distance = distance
                nearest_label = obs["label"]
        return nearest_label

    ###############################################################################
    # Função para transformar grafo em segmentos
    ###############################################################################
    @staticmethod
    def graph_to_segments(G):
        segments = []
        for edge in G.edges():
            (x1, y1), (x2, y2) = edge
            segments.append((x1, y1, x2, y2))
        return segments

    ###############################################################################
    # Função para encontrar subgrafos não conectados
    ###############################################################################
    @staticmethod
    def find_disconnected_subgraphs(G):
        return [G.subgraph(c).copy() for c in nx.connected_components(G)]

    ###############################################################################
    # Função para encontrar o ponto mais próximo entre dois subgrafos
    ###############################################################################
    @staticmethod
    def find_closest_connection(subgraphs):
        min_distance = float('inf')
        best_connection = None

        for i, sg1 in enumerate(subgraphs):
            for j, sg2 in enumerate(subgraphs):
                if i >= j:
                    continue

                for node1 in sg1.nodes():
                    for node2 in sg2.nodes():
                        dist = math.dist(node1, node2)
                        if dist < min_distance:
                            min_distance = dist
                            best_connection = (node1, node2)

        return best_connection

    ###############################################################################
    # Função para conectar todos os subgrafos
    ###############################################################################
    @staticmethod
    def connect_disconnected_subgraphs(G):
        subgraphs = SegmentUtils.find_disconnected_subgraphs(G)
        new_segments = []

        while len(subgraphs) > 1:
            node1, node2 = SegmentUtils.find_closest_connection(subgraphs)
            distance = math.dist(node1, node2)
            G.add_edge(node1, node2, weight=distance)
            new_segments.append((node1[0], node1[1], node2[0], node2[1]))
            subgraphs = SegmentUtils.find_disconnected_subgraphs(G)

        return G, new_segments

    ###############################################################################
    # Função para carregar o grafo
    ###############################################################################
    @staticmethod
    def load_graph(filename="graph.pkl"):
        with open(filename, "rb") as f:
            return pickle.load(f)

    ###############################################################################
    # Função para carregar a lista de segmentos
    ###############################################################################
    @staticmethod
    def load_segments(filename="segments.json"):
        with open(filename, "r") as f:
            return json.load(f)

    ###############################################################################
    # Função para salvar o grafo
    ###############################################################################
    def save_graph(G, filename="graph.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(G, f)

    ###############################################################################
    # Função para salvar a lista de segmentos
    ###############################################################################
    def save_segments(segments, filename="segments.json"):
        with open(filename, "w") as f:
            json.dump(segments, f)


    @staticmethod
    def adjust_nodes_on_segments(segments, threshold=1.0):
        new_segments = []
        nodes = set()

        for segment in segments:
            x1, y1, x2, y2 = segment
            nodes.add((x1, y1))
            nodes.add((x2, y2))

        for node in list(nodes):
            nx, ny = node
            for segment in segments:
                x1, y1, x2, y2 = segment
                if (nx, ny) != (x1, y1) and (nx, ny) != (x2, y2):
                    d = math.dist((nx, ny), (x1, y1)) + math.dist((nx, ny), (x2, y2)) - math.dist((x1, y1), (x2, y2))
                    if abs(d) < threshold:
                        new_segments.append((x1, y1, nx, ny))
                        new_segments.append((nx, ny, x2, y2))
                        break
            else:
                new_segments.append((x1, y1,x2, y2))

        return new_segments

    @staticmethod
    def return_segments(file_path, sheet_name, padding, margin, endpoint_threshold=3.0, center_threshold=5.0,
                        parallel_threshold = 15.0, final_threshold = 4):

        # Carregar obstáculos
        loader = ObstacleLoader(file_path, sheet_name)

        obstacles = loader.get_obstacles()

        # AABBs
        aabbs = AABBUtils.get_aabbs(obstacles, margin)

        # Determinar extents
        x_min = min(o["pos"][0] for o in obstacles) - padding
        x_max = max(o["pos"][0] for o in obstacles) + padding
        y_min = min(o["pos"][1] for o in obstacles) - padding
        y_max = max(o["pos"][1] for o in obstacles) + padding

        # Gera caminhos horizontais e verticais
        horizontal_paths, vertical_paths = SegmentUtils.get_paths(aabbs, x_min, x_max, y_min, y_max)

        # Subdivisão fora de AABB (índice par-ímpar)
        valid_segments = SegmentUtils.split_and_filter_paths(horizontal_paths, vertical_paths, aabbs)

        # Filtrar pelos endpoints e centro
        filtered_segments = SegmentUtils.filter_segments_by_distance(valid_segments, aabbs,
                                                                     endpoint_threshold=3.0,
                                                                     center_threshold=5.0)

        # Filtra paralelos duplicados
        prefinal_segments = SegmentUtils.filter_similar_segments(filtered_segments, aabbs,
                                                                 parallel_threshold=15.0)

        # Adiciona subsegmentos do perímetro
        final_segments = SegmentUtils.add_perimeter_segments(aabbs, final_threshold, prefinal_segments)

        return final_segments


    @staticmethod
    def subdivide_edge(x1, y1, x2, y2, threshold):
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return []
        n = math.ceil(length / threshold)
        step = 1.0 / n
        subs = []
        for i in range(n):
            tA = i * step
            tB = (i + 1) * step
            Ax = x1 + dx * tA
            Ay = y1 + dy * tA
            Bx = x1 + dx * tB
            By = y1 + dy * tB
            subs.append((Ax, Ay, Bx, By))
        return subs

    @staticmethod
    def get_paths(aabbs, x_min, x_max, y_min, y_max):
        """
        Gera caminhos horizontais e verticais que tangenciam o topo, fundo,
        esquerda e direita de cada AABB.
        """
        horizontal_paths = set()
        vertical_paths = set()

        # Para cada AABB, pegamos as 2 linhas horizontais e 2 linhas verticais
        # correspondentes às suas bordas
        for (aabb_x, aabb_y), aabb_w, aabb_h in aabbs:
            # Topo e fundo da AABB
            y_top = aabb_y
            y_bottom = aabb_y + aabb_h

            # Esquerda e direita da AABB
            x_left = aabb_x
            x_right = aabb_x + aabb_w

            # Garantir que aabb_w e aabb_h sejam positivos
            # (caso haja alguma AABB degenerada)
            if aabb_w <= 0 or aabb_h <= 0:
                continue

            # Adicionar caminhos horizontais (y constante, x variando [x_min..x_max])
            # Top
            if y_min <= y_top <= y_max:
                horizontal_paths.add((y_top, x_min, x_max))
            # Bottom
            if y_min <= y_bottom <= y_max:
                horizontal_paths.add((y_bottom, x_min, x_max))

            # Adicionar caminhos verticais (x constante, y variando [y_min..y_max])
            # Left
            if x_min <= x_left <= x_max:
                vertical_paths.add((x_left, y_min, y_max))
            # Right
            if x_min <= x_right <= x_max:
                vertical_paths.add((x_right, y_min, y_max))

        # Converter sets para listas antes de retornar
        return list(horizontal_paths), list(vertical_paths)

    @staticmethod
    def create_perimeter_segments(aabb, threshold):
        ((ax, ay), aw, ah) = aabb
        if aw <= 0 or ah <= 0:
            return []
        top_left = (ax, ay)
        top_right = (ax + aw, ay)
        bottom_right = (ax + aw, ay + ah)
        bottom_left = (ax, ay + ah)

        segs = []
        segs += SegmentUtils.subdivide_edge(top_left[0], top_left[1],
                                            top_right[0], top_right[1], threshold)
        segs += SegmentUtils.subdivide_edge(top_right[0], top_right[1],
                                            bottom_right[0], bottom_right[1], threshold)
        segs += SegmentUtils.subdivide_edge(bottom_right[0], bottom_right[1],
                                            bottom_left[0], bottom_left[1], threshold)
        segs += SegmentUtils.subdivide_edge(bottom_left[0], bottom_left[1],
                                            top_left[0], top_left[1], threshold)
        return segs

    @staticmethod
    def add_perimeter_segments(aabbs, segments, threshold_ponto_por_distancia=4 ):
        """
        Gera subsegmentos no perímetro de cada AABB e adiciona em 'segments'.
        """
        for aabb in aabbs:
            subs = SegmentUtils.create_perimeter_segments(aabb, threshold_ponto_por_distancia)
            segments.extend(subs)
        return segments

    @staticmethod
    def segment_center(seg):
        x1, y1, x2, y2 = seg
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @staticmethod
    def center_distance(s1, s2):
        c1 = SegmentUtils.segment_center(s1)
        c2 = SegmentUtils.segment_center(s2)
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    @staticmethod
    def orientation_and_length(seg):
        x1, y1, x2, y2 = seg
        if math.isclose(y1, y2, abs_tol=1e-9):
            return ('H', abs(x2 - x1))
        elif math.isclose(x1, x2, abs_tol=1e-9):
            return ('V', abs(y2 - y1))
        else:
            return None

    @staticmethod
    def distance_point_to_aabb(px, py, aabb):
        """
        (ax, ay) = canto sup esquerdo, w,h
        """
        (ax, ay), aw, ah = aabb
        rx1, ry1 = ax, ay
        rx2, ry2 = ax + aw, ay + ah
        dx, dy = 0, 0
        if px < rx1:
            dx = rx1 - px
        elif px > rx2:
            dx = px - rx2
        if py < ry1:
            dy = ry1 - py
        elif py > ry2:
            dy = py - ry2
        return math.hypot(dx, dy)

    @staticmethod
    def min_dist_center_to_aabbs(segment, aabbs):
        mx, my = SegmentUtils.segment_center(segment)
        dist_min = float('inf')
        for aabb in aabbs:
            d = SegmentUtils.distance_point_to_aabb(mx, my, aabb)
            if d < dist_min:
                dist_min = d
        return dist_min

    @staticmethod
    def filter_similar_segments(segments, aabbs, parallel_threshold):
        """
        Agrupa segs por (orientacao, length).
        Em cada grupo, BFS dos que estao a < parallel_threshold de distancia (centro).
        Fica so com 1 => o de centro mais proximo de algum AABB.
        """
        from collections import deque
        groups = {}
        for seg in segments:
            key = SegmentUtils.orientation_and_length(seg)
            if not key:
                continue
            if key not in groups:
                groups[key] = []
            groups[key].append(seg)

        filtered = []
        for key, segs in groups.items():
            n = len(segs)
            if n <= 1:
                if n == 1:
                    filtered.append(segs[0])
                continue
            adj = [[] for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    distc = SegmentUtils.center_distance(segs[i], segs[j])
                    if distc < parallel_threshold:
                        adj[i].append(j)
                        adj[j].append(i)
            visited = [False] * n
            for start in range(n):
                if not visited[start]:
                    visited[start] = True
                    queue = deque([start])
                    cluster = [start]
                    while queue:
                        curr = queue.popleft()
                        for neigh in adj[curr]:
                            if not visited[neigh]:
                                visited[neigh] = True
                                queue.append(neigh)
                                cluster.append(neigh)
                    # cluster => so 1
                    best_seg = None
                    best_dist = float('inf')
                    for idx in cluster:
                        seg_ = segs[idx]
                        dC = SegmentUtils.min_dist_center_to_aabbs(seg_, aabbs)
                        if dC < best_dist:
                            best_dist = dC
                            best_seg = seg_
                    filtered.append(best_seg)
        return filtered

    @staticmethod
    def split_and_filter_paths(horizontal_paths, vertical_paths, aabbs):
        """
        Lógica par-ímpar => subsegmentos fora da AABB.
        """
        valid = []
        # horizontais
        for (y, xs, xe) in horizontal_paths:
            cpoints = [xs, xe]
            for (aabb_xy, aw, ah) in aabbs:
                ax, ay = aabb_xy
                if ay <= y <= ay + ah:
                    lx = ax
                    rx = ax + aw
                    if xs <= lx <= xe: cpoints.append(lx)
                    if xs <= rx <= xe: cpoints.append(rx)
            cpoints = sorted(set(cpoints))
            for i in range(len(cpoints) - 1):
                idx1 = i + 1
                idx2 = i + 2
                xA = cpoints[i]
                xB = cpoints[i + 1]
                if (idx1 % 2 == 1) and (idx2 % 2 == 0):
                    valid.append((xA, y, xB, y))

        # verticais
        for (x, ys, ye) in vertical_paths:
            cpoints = [ys, ye]
            for (aabb_xy, aw, ah) in aabbs:
                ax, ay = aabb_xy
                if ax <= x <= ax + aw:
                    bot = ay
                    top = ay + ah
                    if ys <= bot <= ye: cpoints.append(bot)
                    if ys <= top <= ye: cpoints.append(top)
            cpoints = sorted(set(cpoints))
            for i in range(len(cpoints) - 1):
                idx1 = i + 1
                idx2 = i + 2
                yA = cpoints[i]
                yB = cpoints[i + 1]
                if (idx1 % 2 == 1) and (idx2 % 2 == 0):
                    valid.append((x, yA, x, yB))
        return valid

    @staticmethod
    def filter_segments_by_distance(segments, aabbs, endpoint_threshold, center_threshold):
        """
        So mantem se:
         - ambas extremidades < endpoint_threshold
         - centro < center_threshold
        """
        valids = []
        for (x1, y1, x2, y2) in segments:
            dist_min_e1 = float('inf')
            dist_min_e2 = float('inf')
            mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            dist_min_center = float('inf')
            for aabb in aabbs:
                d1 = SegmentUtils.distance_point_to_aabb(x1, y1, aabb)
                d2 = SegmentUtils.distance_point_to_aabb(x2, y2, aabb)
                dC = SegmentUtils.distance_point_to_aabb(mx, my, aabb)
                if d1 < dist_min_e1: dist_min_e1 = d1
                if d2 < dist_min_e2: dist_min_e2 = d2
                if dC < dist_min_center: dist_min_center = dC
            if (dist_min_e1 < endpoint_threshold and
                    dist_min_e2 < endpoint_threshold and
                    dist_min_center < center_threshold):
                valids.append((x1, y1, x2, y2))
        return valids
