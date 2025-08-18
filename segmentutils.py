###############################################################################
# CLASSE: SegmentUtils
###############################################################################
import math

import pandas as pd
from sympy import false

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
import itertools
import matplotlib.colors as mcolors




class SegmentUtils:

    @staticmethod
    def load_observation_points_from_json(json_path):
        import json
        import os

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Arquivo JSON não encontrado: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data  # dict: {label: [[lat, lon], [lat, lon], ...]}


    @staticmethod
    def save_observation_points_to_kml(obstacles, perimeter_points, threshold, xlsx_path, output_folder, offset_lat_meters=0.0,
                                       offset_lon_meters=0.0):
        import os
        import math
        from xml.dom.minidom import Document
        from ajusteplanilha import AjustePlanilha

        def distance(p1, p2):
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Converter pontos (x, y) para (lat, lon)
        pontos_xy = [(px, py) for px, py, _ in perimeter_points]
        pontos_xy_offset = [(px+offset_lon_meters, py+offset_lat_meters) for px, py, _ in perimeter_points]
        gps_coords = AjustePlanilha.metros_para_geocoordenadas(pontos_xy_offset, xlsx_path)
        coord_map = dict(zip(pontos_xy, gps_coords))

        for obs in obstacles:
            ox, oy = obs["pos"]
            label = obs["label"]
            pontos_obs = []

            for px, py, _ in perimeter_points:
                if distance((ox, oy), (px, py)) <= threshold:
                    lat, lon = coord_map[(px, py)]
                    pontos_obs.append((lat, lon))

            # Criar documento KML
            doc = Document()
            kml = doc.createElement("kml")
            kml.setAttribute("xmlns", "http://www.opengis.net/kml/2.2")
            doc.appendChild(kml)

            document = doc.createElement("Document")
            kml.appendChild(document)

            for lat, lon in pontos_obs:
                placemark = doc.createElement("Placemark")

                point = doc.createElement("Point")
                coordinates = doc.createElement("coordinates")
                coordinates.appendChild(doc.createTextNode(f"{lon},{lat},0"))

                point.appendChild(coordinates)
                placemark.appendChild(point)
                document.appendChild(placemark)

            filename = os.path.join(output_folder, f"{label}.kml")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(doc.toprettyxml(indent="  "))

            print(f"✅ KML salvo com deslocamento: {filename}")

    @staticmethod
    def save_observation_points_to_json(obstacles, perimeter_points, threshold, json_path, xlsx_path):
        import math
        import json
        from ajusteplanilha import AjustePlanilha

        def distance(p1, p2):
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        # Converte os pontos para GPS usando a planilha de parâmetros
        try:
            pontos_para_converter = [(px, py) for px, py, _ in perimeter_points]
            gps_coords = AjustePlanilha.metros_para_geocoordenadas(pontos_para_converter, xlsx_path)
            coord_map = dict(zip(pontos_para_converter, gps_coords))
        except Exception as e:
            print(f"[ERRO] Falha ao converter pontos para GPS: {e}")
            return

        # Agrupa pontos por obstáculo
        obs_points_map = {obs["label"]: [] for obs in obstacles}
        for obs in obstacles:
            ox, oy = obs["pos"]
            label = obs["label"]
            for px, py, ponto_label in perimeter_points:
                if distance((ox, oy), (px, py)) <= threshold:
                    lat, lon = coord_map[(px, py)]
                    obs_points_map[label].append({
                        "label": ponto_label,
                        "coord": [lat, lon]
                    })

        # Salva em JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(obs_points_map, f, indent=2)

        print(f"✅ Pontos de observação com rótulo salvos em GPS no arquivo: {json_path}")

    @staticmethod
    def save_observation_points_to_excel(obstacles, perimeter_points, threshold, xlsx_path):
        import math
        import openpyxl
        from openpyxl import load_workbook
        from ajusteplanilha import AjustePlanilha

        def distance(p1, p2):
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        # Carrega os parâmetros de conversão via AjustePlanilha
        try:
            pontos_para_converter = [(px, py) for px, py, _ in perimeter_points]
            gps_coords = AjustePlanilha.metros_para_geocoordenadas(pontos_para_converter, xlsx_path)
            coord_map = dict(zip(pontos_para_converter, gps_coords))
        except Exception as e:
            print(f"[ERRO] Falha ao converter pontos para GPS: {e}")
            return

        # Agrupar pontos próximos de cada obstáculo
        obs_points_map = {obs["label"]: [] for obs in obstacles}
        for obs in obstacles:
            ox, oy = obs["pos"]
            label = obs["label"]
            for px, py, _ in perimeter_points:
                if distance((ox, oy), (px, py)) <= threshold:
                    obs_points_map[label].append(coord_map[(px, py)])

        # Abrir planilha
        try:
            wb = load_workbook(xlsx_path)
        except FileNotFoundError:
            print(f"[ERRO] Arquivo '{xlsx_path}' não encontrado.")
            return

        # Apagar aba antiga se existir
        if "Pontos de Observacao" in wb.sheetnames:
            del wb["Pontos de Observacao"]

        ws = wb.create_sheet("Pontos de Observacao")

        row = 1
        for label, points in obs_points_map.items():
            ws.cell(row=row, column=1, value=f"Obstáculo: {label}")
            row += 1
            for lat, lon in points:
                ws.cell(row=row, column=1, value=lat)
                ws.cell(row=row, column=2, value=lon)
                row += 1
            row += 1

        wb.save(xlsx_path)
        print(f"✅ Pontos de observação em coordenadas GPS salvos no arquivo: {xlsx_path}")


    @staticmethod
    def xml_to_graph(graphxml):
        # Criar um grafo não direcionado
        G = nx.Graph()

        # Adicionar estados
        G.add_nodes_from(graphxml['states'])

        # Adicionar transições
        for (current_state, target_state), nweight in graphxml['transitions'].items():
            G.add_edge(current_state, target_state, label=nweight, weight=nweight)

        # Definir estados finais
        if 'accepting_states' in graphxml and graphxml['accepting_states']:
            for state in graphxml['accepting_states']:
                G.nodes[state]['accepting_state'] = True
                G.nodes[state]['shape'] = 'doublecircle'

        if 'start' in graphxml and graphxml['start']:
            G.graph['start'] = graphxml['start']
            G.nodes[graphxml['start']]['color'] = 'red'

        # Agrupar nós com mesmo nome e índices diferentes
        node_groups = {}
        for node in G.nodes:
            if '_' in node:
                base_name = node.rsplit('_', 1)[0]  # Obtém o nome sem o índice
                if base_name not in node_groups:
                    node_groups[base_name] = []
                node_groups[base_name].append(node)

        # Gerar cores distintas para cada grupo
        color_palette = itertools.cycle(mcolors.TABLEAU_COLORS.values())
        node_colors = {}

        for base_name, nodes in node_groups.items():
            color = next(color_palette)
            for node in nodes:
                G.nodes[node]['color'] = color

        return G

    @staticmethod
    def generate_segments_between_aabbs(aabbs, step=1.0):
        import numpy as np

        segments = []

        x_min = min(x for (x, _), _, _ in aabbs)
        x_max = max(x + w for (x, _), w, _ in aabbs)
        y_min = min(y for (_, y), _, _ in aabbs)
        y_max = max(y + h for (_, y), _, h in aabbs)

        ys = np.arange(y_min, y_max + step, step)
        xs = np.arange(x_min, x_max + step, step)

        # Horizontais: varrendo de y_min a y_max
        for y in ys:
            cut_ranges = []
            for (aabb_x, aabb_y), w, h in aabbs:
                if aabb_y <= y <= aabb_y + h:
                    cut_ranges.append((aabb_x, aabb_x + w))
            cut_ranges.sort()

            for i in range(len(cut_ranges) - 1):
                x1 = cut_ranges[i][1]
                x2 = cut_ranges[i + 1][0]
                if x2 > x1:
                    segments.append((x1, y, x2, y))

        # Verticais: varrendo de x_min a x_max
        for x in xs:
            cut_ranges = []
            for (aabb_x, aabb_y), w, h in aabbs:
                if aabb_x <= x <= aabb_x + w:
                    cut_ranges.append((aabb_y, aabb_y + h))
            cut_ranges.sort()

            for i in range(len(cut_ranges) - 1):
                y1 = cut_ranges[i][1]
                y2 = cut_ranges[i + 1][0]
                if y2 > y1:
                    segments.append((x, y1, x, y2))

        return segments

    @staticmethod
    def generate_perimeter_segments_and_labeled_points(segments, aabbs, obstacles, threshold=3.0):
        import math
        import random
        from collections import defaultdict

        def point_distance(p1, p2):
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        def nearest_obstacle_label(x, y):
            best_label = None
            best_dist = float("inf")
            for obs in obstacles:
                ox, oy = obs["pos"]
                label = obs["label"]
                dist = math.hypot(x - ox, y - oy)
                if dist < best_dist:
                    best_dist = dist
                    best_label = label
            return best_label

        def is_colinear(p, a, b, tol=1e-6):
            x0, y0 = p
            x1, y1 = a
            x2, y2 = b
            area = abs((x1 * (y2 - y0) + x2 * (y0 - y1) + x0 * (y1 - y2)) / 2.0)
            return area < tol

        def segment_intersects_inside(x1, y1, x2, y2, aabb):
            (ax, ay), w, h = aabb
            ax2 = ax + w
            ay2 = ay + h

            if (x1 < ax and x2 < ax) or (x1 > ax2 and x2 > ax2) or \
                    (y1 < ay and y2 < ay) or (y1 > ay2 and y2 > ay2):
                return False

            on_left = math.isclose(x1, ax) and math.isclose(x2, ax)
            on_right = math.isclose(x1, ax2) and math.isclose(x2, ax2)
            on_bottom = math.isclose(y1, ay) and math.isclose(y2, ay)
            on_top = math.isclose(y1, ay2) and math.isclose(y2, ay2)

            if (on_left or on_right) and (ay <= y1 <= ay2 and ay <= y2 <= ay2):
                return False
            if (on_bottom or on_top) and (ax <= x1 <= ax2 and ax <= x2 <= ax2):
                return False

            return True

        # Mapeamento dos pontos por AABB
        points_by_aabb = defaultdict(set)
        final_segments = list(segments)
        labeled_points = []
        label_counter = 1

        # Passo 1: Associar pontos dos segmentos aos AABBs
        for x1, y1, x2, y2 in segments:
            for i, aabb in enumerate(aabbs):
                (ax, ay), w, h = aabb
                ax2, ay2 = ax + w, ay + h
                label = f"aabb_{i}"
                for px, py in [(x1, y1), (x2, y2)]:
                    on_left = math.isclose(px, ax) and ay <= py <= ay2
                    on_right = math.isclose(px, ax2) and ay <= py <= ay2
                    on_bottom = math.isclose(py, ay) and ax <= px <= ax2
                    on_top = math.isclose(py, ay2) and ax <= px <= ax2
                    if on_left or on_right or on_bottom or on_top:
                        points_by_aabb[label].add((px, py))

        # Passo 2: Gerar novos pontos no perímetro dos AABBs
        for i, aabb in enumerate(aabbs):
            (x1, y1), w, h = aabb
            x2, y2 = x1 + w, y1 + h
            label = f"aabb_{i}"
            all_points = points_by_aabb[label]

            sides = [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                     ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]

            for (sx, sy), (ex, ey) in sides:
                all_points.add((sx, sy))
                all_points.add((ex, ey))
                length = point_distance((sx, sy), (ex, ey))
                steps = max(2, int(length / (threshold / 2)))
                for j in range(steps + 1):
                    px = sx + j * (ex - sx) / steps
                    py = sy + j * (ey - sy) / steps
                    if all(point_distance((px, py), p) >= threshold for p in all_points):
                        all_points.add((px, py))

        # Passo 3: Conectar pontos de cada AABB formando ciclo fechado
        for i, aabb in enumerate(aabbs):
            inner_label_counter = 1
            label = f"aabb_{i}"
            points = list(points_by_aabb[label])
            random.shuffle(points)

            graph = defaultdict(set)
            connection_count = defaultdict(int)

            for p1 in points:
                for p2 in points:
                    if p1 == p2 or p2 in graph[p1]:
                        continue
                    if len(graph[p1]) >= 2 or len(graph[p2]) >= 2:
                        continue
                    if segment_intersects_inside(p1[0], p1[1], p2[0], p2[1], aabb):
                        continue
                    if any(
                            is_colinear(p, p1, p2) and
                            point_distance(p1, p, ) + point_distance(p, p2) <= point_distance(p1, p2) + 1e-6
                            for p in points if p != p1 and p != p2
                    ):
                        continue
                    graph[p1].add(p2)
                    graph[p2].add(p1)
                    final_segments.append((p1[0], p1[1], p2[0], p2[1]))

            # Verificação de ciclo fechado
            degrees = [len(neigh) for neigh in graph.values()]
            if not all(deg == 2 for deg in degrees):
                print(f"[!] AABB {label} NÃO formou ciclo fechado com {len(points)} pontos")

            # Adicionar rótulos
            for p in points:
                label_obs = nearest_obstacle_label(p[0], p[1])
                labeled_points.append((p[0], p[1], f"{label_obs}.{label_counter}"))
                inner_label_counter += 1
                label_counter += 1

        return final_segments, labeled_points

    @staticmethod
    def connect_adjacent_perimeter_points(aabbs, obstacles, points, threshold=5.0):
        """
        Conecta pontos de perímetro entre AABBs diretamente vizinhos
        (esquerda, direita, acima, abaixo), se a distância for < threshold.

        Retorna segmentos no formato:
            {
                "start": (x1, y1),
                "end": (x2, y2),
                "label": None,
                "tipo": "ponte"
            }
        """
        import math
        from collections import defaultdict

        # Mapeia cada AABB ao label do obstáculo mais próximo
        aabb_map = {}
        for ((x, y), w, h) in aabbs:
            cx, cy = x + w / 2, y + h / 2
            min_dist = float("inf")
            best_label = None
            for obs in obstacles:
                ox, oy = obs["pos"]
                label = obs.get("label", "unknown")
                dist = math.hypot(cx - ox, cy - oy)
                if dist < min_dist:
                    min_dist = dist
                    best_label = label
            if best_label:
                aabb_map[best_label] = {"x": x, "y": y, "w": w, "h": h}

        # Organiza os pontos por label do obstáculo
        label_to_points = defaultdict(list)
        for x, y, label in points:
            label_to_points[label].append((x, y))

        new_segments = []

        def is_neighbor(l1, l2, tolerance=12.0):  # Tolerância em metros
            a = aabb_map.get(l1)
            b = aabb_map.get(l2)
            if not a or not b:
                return False

            ax1, ay1, aw1, ah1 = a["x"], a["y"], a["w"], a["h"]
            bx1, by1, bw1, bh1 = b["x"], b["y"], b["w"], b["h"]

            ax2 = ax1 + aw1
            ay2 = ay1 + ah1
            bx2 = bx1 + bw1
            by2 = by1 + bh1

            # Adjacência horizontal (esquerda-direita)
            horizontal_adjacent = (
                    (abs(ax2 - bx1) <= tolerance or abs(bx2 - ax1) <= tolerance) and
                    not (ay2 < by1 or by2 < ay1)  # sobreposição vertical
            )

            # Adjacência vertical (cima-baixo)
            vertical_adjacent = (
                    (abs(ay2 - by1) <= tolerance or abs(by2 - ay1) <= tolerance) and
                    not (ax2 < bx1 or bx2 < ax1)  # sobreposição horizontal
            )

            return horizontal_adjacent or vertical_adjacent

        for label_a, pts_a in label_to_points.items():
            for label_b, pts_b in label_to_points.items():
                if is_neighbor(label_a, label_b):
                    print(f"👉 AABBs vizinhos: {label_a} <--> {label_b}")
                if label_a == label_b:
                    continue
                if not is_neighbor(label_a, label_b):
                    continue

                for (x1, y1) in pts_a:
                    for (x2, y2) in pts_b:
                        d = math.hypot(x2 - x1, y2 - y1)
                        if d < threshold:
                            new_segments.append({
                                "start": (x1, y1),
                                "end": (x2, y2),
                                "label": None,
                                "tipo": "ponte"
                            })

        return new_segments




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
        import json
        import networkx as nx

        with open(filename, "r") as f:
            graph_data = json.load(f)

        G = nx.Graph()

        # Restaurar nós com labels
        for node_str, attr in graph_data["nodes"].items():
            # 'node_str' é a chave do dicionário - ou seja, o "nome" do nó como string
            # Basta usá-la diretamente como ID do nó
            G.add_node(node_str, **attr)

        # Restaurar arestas com peso
        for u_str, v_str, weight in graph_data["edges"]:
            # Também usamos as strings diretamente
            G.add_edge(u_str, v_str, weight=weight)

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
    def has_islands(G):
        """
        Verifica se o grafo G possui ilhas (componentes desconexos).
        Retorna:
            - True se houver mais de uma ilha;
            - False se for totalmente conectado;
            - Lista com os componentes desconexos.
        """
        components = list(nx.connected_components(G))
        has_isolated = len(components) > 1
        return has_isolated

    @staticmethod
    def create_graph_with_passage_points_new(segments, passage_points, points, obstacles):
        import networkx as nx
        from collections import defaultdict
        import math

        G = nx.Graph()
        obstacle_node_dict = defaultdict(list)
        node_labels = {}
        passage_label_map = {}
        point_label_map = {}

        # Conjunto de pontos rotulados manualmente (x, y) → label
        for px, py, label in points:
            # Usamos round para evitar floats muito extensos
            point_label_map[(round(px, 4), round(py, 4))] = label

        # Conjunto de pontos de interseção (x, y) sem rótulo
        passage_points_set = set((round(px, 4), round(py, 4)) for px, py in passage_points)
        label_counter = 1

        # Coletar todos os nós (tuplas) que aparecem nos segmentos
        unique_nodes = set()
        for x1, y1, x2, y2 in segments:
            p1 = (round(x1, 4), round(y1, 4))
            p2 = (round(x2, 4), round(y2, 4))
            unique_nodes.add(p1)
            unique_nodes.add(p2)

        # A) CRIAR LABEL PARA CADA (x,y)
        #    E ADICIONAR NO GRAFO USANDO label COMO ID DO NÓ
        for node_pos in sorted(unique_nodes):
            if node_pos in point_label_map:
                label = point_label_map[node_pos]
            elif node_pos in passage_points_set:
                label = f"pp_{label_counter}"
                label_counter += 1
            else:
                nearest_label = SegmentUtils.get_nearest_obstacle_label(node_pos, obstacles)
                index = len(obstacle_node_dict[nearest_label])
                label = f"{nearest_label}_{index}"
                obstacle_node_dict[nearest_label].append(node_pos)

            # Guardar "label" associado a essa coord.
            node_labels[node_pos] = label

            # Adicionar nó ao grafo: nome = label
            # Atributos:
            #   - label: a string de identificação
            #   - pos:   a tupla (x, y)
            G.add_node(label, label=label, pos=node_pos)

        # B) CRIAR AS ARESTAS USANDO OS LABELS DE p1 E p2
        for x1, y1, x2, y2 in segments:
            p1 = (round(x1, 4), round(y1, 4))
            p2 = (round(x2, 4), round(y2, 4))
            if p1 not in node_labels or p2 not in node_labels:
                # Algum ponto não foi rotulado, ignora
                continue

            label1 = node_labels[p1]
            label2 = node_labels[p2]
            dist = math.dist(p1, p2)

            G.add_edge(label1, label2, weight=dist)

        return G

    @staticmethod
    def create_graph_with_passage_points(segments, passage_points, points, obstacles):
        import networkx as nx
        from collections import defaultdict
        import math

        G = nx.Graph()
        obstacle_node_dict = defaultdict(list)
        node_labels = {}
        passage_label_map = {}
        point_label_map = {}

        # Conjunto de pontos rotulados manualmente (x, y) → label
        for px, py, label in points:
            point_label_map[(round(px, 4), round(py, 4))] = label

        # Conjunto de pontos de interseção (x, y) sem rótulo
        passage_points_set = set((round(px, 4), round(py, 4)) for px, py in passage_points)
        label_counter = 1

        unique_nodes = set()
        for x1, y1, x2, y2 in segments:
            p1 = (round(x1, 4), round(y1, 4))
            p2 = (round(x2, 4), round(y2, 4))
            unique_nodes.add(p1)
            unique_nodes.add(p2)

        for node in sorted(unique_nodes):
            if node in point_label_map:
                label = point_label_map[node]
            elif node in passage_points_set:
                label = f"pp_{label_counter}"
                label_counter += 1
            else:
                nearest_label = SegmentUtils.get_nearest_obstacle_label(node, obstacles)
                index = len(obstacle_node_dict[nearest_label])
                label = f"{nearest_label}_{index}"
                obstacle_node_dict[nearest_label].append(node)

            node_labels[node] = label
            G.add_node(node, label=label)

        for x1, y1, x2, y2 in segments:
            p1 = (round(x1, 4), round(y1, 4))
            p2 = (round(x2, 4), round(y2, 4))
            dist = math.dist(p1, p2)
            G.add_edge(p1, p2, weight=dist)

        return G

    @staticmethod
    def resolve_segment_intersections(segments, threshold=1.0):
        def is_horizontal(s):
            return math.isclose(s[1], s[3], abs_tol=1e-6)

        def is_vertical(s):
            return math.isclose(s[0], s[2], abs_tol=1e-6)

        def distance(p1, p2):
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        horizontal_segments = []
        vertical_segments = []
        for seg in segments:
            if is_horizontal(seg):
                horizontal_segments.append(seg)
            elif is_vertical(seg):
                vertical_segments.append(seg)

        new_segments = []
        new_points = []

        for h in horizontal_segments:
            xh1, yh, xh2, _ = h
            xh_min, xh_max = sorted([xh1, xh2])

            for v in vertical_segments:
                xv, yv1, _, yv2 = v
                yv_min, yv_max = sorted([yv1, yv2])

                # Testa se intersectam
                if (xh_min < xv < xh_max) and (yv_min < yh < yv_max):
                    ip = (xv, yh)

                    # Checar se as divisões seriam válidas
                    if (
                            distance((xh1, yh), ip) < threshold or
                            distance((xh2, yh), ip) < threshold or
                            distance((xv, yv1), ip) < threshold or
                            distance((xv, yv2), ip) < threshold
                    ):
                        continue

                    # Quebrar h e v em 2 cada
                    new_segments.extend([
                        (xh1, yh, xv, yh),  # h1
                        (xv, yh, xh2, yh),  # h2
                        (xv, yv1, xv, yh),  # v1
                        (xv, yh, xv, yv2)  # v2
                    ])
                    new_points.append(ip)
                else:
                    # Sem interseção: manter originais
                    continue

        # Agora precisamos adicionar os segmentos que **não foram quebrados**
        # Ou seja, aqueles que não participaram de interseção

        broken_set = set()
        for s in new_segments:
            broken_set.add(((s[0], s[1]), (s[2], s[3])))

        # Para evitar duplicidade, normalizamos extremidades
        def normalize(p1, p2):
            return tuple(sorted([p1, p2]))

        original_set = set()
        for s in segments:
            p1 = (s[0], s[1])
            p2 = (s[2], s[3])
            original_set.add(normalize(p1, p2))

        new_normalized = set(normalize((s[0], s[1]), (s[2], s[3])) for s in new_segments)
        untouched = original_set - new_normalized

        untouched_segments = [(p1[0], p1[1], p2[0], p2[1]) for p1, p2 in untouched]
        final_segments = untouched_segments + new_segments

        return final_segments, new_points

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
            G.add_edge((x2, y2), (x1, y1), weight=distance)

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
    def fix_missing_connections(G, aabbs):
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
                        if SegmentUtils.path_is_clear(n, neighbor_coord, aabbs):
                            G.add_edge(n, neighbor_coord, weight=distance_value)
                            G.add_edge(neighbor_coord, n, weight=distance_value)

        return G  # Retorna para testes

    @staticmethod
    def path_is_clear(n1, n2, aabbs):
        x1, y1 = n1
        x2, y2 = n2
        for aabb in aabbs:
            if SegmentUtils.segment_intersects_aabb(x1, y1, x2, y2, aabb):
                return False
        return True

    @staticmethod
    def segment_intersects_aabb(x1, y1, x2, y2, aabb):
        (ax, ay), aw, ah = aabb
        rx1, ry1 = ax, ay
        rx2, ry2 = ax + aw, ay + ah

        # Cohen-Sutherland line clipping ou teste rápido de separação
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        rect_edges = [
            ((rx1, ry1), (rx2, ry1)),
            ((rx2, ry1), (rx2, ry2)),
            ((rx2, ry2), (rx1, ry2)),
            ((rx1, ry2), (rx1, ry1)),
        ]

        for (p1, p2) in rect_edges:
            if intersect((x1, y1), (x2, y2), p1, p2):
                return True
        return False

    @staticmethod
    def build_safe_graph(G, aabbs):
        G_safe = nx.Graph()
        for u, v, data in G.edges(data=True):
            if SegmentUtils.path_is_clear(u, v, aabbs):
                G_safe.add_edge(u, v, **data)
        for n, attrs in G.nodes(data=True):
            G_safe.add_node(n, **attrs)
        return G_safe

    @staticmethod
    def fix_missing_connections_safe_new(G, aabbs):
        import math
        from collections import defaultdict

        # 1) G_safe é um grafo "seguro" para procurar caminhos
        #    Presumimos que 'build_safe_graph' também cria um Graph
        #    cujos nós são nomes (strings), e que cada nó
        #    possua G_safe.nodes[node]["pos"] = (x, y).
        G_safe = SegmentUtils.build_safe_graph(G, aabbs)

        # 2) Agrupa nós pelo “base” do label (tudo antes do último "_")
        groups = defaultdict(list)
        for node in G.nodes():
            label = G.nodes[node].get("label", "unknown")
            base = label.rsplit("_", 1)[0]  # Divide a partir do último '_'
            groups[base].append(node)

        # 3) Para cada grupo, se o nó n estiver “isolado” (sem vizinhos do mesmo base),
        #    procura nós mais próximos do mesmo grupo e cria caminhos.
        for base, nodes in groups.items():
            for n in nodes:
                neighbors = list(G.neighbors(n))
                cluster_neighbors = []
                for nbr in neighbors:
                    nbr_label = G.nodes[nbr].get("label", "")
                    nbr_base = nbr_label.rsplit("_", 1)[0]
                    if nbr_base == base:
                        cluster_neighbors.append(nbr)

                # Se não há nenhum vizinho do mesmo cluster, consideramos "isolado"
                if not cluster_neighbors:
                    # Vamos calcular a distância de n para os outros do mesmo base
                    # Extraindo as posições (x, y)
                    pxn, pyn = G.nodes[n]["pos"]  # n é string, mas pos é tupla

                    distances = []
                    for other in nodes:
                        if other == n:
                            continue
                        pxo, pyo = G.nodes[other]["pos"]
                        d = math.dist((pxn, pyn), (pxo, pyo))
                        distances.append((other, d))

                    # Pega os 2 nós mais próximos
                    closest = sorted(distances, key=lambda x: x[1])[:2]

                    # 4) Para cada “mais próximo”, tenta achar path no G_safe
                    for target, _ in closest:
                        if G_safe.has_node(n) and G_safe.has_node(target):
                            try:
                                path = nx.shortest_path(
                                    G_safe, source=n, target=target, weight='weight'
                                )
                                # Adiciona arestas desse path no grafo original G
                                for i in range(len(path) - 1):
                                    u, v = path[i], path[i + 1]
                                    if not G.has_edge(u, v):
                                        # Distância entre pos(u) e pos(v)
                                        pxu, pyu = G.nodes[u]["pos"]
                                        pxv, pyv = G.nodes[v]["pos"]
                                        dist_uv = math.dist((pxu, pyu), (pxv, pyv))
                                        G.add_edge(u, v, weight=dist_uv)

                            except nx.NetworkXNoPath:
                                continue

        return G

    @staticmethod
    def fix_missing_connections_safe(G, aabbs):
        from collections import defaultdict
        G_safe = SegmentUtils.build_safe_graph(G, aabbs)

        groups = defaultdict(list)
        for node in G.nodes():
            label = G.nodes[node].get("label", "unknown")
            base = label.rsplit("_", 1)[0]
            groups[base].append(node)

        for base, nodes in groups.items():
            for n in nodes:
                neighbors = list(G.neighbors(n))
                cluster_neighbors = [nbr for nbr in neighbors
                                     if G.nodes[nbr].get("label", "").rsplit("_", 1)[0] == base]

                if not cluster_neighbors:
                    # Isolado
                    distances = [(other, math.dist(n, other)) for other in nodes if other != n]
                    closest = sorted(distances, key=lambda x: x[1])[:2]

                    for target, _ in closest:
                        if G_safe.has_node(n) and G_safe.has_node(target):
                            try:
                                path = nx.shortest_path(G_safe, source=n, target=target, weight='weight')
                                # Se for possível, adiciona as arestas do caminho no grafo original
                                for i in range(len(path) - 1):
                                    u, v = path[i], path[i + 1]
                                    if not G.has_edge(u, v):
                                        dist = math.dist(u, v)
                                        G.add_edge(u, v, weight=dist)
                            except nx.NetworkXNoPath:
                                continue
        return G

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
        for u, v in G.edges():
            # Verifica se ambos os nós têm atributo 'pos'
            if "pos" in G.nodes[u] and "pos" in G.nodes[v]:
                x1, y1 = G.nodes[u]["pos"]
                x2, y2 = G.nodes[v]["pos"]
                segments.append((x1, y1, x2, y2))
        return segments


    def graph_to_segments_old(G):
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
    def get_paths(aabbs, x_min, x_max, y_min, y_max, comments = False):
        """
        Gera caminhos horizontais e verticais que tangenciam o topo, fundo,
        esquerda e direita de cada AABB.
        """
        if comments:
            print("🔹 Gerando caminhos horizontais e verticais...")

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
    def split_and_filter_paths(horizontal_paths, vertical_paths, aabbs, comments=false):
        """
        Lógica par-ímpar => subsegmentos fora da AABB.
        """
        if comments:
            print("🔹 Subdividindo caminhos fora das AABBs...")

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
