import math
import numpy as np
import networkx as nx
from shapely.geometry import box
from shapely.ops import unary_union
import matplotlib.pyplot as plt

###############################################################################
# CLASSE: AABBUtils
###############################################################################
class AABBUtils:

    @staticmethod
    def connect_neighbor_aabbs(aabbs, points, threshold):
        """
        Conecta AABBs vizinhas usando pontos existentes nos perímetros.

        Parâmetros:
            - aabbs: Lista de AABBs [((x,y),w,h), ...]
            - points: Lista de pontos [(x,y,label)]
            - threshold: Distância máxima para conexão

        Retorna:
            - Lista de novos segmentos [(x1,y1,x2,y2)]
        """
        from collections import defaultdict
        import math

        # Primeiro, mapeia cada ponto à sua AABB correspondente
        point_to_aabb = []
        for (x, y, label) in points:
            point_assigned = False
            for idx, ((ax, ay), aw, ah) in enumerate(aabbs):
                # Verifica se o ponto está dentro dos limites da AABB (com tolerância)
                inside_aabb = (ax - 0.1 <= x <= ax + aw + 0.1) and (ay - 0.1 <= y <= ay + ah + 0.1)

                if not inside_aabb:
                    continue

                # Verifica se está no perímetro (bordas da AABB)
                on_left_edge = math.isclose(x, ax, abs_tol=0.15)
                on_right_edge = math.isclose(x, ax + aw, abs_tol=0.15)
                on_top_edge = math.isclose(y, ay + ah, abs_tol=0.15)
                on_bottom_edge = math.isclose(y, ay, abs_tol=0.15)

                on_perimeter = (on_left_edge or on_right_edge or on_top_edge or on_bottom_edge)

                if on_perimeter:
                    point_to_aabb.append((x, y, idx))
                    point_assigned = True
                    break

            if not point_assigned:
                # Ponto está dentro da AABB mas não no perímetro - podemos ignorar?
                pass

        # Agrupa pontos por AABB
        aabb_points = defaultdict(list)
        for x, y, aabb_idx in point_to_aabb:
            aabb_points[aabb_idx].append((x, y))

        segments = []

        # Verifica vizinhança entre cada par de AABBs
        for i in range(len(aabbs)):
            for j in range(i + 1, len(aabbs)):
                if not AABBUtils.are_direct_neighbors(aabbs[i], aabbs[j]):
                    continue

                # Encontra o par de pontos mais próximo entre as AABBs
                min_dist = float('inf')
                best_pair = None

                for p1 in aabb_points.get(i, []):
                    for p2 in aabb_points.get(j, []):
                        dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                        if dist < min_dist and dist <= threshold:
                            if AABBUtils.segment_clear_of_aabbs(p1, p2, aabbs):
                                min_dist = dist
                                best_pair = (p1, p2)

                if best_pair:
                    segments.append((*best_pair[0], *best_pair[1]))

        return segments

    @staticmethod
    def are_direct_neighbors(aabb1, aabb2):
        """
        Verifica se duas AABBs são vizinhas diretas (esquerda, direita, cima ou abaixo)
        """
        ((x1, y1), w1, h1) = aabb1
        ((x2, y2), w2, h2) = aabb2

        x1_end = x1 + w1
        y1_end = y1 + h1
        x2_end = x2 + w2
        y2_end = y2 + h2

        # Vizinho à esquerda/direita
        horizontal_neighbor = (
                (math.isclose(x1, x2_end) or math.isclose(x1_end, x2)) and
                not (y1_end < y2 or y2_end < y1)
        )

        # Vizinho acima/abaixo
        vertical_neighbor = (
                (math.isclose(y1, y2_end) or math.isclose(y1_end, y2)) and
                not (x1_end < x2 or x2_end < x1)
        )

        return horizontal_neighbor or vertical_neighbor

    @staticmethod
    def segment_clear_of_aabbs(p1, p2, aabbs):
        """
        Verifica se o segmento entre p1 e p2 não intersecta nenhuma AABB
        """
        x1, y1 = p1
        x2, y2 = p2

        for aabb in aabbs:
            if AABBUtils.segment_intersects_aabb(x1, y1, x2, y2, aabb):
                return False
        return True

    @staticmethod
    def segment_intersects_aabb(x1, y1, x2, y2, aabb):
        """
        Verifica se um segmento de linha intersecta uma AABB
        """
        (ax, ay), aw, ah = aabb
        rx1, ry1 = ax, ay
        rx2, ry2 = ax + aw, ay + ah

        # Teste de separação de eixos
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        rect_edges = [
            ((rx1, ry1), (rx2, ry1)),
            ((rx2, ry1), (rx2, ry2)),
            ((rx2, ry2), (rx1, ry2)),
            ((rx1, ry2), (rx1, ry1))
        ]

        for (p1, p2) in rect_edges:
            if intersect((x1, y1), (x2, y2), p1, p2):
                return True
        return False

    @staticmethod
    def generate_aabbs_perimeter_observation_points(obstacles, margin, threshold=2.0):
        """
        Gera pontos ao longo do perímetro de cada AABB com distância máxima entre pontos ≤ threshold.
        Para cada ponto, associa o label do obstáculo mais próximo.

        Parâmetros:
            - obstacles: lista de {"pos": (x, y), "label": ...}
            - threshold: distância máxima entre pontos consecutivos no perímetro

        Retorna:
            - Lista de AABBs [((x, y), w, h), ...]
            - Lista de pontos [(x, y, label_obstaculo)]
            - Lista de segmentos [(x1, y1, x2, y2)]
        """
        aabbs = AABBUtils.get_aabbs(obstacles, margin)
        points = []
        segments = []

        def nearest_obstacle_label(x, y):
            min_dist = float("inf")
            best_label = None
            for obs in obstacles:
                ox, oy = obs["pos"]
                label = obs.get("label", "unknown")
                dist = math.hypot(x - ox, y - oy)
                if dist < min_dist:
                    min_dist = dist
                    best_label = label
            return best_label

        for (aabb_x, aabb_y), w, h in aabbs:
            # Lados da AABB (top, right, bottom, left)
            edges = [
                ((aabb_x, aabb_y + h), (aabb_x + w, aabb_y + h)),  # Top
                ((aabb_x + w, aabb_y + h), (aabb_x + w, aabb_y)),  # Right
                ((aabb_x + w, aabb_y), (aabb_x, aabb_y)),  # Bottom
                ((aabb_x, aabb_y), (aabb_x, aabb_y + h))  # Left
            ]

            for start, end in edges:
                x1, y1 = start
                x2, y2 = end
                dist = math.hypot(x2 - x1, y2 - y1)
                num_points = max(1, int(math.ceil(dist / threshold)))

                edge_points = [
                    (x1 + i * (x2 - x1) / num_points,
                     y1 + i * (y2 - y1) / num_points)
                    for i in range(num_points + 1)
                ]

                for i in range(len(edge_points) - 1):
                    segments.append((*edge_points[i], *edge_points[i + 1]))

                for (x, y) in edge_points:
                    points.append((x, y, nearest_obstacle_label(x, y)))

        return aabbs, points, segments

    @staticmethod
    def convert_graph_to_dict(G):
        """
        Converte um grafo NetworkX em um dicionário no formato especificado,
        mas pegando o nome do nó a partir de G.nodes[node]["label"] (se existir).
        """
        grafo_mapa = {"states": set(), "transitions": {}}

        # Para cada nó do grafo, obtemos label do atributo ou, se não houver, o próprio node
        for node in G.nodes():
            label_node = G.nodes[node].get("label", node)
            grafo_mapa["states"].add(label_node)

        # Para cada aresta, também obtemos 'label' do nó de origem e destino
        for u, v, data in G.edges(data=True):
            label_u = G.nodes[u].get("label", u)
            label_v = G.nodes[v].get("label", v)
            weight = data.get("weight", 1.0)

            grafo_mapa["transitions"][(label_u, label_v)] = (weight, 1)
            grafo_mapa["transitions"][(label_v, label_u)] = (weight, 1)

        return grafo_mapa


    # @staticmethod
    # def merge_overlapping_aabbs(aabbs):
    #     merged = []
    #     while aabbs:
    #         base = aabbs.pop(0)
    #         bx, by = base[0]
    #         bw, bh = base[1], base[2]
    #         merged_flag = False

    #         for i, (other_pos, other_w, other_h) in enumerate(merged):
    #             ox, oy = other_pos
    #             # Se sobrepõem
    #             if not (bx + bw < ox or ox + other_w < bx or by + bh < oy or oy + other_h < by):
    #                 new_x = min(bx, ox)
    #                 new_y = min(by, oy)
    #                 new_w = max(bx + bw, ox + other_w) - new_x
    #                 new_h = max(by + bh, oy + other_h) - new_y
    #                 merged[i] = ((new_x, new_y), new_w, new_h)
    #                 merged_flag = True
    #                 break

    #         if not merged_flag:
    #             merged.append(base)
    #     return merged
    @staticmethod
    def plot_union_polygon(union_coords, color='skyblue'):
        x, y = zip(*union_coords)
        plt.figure(figsize=(6, 6))
        plt.fill(x, y, color=color, edgecolor='black', linewidth=1.5)
        plt.plot(x, y, color='black')  # contorno
        plt.title("Polígono resultante da união de AABBs")
        plt.axis('equal')
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def merge_overlapping_aabbs(aabbs, threshold):
        merged = []
        while aabbs:
            base = aabbs.pop(0)
            bx, by = base[0]
            bw, bh = base[1], base[2]
            merged_flag = False

            for i, (other_pos, other_w, other_h) in enumerate(merged):
                ox, oy = other_pos

                # Se sobrepõem
                if not (bx + bw < ox or ox + other_w < bx or by + bh < oy or oy + other_h < by):

                    new_x = min(bx, ox)
                    new_y = min(by, oy)
                    new_w = max(bx + bw, ox + other_w) - new_x
                    new_h = max(by + bh, oy + other_h) - new_y

                    if new_w > threshold or new_h > threshold:
                        break

                    merged[i] = ((new_x, new_y), new_w, new_h)
                    merged_flag = True
                    break

            if not merged_flag:
                merged.append(base)
        return merged

    @staticmethod
    def get_aabbs(obstacles, margin, threshold=6.0):
        """
        Cria AABBs a partir de obstacles, adicionando 'margin'.
        Retorna lista [((ax, ay), w, h, label), ...].
        """
        aabbs = []
        for obs in obstacles:
            x, y = obs["pos"]
            w, h = obs["size"]
            
            # Canto inferior esquerdo da AABB
            aabb_x = x - (w / 2 + margin)
            aabb_y = y - (h / 2 + margin)
            aabb_w = w + 2 * margin
            aabb_h = h + 2 * margin

            # Agora cada AABB carrega o label do obstáculo original
            aabbs.append(((aabb_x, aabb_y), aabb_w, aabb_h))

        # Dupla fusão para garantir, mantendo os labels junto dos AABBs
        merged_aabbs = AABBUtils.merge_overlapping_aabbs(
            AABBUtils.merge_overlapping_aabbs(aabbs, threshold), threshold
        )

        return merged_aabbs

    @staticmethod
    def distance_between_aabbs(aabb1, aabb2):
        """
        Distância min entre 2 AABBs (ax, ay, w, h). Se sobrepõem => 0.
        """
        ((ax1, ay1), w1, h1) = aabb1
        ((ax2, ay2), w2, h2) = aabb2

        x1_min, x1_max = ax1, ax1 + w1
        y1_min, y1_max = ay1, ay1 + h1
        x2_min, x2_max = ax2, ax2 + w2
        y2_min, y2_max = ay2, ay2 + h2

        # sobrepõe => 0
        overlap_x = not (x1_max < x2_min or x2_max < x1_min)
        overlap_y = not (y1_max < y2_min or y2_max < y1_min)
        if overlap_x and overlap_y:
            return 0.0

        # dist em X
        if x1_max < x2_min:
            dx = x2_min - x1_max
        elif x2_max < x1_min:
            dx = x1_min - x2_max
        else:
            dx = 0.0

        # dist em Y
        if y1_max < y2_min:
            dy = y2_min - y1_max
        elif y2_max < y1_min:
            dy = y1_min - y2_max
        else:
            dy = 0.0

        return math.hypot(dx, dy)

    @staticmethod
    def cluster_aabbs_scipy(aabbs, threshold, method='single'):
        """
        Clusteriza AABBs via scipy (hierárquico).
        Retorna array 'labels'.
        """
        n = len(aabbs)
        if n <= 1:
            return [1]*n

        dist_matrix = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1, n):
                dist = AABBUtils.distance_between_aabbs(aabbs[i], aabbs[j])
                dist_matrix[i,j] = dist
                dist_matrix[j,i] = dist

        from scipy.spatial.distance import squareform
        from scipy.cluster.hierarchy import linkage, fcluster

        dist_cond = squareform(dist_matrix, checks=False)
        Z = linkage(dist_cond, method=method)
        labels = fcluster(Z, t=threshold, criterion='distance')
        return labels