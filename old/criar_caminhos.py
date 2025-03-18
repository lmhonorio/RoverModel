import numpy as np
import matplotlib.pyplot as plt
import time
from roverclass import ObstacleLoader
import math
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.cm as cm

def subdivide_edge(x1, y1, x2, y2, threshold):
    """
    Recebe uma aresta (x1,y1) -> (x2,y2) (horizontal ou vertical)
    e subdivide em segmentos de comprimento máx. 'threshold'.
    Retorna lista de subsegmentos [(xA, yA, xB, yB), ...].
    """
    # Comprimento total da aresta
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)

    # Evitar subdivisão se a aresta é degenerada
    if length == 0:
        return []

    # Número de subdivisões
    n = math.ceil(length / threshold)

    # Passo fracionário [0..1] para cada subsegmento
    subsegments = []
    step = 1.0 / n

    for i in range(n):
        tA = i * step
        tB = (i+1) * step

        # Coordenadas do ponto A e ponto B no espaço
        Ax = x1 + dx * tA
        Ay = y1 + dy * tA
        Bx = x1 + dx * tB
        By = y1 + dy * tB

        subsegments.append((Ax, Ay, Bx, By))

    return subsegments

def create_perimeter_segments(aabb, threshold):
    """
    Recebe aabb = ((ax, ay), aw, ah).
    Retorna lista de subsegmentos (x1,y1,x2,y2) que percorrem
    todo o perímetro em sentido anti-horário (ou horário),
    subdividindo cada aresta com max. length = threshold.
    """
    (ax, ay), aw, ah = aabb
    # Se aw<=0 ou ah<=0, sem perímetro válido
    if aw <= 0 or ah <= 0:
        return []

    # 4 arestas
    top_left     = (ax, ay)
    top_right    = (ax + aw, ay)
    bottom_right = (ax + aw, ay + ah)
    bottom_left  = (ax, ay + ah)

    segments = []
    # 1) Top
    segments += subdivide_edge(top_left[0], top_left[1], top_right[0], top_right[1], threshold)
    # 2) Right
    segments += subdivide_edge(top_right[0], top_right[1], bottom_right[0], bottom_right[1], threshold)
    # 3) Bottom
    segments += subdivide_edge(bottom_right[0], bottom_right[1], bottom_left[0], bottom_left[1], threshold)
    # 4) Left
    segments += subdivide_edge(bottom_left[0], bottom_left[1], top_left[0], top_left[1], threshold)

    return segments


def add_perimeter_segments(aabbs, threshold, segments):
    """
    Para cada AABB em 'aabbs', gera subsegmentos do perímetro
    de comprimento <= threshold e adiciona a 'segments'.

    Retorna a lista 'segments' atualizada (pode ser a mesma).
    """
    for aabb in aabbs:
        subs = create_perimeter_segments(aabb, threshold)
        segments.extend(subs)
    return segments


def orientation_and_length(segment):
    """
    Dado um segmento (x1, y1, x2, y2),
    retorna ('H', length) se for horizontal,
    ('V', length) se for vertical,
    ou None se não for estritamente horizontal nem vertical.

    length é sempre positivo (abs).
    """
    x1, y1, x2, y2 = segment
    if math.isclose(y1, y2, abs_tol=1e-9):
        # Horizontal
        length = abs(x2 - x1)
        return ('H', length)
    elif math.isclose(x1, x2, abs_tol=1e-9):
        # Vertical
        length = abs(y2 - y1)
        return ('V', length)
    else:
        # Nem horizontal nem vertical
        return None

def segment_center(segment):
    """
    Retorna (mx, my), o centro do segmento (x1,y1,x2,y2).
    """
    x1, y1, x2, y2 = segment
    return ((x1 + x2)/2, (y1 + y2)/2)

def min_dist_center_to_aabbs(segment, aabbs, distance_point_to_aabb):
    """
    Calcula a distância do centro do segmento ao AABB mais próximo.
    Utiliza 'distance_point_to_aabb(px,py,aabb)' já existente.
    """
    mx, my = segment_center(segment)
    dist_min = float('inf')
    for aabb in aabbs:
        d = distance_point_to_aabb(mx, my, aabb)
        if d < dist_min:
            dist_min = d
    return dist_min

def center_distance(s1, s2):
    """
    Distância euclidiana entre os centros dos segmentos s1 e s2.
    """
    c1 = segment_center(s1)
    c2 = segment_center(s2)
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    return math.hypot(dx, dy)

def distance_between_aabbs(aabb1, aabb2):
    """
    Retorna a menor distância entre duas AABBs (ax, ay, w, h),
    assumindo (ax, ay) como canto superior esquerdo.
    Se elas se sobrepõem, a distância é 0.
    """
    ((ax1, ay1), w1, h1) = aabb1
    ((ax2, ay2), w2, h2) = aabb2

    # bounding box 1
    x1_min, x1_max = ax1, ax1 + w1
    y1_min, y1_max = ay1, ay1 + h1

    # bounding box 2
    x2_min, x2_max = ax2, ax2 + w2
    y2_min, y2_max = ay2, ay2 + h2

    # Se há sobreposição => dist = 0
    overlap_x = not (x1_max < x2_min or x2_max < x1_min)
    overlap_y = not (y1_max < y2_min or y2_max < y1_min)
    if overlap_x and overlap_y:
        return 0.0

    # Distância em X
    if x1_max < x2_min:
        dx = x2_min - x1_max
    elif x2_max < x1_min:
        dx = x1_min - x2_max
    else:
        dx = 0.0

    # Distância em Y
    if y1_max < y2_min:
        dy = y2_min - y1_max
    elif y2_max < y1_min:
        dy = y1_min - y2_max
    else:
        dy = 0.0

    return math.sqrt(dx * dx + dy * dy)


def filter_similar_segments(segments, aabbs, parallel_threshold, distance_point_to_aabb):
    """
    1) Agrupa segmentos por orientação (H/V) e mesmo comprimento.
    2) Em cada grupo, se a distância entre centros de dois segmentos < parallel_threshold,
       eles estão 'conectados'. Formamos clusters.
    3) Em cada cluster, só mantemos 1: o que tem o menor dist do centro até o AABB mais próximo.
    4) Retorna a nova lista de segmentos após essa filtragem.

    segments: [(x1, y1, x2, y2), ...]
    aabbs: [((ax, ay), w, h), ...]
    parallel_threshold: float
    distance_point_to_aabb: função existente p/ dist ponto->AABB
    """
    # 1) Agrupar por (orientation, length)
    groups = {}  # dict => (orientation, length) -> lista de segments
    for seg in segments:
        key = orientation_and_length(seg)
        if not key:
            # ignora se não for H nem V
            continue
        # key = ('H', length) ou ('V', length)
        if key not in groups:
            groups[key] = []
        groups[key].append(seg)

    filtered_segments = []

    # 2) Para cada grupo, cria grafo e clusteriza
    for key, segs in groups.items():
        n = len(segs)
        if n == 1:
            # só 1 segmento => mantemos
            filtered_segments.append(segs[0])
            continue

        # Construir lista de adjacência
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                dist_c = center_distance(segs[i], segs[j])
                if dist_c < parallel_threshold:
                    # Conectar i e j
                    adj[i].append(j)
                    adj[j].append(i)

        visited = [False]*n

        # BFS/DFS => clusters
        for start in range(n):
            if not visited[start]:
                queue = [start]
                visited[start] = True
                cluster = [start]
                while queue:
                    curr = queue.pop(0)
                    for neigh in adj[curr]:
                        if not visited[neigh]:
                            visited[neigh] = True
                            queue.append(neigh)
                            cluster.append(neigh)
                # cluster pronto => manter só 1
                best_seg = None
                best_dist = float('inf')
                for idx in cluster:
                    seg_ = segs[idx]
                    dist_center = min_dist_center_to_aabbs(seg_, aabbs, distance_point_to_aabb)
                    if dist_center < best_dist:
                        best_dist = dist_center
                        best_seg = seg_
                filtered_segments.append(best_seg)

    return filtered_segments




def cluster_aabbs_scipy(aabbs, threshold, method='single'):
    """
    Dado aabbs = [((ax, ay), w, h), ...],
    clusteriza usando hierárquico do scipy, com 'method' = 'single', 'complete', etc.
    'threshold' define a distância de corte para separar os clusters.

    Retorna 'labels', onde labels[i] = rótulo do cluster ao qual aabb i pertence.
    """

    n = len(aabbs)
    # Se houver 0 ou 1 AABB, não precisamos clusterizar
    if n <= 1:
        return [1]*n  # tudo no cluster 1

    # Cria matriz NxN de distâncias
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = distance_between_aabbs(aabbs[i], aabbs[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Converte para formato condensado
    dist_condensed = squareform(dist_matrix, checks=False)

    # Clustering hierárquico
    Z = linkage(dist_condensed, method=method)

    # fcluster para cortar em 'threshold'
    labels = fcluster(Z, t=threshold, criterion='distance')
    return labels


def plot_clusters_aabbs(aabbs, labels):
    """
    Dado:
      - aabbs = [((ax, ay), aw, ah), ...]
      - labels = array/list de inteiros, mesmo tamanho de 'aabbs'
        que indica a qual cluster cada aabb pertence (ex. [1,1,2,3,...])

    Plota cada AABB colorido conforme o cluster.
    """
    # Gera um colormap com base no tab10 (10 cores)
    cmap = cm.get_cmap('tab10')

    # Descobrir quantos clusters temos
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.grid(color='lightgray', linestyle='--', linewidth=0.5)

    for i, ((ax_, ay_), w_, h_) in enumerate(aabbs):
        cluster_id = labels[i]

        # A cor será baseada em cluster_id
        color_index = (cluster_id - 1) % 10  # se cluster_id começa em 1
        color = cmap(color_index)  # RGBA

        # Desenhar retângulo
        rect = plt.Rectangle(
            (ax_, ay_), w_, h_,
            edgecolor=color, facecolor='none', linewidth=2
        )
        ax.add_patch(rect)

        # Podemos também adicionar texto indicando a qual cluster pertence
        # Ex.: ax.text(ax_ + w_/2, ay_ + h_/2, str(cluster_id),
        #        color=color, ha='center', va='center')

    ax.set_aspect('equal', 'box')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"AABBs Coloridas por Cluster (total = {num_clusters})")
    plt.show()






def plot_obstacles_aabbs_segments(obstacles, aabbs, segments):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.grid(color='lightgray', linestyle='--', linewidth=0.3)

    # Plotar obstáculos
    for obs in obstacles:
        x, y = obs["pos"]
        w, h = obs["size"]
        rect = plt.Rectangle((x - w / 2, y - h / 2), w, h, color='red', alpha=0.7)
        ax.add_patch(rect)

    # Plotar AABBs
    for (aabb_x, aabb_y), aabb_w, aabb_h in aabbs:
        aabb_rect = plt.Rectangle((aabb_x, aabb_y), aabb_w, aabb_h, edgecolor='blue', facecolor='none', linewidth=1.5)
        ax.add_patch(aabb_rect)

    # Plotar segmentos
    for x1, y1, x2, y2 in segments:
        ax.plot([x1, x2], [y1, y2], color='green', linestyle='-', linewidth=1.5)


    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Obstáculos, AABBs e Segmentos Válidos")
    plt.show()


def get_aabbs(obstacles, margin):
    aabbs = []
    for obs in obstacles:
        x, y = obs["pos"]
        w, h = obs["size"]
        aabb_x = x - (w / 2 + margin)
        aabb_y = y - (h / 2 + margin)
        aabb_w = w + 2 * margin
        aabb_h = h + 2 * margin
        aabbs.append(((aabb_x, aabb_y), aabb_w, aabb_h))
    return merge_overlapping_aabbs(merge_overlapping_aabbs(aabbs))


def merge_overlapping_aabbs(aabbs):
    merged = []
    while aabbs:
        base = aabbs.pop(0)
        bx, by = base[0]
        bw, bh = base[1], base[2]
        merged_flag = False

        for i, (other_pos, other_w, other_h) in enumerate(merged):
            ox, oy = other_pos
            if not (bx + bw < ox or ox + other_w < bx or by + bh < oy or oy + other_h < by):
                # Mesclar os AABBs sobrepostos
                new_x = min(bx, ox)
                new_y = min(by, oy)
                new_w = max(bx + bw, ox + other_w) - new_x
                new_h = max(by + bh, oy + other_h) - new_y
                merged[i] = ((new_x, new_y), new_w, new_h)
                merged_flag = True
                break

        if not merged_flag:
            merged.append(base)
    return merged


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


def split_segments(paths, is_horizontal, intersections):
    split_segments = []

    # Pré-processar os pontos de interseção organizados
    intersection_points = {p[0 if is_horizontal else 1] for p in intersections}

    for coord, start, end in paths:
        # Criar lista de split points sem duplicatas
        split_points = {start, end} | {p for p in intersection_points if start <= p <= end}

        # Ordenar os pontos de segmentação
        split_points = sorted(split_points)

        # Criar segmentos
        for i in range(len(split_points) - 1):
            seg_start, seg_end = split_points[i], split_points[i + 1]
            segment = (seg_start, coord, seg_end, coord) if is_horizontal else (coord, seg_start, coord, seg_end)
            split_segments.append(segment)

    return split_segments


def split_and_filter_paths(horizontal_paths, vertical_paths, aabbs):
    """
    Para cada caminho horizontal e vertical, cria segmentos fora das AABBs
    seguindo a lógica de índice ímpar-par (válido) e par-ímpar (inválido).
    """
    valid_segments = []

    # 1) Processar caminhos horizontais
    for (y, x_start, x_end) in horizontal_paths:
        # Constrói a lista de 'pontos de colisão' + bordas do caminho
        collision_points = [x_start, x_end]

        # Para cada AABB, verificar se a linha horizontal 'y' está dentro da faixa vertical do retângulo
        # Se sim, adicionar as fronteiras em x (aabb_x e aabb_x + aabb_w) à lista
        for (aabb_pos, aabb_w, aabb_h) in aabbs:
            aabb_x, aabb_y = aabb_pos
            # Verifica se a linha horizontal cruza aabb_y..(aabb_y+aabb_h)
            if aabb_y <= y <= aabb_y + aabb_h:
                # AABB cruza a linha, então pegue os limites em x
                x_left = aabb_x
                x_right = aabb_x + aabb_w

                # Somente adiciona se estiver dentro do trecho (x_start..x_end)
                if x_start <= x_left <= x_end:
                    collision_points.append(x_left)
                if x_start <= x_right <= x_end:
                    collision_points.append(x_right)

        # Ordena e remove duplicatas
        collision_points = sorted(set(collision_points))

        # Agora, percorre pares consecutivos de pontos para decidir se é válido
        # Índices (i+1) ímpar → (i+2) par => segmento válido (fora da AABB)
        # Índices (i+1) par → (i+2) ímpar => segmento inválido (dentro da AABB)
        # Vamos indexar de 1 para seguir a convenção impar-par
        for i in range(len(collision_points) - 1):
            # i = 0 => esse ponto tem "índice 1"
            # i = 1 => esse ponto tem "índice 2", etc.
            idx1 = i + 1      # índice do primeiro ponto
            idx2 = i + 2      # índice do segundo ponto
            xA = collision_points[i]
            xB = collision_points[i + 1]

            # Se idx1 for ímpar e idx2 for par => segmento válido
            if (idx1 % 2 == 1) and (idx2 % 2 == 0):
                # Segmento válido
                valid_segments.append((xA, y, xB, y))

    # 2) Processar caminhos verticais
    for (x, y_start, y_end) in vertical_paths:
        # Constrói a lista de 'pontos de colisão' + bordas do caminho
        collision_points = [y_start, y_end]

        # Para cada AABB, verificar se a linha vertical 'x' está dentro da faixa horizontal do retângulo
        # Se sim, adicionar as fronteiras em y (aabb_y e aabb_y + aabb_h) à lista
        for (aabb_pos, aabb_w, aabb_h) in aabbs:
            aabb_x, aabb_y = aabb_pos
            # Verifica se a linha vertical cruza aabb_x..(aabb_x+aabb_w)
            if aabb_x <= x <= aabb_x + aabb_w:
                # AABB cruza a linha, então pegue os limites em y
                y_bottom = aabb_y
                y_top = aabb_y + aabb_h

                # Somente adiciona se estiver dentro do trecho (y_start..y_end)
                if y_start <= y_bottom <= y_end:
                    collision_points.append(y_bottom)
                if y_start <= y_top <= y_end:
                    collision_points.append(y_top)

        # Ordena e remove duplicatas
        collision_points = sorted(set(collision_points))

        # Índice ímpar → índice par => segmento válido (fora da AABB)
        for i in range(len(collision_points) - 1):
            idx1 = i + 1
            idx2 = i + 2
            yA = collision_points[i]
            yB = collision_points[i + 1]

            if (idx1 % 2 == 1) and (idx2 % 2 == 0):
                valid_segments.append((x, yA, x, yB))

    return valid_segments


import math

def distance_point_to_aabb(px, py, aabb):
    """
    Retorna a distância mínima do ponto (px, py) a um AABB cujo
    canto superior esquerdo é (ax, ay), com largura aw e altura ah.
    """
    (ax, ay), aw, ah = aabb
    rx1, ry1 = ax, ay
    rx2, ry2 = ax + aw, ay + ah

    # Determina dx
    if px < rx1:
        dx = rx1 - px
    elif px > rx2:
        dx = px - rx2
    else:
        dx = 0

    # Determina dy
    if py < ry1:
        dy = ry1 - py
    elif py > ry2:
        dy = py - ry2
    else:
        dy = 0

    return math.sqrt(dx*dx + dy*dy)


def distance_endpoints_to_aabb(segment, aabb):
    """
    Retorna a menor distância entre as extremidades do segmento (x1, y1, x2, y2)
    e o AABB. Ignora a possibilidade de a parte central do segmento estar mais próxima.
    """
    x1, y1, x2, y2 = segment
    d1 = distance_point_to_aabb(x1, y1, aabb)
    d2 = distance_point_to_aabb(x2, y2, aabb)
    return min(d1, d2)


def filter_segments_by_distance(segments, aabbs, endpoint_threshold, center_threshold):
    """
    Retorna apenas os segmentos (x1, y1, x2, y2) que satisfazem DUAS condições:
      1) Cada extremidade (x1,y1) e (x2,y2) está a menos que 'endpoint_threshold'
         do AABB mais próximo.
      2) O centro do segmento também está a menos que 'center_threshold' do AABB mais próximo.

    Caso contrário, o segmento é removido.
    """

    valid = []
    for seg in segments:
        x1, y1, x2, y2 = seg

        # 1) Distância dos endpoints ao AABB mais próximo
        dist_min_e1 = float('inf')
        dist_min_e2 = float('inf')

        # 2) Distância do centro ao AABB mais próximo
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        dist_min_center = float('inf')

        # Percorre todos os AABBs para achar as menores distâncias
        for aabb in aabbs:
            d1 = distance_point_to_aabb(x1, y1, aabb)
            d2 = distance_point_to_aabb(x2, y2, aabb)
            dC = distance_point_to_aabb(mx, my, aabb)

            if d1 < dist_min_e1:
                dist_min_e1 = d1
            if d2 < dist_min_e2:
                dist_min_e2 = d2
            if dC < dist_min_center:
                dist_min_center = dC

        # Verifica condições:
        # - ambas extremidades < endpoint_threshold
        # - centro < center_threshold
        if (
            dist_min_e1 < endpoint_threshold
            and dist_min_e2 < endpoint_threshold
            and dist_min_center < center_threshold
        ):
            valid.append(seg)

    return valid

def clusters_to_labels(clusters, n):
    """
    clusters: lista de listas. Ex: [[0,2,3],[1],[4,5]]
    n: qtd total de AABBs
    Retorna um array 'labels' de tamanho n.
    """
    import numpy as np
    labels = np.zeros(n, dtype=int)
    for i, cluster in enumerate(clusters, start=1):
        for idx in cluster:
            labels[idx] = i
    return labels


def plot_segments_with_vertices(segments, raio=2):
    """
    Plota cada segmento (x1, y1, x2, y2) em verde,
    com os vértices (endpoints) como círculos vermelhos de raio 2.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.grid(True, linestyle='--', color='lightgray', alpha=0.7)

    for (x1, y1, x2, y2) in segments:
        # Plotar a reta em verde
        ax.plot([x1, x2], [y1, y2], color='green', linewidth=1.5)

        # Criar círculos vermelhos nos endpoints
        c1 = plt.Circle((x1, y1), raio, color='red', fill=True)
        c2 = plt.Circle((x2, y2), raio, color='red', fill=True)

        # Adicionar ao Axes
        ax.add_patch(c1)
        ax.add_patch(c2)

    ax.set_aspect('equal', 'box')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Segmentos em verde e vértices em vermelho (raio=2)")
    plt.show()

def plot_obstacles_aabbs_and_paths(obstacles, aabbs, horizontal_paths, vertical_paths,
                                   x_min, x_max, y_min, y_max):
    """
    Plota:
      - Obstáculos (vermelho),
      - AABBs (azul),
      - Caminhos horizontais e verticais (verde),
      - Pontos de início/fim dos caminhos (vermelho).
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.grid(color='lightgray', linestyle='--', linewidth=0.5)

    # 1) Plotar obstáculos
    #    Cada obstáculo = {"pos": (x, y), "size": (w, h)}
    #    Se o "pos" for o centro, subtraímos w/2 e h/2 para desenhar
    for obs in obstacles:
        x_obs, y_obs = obs["pos"]
        w, h = obs["size"]
        # Ajuste se (x_obs, y_obs) for centro:
        x_rect = x_obs - w / 2
        y_rect = y_obs - h / 2
        rect = plt.Rectangle((x_rect, y_rect), w, h, color='red', alpha=0.4)
        ax.add_patch(rect)

    # 2) Plotar AABBs (azul)
    for (aabb_x, aabb_y), aabb_w, aabb_h in aabbs:
        aabb_rect = plt.Rectangle(
            (aabb_x, aabb_y), aabb_w, aabb_h,
            edgecolor='blue', facecolor='none', linewidth=1.5
        )
        ax.add_patch(aabb_rect)

    # 3) Plotar caminhos horizontais (verde)
    #    + pontos finais em vermelho
    for y, x_start, x_end in horizontal_paths:
        ax.plot([x_start, x_end], [y, y], color='green', linestyle='-', linewidth=1.5)
        # Pontos de início/fim
        # ax.plot(x_start, y, 'ro')  # início
        # ax.plot(x_end,   y, 'ro')  # fim

    # 4) Plotar caminhos verticais (verde)
    #    + pontos finais em vermelho
    for x, y_start, y_end in vertical_paths:
        ax.plot([x, x], [y_start, y_end], color='green', linestyle='-', linewidth=1.5)
        # Pontos de início/fim
        ax.plot(x, y_start, 'ro')  # início
        ax.plot(x, y_end,   'ro')  # fim

    # 5) Ajustar limites do plot
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Obstáculos, AABBs e Caminhos")

    plt.show()


# Caminho do arquivo e nome da planilha
file_path = "obstaculos_processado2.xlsx"
sheet_name = "Parnaiba3_Transformado"
padding = 15
margin = 3

# Carregar obstáculos
obstacle_loader = ObstacleLoader(file_path, sheet_name)
obstacles = obstacle_loader.get_obstacles()

# Obter limites do grid
x_min = min(obs["pos"][0] for obs in obstacles) - padding
x_max = max(obs["pos"][0] for obs in obstacles) + padding
y_min = min(obs["pos"][1] for obs in obstacles) - padding
y_max = max(obs["pos"][1] for obs in obstacles) + padding

# Calcular AABBs e mesclar sobrepostos
aabbs = get_aabbs(obstacles, margin)

# Calcular caminhos horizontais e verticais
horizontal_paths, vertical_paths = get_paths(aabbs, x_min, x_max, y_min, y_max)

# plot_obstacles_aabbs_and_paths(obstacles, aabbs, horizontal_paths, vertical_paths,x_min, x_max, y_min, y_max)

# Gerar segmentos válidos
valid_segments = split_and_filter_paths(horizontal_paths, vertical_paths, aabbs)

filtered_segments = filter_segments_by_distance(valid_segments, aabbs, endpoint_threshold=3.0, center_threshold=10)

# Criar figura
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.grid(color='lightgray', linestyle='--', linewidth=0.5)

#plot_obstacles_aabbs_segments(obstacles, aabbs, valid_segments)

plot_obstacles_aabbs_segments(obstacles, aabbs, filtered_segments)

final_segments = filter_similar_segments(
    segments=filtered_segments,
    aabbs=aabbs,
    parallel_threshold=10.0,  # ex.
    distance_point_to_aabb=distance_point_to_aabb
)

plot_obstacles_aabbs_segments(obstacles, aabbs, final_segments)


updated_segments = add_perimeter_segments(aabbs, 4, final_segments)

plot_segments_with_vertices(updated_segments, raio = 0.5)

# labels = cluster_aabbs_scipy(aabbs, threshold=5.0, method='single')
#
# n = len(aabbs)  # digamos que = 6
# labels_array = clusters_to_labels(clusters, n)
#
# plot_clusters_aabbs(labels,aabbs)



