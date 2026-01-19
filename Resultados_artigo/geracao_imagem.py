import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, LineString, box
from shapely.ops import unary_union
import math

# --- 1. CONFIGURAÇÃO DOS OBSTÁCULOS (Réplica da sua imagem) ---
# Coordenadas aproximadas baseadas na imagem 'testeAABB.png'
# Formato: (x_centro, y_centro, largura, altura)
# obstacles_data = [
#     (3.5, 11.5, 4, 3),    # Topo Esquerda
#     (14.0, 11.0, 7, 5.5), # Topo Direita (Grandão)
#     (5.5, 6.0, 5, 5),     # Meio Esquerda (Quadrado)
#     (9.0, 3.5, 3, 2.5),   # Baixo Meio (Pequeno)
#     (14.0, 3.0, 6, 3.5)   # Baixo Direita
# ]

obstacles_data = [
    (3.5, 11.5, 4, 3),    # Topo Esquerda
    (14.0, 11.0, 7, 5.5), # Topo Direita (Grandão)
    (5.5, 6.0, 5, 5),     # Meio Esquerda (Quadrado)
    (9.0, 3.5, 3, 2.5),   # Baixo Meio (Pequeno)
    (14.0, 3.0, 6, 3.5)   # Baixo Direita
]

# Configurações do Algoritmo
WORKSPACE_LIMITS = (0, 19, 0, 15) # xmin, xmax, ymin, ymax
POINTS_PER_METER = 2.0            # Aumentei um pouco a densidade para linhas mais suaves
MARGIN = 0.0                      # Margem zero para colar no obstáculo visualmente

# --- 2. FUNÇÕES DO SEU ALGORITMO (Adaptadas para Standalone) ---

def create_polygons(obs_data):
    polys = []
    for (cx, cy, w, h) in obs_data:
        x0 = cx - w / 2.0
        y0 = cy - h / 2.0
        # Cria o box
        p = box(x0, y0, x0 + w, y0 + h)
        polys.append(p)
    return polys

def sample_boundary_points(polygons, density=1.0):
    samples = []
    for poly in polygons:
        perim = poly.length
        # Garante pelo menos um número razoável de pontos por polígono
        n = max(20, int(math.ceil(perim * density)))
        for frac in np.linspace(0.0, 1.0, n, endpoint=False):
            pt = poly.exterior.interpolate(frac, normalized=True)
            samples.append((pt.x, pt.y))
    return np.array(samples)

def compute_gvd(polygons, samples, limits):
    # 1. Computar Voronoi das amostras
    vor = Voronoi(samples)
    
    # 2. Definir Espaço Livre (Workspace - Obstaculos)
    xmin, xmax, ymin, ymax = limits
    workspace_box = box(xmin, ymin, xmax, ymax)
    obstacles_union = unary_union(polygons)
    free_space = workspace_box.difference(obstacles_union)
    
    valid_lines = []

    # 3. Processar cada aresta (ridge) do Voronoi
    # (Focando apenas nas arestas finitas para visualização limpa)
    for (pidx0, pidx1), ridge_vertices in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 not in ridge_vertices:
            # Caso finito (segmento de reta entre dois vértices do Voronoi)
            v0 = vor.vertices[ridge_vertices[0]]
            v1 = vor.vertices[ridge_vertices[1]]
            line = LineString([v0, v1])
            
            # 4. Clipar com o espaço livre (Crucial!)
            if line.intersects(free_space):
                intersection = line.intersection(free_space)
                if not intersection.is_empty:
                    # Lida com o caso da interseção resultar em múltiplas linhas
                    if intersection.geom_type == 'LineString':
                        if intersection.length > 0.05: # Filtra linhas muito pequenas
                            valid_lines.append(intersection)
                    elif intersection.geom_type == 'MultiLineString':
                        for l in intersection.geoms:
                            if l.length > 0.05:
                                valid_lines.append(l)

    return valid_lines, obstacles_union

# --- 3. EXECUÇÃO E PLOTAGEM ---

# Gerar dados
polygons = create_polygons(obstacles_data)
samples = sample_boundary_points(polygons, density=POINTS_PER_METER)
lines, obs_union = compute_gvd(polygons, samples, WORKSPACE_LIMITS)

# Configurar Plot
fig, ax = plt.subplots(figsize=(10, 8))
# Garante que o fundo do eixo seja branco
ax.set_facecolor('white') 
ax.set_xlim(WORKSPACE_LIMITS[0], WORKSPACE_LIMITS[1])
ax.set_ylim(WORKSPACE_LIMITS[2], WORKSPACE_LIMITS[3])
ax.set_aspect('equal')

# --- ALTERAÇÃO AQUI: Fundo com Pontinhos REMOVIDO ---
# xs = np.arange(WORKSPACE_LIMITS[0], WORKSPACE_LIMITS[1], 1)
# ys = np.arange(WORKSPACE_LIMITS[2], WORKSPACE_LIMITS[3], 1)
# X, Y = np.meshgrid(xs, ys)
# ax.scatter(X, Y, c='#B0C4DE', s=15, alpha=0.5, zorder=1) 
# ----------------------------------------------------

# 2. Desenhar Obstáculos (Vermelho com Borda Grossa)
for poly in polygons:
    x, y = poly.exterior.xy
    # Cores ajustadas para ficarem iguais à sua imagem de referência limpa
    ax.fill(x, y, color='#E3888B', alpha=1.0, zorder=2) # Preenchimento Rosa/Vermelho suave
    ax.plot(x, y, color='#B23A33', linewidth=3, zorder=3) # Borda Vermelho Escuro

# 3. Desenhar o GVD (Linhas Pretas Finas)
for line in lines:
    x, y = line.xy
    # Zorder alto para ficar por cima de tudo
    ax.plot(x, y, color='black', linewidth=1.0, alpha=1.0, zorder=10) 

# 4. Estilo Final Clean
# Remove título e eixos para ficar só o diagrama
ax.axis('off') 

plt.tight_layout(pad=0) # Remove margens brancas extras
plt.show()