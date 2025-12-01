"""
Script para análise comparativa entre o grafo AABB e o grafo GVD (Voronoi).

Este script gera três grafos:
1. Grafo AABB: O método principal, com margem de 3.5 e threshold de 20.
2. Grafo GVD com Margem: GVD gerado sobre obstáculos com margem de 3.5.
3. Grafo GVD sem Margem: GVD gerado sobre os obstáculos originais (margem 0).

Realiza as seguintes análises e gera os seguintes artefatos:
- Tabela Comparativa de Métricas: Imprime no console uma tabela com estatísticas
  dos grafos (vértices, arestas, comprimento, ilhas, segurança, acessibilidade).
- Histograma de Distâncias de Segurança: Salva um PNG (`comparacao_distancias_seguranca.png`)
  mostrando a distribuição da distância das arestas aos obstáculos para cada grafo.
- Plot de Caminhos Mais Curtos: Salva um PNG (`comparacao_caminhos.png`) mostrando
  o caminho mais curto entre dois pontos de exemplo nos três grafos.

Uso: Executar a partir da raiz do repositório: `python Resultados_artigo/analise_comparativa.py`
"""

import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import nearest_points, unary_union
import pandas as pd

# Garantir que a raiz do repositório esteja no path para importações
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from roverclass import ObstacleLoader
from CriarPontosObservacao import build_graph
from Resultados_artigo.gvd_from_equipment import (
    obstacle_polygons_from_obstacles,
    sample_boundary_points,
    build_gvd_by_clipping,
    workspace_bbox_from_polygons,
)

RESULTS_DIR = os.path.dirname(__file__)

def get_edge_clearance(graph, obstacle_union):
    """Calcula a distância mínima de cada aresta do grafo para a união de obstáculos."""
    clearances = []
    if not isinstance(obstacle_union, (Polygon, MultiPolygon)):
        return clearances

    for u, v in graph.edges():
        p1 = graph.nodes[u].get('pos')
        p2 = graph.nodes[v].get('pos')
        if p1 is None or p2 is None:
            continue
        
        edge_line = LineString([p1, p2])
        # Usamos o centro da aresta como uma aproximação para o cálculo da distância
        mid_point = edge_line.interpolate(0.5, normalized=True)
        
        # Encontra o ponto mais próximo no obstáculo e calcula a distância
        _, p_nearest = nearest_points(mid_point, obstacle_union)
        dist = mid_point.distance(p_nearest)
        clearances.append(dist)
    return clearances

def get_obs_point_accessibility(graph, obs_points):
    """Calcula a distância média de cada ponto de observação ao nó mais próximo no grafo."""
    if not obs_points or graph.number_of_nodes() == 0:
        return float('inf')

    graph_nodes_coords = np.array([data['pos'] for _, data in graph.nodes(data=True)])
    total_dist = 0
    
    for p_obs in obs_points:
        obs_coord = np.array(p_obs[:2])
        distances = np.linalg.norm(graph_nodes_coords - obs_coord, axis=1)
        total_dist += np.min(distances)
        
    return total_dist / len(obs_points)

def find_farthest_nodes(graph):
    """Encontra um par de nós que estão distantes no grafo."""
    if graph.number_of_nodes() < 2:
        return None, None
    
    # Heurística: pega um nó e encontra o mais distante dele
    nodes = list(graph.nodes)
    p1 = nodes[0]
    lengths = nx.single_source_shortest_path_length(graph, p1)
    p2 = max(lengths, key=lengths.get)
    
    # Agora, a partir de p2, encontre o mais distante
    lengths2 = nx.single_source_shortest_path_length(graph, p2)
    p1_new = max(lengths2, key=lengths2.get)
    
    return p1_new, p2

def main():
    # --- Configuração ---
    file_path = os.path.join(REPO_ROOT, 'planilhas', 'equipment_processado.xlsx')
    sheet_name = 'Parnaiba3_Transformado'
    
    # Parâmetros para os cenários
    AABB_MARGIN = 3.5
    AABB_THRESHOLD = 20
    GVD_MARGIN = 3.5

    print("Iniciando análise comparativa entre grafos AABB e GVD...")

    # --- Carregamento de Dados ---
    print("1. Carregando obstáculos...")
    loader = ObstacleLoader(file_path, sheet_name)
    obstacles = loader.get_obstacles()
    # Polígonos de obstáculos originais (sem margem) para cálculo de folga
    obstacle_polys_no_margin = obstacle_polygons_from_obstacles(obstacles, margin=0)
    obstacle_union_no_margin = unary_union(obstacle_polys_no_margin)

    # --- Geração dos Grafos ---
    print("2. Gerando os três grafos para comparação...")
    # Cenário 1: Grafo AABB
    G_aabb, _, obs_pts_list, _ = build_graph(file_path, sheet_name, margin=AABB_MARGIN, threshold=AABB_THRESHOLD)
    
    # Cenário 2: Grafo GVD com Margem
    polys_gvd_margin = obstacle_polygons_from_obstacles(obstacles, margin=GVD_MARGIN)
    samples_gvd_margin = sample_boundary_points(polys_gvd_margin, points_per_meter=1.0)
    G_gvd_margin, _, _ = build_gvd_by_clipping(polys_gvd_margin, samples_gvd_margin)

    # Cenário 3: Grafo GVD sem Margem
    samples_gvd_no_margin = sample_boundary_points(obstacle_polys_no_margin, points_per_meter=1.0)
    G_gvd_no_margin, _, _ = build_gvd_by_clipping(obstacle_polys_no_margin, samples_gvd_no_margin)

    graphs = {
        f"AABB (margin={AABB_MARGIN})": G_aabb,
        f"GVD (margin={GVD_MARGIN})": G_gvd_margin,
        "GVD (margin=0)": G_gvd_no_margin,
    }

    # --- Análise Quantitativa ---
    print("\n3. Calculando métricas e gerando tabela comparativa...")
    results = []
    obs_points_coords = [p[:2] for p in obs_pts_list]

    for name, G in graphs.items():
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        total_length = sum(d['weight'] for _, _, d in G.edges(data=True))
        num_islands = nx.number_connected_components(G) if num_nodes > 0 else 0
        
        clearances = get_edge_clearance(G, obstacle_union_no_margin)
        avg_clearance = np.mean(clearances) if clearances else 0
        
        accessibility = get_obs_point_accessibility(G, obs_points_coords)

        results.append({
            "Grafo": name,
            "Vértices": num_nodes,
            "Arestas": num_edges,
            "Comprimento Total (m)": f"{total_length:.1f}",
            "Nº de Ilhas": num_islands,
            "Folga Média (m)": f"{avg_clearance:.2f}",
            "Acessibilidade Obs. (m)": f"{accessibility:.2f}"
        })

    df = pd.DataFrame(results)
    print("\n--- Tabela de Resultados ---")
    print(df.to_string(index=False))

    # --- Análise Visual (Plots) ---
    print("\n4. Gerando plots comparativos...")

    # Plot 1: Histograma de Distâncias de Segurança (Folga)
    plt.figure(figsize=(12, 7))
    for name, G in graphs.items():
        clearances = get_edge_clearance(G, obstacle_union_no_margin)
        if clearances:
            plt.hist(clearances, bins=50, alpha=0.6, label=name, density=True)
    plt.title("Distribuição da Distância de Segurança das Arestas aos Obstáculos")
    plt.xlabel("Distância (folga) da aresta ao obstáculo mais próximo (m)")
    plt.ylabel("Densidade")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    out_path_hist = os.path.join(RESULTS_DIR, "comparacao_distancias_seguranca.png")
    plt.savefig(out_path_hist, dpi=300)
    print(f"   - Histograma salvo em: {out_path_hist}")
    plt.close()

    # Plot 2: Comparação de Caminhos Mais Curtos
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Desenha os obstáculos
    xmin, xmax, ymin, ymax = workspace_bbox_from_polygons(obstacle_polys_no_margin)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    for poly in obstacle_polys_no_margin:
        ax.fill(*poly.exterior.xy, color='lightgray', ec='black')

    # Encontra pontos de início e fim (usando o grafo AABB como referência)
    start_node, end_node = find_farthest_nodes(G_aabb)
    
    if start_node and end_node:
        start_pos = G_aabb.nodes[start_node]['pos']
        end_pos = G_aabb.nodes[end_node]['pos']
        ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Início')
        ax.plot(end_pos[0], end_pos[1], 'ro', markersize=10, label='Fim')

        colors = {'AABB': 'blue', 'GVD (margin=3.5)': 'orange', 'GVD (margin=0)': 'purple'}
        
        for name, G in graphs.items():
            key_part = name.split(' ')[0] # 'AABB' ou 'GVD'
            color = colors.get(key_part, 'black')
            
            try:
                path_nodes = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
                path_edges = list(zip(path_nodes, path_nodes[1:]))
                for u, v in path_edges:
                    p1 = G.nodes[u]['pos']
                    p2 = G.nodes[v]['pos']
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2.5, label=name if (u,v) == path_edges[0] else "")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                print(f"   - Aviso: Não foi possível encontrar caminho entre os pontos para o grafo '{name}'.")

    ax.set_title("Comparação de Caminho Mais Curto entre Grafos")
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    out_path_paths = os.path.join(RESULTS_DIR, "comparacao_caminhos.png")
    plt.savefig(out_path_paths, dpi=300)
    print(f"   - Plot de caminhos salvo em: {out_path_paths}")
    plt.close()

    print("\nAnálise concluída com sucesso!")

if __name__ == "__main__":
    main()