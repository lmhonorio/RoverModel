"""
Comparative analysis script between the AABB graph and the GVD (Voronoi) graph.

This script generates three graphs:
1. AABB Graph: Main method, using margin=1.5 and threshold=50.
2. GVD with Margin: GVD generated with obstacles expanded by margin=1.5.
3. GVD without Margin: GVD generated from original obstacles (margin=0).

It performs the following analyses and produces the following artifacts:
- Comparative Metrics Table: prints a table with graph statistics
    (vertices, edges, total length, islands, density, tortuosity).
- Shortest-path comparison plot: saves `comparacao_caminhos.png` showing
    the shortest path between two example points on the three graphs.

Usage: Run from repository root: `python Resultados_artigo/analise_comparativa.py`
"""

import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import nearest_points, unary_union
import pandas as pd
import seaborn as sns

    # Ensure repository root is in the import path
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
# Publication-quality defaults for this script's figures
import matplotlib as mpl
mpl.rcParams.update({
    'figure.titlesize': 26,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'legend.title_fontsize': 16,
    'font.size': 16,
})

def find_closest_node(graph, point):
    """Encontra o nó no grafo mais próximo de uma dada coordenada (x, y)."""
    if not graph or graph.number_of_nodes() == 0:
        return None

    nodes_with_pos = {n: data['pos'] for n, data in graph.nodes(data=True) if 'pos' in data}
    if not nodes_with_pos:
        return None

    node_ids, coords = zip(*nodes_with_pos.items())
    closest_idx = np.argmin(np.linalg.norm(np.array(coords) - np.array(point), axis=1))
    return node_ids[closest_idx]


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

def get_average_tortuosity(graph):
    """Calcula a tortuosidade média para pares de nós no maior componente conectado."""
    if graph.number_of_nodes() < 2:
        return float('nan')

    # Foca no maior componente para garantir que os caminhos existam
    connected_components = list(nx.connected_components(graph))
    if not connected_components: return float('nan')
    largest_cc_nodes = max(connected_components, key=len)
    if len(largest_cc_nodes) < 2:
        return float('nan')
    subgraph = graph.subgraph(largest_cc_nodes)

    tortuosities = []
    all_nodes = list(subgraph.nodes)
    
    # Amostragem para evitar consumo excessivo de memória e tempo.
    # Seleciona um número fixo de nós de ORIGEM e para cada um, calcula o caminho para os outros.
    # Isso é muito mais leve do que all_pairs_dijkstra.
    num_source_nodes_sample = min(50, len(all_nodes))
    source_nodes = np.random.choice(all_nodes, num_source_nodes_sample, replace=False)

    for u in source_nodes:
        # Calcula o caminho mais curto de 'u' para todos os outros nós no subgrafo
        path_lengths = nx.single_source_dijkstra_path_length(subgraph, u, weight='weight')
        
        for v, path_length in path_lengths.items():
            if u == v: continue

            p1 = subgraph.nodes[u].get('pos')
            p2 = subgraph.nodes[v].get('pos')
            if p1 is None or p2 is None: continue
            
            euclidean_dist = np.linalg.norm(np.array(p1) - np.array(p2))
            if euclidean_dist > 1e-6: # Evita divisão por zero
                tortuosities.append(path_length / euclidean_dist)
    
    return np.mean(tortuosities) if tortuosities else float('nan')

def plot_radar_chart(df, output_dir):
    # Radar chart removed per user request — keep function as placeholder.
    print("Radar chart generation disabled (plots removed).")

def plot_metrics_comparison(df, output_dir):
    # Barplots generation removed per user request — keep function as placeholder.
    print("Barplot generation disabled (plots removed).")

def main():
    # --- Configuração ---
    file_path = os.path.join(REPO_ROOT, 'planilhas', 'equipment_processado.xlsx')
    sheet_name = 'Parnaiba3_Transformado'
    
    # Parâmetros para os cenários
    AABB_MARGIN = 1.5
    AABB_THRESHOLD = 50
    GVD_MARGIN = 1.5

    print("Starting comparative analysis between AABB and GVD graphs...")

    # --- Carregamento de Dados ---
    print("1. Loading obstacles...")
    loader = ObstacleLoader(file_path, sheet_name)
    obstacles = loader.get_obstacles()
    # Polígonos de obstáculos originais (sem margem) para cálculo de folga
    obstacle_polys_no_margin = obstacle_polygons_from_obstacles(obstacles, margin=0)
    obstacle_union_no_margin = unary_union(obstacle_polys_no_margin)

    # --- Geração dos Grafos ---
    print("2. Generating the three graphs for comparison...")
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
    print("\n3. Calculating metrics and generating comparative table...")
    results = []
    obs_points_coords = [p[:2] for p in obs_pts_list]

    for name, G in graphs.items():
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        total_length = sum(d['weight'] for _, _, d in G.edges(data=True))
        num_islands = nx.number_connected_components(G) if num_nodes > 0 else 0
        density = nx.density(G) if num_nodes > 1 else 0

        fragmentation_index = 0.0
        if num_nodes > 0:
            largest_cc = max(nx.connected_components(G), key=len, default=set())
            fragmentation_index = 1.0 - (len(largest_cc) / num_nodes)

        avg_tortuosity = get_average_tortuosity(G)

        results.append({
            "Graph": name,
            "Vertices": num_nodes,
            "Edges": num_edges,
            "Total Length (m)": f"{total_length:.1f}",
            "Density": f"{density:.4f}",
            "Average Tortuosity": f"{avg_tortuosity:.3f}",
            "Fragmentation Index": f"{fragmentation_index:.3f}",
            "Number of Islands": num_islands,
        })

    df = pd.DataFrame(results)
    print("\n--- Results Table ---")
    print(df.to_string(index=False))

    # Save the table to a CSV file
    csv_path = os.path.join(RESULTS_DIR, "comparacao_resultados.csv")
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nResults table saved to: {csv_path}")

    # Copia o DataFrame para não alterar o original que será usado no plot de caminhos
    df_plot = df.copy()

    # --- Análise Visual (Plots) ---
    print("\n4. Generating comparative visualizations...")

    # Plot 1: Comparação de Caminhos Mais Curtos
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Desenha os obstáculos
    xmin, xmax, ymin, ymax = workspace_bbox_from_polygons(obstacle_polys_no_margin)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    for poly in obstacle_polys_no_margin:
        ax.fill(*poly.exterior.xy, color='lightgray', ec='black', linewidth=1.5)

    # Encontra pontos de início e fim (usando o grafo AABB como referência)
    start_node, end_node = find_farthest_nodes(G_aabb)
    
    if start_node and end_node:
        start_pos = G_aabb.nodes[start_node]['pos']
        end_pos = G_aabb.nodes[end_node]['pos']
        ax.plot(start_pos[0], start_pos[1], 'go', markersize=16, markeredgewidth=1.5, label='Start')
        ax.plot(end_pos[0], end_pos[1], 'ro', markersize=16, markeredgewidth=1.5, label='End')

        colors = {
            f"AABB (margin={AABB_MARGIN})": ('blue', '-'),
            f"GVD (margin={GVD_MARGIN})": ('darkorange', '--'),
            "GVD (margin=0)": ('purple', ':')
        }

        for name, G in graphs.items():
            color, style = colors.get(name, ('black', '-'))

            # Find the closest nodes to the start/end positions for this specific graph
            current_start_node = find_closest_node(G, start_pos)
            current_end_node = find_closest_node(G, end_pos)

            if not current_start_node or not current_end_node:
                print(f"   - Warning: Could not find start/end nodes for graph '{name}'.")
                continue

            try:
                path_nodes = nx.shortest_path(G, source=current_start_node, target=current_end_node, weight='weight')
                path_length = nx.shortest_path_length(G, source=current_start_node, target=current_end_node, weight='weight')
                label_text = f"{name} ({path_length:.1f} m)"
                path_edges = list(zip(path_nodes, path_nodes[1:]))
                for u, v in path_edges:
                    p1 = G.nodes[u]['pos']
                    p2 = G.nodes[v]['pos']
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linestyle=style, linewidth=3.5, label=label_text if (u,v) == path_edges[0] else "")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                print(f"   - Warning: No path found between start and end for graph '{name}'.")

    ax.set_title("Shortest-path Comparison Between Graphs", fontsize=26)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel('X (meters)', fontsize=20)
    ax.set_ylabel('Y (meters)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    out_path_paths = os.path.join(RESULTS_DIR, "comparacao_caminhos.png")
    plt.savefig(out_path_paths, dpi=300)
    print(f"   - Path comparison plot saved to: {out_path_paths}")
    plt.close()

    # Barplots and radar chart generation were disabled per request.

    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()