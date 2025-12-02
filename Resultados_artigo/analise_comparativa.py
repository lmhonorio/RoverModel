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
import seaborn as sns

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
    """Gera um radar chart para comparar as métricas normalizadas dos grafos."""
    print("   - Gerando Radar Chart para comparação de trade-offs...")
    
    # Métricas para o radar chart. Escolhemos as que melhor representam os trade-offs.
    # Nota: Para 'Tortuosidade' e 'Fragmentação', valores menores são melhores. Inverteremos a escala.
    metrics = ['Vértices', 'Comprimento Total (m)', 'Tortuosidade Média', 'Índice Fragmentação']
    df_radar = df[['Grafo'] + metrics].copy()

    # Normaliza os dados (escala de 0 a 1)
    for metric in metrics:
        min_val = df_radar[metric].min()
        max_val = df_radar[metric].max()
        if max_val - min_val > 0:
            df_radar[metric] = (df_radar[metric] - min_val) / (max_val - min_val)
        else:
            df_radar[metric] = 0.5 # Valor neutro se todos forem iguais

    # Inverte a escala para métricas onde "menos é mais"
    for metric in ['Tortuosidade Média', 'Índice Fragmentação']:
        df_radar[metric] = 1 - df_radar[metric]

    labels = df_radar.columns[1:]
    num_vars = len(labels)

    # Ângulos para o radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, row in df_radar.iterrows():
        values = row.drop('Grafo').tolist()
        values += values[:1]
        ax.plot(angles, values, label=row['Grafo'], linewidth=2)
        ax.fill(angles, values, alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([l.replace(' ', '\n') for l in labels])
    plt.title('Comparação de Trade-offs dos Métodos (Radar Chart)', size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig(os.path.join(output_dir, "comparacao_radar_chart.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(df, output_dir):
    """Gera gráficos de barras comparando as métricas dos diferentes grafos."""
    print("   - Gerando plots de comparação de métricas...")
    
    # Garante que as colunas de métricas sejam numéricas para plotagem
    for col in df.columns:
        if col != 'Grafo':
            df[col] = pd.to_numeric(df[col])

    # Define as métricas para os plots
    metrics_to_plot = {
        'Complexidade': ['Vértices', 'Arestas'],
        'Eficiência e Coesão': ['Comprimento Total (m)', 'Tortuosidade Média', 'Índice Fragmentação']
    }

    sns.set_theme(style="whitegrid")
    palette = "viridis"

    # Plot para Complexidade
    df_melted_complexity = df.melt(id_vars='Grafo', value_vars=metrics_to_plot['Complexidade'], var_name='Métrica', value_name='Valor')
    plt.figure(figsize=(10, 6))
    ax1 = sns.barplot(x='Métrica', y='Valor', hue='Grafo', data=df_melted_complexity, palette=palette)
    ax1.set_title('Comparação de Complexidade do Grafo', fontsize=16)
    ax1.set_ylabel('Contagem')
    ax1.set_yscale('log') # Usar escala logarítmica para lidar com grandes diferenças
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparacao_metricas_complexidade.png"), dpi=300)
    plt.close()

    # Plot para Eficiência e Coesão
    df_melted_efficiency = df.melt(id_vars='Grafo', value_vars=metrics_to_plot['Eficiência e Coesão'], var_name='Métrica', value_name='Valor')
    g = sns.catplot(x='Métrica', y='Valor', hue='Grafo', data=df_melted_efficiency, kind='bar', palette=palette, height=6, aspect=1.5, sharey=False)
    g.fig.suptitle('Comparação de Eficiência e Coesão do Grafo', y=1.03, fontsize=16)
    g.set_axis_labels("Métrica", "Valor")
    g.set_xticklabels(rotation=10)
    g.despine(left=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparacao_metricas_eficiencia.png"), dpi=300)
    plt.close()
    print(f"   - Plots de métricas salvos em: {output_dir}")

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
        density = nx.density(G) if num_nodes > 1 else 0

        fragmentation_index = 0.0
        if num_nodes > 0:
            largest_cc = max(nx.connected_components(G), key=len, default=set())
            fragmentation_index = 1.0 - (len(largest_cc) / num_nodes)

        avg_tortuosity = get_average_tortuosity(G)

        results.append({
            "Grafo": name,
            "Vértices": num_nodes,
            "Arestas": num_edges,
            "Comprimento Total (m)": f"{total_length:.1f}",
            "Densidade": f"{density:.4f}",
            "Tortuosidade Média": f"{avg_tortuosity:.3f}",
            "Índice Fragmentação": f"{fragmentation_index:.3f}",
            "Nº de Ilhas": num_islands,
        })

    df = pd.DataFrame(results)
    print("\n--- Tabela de Resultados ---")
    print(df.to_string(index=False))

    # Salva a tabela em um arquivo CSV
    csv_path = os.path.join(RESULTS_DIR, "comparacao_resultados.csv")
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nTabela de resultados salva em: {csv_path}")

    # Copia o DataFrame para não alterar o original que será usado no plot de caminhos
    df_plot = df.copy()

    # --- Análise Visual (Plots) ---
    print("\n4. Gerando plots comparativos...")

    # Plot 1: Comparação de Caminhos Mais Curtos
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
        colors = {
            f"AABB (margin={AABB_MARGIN})": ('blue', '-'),
            f"GVD (margin={GVD_MARGIN})": ('darkorange', '--'),
            "GVD (margin=0)": ('purple', ':')
        }

        for name, G in graphs.items():
            color, style = colors.get(name, ('black', '-'))
            
            # Encontra os nós mais próximos das posições de início/fim para este grafo específico
            current_start_node = find_closest_node(G, start_pos)
            current_end_node = find_closest_node(G, end_pos)

            if not current_start_node or not current_end_node:
                print(f"   - Aviso: Não foi possível encontrar nós de início/fim para o grafo '{name}'.")
                continue

            try:
                path_nodes = nx.shortest_path(G, source=current_start_node, target=current_end_node, weight='weight')
                path_length = nx.shortest_path_length(G, source=current_start_node, target=current_end_node, weight='weight')
                label_text = f"{name} ({path_length:.1f} m)"
                path_edges = list(zip(path_nodes, path_nodes[1:]))
                for u, v in path_edges:
                    p1 = G.nodes[u]['pos']
                    p2 = G.nodes[v]['pos']
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linestyle=style, linewidth=2.5, label=label_text if (u,v) == path_edges[0] else "")
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

    # Plot 2: Gráficos de Barras das Métricas
    plot_metrics_comparison(df_plot, RESULTS_DIR)

    # Plot 3: Radar Chart
    plot_radar_chart(df_plot, RESULTS_DIR)

    print("\nAnálise concluída com sucesso!")

if __name__ == "__main__":
    main()