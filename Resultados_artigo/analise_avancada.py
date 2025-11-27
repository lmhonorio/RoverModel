"""Experimento Avançado: Análise de Sensibilidade da Topologia do Grafo

Este script realiza uma análise de sensibilidade dos parâmetros `margin` e
`threshold` na estrutura do grafo de observação.

Ele calcula métricas avançadas de grafos (densidade, nós isolados,
componente gigante) e gera duas visualizações principais para análise:

1.  Gráficos de Linha de Interação: Mostram como cada métrica varia com o
    `margin` para diferentes níveis de `threshold`.
2.  Matriz de Topologia: Uma grade que exibe a estrutura visual do grafo
    para cada combinação de `(margin, threshold)`.

Uso: python3 Resultados_artigo/analise_avancada.py
"""
import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Garante que os módulos locais (roverclass, etc.) possam ser importados
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from CriarPontosObservacao import build_graph

RESULTS_DIR = os.path.dirname(__file__)


def run_graph_sensitivity_analysis(file_path, sheet_name, margins, thresholds):
    """
    Executa a análise de sensibilidade, calcula métricas e armazena os grafos.
    """
    results = []
    print("Iniciando análise de sensibilidade do grafo...")

    for m in margins:
        for t in thresholds:
            print(f"Processando: margin={m}, threshold={t}...")
            g0 = time.perf_counter()
            G, _, obs_points, _ = build_graph(
                file_path, sheet_name, margin=m, threshold=t, plotting=False
            )
            elapsed = time.perf_counter() - g0

            # Cálculo das métricas
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            density = nx.density(G) if n_nodes > 1 else 0
            isolated_nodes = nx.number_of_isolates(G)
            n_components = nx.number_connected_components(G)

            giant_ratio = 0
            if n_nodes > 0:
                largest_cc = max(nx.connected_components(G), key=len, default=set())
                giant_ratio = len(largest_cc) / n_nodes

            results.append({
                "margin": m,
                "threshold": t,
                "nodes": n_nodes,
                "edges": n_edges,
                "density": density,
                "isolated_nodes": isolated_nodes,
                "components": n_components,
                "giant_component_ratio": giant_ratio,
                "graph": G,
                "pos": {i: (p[0], p[1]) for i, p in enumerate(obs_points)},
                "time_s": elapsed,
            })

    return pd.DataFrame(results)


def plot_interaction_graphs(df, metrics, output_dir):
    """
    Gera gráficos de linha de interação para as métricas especificadas.
    """
    print("Gerando gráficos de linha de interação...")
    thresholds = sorted(df['threshold'].unique())

    for metric in metrics:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))

        for t in thresholds:
            subset = df[df['threshold'] == t]
            ax.plot(subset['margin'], subset[metric], marker='o', linestyle='-', label=f'Threshold = {t}')

        ax.set_xlabel("Margin", fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f"Análise de Sensibilidade: {metric.replace('_', ' ').title()} vs. Margin", fontsize=14)
        ax.legend(title="Threshold")
        ax.grid(True)

        filename = os.path.join(output_dir, f"interaction_plot_{metric}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f" - Gráfico salvo em: {filename}")


def plot_topology_matrix(df, output_dir):
    """
    Gera uma matriz visual com a topologia dos grafos.
    """
    print("Gerando matriz de topologia...")
    margins = sorted(df['margin'].unique())
    thresholds = sorted(df['threshold'].unique())
    
    # Ajusta o tamanho da figura dinamicamente
    fig, axes = plt.subplots(
        nrows=len(margins), 
        ncols=len(thresholds), 
        figsize=(4 * len(thresholds), 4 * len(margins)),
        squeeze=False # Garante que 'axes' seja sempre 2D
    )

    for i, m in enumerate(margins):
        for j, t in enumerate(thresholds):
            ax = axes[i, j]
            data = df[(df['margin'] == m) & (df['threshold'] == t)].iloc[0]
            G = data['graph']
            pos = data['pos']

            nx.draw(
                G, pos, ax=ax, 
                node_size=10, 
                width=0.5,
                node_color='skyblue',
                edge_color='gray'
            )
            
            ax.set_title(f"Margin={m}, Threshold={t}", fontsize=10)
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    plt.tight_layout(pad=3.0)
    filename = os.path.join(output_dir, "topology_matrix.png")
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f" - Matriz de topologia salva em: {filename}")


def main():
    """
    Função principal para executar a análise.
    """
    # --- Configuração do Experimento ---
    file_path = os.path.abspath(os.path.join(RESULTS_DIR, "..", "planilhas", "equipment_processado.xlsx"))
    sheet_name = "Parnaiba3_Transformado"

    # Use uma grade maior para uma análise mais rica
    # margins = [1.5, 2.0, 2.5, 3.0]
    # thresholds = [10, 20, 30, 40]
    margins = [2.5, 3.0]
    thresholds = [10, 20]

    # --- Execução ---
    df_results = run_graph_sensitivity_analysis(file_path, sheet_name, margins, thresholds)

    # Salva os dados brutos em CSV
    csv_path = os.path.join(RESULTS_DIR, "advanced_graph_metrics.csv")
    # Exclui colunas de objetos para salvar o CSV
    df_to_save = df_results.drop(columns=['graph', 'pos'])
    df_to_save.to_csv(csv_path, index=False, float_format='%.5f')
    print(f"\nMétricas avançadas salvas em: {csv_path}")

    # --- Visualização ---
    metrics_to_plot = [
        'density',
        'isolated_nodes',
        'giant_component_ratio',
        'nodes',
        'edges',
        'components'
    ]
    plot_interaction_graphs(df_results, metrics_to_plot, RESULTS_DIR)
    plot_topology_matrix(df_results, RESULTS_DIR)

    print("\nAnálise concluída com sucesso!")


if __name__ == "__main__":
    main()