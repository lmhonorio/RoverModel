"""Advanced Experiment: Graph Topology Sensitivity Analysis

This script performs a sensitivity analysis of the `margin` and `threshold`
parameters on the structure of the observation graph.

It calculates advanced graph metrics (density, isolated nodes, largest
component) and generates several key visualizations for analysis:

1.  Metric Heatmaps: Show how each metric varies with `margin` and `threshold`.
2.  Topology Matrix: A grid displaying the visual structure of the graph for
    each `(margin, threshold)` combination.
3.  Correlation and Distribution Plots: Statistical analysis of the relationships
    between metrics.

Usage: python3 Resultados_artigo/analise_avancada.py
"""
import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

# Ensures local modules (roverclass, etc.) can be imported
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from CriarPontosObservacao import build_graph

RESULTS_DIR = os.path.dirname(__file__)


def run_graph_sensitivity_analysis(file_path, sheet_name, margins, thresholds):
    """
    Runs the sensitivity analysis, calculates metrics, and stores the graphs.
    """
    results = []
    print("Starting graph sensitivity analysis...")

    for m in margins:
        for t in thresholds:
            print(f"Processing: margin={m}, threshold={t}...")
            g0 = time.perf_counter()
            G, _, obs_points, _ = build_graph(
                file_path, sheet_name, margin=m, threshold=t, plotting=False
            )
            elapsed = time.perf_counter() - g0

            # Metric calculation
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            density = nx.density(G) if n_nodes > 1 else 0
            isolated_nodes = nx.number_of_isolates(G)
            n_components = nx.number_connected_components(G)

            largest_component_ratio = 0
            fragmentation_index = 1.0  # Maximum fragmentation if there are no nodes
            if n_nodes > 0:
                largest_cc = max(nx.connected_components(G), key=len, default=set())
                largest_component_ratio = len(largest_cc) / n_nodes
                fragmentation_index = 1.0 - largest_component_ratio

            results.append({
                "margin": m,
                "threshold": t,
                "nodes": n_nodes,
                "edges": n_edges,
                "components": n_components,
                "density": density,
                "isolated_nodes": isolated_nodes,
                "largest_component_ratio": largest_component_ratio,
                "fragmentation_index": fragmentation_index,
                "graph": G,
                "pos": {i: (p[0], p[1]) for i, p in enumerate(obs_points)},
                "time_s": elapsed,
            })

    return pd.DataFrame(results)


def plot_metric_heatmaps(df, metrics, output_dir):
    """
    Generates heatmaps to visualize the impact of parameters on graph metrics.
    """
    print("Generating metric heatmaps...")
    sns.set_theme(style="whitegrid")

    for metric in metrics:
        # Pivot the data for the heatmap format
        pivot_table = df.pivot(index="margin", columns="threshold", values=metric)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".2f" if metric in ['nodes', 'edges', 'components', 'isolated_nodes'] else ".3f",
            cmap="viridis",
            linewidths=.5,
            ax=ax
        )
        
        metric_title = metric.replace('_', ' ').title()
        ax.set_title(f'Sensitivity Heatmap: {metric_title}', fontsize=16, pad=20)
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Margin', fontsize=12)
        filename = os.path.join(output_dir, f"heatmap_{metric}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f" - Plot saved to: {filename}")


def plot_correlation_matrix(df, metrics, output_dir):
    """
    Generates a correlation matrix (pairplot) between the graph metrics.
    """
    print("Generating correlation matrix (pairplot)...")
    sns.set_theme(style="ticks")
    
    # Select only the metric and parameter columns
    df_subset = df[metrics + ['margin', 'threshold']]
    
    g = sns.pairplot(df_subset, hue="threshold", palette="viridis", diag_kind="kde")
    g.figure.suptitle("Correlation Matrix and Metric Distribution", y=1.02, fontsize=16)
    
    filename = os.path.join(output_dir, "correlation_pairplot.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" - Pairplot saved to: {filename}")


def plot_distribution_analysis(df, metric, group_by, output_dir):
    """
    Creates a boxplot and a stripplot to analyze the distribution of a metric.
    """
    print(f"Generating distribution analysis for '{metric}'...")
    sns.set_theme(style="whitegrid")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sns.boxplot(x=group_by, y=metric, data=df, ax=ax, palette="coolwarm")
    sns.stripplot(x=group_by, y=metric, data=df, ax=ax, color=".25", size=6, jitter=True, alpha=0.7)
    
    metric_title = metric.replace('_', ' ').title()
    group_by_title = group_by.replace('_', ' ').title()
    
    ax.set_title(f'Distribution of "{metric_title}" by "{group_by_title}"', fontsize=16)
    ax.set_xlabel(group_by_title, fontsize=12)
    ax.set_ylabel(metric_title, fontsize=12)
    
    filename = os.path.join(output_dir, f"distribution_{metric}_by_{group_by}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" - Distribution plot saved to: {filename}")


def main():
    """
    Main function to run the analysis.
    """
    # --- Experiment Setup ---
    file_path = os.path.abspath(os.path.join(RESULTS_DIR, "..", "planilhas", "equipment_processado.xlsx"))
    sheet_name = "Parnaiba3_Transformado"

    # Use a larger grid for a richer analysis
    margins = [1.5, 2.0, 2.5, 3.0, 3.5]
    thresholds = [10, 20, 30, 40, 50]

    # --- Execution ---
    df_results = run_graph_sensitivity_analysis(file_path, sheet_name, margins, thresholds)

    # Save raw data to CSV
    csv_path = os.path.join(RESULTS_DIR, "advanced_graph_metrics.csv")
    # Exclude object columns to save the CSV
    df_to_save = df_results.drop(columns=['graph', 'pos'])
    df_to_save.to_csv(csv_path, index=False, float_format='%.5f')
    print(f"\nAdvanced metrics saved to: {csv_path}")

    # --- Visualization ---
    metrics_to_plot = [
        'nodes',
        'edges',
        'components',
        'density',
        'isolated_nodes',
        'largest_component_ratio',
        'fragmentation_index'
    ]
    
    # Generate the new statistical plots
    plot_metric_heatmaps(df_results, metrics_to_plot, RESULTS_DIR)
    plot_correlation_matrix(df_results, metrics_to_plot, RESULTS_DIR)
    plot_distribution_analysis(df_results, metric='largest_component_ratio', group_by='margin', output_dir=RESULTS_DIR)
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()