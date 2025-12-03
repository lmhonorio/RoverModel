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
import numpy as np
import runpy

# Ensures local modules (roverclass, etc.) can be imported
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from CriarPontosObservacao import build_graph

RESULTS_DIR = os.path.dirname(__file__)

# Matplotlib/Seaborn default font sizes tuned for publication-quality figures
plt.rcParams.update({
    'figure.titlesize': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 13,
    'font.size': 12,
})


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
            fmt=".2f" if metric in ['nodes', 'edges', 'components'] else ".3f",
            annot_kws={'fontsize': 12},
            cmap="viridis",
            linewidths=.5,
            ax=ax
        )
        
        metric_title = metric.replace('_', ' ').title()
        ax.set_title(f'Sensitivity Heatmap: {metric_title}', fontsize=18, pad=20)
        ax.set_xlabel('Threshold', fontsize=14)
        ax.set_ylabel('Margin', fontsize=14)
        # Tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)
        # Colorbar ticks size
        try:
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=12)
        except Exception:
            pass
        filename = os.path.join(output_dir, f"heatmap_{metric}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f" - Plot saved to: {filename}")


def plot_correlation_matrix(df, metrics, output_dir):
    """
    Generates a correlation matrix heatmap between the graph metrics.
    """
    print("Generating correlation matrix heatmap...")
    sns.set_theme(style="white")

    # Select metric columns plus margin and threshold for correlation (if present)
    cols = [c for c in (metrics + ['margin', 'threshold']) if c in df.columns]
    if not cols:
        raise ValueError('No metric columns found for correlation heatmap')

    df_metrics = df[cols].copy()
    # Ensure numeric dtype
    df_metrics = df_metrics.apply(pd.to_numeric, errors='coerce')

    corr = df_metrics.corr()

    # Mask the upper triangle for cleaner display
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        annot_kws={'fontsize': 12},
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title("Correlation Matrix (Heatmap)", fontsize=18, pad=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    try:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
    except Exception:
        pass
    filename = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" - Correlation heatmap saved to: {filename}")


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

    ax.set_title(f'Distribution of "{metric_title}" by "{group_by_title}"', fontsize=18)
    ax.set_xlabel(group_by_title, fontsize=14)
    ax.set_ylabel(metric_title, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    filename = os.path.join(output_dir, f"distribution_{metric}_by_{group_by}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" - Distribution plot saved to: {filename}")


def plot_tradeoff_scatter(df, output_dir):
    """
    Scatter plot showing the trade-off between number of nodes (complexity)
    and largest component ratio (connectivity).

    - X: nodes
    - Y: largest_component_ratio
    - Color: margin (shows decision groups)
    - Size: threshold (bubble size)
    """
    print("Generating trade-off scatter plot...")

    # Prepare dataframe and coerce numeric types
    cols = ['nodes', 'largest_component_ratio', 'margin', 'threshold']
    available = [c for c in cols if c in df.columns]
    if 'nodes' not in available or 'largest_component_ratio' not in available:
        print(" - Required columns for trade-off plot are missing. Skipping.")
        return

    df2 = df.copy()
    # Coerce
    df2['nodes'] = pd.to_numeric(df2['nodes'], errors='coerce')
    df2['largest_component_ratio'] = pd.to_numeric(df2['largest_component_ratio'], errors='coerce')
    if 'threshold' in df2.columns:
        df2['threshold'] = pd.to_numeric(df2['threshold'], errors='coerce')
    else:
        # fallback constant size
        df2['threshold'] = 1
    if 'margin' in df2.columns:
        # keep numeric if possible, otherwise treat as categorical
        try:
            df2['margin'] = pd.to_numeric(df2['margin'], errors='coerce')
            margin_is_numeric = True
        except Exception:
            margin_is_numeric = False
    else:
        df2['margin'] = 0
        margin_is_numeric = True

    # Drop rows with missing x/y
    df2 = df2.dropna(subset=['nodes', 'largest_component_ratio'])

    plt.figure(figsize=(10, 7))
    sns.set_style('whitegrid')

    # Use seaborn scatterplot with size mapping
    scatter = sns.scatterplot(
        data=df2,
        x='nodes',
        y='largest_component_ratio',
        hue='margin' if 'margin' in df2.columns else None,
        size='threshold',
        sizes=(40, 300),
        palette='viridis',
        alpha=0.85,
        edgecolor='k'
    )

    scatter.set_title('Trade-off: Nodes vs Largest Component Ratio', fontsize=18)
    scatter.set_xlabel('Number of Nodes (Complexity)', fontsize=14)
    scatter.set_ylabel('Largest Component Ratio (Connectivity)', fontsize=14)
    scatter.tick_params(axis='both', which='major', labelsize=12)

    # Improve legend: collapse size and hue legends
    handles, labels = scatter.get_legend_handles_labels()
    # seaborn creates combined legend entries; keep default but make it readable
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=12, title_fontsize=13)

    filename = os.path.join(output_dir, 'tradeoff_scatter.png')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" - Trade-off scatter saved to: {filename}")


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

    # CSV path (if exists, use it instead of recomputing)
    csv_path = os.path.join(RESULTS_DIR, "advanced_graph_metrics.csv")

    # --- Execution: prefer existing CSV to avoid heavy recomputation ---
    if os.path.exists(csv_path):
        print(f"Found existing CSV at {csv_path}. Loading metrics and skipping graph reconstruction...")
        df_results = pd.read_csv(csv_path)

        # Try to restore sensible dtypes for plotting (optional)
        if 'threshold' in df_results.columns:
            try:
                df_results['threshold'] = df_results['threshold'].astype(int)
            except Exception:
                # leave as-is if conversion fails
                pass
        if 'margin' in df_results.columns:
            try:
                df_results['margin'] = df_results['margin'].astype(float)
            except Exception:
                pass

        print("CSV loaded. Proceeding to generate plots.")
    else:
        print("CSV not found — running full sensitivity analysis (this may take a while)...")
        # --- Execution (full run) ---
        df_results = run_graph_sensitivity_analysis(file_path, sheet_name, margins, thresholds)

        # Save raw data to CSV (exclude heavy/object columns)
        df_to_save = df_results.drop(columns=['graph', 'pos'])
        df_to_save.to_csv(csv_path, index=False, float_format='%.5f')
        print(f"\nAdvanced metrics saved to: {csv_path}")

    # --- Visualization ---
    metrics_to_plot = [
        'nodes',
        'edges',
        'components',
        'density',
        'largest_component_ratio',
        'fragmentation_index'
    ]
    
    # Generate the new statistical plots
    plot_metric_heatmaps(df_results, metrics_to_plot, RESULTS_DIR)
    plot_correlation_matrix(df_results, metrics_to_plot, RESULTS_DIR)
    plot_tradeoff_scatter(df_results, RESULTS_DIR)
    plot_distribution_analysis(df_results, metric='largest_component_ratio', group_by='margin', output_dir=RESULTS_DIR)
    
    # --- Comparative analysis plots (from analise_comparativa) ---
    comp_csv = os.path.join(RESULTS_DIR, "comparacao_resultados.csv")
    if not os.path.exists(comp_csv):
        # Try to run the comparative analysis script to produce the CSV and path plot
        try:
            print("Comparative CSV not found — running Resultados_artigo/analise_comparativa.py to generate comparative metrics...")
            runpy.run_path(os.path.join(RESULTS_DIR, 'analise_comparativa.py'), run_name='__main__')
        except Exception as e:
            print(f" - Could not run comparative analysis script: {e}")

    if os.path.exists(comp_csv):
        try:
            df_comp = pd.read_csv(comp_csv)
            # Coerce numeric columns that may be strings
            for col in ['Vértices', 'Tortuosidade Média', 'Nº de Ilhas']:
                if col in df_comp.columns:
                    df_comp[col] = pd.to_numeric(df_comp[col], errors='coerce')

            def plot_comparative_bars(df_comp, output_dir):
                """Generate the three comparative barplots requested by the user."""
                sns.set_theme(style='whitegrid')

                # 1) Efficiency (Vértices)
                if 'Vértices' in df_comp.columns:
                    plt.figure(figsize=(8,6))
                    ax = sns.barplot(x='Grafo', y='Vértices', data=df_comp, palette='viridis')
                    ax.set_title('Eficiência Computacional (Número de Vértices)')
                    ax.set_ylabel('Número de Vértices')
                    ax.set_xlabel('Grafo')
                    ax.tick_params(axis='x', rotation=15)
                    # annotate
                    for p in ax.patches:
                        h = p.get_height()
                        ax.text(p.get_x() + p.get_width()/2., h + max(df_comp['Vértices'].max()*0.01,1), f"{int(h):,}", ha='center', fontsize=11)
                    out = os.path.join(output_dir, 'comparacao_vertices.png')
                    plt.tight_layout()
                    plt.savefig(out, dpi=300)
                    plt.close()
                    print(f" - Saved: {out}")

                # 2) Tortuosidade (only that metric)
                if 'Tortuosidade Média' in df_comp.columns:
                    plt.figure(figsize=(8,6))
                    ax = sns.barplot(x='Grafo', y='Tortuosidade Média', data=df_comp, palette='magma')
                    ax.set_title('Tortuosidade Média dos Caminhos')
                    ax.set_ylabel('Tortuosidade Média')
                    ax.set_xlabel('Grafo')
                    ax.tick_params(axis='x', rotation=15)
                    for p in ax.patches:
                        h = p.get_height()
                        ax.text(p.get_x() + p.get_width()/2., h + max(df_comp['Tortuosidade Média'].max()*0.01,0.01), f"{h:.2f}", ha='center', fontsize=11)
                    out = os.path.join(output_dir, 'comparacao_tortuosidade.png')
                    plt.tight_layout()
                    plt.savefig(out, dpi=300)
                    plt.close()
                    print(f" - Saved: {out}")

                # 3) Nº de Ilhas (log scale)
                if 'Nº de Ilhas' in df_comp.columns:
                    plt.figure(figsize=(8,6))
                    ax = sns.barplot(x='Grafo', y='Nº de Ilhas', data=df_comp, palette='cividis')
                    ax.set_yscale('log')
                    ax.set_title('Número de Ilhas (Componentes Conectadas) - Escala Log')
                    ax.set_ylabel('Nº de Ilhas (log scale)')
                    ax.set_xlabel('Grafo')
                    ax.tick_params(axis='x', rotation=15)
                    for p in ax.patches:
                        h = p.get_height()
                        # annotate with raw integer value
                        ax.text(p.get_x() + p.get_width()/2., h * 1.1, f"{int(h):,}", ha='center', fontsize=11)
                    out = os.path.join(output_dir, 'comparacao_ilhas_log.png')
                    plt.tight_layout()
                    plt.savefig(out, dpi=300)
                    plt.close()
                    print(f" - Saved: {out}")

            plot_comparative_bars(df_comp, RESULTS_DIR)
        except Exception as e:
            print(f" - Could not generate comparative plots: {e}")
    else:
        print(" - Comparative CSV still not available; skipping comparative barplots.")
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()