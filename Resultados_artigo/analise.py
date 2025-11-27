"""Experimento: grade margin x threshold para AABB merging

Gera um CSV com métricas (nº de AABBs, área total, tempo) para cada par
de (margin, threshold) e salva heatmaps das métricas em
`Resultados_artigo/`.

Uso: python3 Resultados_artigo/analise.py
"""
import time
import csv
import os
from collections import OrderedDict

import sys
# Ensure repository root is on sys.path so local imports (roverclass, CriarPontosObservacao, etc.) work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib.pyplot as plt

from roverclass import ObstacleLoader
from aabbutils import AABBUtils
from CriarPontosObservacao import build_graph
import networkx as nx


RESULTS_DIR = os.path.dirname(__file__)


def run_grid_experiment(obstacles, margins, thresholds, out_csv_path=None):
	rows = []

	# matrices for heatmaps
	num_aabbs_mat = np.zeros((len(margins), len(thresholds)), dtype=int)
	total_area_mat = np.zeros((len(margins), len(thresholds)), dtype=float)
	time_mat = np.zeros((len(margins), len(thresholds)), dtype=float)

	for i, m in enumerate(margins):
		for j, t in enumerate(thresholds):
			t0 = time.perf_counter()
			try:
				aabbs = AABBUtils.get_aabbs(obstacles, margin=m, threshold=t)
			except TypeError:
				# backward compatibility: some calls in repo pass positional args
				aabbs = AABBUtils.get_aabbs(obstacles, m, t)
			elapsed = time.perf_counter() - t0

			num_aabbs = len(aabbs)
			total_area = sum([w * h for ((x, y), w, h) in aabbs])

			num_aabbs_mat[i, j] = num_aabbs
			total_area_mat[i, j] = total_area
			time_mat[i, j] = elapsed

			rows.append(OrderedDict([
				("margin", m),
				("threshold", t),
				("num_aabbs", num_aabbs),
				("total_area", total_area),
				("elapsed_s", round(elapsed, 4)),
			]))

			print(f"margin={m:4} threshold={t:3} -> aabbs={num_aabbs} area={total_area:.1f} t={elapsed:.3f}s")

	if out_csv_path:
		with open(out_csv_path, "w", newline="") as f:
			writer = csv.DictWriter(f, fieldnames=rows[0].keys())
			writer.writeheader()
			for r in rows:
				writer.writerow(r)

	return {
		"margins": margins,
		"thresholds": thresholds,
		"num_aabbs": num_aabbs_mat,
		"total_area": total_area_mat,
		"time": time_mat,
		"rows": rows,
	}


def plot_heatmap(mat, x_labels, y_labels, title, out_path, cmap="viridis", fmt=None):
	plt.figure(figsize=(8, 6))
	im = plt.imshow(mat, origin="lower", cmap=cmap, aspect="auto")
	plt.colorbar(im, fraction=0.046, pad=0.04)

	plt.xticks(range(len(x_labels)), x_labels)
	plt.yticks(range(len(y_labels)), y_labels)
	plt.xlabel("threshold")
	plt.ylabel("margin")
	plt.title(title)

	if fmt is not None:
		for (i, j), val in np.ndenumerate(mat):
			plt.text(j, i, fmt.format(val), ha="center", va="center", color="white", fontsize=8)

	plt.tight_layout()
	plt.savefig(out_path, dpi=200)
	plt.close()


def main():
	# configuração — adapte caminhos se necessário
	# Usa caminho absoluto baseado na localização deste script para evitar erros
	file_path = os.path.abspath(os.path.join(RESULTS_DIR, "..", "planilhas", "equipment_processado.xlsx"))
	sheet_name = "Parnaiba3_Transformado"

	margins = [1.5, 2.0]
	thresholds = [10, 20]
	
    # margins = [1.5, 2.0, 2.5, 3.0, 3.5]
	# thresholds = [10, 20, 30, 40, 50]

	print("Carregando obstáculos...", file_path)
	loader = ObstacleLoader(file_path, sheet_name)
	obstacles = loader.get_obstacles()
	print(f"Obstáculos carregados: {len(obstacles)}")

	out_csv = os.path.join(RESULTS_DIR, "grid_margin_threshold_results.csv")
	print("Executando grid experiment — isso pode demorar alguns segundos por ponto...")
	res = run_grid_experiment(obstacles, margins, thresholds, out_csv_path=out_csv)

	# salvar heatmaps
	num_path = os.path.join(RESULTS_DIR, "heatmap_num_aabbs.png")
	area_path = os.path.join(RESULTS_DIR, "heatmap_total_area.png")
	time_path = os.path.join(RESULTS_DIR, "heatmap_time_s.png")

	# Note: imshow expects shape (rows=margins, cols=thresholds)
	plot_heatmap(res["num_aabbs"], thresholds, margins, "Num AABBs (margin x threshold)", num_path, cmap="magma", fmt="{}")
	plot_heatmap(res["total_area"], thresholds, margins, "Total AABB Area", area_path, cmap="plasma", fmt="{:.0f}")
	plot_heatmap(res["time"], thresholds, margins, "Elapsed time (s)", time_path, cmap="viridis", fmt="{:.2f}")

	print("Resultados salvos em:")
	print(" - ", out_csv)
	print(" - ", num_path)
	print(" - ", area_path)
	print(" - ", time_path)

	# ------------------ Experimento de grafo final ------------------
	print("\nExecutando grid experiment no grafo final (build_graph)...")
	graph_csv = os.path.join(RESULTS_DIR, "grid_graph_metrics.csv")
	# matrices
	nodes_mat = np.zeros((len(margins), len(thresholds)), dtype=int)
	edges_mat = np.zeros((len(margins), len(thresholds)), dtype=int)
	components_mat = np.zeros((len(margins), len(thresholds)), dtype=int)
	avgdeg_mat = np.zeros((len(margins), len(thresholds)), dtype=float)

	rows = []

	for i, m in enumerate(margins):
		for j, t in enumerate(thresholds):
			print(f"building graph for margin={m} threshold={t} ...")
			g0 = time.perf_counter()
			G_corrigido, aabbs, observation_points, obstacles2 = build_graph(file_path, sheet_name, margin=m, threshold=t, plotting=False)
			elapsed = time.perf_counter() - g0

			n_nodes = G_corrigido.number_of_nodes()
			n_edges = G_corrigido.number_of_edges()
			n_components = nx.number_connected_components(G_corrigido)
			deg_seq = [d for _, d in G_corrigido.degree()]
			avg_deg = float(np.mean(deg_seq)) if len(deg_seq) else 0.0

			nodes_mat[i, j] = n_nodes
			edges_mat[i, j] = n_edges
			components_mat[i, j] = n_components
			avgdeg_mat[i, j] = avg_deg

			rows.append({
				"margin": m,
				"threshold": t,
				"nodes": n_nodes,
				"edges": n_edges,
				"components": n_components,
				"avg_degree": round(avg_deg, 3),
				"elapsed_s": round(elapsed, 3),
			})

	# salva CSV
	with open(graph_csv, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=rows[0].keys())
		writer.writeheader()
		for r in rows:
			writer.writerow(r)

	# heatmaps
	plot_heatmap(nodes_mat, thresholds, margins, "Graph: # nodes", os.path.join(RESULTS_DIR, "heatmap_graph_nodes.png"), cmap="magma", fmt="{}")
	plot_heatmap(edges_mat, thresholds, margins, "Graph: # edges", os.path.join(RESULTS_DIR, "heatmap_graph_edges.png"), cmap="plasma", fmt="{}")
	plot_heatmap(avgdeg_mat, thresholds, margins, "Graph: avg degree", os.path.join(RESULTS_DIR, "heatmap_graph_avgdeg.png"), cmap="viridis", fmt="{:.2f}")

	print("Graph experiment results saved:")
	print(" - ", graph_csv)
	print(" - ", os.path.join(RESULTS_DIR, "heatmap_graph_nodes.png"))
	print(" - ", os.path.join(RESULTS_DIR, "heatmap_graph_edges.png"))
	print(" - ", os.path.join(RESULTS_DIR, "heatmap_graph_avgdeg.png"))


if __name__ == "__main__":
	main()

