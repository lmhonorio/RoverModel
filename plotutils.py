###############################################################################
# CLASSE: PlotUtils
###############################################################################

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from networkx.drawing.nx_agraph import to_agraph
from pygraphviz import AGraph
import networkx as nx

class PlotUtils:

    @staticmethod
    def plot_graph_with_indexed_labels(G, indexed_labels):
        plt.figure(figsize=(10, 10))
        pos = {node: node for node in G.nodes()}  # Usa as coordenadas dos nós para posicionamento

        # Extrai cores distintas para os labels
        unique_labels = list(set(indexed_labels.values()))
        color_map = {label: plt.cm.rainbow(i / len(unique_labels)) for i, label in enumerate(unique_labels)}

        node_colors = [color_map[indexed_labels[node]] for node in G.nodes()]

        # Desenha o grafo
        nx.draw(G, pos, node_color=node_colors, edge_color='gray', with_labels=True, font_size=8)

        # Adiciona os labels numerados aos nós
        node_labels = {node: indexed_labels[node] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='black')

        plt.show()
    ###############################################################################
    # Função para plotar subgrafos em cores diferentes
    ###############################################################################
    @staticmethod
    def plot_subgraphs(G):
        subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        colors = plt.cm.rainbow(range(len(subgraphs)))

        plt.figure(figsize=(10, 10))
        for subgraph, color in zip(subgraphs, colors):
            pos = {node: node for node in subgraph.nodes()}
            labels = nx.get_node_attributes(subgraph, 'label')  # Obtendo os labels dos nós
            nx.draw(subgraph, pos, node_color=[color], edge_color=[color], with_labels=False)
            nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8, font_color='black')
        plt.show()

    @staticmethod
    def plot_grafo(G, filename, figsize=(35, 35), titulo=None):
        agraph = to_agraph(G)
        agraph.layout(prog='dot')
        agraph.draw(filename)

        plt.figure(figsize=figsize)
        img = plt.imread(filename)
        plt.imshow(img)
        plt.axis('off')

        if titulo:
            plt.title(titulo, fontsize=20, fontweight='bold')

        plt.show()

    @staticmethod
    def plot_clusters_aabbs(aabbs, labels):
        cmap = cm.get_cmap('tab10')
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels)

        fig, ax = plt.subplots(figsize=(8,8))
        ax.grid(True, linestyle='--', color='lightgray', alpha=0.7)

        for i, ((ax_, ay_), w_, h_) in enumerate(aabbs):
            cluster_id=labels[i]
            color_index=(cluster_id-1)%10
            color=cmap(color_index)
            rect=plt.Rectangle((ax_, ay_), w_, h_, edgecolor=color, facecolor='none', linewidth=2)
            ax.add_patch(rect)

        ax.set_aspect('equal','box')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"AABBs Coloridas por Cluster (total = {num_clusters})")
        plt.show()

    @staticmethod
    def plot_obstacles_aabbs(obstacles, aabbs):
        fig, ax = plt.subplots(figsize=(10, 10))  # Ajuste do tamanho do gráfico
        ax.grid(True, linestyle='--', color='lightgray', alpha=0.7)

        # Obstáculos (retângulos vermelhos)
        for obs in obstacles:
            x, y = obs["pos"]
            w, h = obs["size"]
            rect = plt.Rectangle((x - w / 2, y - h / 2), w, h, color='red', alpha=0.5, label="Obstáculo")
            ax.add_patch(rect)

        # AABBs (retângulos azuis)
        for (aabb_x, aabb_y), aabb_w, aabb_h in aabbs:
            arect = plt.Rectangle((aabb_x, aabb_y), aabb_w, aabb_h, edgecolor='blue', facecolor='none', linewidth=1.5,
                                  label="AABB")
            ax.add_patch(arect)

        # Ajuste dos limites do gráfico
        ax.autoscale()

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Obstáculos e AABBs")
        plt.legend(["Obstáculo", "AABB"], loc="upper right")
        plt.show()

    @staticmethod
    def plot_segments_with_vertices(segments, raio=2):
        fig, ax = plt.subplots(figsize=(8,8))
        ax.grid(True, linestyle='--', color='lightgray', alpha=0.7)

        for(x1,y1,x2,y2) in segments:
            ax.plot([x1,x2],[y1,y2],color='green',linewidth=1.5)
            c1=plt.Circle((x1,y1),raio,color='red',fill=True)
            c2=plt.Circle((x2,y2),raio,color='red',fill=True)
            ax.add_patch(c1)
            ax.add_patch(c2)

        ax.set_aspect('equal','box')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Segmentos em verde + vértices vermelhos (raio={raio})")
        plt.show()