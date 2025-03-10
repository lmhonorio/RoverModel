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
    ###############################################################################
    # Função para plotar subgrafos em cores diferentes
    ###############################################################################
    def plot_subgraphs(G):
        subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        colors = plt.cm.rainbow(range(len(subgraphs)))

        plt.figure(figsize=(10, 10))
        for subgraph, color in zip(subgraphs, colors):
            pos = {node: node for node in subgraph.nodes()}
            nx.draw(subgraph, pos, node_color=[color], edge_color=[color], with_labels=False)
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
    def plot_obstacles_aabbs_segments(obstacles, aabbs, segments):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.grid(True, linestyle='--', color='lightgray', alpha=0.7)

        # Obstáculos
        for obs in obstacles:
            x, y=obs["pos"]
            w, h=obs["size"]
            rect=plt.Rectangle((x-w/2,y-h/2), w,h,color='red',alpha=0.7)
            ax.add_patch(rect)

        # AABBs
        for (aabb_x,aabb_y),aabb_w,aabb_h in aabbs:
            arect=plt.Rectangle((aabb_x,aabb_y),aabb_w,aabb_h,edgecolor='blue',facecolor='none',linewidth=1.5)
            ax.add_patch(arect)

        # Segmentos
        for (x1,y1,x2,y2) in segments:
            ax.plot([x1,x2],[y1,y2],color='green',linewidth=1.5)

        ax.set_xlabel("X(m)")
        ax.set_ylabel("Y(m)")
        ax.set_title("Obstáculos, AABBs e Segmentos")
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