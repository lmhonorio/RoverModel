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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import subprocess


class PlotUtils:

    @staticmethod
    def plot_aabbs_obstacles_points(obstacles, aabbs, points, point_color=None, point_radius=0.5):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.grid(True, linestyle='--', color='lightgray', alpha=0.7)

        # Criar mapa de cores por label (se necessário)
        if point_color is None:
            labels = sorted(set(label for _, _, label in points))
            cmap = cm.get_cmap('tab20', len(labels))
            label_colors = {label: cmap(i) for i, label in enumerate(labels)}
        else:
            label_colors = None

        # Plotar obstáculos com cor do label (mesma dos pontos)
        for obs in obstacles:
            (x, y) = obs["pos"]
            w, h = obs["size"]
            x0 = x - w / 2
            y0 = y - h / 2
            label = obs.get("label", "unknown")
            color = point_color if point_color else label_colors.get(label, 'gray')

            rect = plt.Rectangle((x0, y0), w, h, facecolor=color, edgecolor=color, alpha=0.5)
            ax.add_patch(rect)

        # Plotar AABBs (preenchidas, vermelhas claras)
        for (x, y), w, h in aabbs:
            rect = plt.Rectangle((x, y), w, h, facecolor='lightcoral', edgecolor='red', alpha=0.2)
            ax.add_patch(rect)

        # Plotar pontos de observação (sem texto)
        for (px, py, label) in points:
            color = point_color if point_color else label_colors[label]
            ax.plot(px, py, 'o', color=color, markersize=point_radius * 5)

        ax.set_aspect('equal')
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_title("AABBs, Obstacles and Observation Points")
        plt.show()



    @staticmethod
    def plot_mission_graph(G_m):
        """ Plota o grafo de missões G_m """
        plt.figure(figsize=(8, 6))

        pos = nx.spring_layout(G_m)  # Define um layout para os nós
        nx.draw(G_m, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, font_size=10,
                font_weight="bold")

        edge_labels = {(u, v): f"{u} → {v}" for u, v in G_m.edges()}
        nx.draw_networkx_edge_labels(G_m, pos, edge_labels=edge_labels, font_size=8, font_color="red")

        plt.title("Mission Dependency Graph (G_m)")
        plt.show()

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

    @staticmethod
    def plot_rotas_grafo(G, rotas_por_robo):
        pos = nx.get_node_attributes(G, 'pos')

        plt.figure(figsize=(12, 10))

        # Plotar o grafo base
        nx.draw(G, pos, node_color='lightgray', edge_color='gray', node_size=50, with_labels=False)

        # Plotar rotas por robô com cores diferentes
        cores = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
        for i, (robo, rota) in enumerate(rotas_por_robo.items()):
            coords_rota = np.array([pos[node] for node in rota])
            plt.plot(coords_rota[:, 0], coords_rota[:, 1], marker='o', linestyle='-', linewidth=2,
                     color=cores[i % len(cores)], label=f'Rota {robo}')

        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.title("Optimized Routes per Robot (Graph)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    @staticmethod
    def plot_rotas_reais(G_robot, rotas_por_robo, pontos_vistoria):
        plt.figure(figsize=(12, 10))

        cores = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
        for i, (robo, rota) in enumerate(rotas_por_robo.items()):
            caminho_completo = []

            for j in range(len(rota) - 1):
                u, v = rota[j], rota[j + 1]

                subpath = nx.shortest_path(G_robot, source=u, target=v, weight="weight")

                if caminho_completo and subpath[0] == caminho_completo[-1]:
                    caminho_completo.extend(subpath[1:])
                else:
                    caminho_completo.extend(subpath)

            coords_caminho = np.array([G_robot.nodes[n]['pos'] for n in caminho_completo])

            # Plota todo o caminho em linha
            plt.plot(coords_caminho[:, 0], coords_caminho[:, 1],
                     linestyle='-', linewidth=2,
                     color=cores[i % len(cores)], label=f'Rota {robo}')

            # Diferencia pontos de vistoria dos pontos intermediários
            for ponto in caminho_completo:
                x, y = G_robot.nodes[ponto]['pos']
                if ponto in pontos_vistoria:
                    plt.plot(x, y, marker='o', markersize=10, color='yellow', markeredgecolor='black', zorder=5)
                else:
                    plt.plot(x, y, marker='.', markersize=5, color='gray', zorder=4)

            # Marca posição inicial claramente
            plt.plot(coords_caminho[0, 0], coords_caminho[0, 1], marker='s', color='black', markersize=12, zorder=6)

            # Setas indicando o percurso
            for k in range(len(coords_caminho) - 1):
                plt.arrow(coords_caminho[k, 0], coords_caminho[k, 1],
                          coords_caminho[k + 1, 0] - coords_caminho[k, 0],
                          coords_caminho[k + 1, 1] - coords_caminho[k, 1],
                          shape='full', lw=0, length_includes_head=True, head_width=1.0,
                          color=cores[i % len(cores)], alpha=0.4, zorder=3)

        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.title("Optimized Routes (Inspection vs Passage)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()




    ###############################################################################
    # Função para plotar subgrafos em cores diferentes
    ###############################################################################
    @staticmethod
    def plot_grafo_distance(G, filename=r"grafo2.png", figsize=(12, 12), titulo=None):
        """
        Gera um arquivo de imagem (filename) do grafo usando Graphviz
        e exibe com matplotlib. Tenta refletir 'weight' como distância,
        configurando o atributo 'len' em cada aresta antes de chamar layout='neato'.

        """
        try:
            # Try to find Graphviz in common paths
            graphviz_paths = [
                r"C:\Program Files\Graphviz\bin",
                r"C:\Program Files (x86)\Graphviz\bin",
            ]

            for path in graphviz_paths:
                if os.path.exists(path):
                    os.environ["PATH"] += os.pathsep + path
                    break
            else:
                raise RuntimeError("Graphviz not found. Install it from https://graphviz.org/download/")

            A = to_agraph(G)
            A.graph_attr.update(size="30,30", ratio="compress")
            # Optional: Adjust edge lengths based on weights
            scale = 0.1
            for u, v in G.edges():
                w = G[u][v].get("weight", 1.0)
                edge = A.get_edge(u, v)
                edge.attr["len"] = str(w * scale)


            A.layout(prog="dot")
            A.draw(filename)  # Save the image
            # Display using matplotlib
            plt.figure(figsize=figsize)
            img = mpimg.imread(filename)
            plt.imshow(img)
            plt.axis("off")
            if titulo:
                plt.title(titulo, fontsize=20, fontweight="bold")
            plt.show()

        except Exception as e:
            print(f"Error plotting graph: {e}")
            raise
        # temp_dot = "temp_graph.dot"
        # nx.drawing.nx_pydot.write_dot(G, temp_dot)
        #
        # graphviz_bin = r"C:\Program Files\Graphviz\bin"  # Ajuste o caminho aqui se precisar
        # cmd = f'"{os.path.join(graphviz_bin, "dot")}" -Tpng {temp_dot} -o {filename}'
        #
        # subprocess.run(cmd, shell=True, check=True)
        #
        # plt.figure(figsize=figsize)
        # img = mpimg.imread(filename)
        # plt.imshow(img)
        # plt.axis('off')
        #
        # if titulo:
        #     plt.title(titulo, fontsize=20, fontweight='bold')
        #
        # plt.show()

        # A = to_agraph(G)
        #
        # # Ajuste do 'len' para refletir peso nas arestas (opcional)
        # scale = 0.1  # fator de escala caso os pesos sejam grandes
        # for u, v in G.edges():
        #     w = G[u][v].get("weight", 1.0)
        #     edge = A.get_edge(u, v)
        #     edge.attr["len"] = str(w * scale)
        #
        # #####################################################################
        # # Reduzindo o tamanho da imagem final (em polegadas)
        # # Exemplo: "size" = "15,15" indica 15"x15" para o layout.
        # # "ratio" ajuda a manter a proporção / evitar distorção
        # A.graph_attr["size"] = "15,15"
        # A.graph_attr["ratio"] = "fill"
        # #####################################################################
        #
        # # Em vez de 'dot', use 'neato' (tenta respeitar distâncias)
        # A.layout(prog="dot")
        # A.draw(filename)
        #
        # # Carrega e exibe o PNG
        # plt.figure(figsize=figsize)
        # img = mpimg.imread(filename)
        # plt.imshow(img)
        # plt.axis("off")
        #
        # if titulo:
        #     plt.title(titulo, fontsize=20, fontweight="bold")
        #
        # plt.show()

    @staticmethod
    def plot_robot_graph(G_robot):
        """
        Plota o grafo G_robot usando as posições (x,y) guardadas em G_robot.nodes[node]["pos"].
        Aplica fatores de escala em x e y.
        Se algum nó não tiver 'pos', posiciona em (0,0) apenas para não quebrar.
        """

        # 1) Tentar Kamada-Kawai layout (bom para tentar preservar distâncias)
        pos = nx.kamada_kawai_layout(G_robot, weight='weight')

        # Se preferir Spring layout, também pode fazer:
        # pos = nx.spring_layout(G_robot, weight='weight', k=0.15, iterations=100)
        # Ajuste 'k' (comprimento ideal das arestas) e 'iterations' conforme precisar.

        # 2) Desenha o grafo
        plt.figure()
        nx.draw(
            G_robot,
            pos=pos,
            with_labels=True,
            node_color="yellow",
            edge_color="blue"
        )


        plt.title("Robot Graph - Distance-based Layout (Kamada-Kawai)")
        plt.axis("equal")  # para evitar distorção dos eixos
        plt.show()

    @staticmethod
    def plot_subgraphs(G, scale_x=1.0, scale_y=1.0):
        import matplotlib.pyplot as plt
        import networkx as nx

        # Para cada conjunto de nós que formam um componente conexo, criamos um subgrafo
        subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        colors = plt.cm.rainbow(range(len(subgraphs)))
        if len(subgraphs) == 1:
            colors = [plt.cm.rainbow(0.20)]

        plt.figure()

        for subgraph, color in zip(subgraphs, colors):
            # Montamos o dicionário 'pos' a partir das coordenadas de cada nó
            pos = {}
            for node in subgraph.nodes():
                # Se o nó tiver o atributo 'pos' (x,y)
                if "pos" in subgraph.nodes[node]:
                    x, y = subgraph.nodes[node]["pos"]
                    pos[node] = (x * scale_x, y * scale_y)
                else:
                    # Se não tiver, podemos colocar (0,0) ou pular
                    # Aqui, só vamos jogar (0,0) para não quebrar o draw.
                    pos[node] = (0, 0)

            # Se cada nó também tiver atributo 'label', podemos usar esse dicionário para desenhar
            labels = nx.get_node_attributes(subgraph, 'label')

            # Desenha o subgrafo com as posições calculadas
            nx.draw(
                subgraph,
                pos,
                node_color=[color],
                edge_color=color,
                with_labels=False
            )

            # Desenha o texto de cada nó (usando labels)
            nx.draw_networkx_labels(
                subgraph, pos,
                labels=labels,
                font_size=6,
                font_color='black'
            )

        plt.axis("equal")
        plt.show()

    @staticmethod
    def plot_subgraphs_old(G, scale_x=1.0, scale_y=1.0):
        import matplotlib.pyplot as plt
        import networkx as nx

        subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        colors = plt.cm.rainbow(range(len(subgraphs)))
        if len(subgraphs) == 1:
            colors = [plt.cm.rainbow(0.20)]

        plt.figure()

        for subgraph, color in zip(subgraphs, colors):
            # Aplica a escala aos nós
            pos = {node: (node[0] * scale_x, node[1] * scale_y) for node in subgraph.nodes()}
            labels = nx.get_node_attributes(subgraph, 'label')

            nx.draw(subgraph, pos, node_color=[color], edge_color=[color], with_labels=False)
            nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=6, font_color='black')

        plt.axis("equal")
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
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_title(f"AABBs Colored by Cluster (total = {num_clusters})")
        plt.show()

    @staticmethod
    def plot_segments_aabbs_vertices(segments, aabbs, raio=0.5):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.grid(True, linestyle='--', color='lightgray', alpha=0.7)

        # Plotar AABBs preenchidas em vermelho claro
        for (aabb_x, aabb_y), aabb_w, aabb_h in aabbs:
            rect = plt.Rectangle(
                (aabb_x, aabb_y), aabb_w, aabb_h,
                facecolor='lightcoral', edgecolor='red', alpha=0.4
            )
            ax.add_patch(rect)

        # Plotar segmentos e vértices
        for (x1, y1, x2, y2) in segments:
            # Linha verde para o segmento
            ax.plot([x1, x2], [y1, y2], color='green', linewidth=1.5)
            # Círculos vermelhos nas extremidades
            c1 = plt.Circle((x1, y1), raio, color='red', fill=True)
            c2 = plt.Circle((x2, y2), raio, color='red', fill=True)
            ax.add_patch(c1)
            ax.add_patch(c2)

        ax.set_aspect('equal', 'box')
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_title("AABBs (light red), Segments (green), Vertices (red)")
        plt.show()

    @staticmethod
    def plot_obstacles_aabbs(obstacles, aabbs):
        fig, ax = plt.subplots()  # Ajuste do tamanho do gráfico
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

        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_title("Obstacles and AABBs")
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
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_title(f"Segments in green + red vertices (radius={raio})")
        plt.show()