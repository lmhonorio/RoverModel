"""
Visualização Manim: pipeline do PlanejadorHeterogeneo.py
Autor: <seu‑nome>
Requer: manim>=0.18  •  networkx  •  numpy  •  (seus módulos)
"""

from manim import *
from manim import config
config.media_dir = "C://Users//lmhon//Documents" # fora da pasta sincronizada
config.disable_caching = True             # evita WinError 32

import numpy as np, networkx as nx
from segmentutils import SegmentUtils
from aabbutils import AABBUtils
from tspOptimization import FixedTaskPlanner
from old.multigraphplanner import MultiGraphPlanner

# ---------------------- ARQUIVOS DO SCRIPT ORIGINAL --------------------
G_FILE   = "./jsons/graph9_new.json"
OBS_FILE = "./jsons/obp_6.json"
MISSIONS = ['b_busip4', 'ef_reator1', 'ls_pr4']

# CARREGA O GRAFO COMPLETO QUE PlotUtils USA  ⬇⬇⬇
G_MAPA_FILE = "./jsons/graph_mapa_real.json"   # ajuste o caminho
G_mapa = SegmentUtils.load_graph_json(G_FILE)

# posições (x,y) fornecidas no .py
XY_R1 = (-165.9766, -77.6645)
XY_R2 = (  87.9766,  30.6645)

# ---------------------- CENA ------------------------------------------
class PipelinePlanejador(MovingCameraScene):
    COLORS_ROBOTS = {"R1": GREEN_B, "R2": ORANGE}

    def construct(self):
        self.camera.background_color = "#0d1117"
        axes = Axes(x_range=(-300, 120, 50), y_range=(-150, 80, 50),
                    x_length=12, y_length=8,
                    tips=False, axis_config={"stroke_opacity": 0.15,
                                             "stroke_color": GREY_C})
        axes.add_coordinates(font_size=18)
        self.add(axes)
        self.camera.frame.scale(1.6)

        def caption(txt):
            c = Text(txt, font_size=32, color=YELLOW).to_corner(UL)
            if hasattr(self, "_cap"):
                self.play(ReplacementTransform(self._cap, c), run_time=0.6)
            else:
                self.play(FadeIn(c), run_time=0.6)
            self._cap = c

        # 1) Carregamento do grafo --------------------------------------
        caption("1 / 6  Carregando grafo do ambiente")
        G_full = SegmentUtils.load_graph_json(G_FILE)
        nodes_full = nx.get_node_attributes(G_full, "pos")
        vg_nodes = VGroup(*[
            Dot(axes.c2p(x, y, 0), radius=0.02, color=GREY_A)
            for _, (x, y) in nodes_full.items()
        ])
        vg_edges = VGroup(*[
            Line(axes.c2p(nodes_full[u][0], nodes_full[u][1]),
                 axes.c2p(nodes_full[v][0], nodes_full[v][1]),
                 stroke_color=GREY_D, stroke_width=0.6)
            for u, v in G_full.edges()
        ])
        self.play(Create(vg_edges, run_time=2), FadeIn(vg_nodes), run_time=1.5)

        # 2) Nós mais próximos dos robôs --------------------------------
        caption("2 / 6  Localizando robôs no grafo")
        label_r1, *_ = MultiGraphPlanner.find_nearest_node(G_full, *XY_R1)
        label_r2, *_ = MultiGraphPlanner.find_nearest_node(G_full, *XY_R2)
        robots_pos = {"R1": label_r1, "R2": label_r2}

        dots_robots = VGroup(*[
            Dot(axes.c2p(*nodes_full[label]), radius=0.06,
                color=self.COLORS_ROBOTS[rb]).scale(1.2)
            for rb, label in robots_pos.items()
        ])
        self.play(FadeIn(dots_robots, scale=1.3))

        # 3) Grafo reduzido de inspeção ---------------------------------
        caption("3 / 6  Construindo grafo de inspeção")
        # Carrega pontos de observação e missões
        obs_por_obs = SegmentUtils.load_observation_points_from_json(OBS_FILE)
        mission_pos = MultiGraphPlanner.gerar_mission_positions_from_json(
            obs_por_obs, MISSIONS)
        point_mission_positions = {p: p for pts in mission_pos.values()
                                     for p in pts}

        Greduced = MultiGraphPlanner.build_inspection_graph(
            list(robots_pos.values()), point_mission_positions, G_full)

        # destaca apenas as arestas & nós do grafo reduzido
        nred = nx.get_node_attributes(Greduced, "pos")
        er_lines = VGroup(*[
            Line(axes.c2p(nred[u][0], nred[u][1]),
                 axes.c2p(nred[v][0], nred[v][1]),
                 color=BLUE_B, stroke_width=1.2)
            for u, v in Greduced.edges()
        ])
        nred_dots = VGroup(*[
            Dot(axes.c2p(x, y), radius=0.03, color=BLUE_B)
            for x, y in nred.values()
        ])
        self.play(FadeIn(er_lines), FadeIn(nred_dots))

        # 4) Clusterização balanceada -----------------------------------
        caption("4 / 6  Clusterização balanceada dos pontos")
        mission_exec = {p: ["R1", "R2"] for p in point_mission_positions}
        pontos_por_robo = FixedTaskPlanner.clusterizar_pontos_balanceado(
            Greduced, point_mission_positions, robots_pos, mission_exec)

        # marca cada ponto com cor do robô alocado
        dots_alloc = VGroup()
        for rb, pontos in pontos_por_robo.items():
            for p in pontos:
                x, y = nred[p]
                dots_alloc.add(Dot(axes.c2p(x, y), radius=0.05,
                                   color=self.COLORS_ROBOTS[rb]))
        self.play(FadeIn(dots_alloc, scale=1.2))

        # 5) Rota ótima (TSP‑nearest) -----------------------------------
        caption("5 / 6  TSP nearest‑neighbor por robô")
        rotas = {}
        for rb, pontos in pontos_por_robo.items():
            rotas[rb] = FixedTaskPlanner.tsp_nearest_neighbor(
                Greduced, robots_pos[rb], pontos)

        lines_paths = VGroup()
        for rb, rota in rotas.items():
            for a, b in zip(rota, rota[1:]):
                x1, y1 = nred[a]; x2, y2 = nred[b]
                lines_paths.add(Line(axes.c2p(x1, y1),
                                     axes.c2p(x2, y2),
                                     color=self.COLORS_ROBOTS[rb],
                                     stroke_width=3))
        self.play(Create(lines_paths, run_time=2))

        # 6) Resultado final --------------------------------------------
        caption("6 / 6  Rotas finais por robô")
        legend_items = VGroup(*[
            VGroup(Dot(radius=0.07, color=col), Text(rb, font_size=28, color=WHITE)
                   .next_to(Dot(radius=0.07), RIGHT, buff=0.1))
                   .arrange(RIGHT)
            for rb, col in self.COLORS_ROBOTS.items()
        ]).arrange(DOWN).to_corner(UR)
        self.play(FadeIn(legend_items))

        # === NOVO BLOCO: percorre rotas reais no G_mapa ================
        caption("Rotas reais no mapa (pós‑TSP)")
        real_lines = VGroup()
        pos_mapa = nx.get_node_attributes(G_mapa, "pos")

        for rb, rota in rotas.items():              # rotas já calculadas (TSP)
            color = self.COLORS_ROBOTS[rb]
            for a, b in zip(rota, rota[1:]):
                path_nodes = nx.shortest_path(G_mapa, a, b, weight="weight")
                # converte cada par sucessivo da trajetória
                for u, v in zip(path_nodes, path_nodes[1:]):
                    x1, y1 = pos_mapa[u]; x2, y2 = pos_mapa[v]
                    real_lines.add(
                        Line(axes.c2p(x1, y1), axes.c2p(x2, y2),
                             color=color, stroke_opacity=0.4, stroke_width=5)
                    )

        self.play(Create(real_lines, run_time=3))
        # ===============================================================



        self.wait(2)