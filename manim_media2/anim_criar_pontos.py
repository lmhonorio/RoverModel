"""
Animação didática para ilustrar a rotina main() de CriarPontosObservacao.py
Autor: <seu‑nome>
Requer: Manim v0.18+, NetworkX, pygraphviz (+ dependências do seu projeto)
"""

from manim import *
from aabbutils import AABBUtils
from segmentutils import SegmentUtils
from roverclass import ObstacleLoader
import numpy as np

config.disable_caching = True
config.media_dir = "C://Users//lmhon//Documents"      # fora do Dropbox

# ---------- AJUSTE OS MESMOS CAMINHOS DO SEU SCRIPT ------------------
FILE_PATH   = "../planilhas/obstaculos_processado6.xlsx"
SHEET_NAME  = "Parnaiba3_Transformado"
MARGIN      = 2.5
SEG_STEP    = 4.0
OBS_THRESH  = 3.0
BREAK_DIST  = 2.0
# ---------------------------------------------------------------------


class RotinaCriarPontos(MovingCameraScene):
    """Cena única que faz zoom‑out progressivo mostrando cada artefato novo."""
    SCALE = 0.2                          # m → unidades de cena

    def construct(self):
        self.camera.background_color = "#0d1117"
        self.camera.frame.scale(6)       # zoom‑out inicial (ajuste a gosto)

        # Eixos “fantasma” só para ter c2p – NÃO adicionamos à cena
        axes = Axes(x_range=(-100, 100), y_range=(-100, 100))

        # -------- helpers internos -----------------------------------
        def p2m(xy):               # (x,y) world → manim
            x, y = xy
            return np.array([x * self.SCALE, y * self.SCALE, 0])

        def caption(txt):
            c = Text(txt, font_size=28, color=YELLOW).to_corner(UL)
            if hasattr(self, "_cap"):
                self.play(ReplacementTransform(self._cap, c), run_time=0.5)
            else:
                self.play(FadeIn(c), run_time=0.5)
            self._cap = c

        # -------- 1) Obstáculos --------------------------------------
        loader = ObstacleLoader(FILE_PATH, SHEET_NAME)
        obstacles = loader.get_obstacles()
        obs_mobs = VGroup(*[
            Rectangle(width=o["size"][0]*self.SCALE,
                      height=o["size"][1]*self.SCALE,
                      color=RED_E, fill_color=RED_E, fill_opacity=0.65
                     ).move_to(p2m(o["pos"]))
            for o in obstacles
        ])

        caption("1 / 6  Carregando obstáculos")
        self.play(LaggedStartMap(FadeIn, obs_mobs, lag_ratio=0.2),
                  run_time=1.5)

        # -------- 2) AABBs -------------------------------------------
        aabbs = AABBUtils.get_aabbs(obstacles, margin=MARGIN)
        aabb_mobs = VGroup(*[
            Rectangle(width=aw*self.SCALE, height=ah*self.SCALE,
                      color=BLUE_B, stroke_width=3, fill_color=BLUE_B,
                      fill_opacity=0.1
                     ).move_to(p2m((ax+aw/2, ay+ah/2)))
            for (ax, ay), aw, ah in aabbs
        ])

        caption("2 / 6  Gerando AABBs")
        self.play(LaggedStart(*[
            TransformFromCopy(o, a) for o, a in zip(obs_mobs, aabb_mobs)
        ], lag_ratio=0.15), run_time=1.8)

        # -------- 3) Segmentos entre AABBs ---------------------------
        segments = SegmentUtils.generate_segments_between_aabbs(
            aabbs, SEG_STEP)
        seg_mobs = VGroup(*[
            Line(p2m((x1, y1)), p2m((x2, y2)),
                 color=YELLOW_E, stroke_width=2)
            for x1, y1, x2, y2 in segments
        ])

        caption("3 / 6  Conectando AABBs")
        self.play(Create(seg_mobs, run_time=1.6))

        # -------- 4) Pontos + perímetro ------------------------------
        caption("4 / 6  Pontos de observação")
        segments, obs_points = SegmentUtils.generate_perimeter_segments_and_labeled_points(
            segments, aabbs, obstacles, threshold=OBS_THRESH)

        # pontos de observação – dots verdes pulsantes
        dots_obs = VGroup(*[
            Dot(p2m(p["pos"]), radius=0.07, color=GREEN_B)
            for p in obs_points
        ])
        self.play(FadeIn(dots_obs, scale=1.3), run_time=1.2)
        # pequeno “pulse” concentrado
        self.play(LaggedStartMap(
            lambda d: d.animate.scale(1.4).set_fill(opacity=0.4).set_stroke(width=0),
            dots_obs, lag_ratio=0.02), run_time=0.4)
        self.play(LaggedStartMap(
            lambda d: d.animate.scale(1/1.4).set_fill(opacity=1).set_stroke(width=0),
            dots_obs, lag_ratio=0.02), run_time=0.3)

        # -------- 5) Quebra de interseções ---------------------------
        caption("5 / 6  Quebrando interseções")
        broken, pass_points = SegmentUtils.resolve_segment_intersections(
            segments, BREAK_DIST)

        # substitui linhas por quebradas
        self.play(FadeOut(seg_mobs), run_time=0.3)
        seg_broken_mobs = VGroup(*[
            Line(p2m((x1, y1)), p2m((x2, y2)),
                 color=ORANGE, stroke_width=2)
            for x1, y1, x2, y2 in broken
        ])
        self.play(Create(seg_broken_mobs, run_time=1.4))

        # pontos de passagem – dots laranja com flash
        dots_pass = VGroup(*[
            Dot(p2m(pt), radius=0.06, color=ORANGE)
            for pt in pass_points
        ])
        self.play(FadeIn(dots_pass), run_time=0.8)
        # Flash highlight
        self.play(LaggedStartMap(
            lambda d: Flash(d.get_center(), color=ORANGE, time_width=0.4, run_time=0.6),
            dots_pass, lag_ratio=0.05))

        # -------- 6) Grafo final -------------------------------------
        caption("6 / 6  Construindo grafo final")
        G = SegmentUtils.create_graph_with_passage_points_new(
            broken, pass_points, obs_points, obstacles)
        if SegmentUtils.has_islands(G):
            G = SegmentUtils.fix_missing_connections_safe_new(G, aabbs)

        # desenha grafo (arestas finas roxas)
        e_lines = VGroup()
        for u, v in G.edges():
            x1, y1 = G.nodes[u]["pos"]
            x2, y2 = G.nodes[v]["pos"]
            e_lines.add(Line(p2m((x1, y1)), p2m((x2, y2)),
                             color=PURPLE_B, stroke_width=1.1))
        n_dots = VGroup(*[
            Dot(p2m(G.nodes[n]["pos"]), radius=0.045, color=PURPLE_B)
            for n in G.nodes()
        ])
        self.play(Create(e_lines, run_time=2), FadeIn(n_dots), run_time=1.5)

        caption("Rotina concluída ✔")
        self.wait(2)

