from manim import *
from aabbutils import AABBUtils


# ---------- CONFIGURAÇÕES ------------------------------------
# Obstáculos de exemplo (pos em m, size em m -> largura, altura)
OBSTACLES = [
    {"pos": (-2,  1), "size": (1.2, 0.8), "label": "Obs 1"},
    {"pos": ( 1, -1), "size": (1.0, 1.4), "label": "Obs 2"},
    {"pos": ( 3,  2), "size": (0.6, 0.6), "label": "Obs 3"},
]
MARGIN = 0.5  # margem extra (m)
# --------------------------------------------------------------


class ShowAABBGeneration(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        # Eixos de referência
        axes = Axes(
            x_range=(-5, 5, 1), y_range=(-3, 3, 1),
            x_length=10, y_length=6,
            axis_config={"stroke_opacity": 0.3, "stroke_color": GREY_C}
        ).add_coordinates()
        self.add(axes)

        # Cria grupo de rótulos para manter acima das caixas
        labels_group = VGroup()

        # Desenha obstáculos originais
        obstacle_mobjects = []
        for obs in OBSTACLES:
            rect = self._rect_from_obs(obs, color=RED_E, fill_opacity=0.6)
            obstacle_mobjects.append(rect)

            label = Text(obs["label"], font_size=26, color=WHITE)\
                        .next_to(rect, UP)
            labels_group.add(label)

        self.play(
            *[FadeIn(m) for m in obstacle_mobjects],
            FadeIn(labels_group),
            run_time=1.5
        )
        self._show_stage("Obstáculo original")

        # “Pulsa” o retângulo para chamar atenção
        self.play(
            *[rect.animate.scale(1.1) for rect in obstacle_mobjects],
            run_time=0.8
        )
        self.play(
            *[rect.animate.scale(1 / 1.1) for rect in obstacle_mobjects],
            run_time=0.4
        )

        # Calcula AABBs com a função do utilizador
        aabbs = AABBUtils.get_aabbs(OBSTACLES, margin=MARGIN)

        # Desenha gradualmente as AABBs
        aabb_mobjects = []
        for aabb in aabbs:
            (ax, ay), aw, ah = aabb
            aabb_rect = Rectangle(
                width=aw, height=ah,
                color=BLUE_C, stroke_width=3
            ).move_to(axes.c2p(ax + aw / 2, ay + ah / 2))
            aabb_rect.set_fill(BLUE_C, opacity=0.1)
            aabb_mobjects.append(aabb_rect)

        self._show_stage("Aplicando margem")
        self.play(
            *[TransformFromCopy(o, a) for o, a in zip(obstacle_mobjects, aabb_mobjects)],
            run_time=1.8
        )

        # Destaque final
        self._show_stage("AABB final")
        self.play(
            *[a.animate.set_stroke_width(6) for a in aabb_mobjects],
            run_time=1
        )
        self.wait(2)

    # ---------- FUNÇÕES AUXILIARES -----------------------------
    def _rect_from_obs(self, obs, color, fill_opacity=1):
        x, y = obs["pos"]
        w, h = obs["size"]
        rect = Rectangle(
            width=w, height=h,
            color=color, stroke_width=3
        ).move_to([x, y, 0])
        rect.set_fill(color, opacity=fill_opacity)
        return rect

    def _show_stage(self, text):
        """Atualiza/mostra legenda de etapa no canto."""
        label = Text(text, font_size=28, color=YELLOW).to_corner(UL)
        if hasattr(self, "_stage_label"):
            self.play(ReplacementTransform(self._stage_label, label), run_time=0.6)
        else:
            self.play(FadeIn(label), run_time=0.6)
        self._stage_label = label
