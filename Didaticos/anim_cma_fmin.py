"""
Animação: visão 2‑D do CMA‑ES (cma.fmin) usado para calibrar o rover
Autor: <seu‑nome>
⚠️  Requer: manim>=0.18  •  numpy  •  cma>=3.3
"""

from manim import *
import numpy as np
import cma                              # só para obter amostras & atualização


config.disable_caching = True

# -------------------- CONFIGURAÇÃO DO PROBLEMA --------------------------
DIM      = 11                      # problema real tem 11 parâmetros
IDX_X, IDX_Y = 0, 1                # projetamos p0 e p1 no plano
x0_full  = np.array([0.0150199, 0.34243617, 1.56524389, 1.35486333,
                     0.0142325, 0.60528597, 6.01851242, 3.27074781,
                     0.65726919, 5.63780158, 3.47293682])
sigma0   = 0.2
POP_SIZE = 50                      # menor que 100 para enxergar bem
N_ITERS  = 6                       # mostrar 6 gerações

# Toy‑cost para animação (esfera + ruído)  -------------------------------
def sphere_cost(x):
    return np.sum(x**2) + np.random.randn()*0.001


# -------------------- CENA MANIM ----------------------------------------
class CMAVisualization(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"

        axes = Axes(
            x_range=(-0.2, 1.2, 0.2), y_range=(-0.2, 1.2, 0.2),
            x_length=8, y_length=8,
            axis_config={"stroke_color": GREY_C, "stroke_opacity": 0.4}
        ).add_coordinates()
        axes.to_edge(DOWN)
        self.add(axes)

        # legenda dinâmica
        caption = Text("", font_size=34, color=YELLOW).to_edge(UP)
        def set_caption(txt): self.play(Transform(caption, Text(txt, font_size=34,
                                                                color=YELLOW).to_edge(UP)),
                                        run_time=0.4)
        self.add(caption)

        # inicializa CMA
        es = cma.CMAEvolutionStrategy(x0_full, sigma0,
                                      {'popsize': POP_SIZE, 'verb_disp': 0})

        all_dots = VGroup()
        ellipses = VGroup()

        # loop de gerações ------------------------------------------------
        for it in range(N_ITERS):
            set_caption(f"It. {it+1} – amostragem")
            samples = es.ask()
            costs   = [sphere_cost(s) for s in samples]   # ↙ troque p/ evaluate()

            # ↳ projeta no 2‑D
            xs = [(s[IDX_X], s[IDX_Y]) for s in samples]
            dots = VGroup(*[
                Dot(point=axes.c2p(x, y),
                    color=interpolate_color(RED_E, GREEN_B,
                                            alpha=np.clip((max(costs)-c)/(max(costs)-min(costs)+1e-9), 0, 1)),
                    radius=0.07)
                for (x, y), c in zip(xs, costs)
            ])
            self.play(LaggedStartMap(FadeIn, dots, lag_ratio=0.1), run_time=0.8)
            all_dots.add(dots)

            set_caption(f"It. {it+1} – atualizando distribuição")
            es.tell(samples, costs)
            mu = es.result.xbest      # melhor ponto até aqui
            C  = es.sm.C              # matriz de covariância atual
            # extrai submatriz 2x2 para desenhar elipse
            C2 = C[np.ix_([IDX_X, IDX_Y],[IDX_X, IDX_Y])]
            vals, vecs = np.linalg.eigh(C2)
            width, height = 2*np.sqrt(vals)               # 2σ
            angle = np.degrees(np.arctan2(vecs[1,1], vecs[0,1]))

            ell = Ellipse(width=width, height=height,
                          color=BLUE_B, stroke_opacity=0.6)\
                          .rotate(angle*DEGREES)\
                          .move_to(axes.c2p(mu[IDX_X], mu[IDX_Y]))
            ellipses.add(ell)
            self.play(Create(ell), run_time=0.8)

            self.wait(0.2)

        # destaque da solução
        best = es.result.xbest
        best_dot = Dot(axes.c2p(best[IDX_X], best[IDX_Y]), color=YELLOW, radius=0.09)
        set_caption("Parâmetro ótimo encontrado")
        self.play(FadeIn(best_dot, scale=1.5), run_time=0.8)

        self.wait(2)

