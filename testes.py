import pygame
import math
import numpy as np

from roverclass import MotorModel, SkidSteerRoverModel, RoverController, ObstacleLoader
from segmentutils import SegmentUtils
from aabbutils import AABBUtils
from multigraphplanner import MultiGraphPlanner

# 🚀 Carregar o Grafo JSON
file_path_grafo = "./jsons/graph9.json"
graph_nx = SegmentUtils.load_graph_json(file_path_grafo)
grafo_mapa = AABBUtils.convert_graph_to_dict(graph_nx)

robots_positions = {
    "R1": "PR11_0",
    "R2": "PR11_2"
}
destinations = {
    "R1": "PR13_7",
    "R2": "TPC2_4"
}

planner = MultiGraphPlanner(grafo_mapa, None, None, None, None, None)

def get_xy_from_label(nx_graph, label):
    """ Pega as coordenadas reais do label """
    for node, data in nx_graph.nodes(data=True):
        if data.get("label", "") == label:
            return node
    print(f"[ERRO] Não foi encontrado nenhum nó com label '{label}'.")
    return (0.0, 0.0)

pygame.init()

# 🚀 Configurar Tela Pygame
WIDTH_METERS, HEIGHT_METERS = 600, 600
PIXELS_PER_METER = 2
WIDTH, HEIGHT = int(WIDTH_METERS * PIXELS_PER_METER), int(HEIGHT_METERS * PIXELS_PER_METER)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulação Debug - graph9.json")

clock = pygame.time.Clock()
dt = 0.1

def meters_to_pixels(x, y):
    """ Converte coordenadas do mundo real para pixels na tela """
    return (
        int(WIDTH // 2 + (x * PIXELS_PER_METER)),
        int(HEIGHT // 2 - (y * PIXELS_PER_METER))
    )

# 🚀 Obter Posições dos Rovers
x1i, y1i = get_xy_from_label(graph_nx, robots_positions["R1"])
x2i, y2i = get_xy_from_label(graph_nx, robots_positions["R2"])

print(f"[DEBUG] Posição inicial R1: (x1i={x1i}, y1i={y1i})")
print(f"[DEBUG] Posição inicial R2: (x2i={x2i}, y2i={y2i})")

# 🚀 Converter para Pixels
x1p, y1p = meters_to_pixels(x1i, y1i)
x2p, y2p = meters_to_pixels(x2i, y2i)

print(f"[DEBUG] R1 será desenhado em: ({x1p}, {y1p}) pixels")
print(f"[DEBUG] R2 será desenhado em: ({x2p}, {y2p}) pixels")

def draw_rovers():
    """ Desenha os rovers na tela, garantindo que aparecem corretamente """
    if 0 <= x1p < WIDTH and 0 <= y1p < HEIGHT:
        pygame.draw.circle(screen, (0, 0, 255), (x1p, y1p), 8)  # Rover 1 azul
    else:
        print(f"[ERRO] R1 está fora da tela em ({x1p}, {y1p}) pixels!")

    if 0 <= x2p < WIDTH and 0 <= y2p < HEIGHT:
        pygame.draw.circle(screen, (0, 0, 255), (x2p, y2p), 8)  # Rover 2 azul também
    else:
        print(f"[ERRO] R2 está fora da tela em ({x2p}, {y2p}) pixels!")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))  # Fundo branco

    # 🔴 Desenha um ponto vermelho no centro da tela
    pygame.draw.circle(screen, (255, 0, 0), (WIDTH // 2, HEIGHT // 2), 10)

    # 🚀 Desenha os rovers em azul
    draw_rovers()

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
