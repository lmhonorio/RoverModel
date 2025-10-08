import pygame
import math
import numpy as np

# Classes do rover
from roverclass import MotorModel, SkidSteerRoverModel, RoverController, ObstacleLoader
# Manipulação de grafo
from segmentutils import SegmentUtils
from aabbutils import AABBUtils
from multigraphplanner import MultiGraphPlanner

##################################################
# 1) CARREGA O GRAFO
##################################################
file_path_grafo = "./jsons/graph6.json"  # Ajuste se necessário
graph_nx = SegmentUtils.load_graph_json(file_path_grafo)
grafo_mapa = AABBUtils.convert_graph_to_dict(graph_nx)

# Definindo pontos de partida e destino (labels)
robots_positions = {
    "R1": "b_busip4_3",
    "R2": "ls_pr1_1"
}
destinations = {
    "R1": "ef_pr11_4",
    "R2": "b_busip40_6"
}

planner = MultiGraphPlanner(grafo_mapa, None, None, None, None, None)

def get_xy_from_label(nx_graph, label):
    """
    Retorna (x, y) a partir de um label.
    Se não encontrar, retorna (0,0) e imprime erro.
    """
    for node, data in nx_graph.nodes(data=True):
        if data.get("label", "") == label:
            return node  # 'node' aqui já deve ser (x, y)
    print(f"[ERRO] Label '{label}' não encontrado no grafo.")
    return (0.0, 0.0)

def get_path_from_label(nx_graph, path_labels):
    path_coord = []
    for label in path_labels:
        for node, data in nx_graph.nodes(data=True):
            if data.get("label", "") == label:
                path_coord.append(node)
    return path_coord



##################################################
# 2) PLANEJAR CAMINHOS (LABELS -> (x, y))
##################################################
# A* retorna lista de labels
path_r1_labels = planner.a_star(robots_positions["R1"], destinations["R1"])
path_r2_labels = planner.a_star(robots_positions["R2"], destinations["R2"])

print(f"[DEBUG] R1 path (labels): {path_r1_labels}")
print(f"[DEBUG] R2 path (labels): {path_r2_labels}")

# Converter cada label para coordenadas
path_r1_coords = get_path_from_label(graph_nx, path_r1_labels[0])
path_r2_coords = get_path_from_label(graph_nx, path_r2_labels[0])


print(f"[DEBUG] R1 path (coords): {path_r1_coords}")
print(f"[DEBUG] R2 path (coords): {path_r2_coords}")

##################################################
# 3) INICIALIZA PYGAME E A JANELA
##################################################
pygame.init()
WIDTH_METERS, HEIGHT_METERS = 400, 400
PIXELS_PER_METER = 2
WIDTH, HEIGHT = int(WIDTH_METERS * PIXELS_PER_METER), int(HEIGHT_METERS * PIXELS_PER_METER)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rovers - Caminho em Amarelo")

clock = pygame.time.Clock()
dt = 0.1

def meters_to_pixels(x, y):
    """
    Converte (x, y) do mundo para a tela.
    0,0 do mundo é o centro da tela.
    """
    return (
        int(WIDTH // 2 + x * PIXELS_PER_METER),
        int(HEIGHT // 2 - y * PIXELS_PER_METER)
    )

##################################################
# 4) CARREGA OBSTÁCULOS DO EXCEL
##################################################
file_path_obst = "./planilhas/obstaculos_processado6.xlsx"
sheet_name = "Parnaiba3_Transformado"
obstacle_loader = ObstacleLoader(file_path_obst, sheet_name)
obstacles = obstacle_loader.get_obstacles()

def draw_obstacles():
    """ Desenha cada obstáculo como um retângulo. """
    for obs in obstacles:
        (ox, oy) = obs["pos"]
        (w, h) = obs["size"]
        color = obs["color"]
        px, py = meters_to_pixels(ox, oy)
        rw = int(w * PIXELS_PER_METER)
        rh = int(h * PIXELS_PER_METER)
        rect = pygame.Rect(px - rw // 2, py - rh // 2, rw, rh)
        pygame.draw.rect(screen, color, rect)

##################################################
# 5) FUNÇÃO PARA DESENHAR O CAMINHO (AMARELO)
##################################################
def draw_planned_path(path_coords, color=(255, 255, 0)):
    """
    Desenha linhas ligando cada par de waypoints do caminho 'path_coords' em 'color'.
    Ex: (255,255,0) => amarelo
    """
    if len(path_coords) < 2:
        return
    for i in range(len(path_coords) - 1):
        (xA, yA) = path_coords[i]
        (xB, yB) = path_coords[i+1]
        pA = meters_to_pixels(xA, yA)
        pB = meters_to_pixels(xB, yB)
        pygame.draw.line(screen, color, pA, pB, 2)

##################################################
# 6) CRIAR ROVERS (MODELO + CONTROLE)
##################################################
def criar_rover():
    motor_FL = MotorModel(80, 1.0, 0.2, 0.08, 0.05, 0.01, 0.01, -100, 100, 50, 0.05)
    motor_FR = MotorModel(80, 1.0, 0.2, 0.08, 0.05, 0.01, 0.01, -100, 100, 50, 0.05)
    motor_RL = MotorModel(80, 1.0, 0.2, 0.08, 0.05, 0.01, 0.01, -100, 100, 50, 0.05)
    motor_RR = MotorModel(80, 1.0, 0.2, 0.08, 0.05, 0.01, 0.01, -100, 100, 50, 0.05)

    rover_model = SkidSteerRoverModel(
        m=10, I=3, L=0.5, r=0.5, b=0.3,
        C_d=0.01, C_r=0.01,
        motor_FL=motor_FL,
        motor_FR=motor_FR,
        motor_RL=motor_RL,
        motor_RR=motor_RR
    )
    controller = RoverController(
        kp_heading=2.0, kd_heading=0.0,
        kp_speed=1.0,   kd_speed=0.0,
        pwm_max=100,
        wheelbase=0.5
    )
    return rover_model, controller

rover1_model, rover1_ctrl = criar_rover()
rover2_model, rover2_ctrl = criar_rover()

# Estado inicial de cada rover: Se o path estiver vazio, define (0,0)
if path_r1_coords:
    x1i, y1i = path_r1_coords[0]
else:
    x1i, y1i = 0.0, 0.0

if path_r2_coords:
    x2i, y2i = path_r2_coords[0]
else:
    x2i, y2i = 0.0, 0.0

rover1_state = np.array([x1i, y1i, 0.0, 0.0, 0.0], dtype=float)
rover2_state = np.array([x2i, y2i, 0.0, 0.0, 0.0], dtype=float)

idx1 = 0
idx2 = 0

trajectory_r1 = []
trajectory_r2 = []

##################################################
# 7) FUNÇÃO DE CONTROLE
##################################################
def compute_velocity_and_omega(rover_state, target_waypoint, ctrl):
    x, y, theta, v, omega = rover_state
    dx = target_waypoint[0] - x
    dy = target_waypoint[1] - y

    desired_heading = math.atan2(dy, dx)
    heading_error = desired_heading - theta
    # Normaliza erro para -pi..pi
    heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

    desired_speed = 1.0  # Exemplo
    speed_error = desired_speed - v

    # Controlador P/PD
    u_heading, u_speed = ctrl.compute_control(heading_error, 0.0, speed_error, 0.0)
    return (u_speed, u_heading)

##################################################
# 8) LOOP PRINCIPAL (SIMULAÇÃO)
##################################################
running = True
battery_voltage = 48.0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fundo branco
    screen.fill((255, 255, 255))

    # Desenha obstáculos
    draw_obstacles()

    # Desenha caminhos planejados em amarelo
    draw_planned_path(path_r1_coords, color=(255, 255, 0))
    draw_planned_path(path_r2_coords, color=(255, 255, 0))

    # 8.1) Rover 1
    if idx1 < len(path_r1_coords):
        waypoint1 = path_r1_coords[idx1]
        dist1 = math.hypot(waypoint1[0] - rover1_state[0],
                           waypoint1[1] - rover1_state[1])
        if dist1 < 0.4:
            idx1 += 1
        else:
            v_des, w_des = compute_velocity_and_omega(rover1_state, waypoint1, rover1_ctrl)
            pwm_left, pwm_right = rover1_ctrl.pwm_allocator(v_des, w_des, battery_voltage)
            # print(f"[DEBUG] Rover1 PWM: L={pwm_left:.2f}, R={pwm_right:.2f}")
            pwm_inputs = np.array([pwm_left, pwm_right, pwm_left, pwm_right])
            rover1_state = rover1_model.dynamics(rover1_state, pwm_inputs, dt, battery_voltage)

    # 8.2) Rover 2
    if idx2 < len(path_r2_coords):
        waypoint2 = path_r2_coords[idx2]
        dist2 = math.hypot(waypoint2[0] - rover2_state[0],
                           waypoint2[1] - rover2_state[1])
        if dist2 < 0.4:
            idx2 += 1
        else:
            v_des, w_des = compute_velocity_and_omega(rover2_state, waypoint2, rover2_ctrl)
            pwm_left, pwm_right = rover2_ctrl.pwm_allocator(v_des, w_des, battery_voltage)
            # print(f"[DEBUG] Rover2 PWM: L={pwm_left:.2f}, R={pwm_right:.2f}")
            pwm_inputs = np.array([pwm_left, pwm_right, pwm_left, pwm_right])
            rover2_state = rover2_model.dynamics(rover2_state, pwm_inputs, dt, battery_voltage)

    # Desenha trajetórias em azul
    x1p, y1p = meters_to_pixels(rover1_state[0], rover1_state[1])
    x2p, y2p = meters_to_pixels(rover2_state[0], rover2_state[1])
    trajectory_r1.append((x1p, y1p))
    trajectory_r2.append((x2p, y2p))

    for px, py in trajectory_r1:
        pygame.draw.circle(screen, (0, 0, 255), (px, py), 2)
    for px, py in trajectory_r2:
        pygame.draw.circle(screen, (0, 0, 255), (px, py), 2)

    # Desenha os rovers em azul
    pygame.draw.circle(screen, (0, 0, 255), (x1p, y1p), 6)
    pygame.draw.circle(screen, (0, 0, 255), (x2p, y2p), 6)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
