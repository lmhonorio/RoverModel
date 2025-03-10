import pygame
import math
import numpy as np
from roverclass import MotorModel, SkidSteerRoverModel, ObstacleLoader

# Inicialização do pygame
pygame.init()

# Configurações da tela
WIDTH_METERS, HEIGHT_METERS = 300, 300  # Dimensões do ambiente em metros
PIXELS_PER_METER = 2  # Escala de conversão
WIDTH, HEIGHT = int(WIDTH_METERS * PIXELS_PER_METER), int(HEIGHT_METERS * PIXELS_PER_METER)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulação do Rover Skid-Steering")

# Função para converter metros para pixels
def meters_to_pixels(x, y):
    return int(WIDTH // 2 + x * PIXELS_PER_METER), int(HEIGHT // 2 - y * PIXELS_PER_METER)

# Carregar e processar a imagem do rover
image = pygame.image.load("robo3.png")  # Substitua pelo caminho correto da imagem
rover_width, rover_height = int(image.get_width() * 0.1), int(image.get_height() * 0.1)
image = pygame.transform.scale(image, (rover_width, rover_height))  # Reduz imagem
image = pygame.transform.rotate(image, 90)  # Rotaciona 90° anti-horário para alinhar com o referencial correto
image.set_colorkey((255, 255, 255))  # Remove o fundo branco

# Dimensões do rover em metros
rover_size_meters = (0.6, 0.6)  # Largura e altura do rover em metros

# Parâmetros do rover (em metros)
m, I, L, r, b = 10, 3, 0.5, 0.5, 0.3
C_d, C_r = 0.01, 0.01
max_torque = 80
resistance, inductance, kt, ke, friction, inertia = 1.00, 0.2, 0.08, 0.05, 0.01, 0.01
pwm_min, pwm_max, current_limit, damping_factor = -100, 100, 50, 0.05

# Criando os motores
motor_FL = MotorModel(max_torque, resistance, inductance, kt, ke, friction, inertia, pwm_min, pwm_max, current_limit, damping_factor)
motor_FR = MotorModel(max_torque, resistance, inductance, kt, ke, friction, inertia, pwm_min, pwm_max, current_limit, damping_factor)
motor_RL = MotorModel(max_torque, resistance, inductance, kt, ke, friction, inertia, pwm_min, pwm_max, current_limit, damping_factor)
motor_RR = MotorModel(max_torque, resistance, inductance, kt, ke, friction, inertia, pwm_min, pwm_max, current_limit, damping_factor)

# Criando o modelo do rover
rover_model = SkidSteerRoverModel(m, I, L, r, b, C_d, C_r, motor_FL, motor_FR, motor_RL, motor_RR)

# Estado inicial do rover (em metros)
state = np.array([0, 0, 0, 0, 0])

# Lista de obstáculos [(x, y, largura, altura)] em metros
# obstacles = [
#     {"pos": (2.0, 0.0), "size": (0.5, 1.5), "color": (255, 0, 0)},  # Obstáculo 1
#     {"pos": (-3.0, 1.0), "size": (1.0, 0.5), "color": (255, 0, 0)},  # Obstáculo 2
#     {"pos": (1.0, -2.0), "size": (1.5, 0.5), "color": (255, 0, 0)},  # Obstáculo 3
#     {"pos": (-2.0, -1.5), "size": (0.5, 1.5), "color": (255, 0, 0)}   # Obstáculo 4
# ]

# Carregar obstáculos da planilha
file_path = "obstaculos_processado2.xlsx"
sheet_name = "Parnaiba3_Transformado"
obstacle_loader = ObstacleLoader(file_path, sheet_name)
obstacles = obstacle_loader.get_obstacles()

# Sequências de PWM para movimentação
battery_voltage = 48
pwm_sequences = [
    (np.array([100, 100, 100, 100]), 5.0),  # Move para frente
    (np.array([100, 80, 100, 80]), 2.0),  # Curva para um lado
    (np.array([50, 50, 50, 50]), 4.0),  # Move devagar
    (np.array([50, -50, 50, -50]), 6.0),  # Rotação no próprio eixo
    (np.array([100, 100, 100, 100]), 10.0)  # Rotação no próprio eixo
]

# Configuração do tempo de simulação
dt = 0.1
trajectory = []  # Armazena a trajetória
clock = pygame.time.Clock()

# Função para verificar colisões
def check_collision(x, y):
    for obs in obstacles:
        obs_x, obs_y = obs["pos"]
        obs_w, obs_h = obs["size"]

        # Verifica se há sobreposição entre o rover e o obstáculo
        if (abs(x - obs_x) < (rover_size_meters[0] / 2 + obs_w / 2)) and \
           (abs(y - obs_y) < (rover_size_meters[1] / 2 + obs_h / 2)):
            obs["color"] = (255, 255, 0)  # Amarelo se houver colisão
        else:
            obs["color"] = (255, 0, 0)  # Vermelho se não houver colisão

# Função para desenhar obstáculos
def draw_obstacles():
    for obs in obstacles:
        x, y = obs["pos"]
        w, h = obs["size"]
        color = obs["color"]
        x_px, y_px = meters_to_pixels(x, y)
        w_px, h_px = int(w * PIXELS_PER_METER), int(h * PIXELS_PER_METER)
        rect = pygame.Rect(x_px - w_px // 2, y_px - h_px // 2, w_px, h_px)
        pygame.draw.rect(screen, color, rect)

# Função para desenhar a grade
def draw_grid():
    grid_color = (200, 200, 200)
    for x in range(0, WIDTH, PIXELS_PER_METER):
        pygame.draw.line(screen, grid_color, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, PIXELS_PER_METER):
        pygame.draw.line(screen, grid_color, (0, y), (WIDTH, y))

# Loop de simulação baseado nas sequências PWM
for pwm_inputs, duration in pwm_sequences:
    num_steps = int(duration / dt)
    for _ in range(num_steps):
        screen.fill((255, 255, 255))  # Fundo branco
        draw_grid()  # Desenha a grade

        # Atualiza o estado do rover com base na dinâmica
        state = rover_model.dynamics(state, pwm_inputs, dt, battery_voltage)
        x, y, theta, _, _ = state
        x_px, y_px = meters_to_pixels(x, y)
        trajectory.append((x_px, y_px))

        # Verifica colisões e desenha obstáculos
        check_collision(x, y)
        draw_obstacles()

        # Desenha a trajetória do rover
        for point in trajectory:
            pygame.draw.circle(screen, (0, 0, 255), (int(point[0]), int(point[1])), 1)

        # Rotaciona e desenha a imagem do rover
        # rotated_image = pygame.transform.rotate(image, math.degrees(theta))
        # rect = rotated_image.get_rect(center=(x_px, y_px))
        # screen.blit(rotated_image, rect.topleft)

        pygame.display.flip()  # Atualiza a tela
        clock.tick(30)  # Mantém a simulação em 30 FPS

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                running = False

    # screen.fill((255, 255, 255))  # Fundo branco
    # pygame.display.flip()  # Atualiza a tela
    # clock.tick(30)  # Mantém a simulação em 30 FPS



pygame.quit()
