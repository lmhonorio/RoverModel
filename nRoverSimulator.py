import numpy as np
import math
import heapq
from segmentutils import SegmentUtils
from roverclass import MotorModel, SkidSteerRoverModel
# Roverclass.py deve conter as classes MotorModel e SkidSteerRoverModel
# definidas no anexo.






###############################################################################
# 1) Controlador e Alocador de PWM
###############################################################################





class RoverController:
    """
    Controlador de heading e velocidade (PI/PID simples)
    e alocação para PWM (modelo de equações simplificadas).
    """
    def __init__(self, kp_heading=2.0, kd_heading=0.0,
                 kp_speed=1.0, kd_speed=0.0,
                 pwm_max=100, wheelbase=0.5):
        self.kp_heading = kp_heading
        self.kd_heading = kd_heading
        self.kp_speed   = kp_speed
        self.kd_speed   = kd_speed
        self.pwm_max    = pwm_max
        self.L          = wheelbase  # Distância entre rodas

        # Estados para derivadas
        self.prev_heading_error = 0.0
        self.prev_speed_error   = 0.0

    def compute_control(self, heading_error, heading_error_dot,
                        speed_error, speed_error_dot):
        """
        Retorna (u_heading, u_speed) => sinal de controle para heading e speed.
        """
        # PID/PI simplificado, sem integral por exemplo.
        # heading
        ctrl_heading = self.kp_heading * heading_error + self.kd_heading * heading_error_dot
        # speed
        ctrl_speed   = self.kp_speed   * speed_error   + self.kd_speed   * speed_error_dot
        return ctrl_heading, ctrl_speed

    def pwm_allocator(self, v_des, w_des, battery_voltage):
        """
        Converte (v_des, w_des) -> (pwm_left, pwm_right).
        Modelo simples de skid-steer:
            v_des = (wL + wR)/2 * R
            w_des = (wR - wL)/L * R
        em que R = raio da roda => assumido 1 para simplificar
        e 'pwm' ~ w(angular)
        """
        # Para simplificar, assumimos:
        # w_left  = (v_des - w_des * L/2)/R
        # w_right = (v_des + w_des * L/2)/R
        # E pwm ~ w*(pwm_max / w_max). Faltam escalas exatas, mas é um ex.
        # Se a bateria estiver com 48 V, apenas normalizamos por pwm_max.

        R = 1.0  # Raio fictício
        w_left  = (v_des - w_des*(self.L/2)) / R
        w_right = (v_des + w_des*(self.L/2)) / R

        # Limit w_(left/right) para a pwm
        # Exemplo: pwm ~ w. Ajuste se quiser um mapeamento real.
        pwm_left  = np.clip(w_left,  -self.pwm_max, self.pwm_max)
        pwm_right = np.clip(w_right, -self.pwm_max, self.pwm_max)
        return pwm_left, pwm_right

###############################################################################
# 2) Função para encontrar segmento mais próximo
###############################################################################

def closest_segment(rover_x, rover_y, segments):
    """
    Dado (rover_x, rover_y) e lista de segmentos (x1,y1,x2,y2),
    retorna o índice do segmento mais próximo + a dist.
    """
    best_dist = float('inf')
    best_idx  = None
    for i, seg in enumerate(segments):
        x1, y1, x2, y2 = seg
        dist = point_to_line_segment_dist(rover_x, rover_y, x1, y1, x2, y2)
        if dist < best_dist:
            best_dist = dist
            best_idx  = i
    return best_idx, best_dist

def point_to_line_segment_dist(px, py, x1, y1, x2, y2):
    """
    Distância do ponto (px,py) ao segmento (x1,y1)-(x2,y2).
    """
    seg_len2 = (x2 - x1)**2 + (y2 - y1)**2
    if seg_len2 < 1e-12:
        return math.hypot(px - x1, py - y1) # degenerate
    t = ((px - x1)*(x2 - x1) + (py - y1)*(y2 - y1)) / seg_len2
    if t < 0:
        return math.hypot(px - x1, py - y1)
    elif t > 1:
        return math.hypot(px - x2, py - y2)
    projx = x1 + t*(x2 - x1)
    projy = y1 + t*(y2 - y1)
    return math.hypot(px - projx, py - projy)

###############################################################################
# 3) A* no grafo de segmentos
###############################################################################

def build_graph_from_segments(segments):
    """
    Constrói um grafo cujos nós são 'endpoints' dos segmentos,
    e as arestas são os próprios segmentos.
    Retorna:
      - nodes: lista de (x,y)
      - adj: dict, adj[i] => [(j, cost),...]
    """
    # 1) extrair endpoints
    endpoints = {}
    idx_counter = 0

    def add_point(pt):
        # pt = (x, y). checamos se existe ~
        # usaremos round ou isclose p/ agrupar?
        # Para simplicidade, guardamos exato e assumimos sem duplicatas.
        nonlocal idx_counter
        if pt not in endpoints:
            endpoints[pt] = idx_counter
            idx_counter += 1

    for (x1, y1, x2, y2) in segments:
        add_point((x1,y1))
        add_point((x2,y2))

    # Montar 'nodes'
    nodes = [None]*len(endpoints)
    for pt, idx in endpoints.items():
        nodes[idx] = pt

    # 2) construir adj
    adj = { i:[] for i in range(len(nodes))}

    for (x1, y1, x2, y2) in segments:
        i = endpoints[(x1,y1)]
        j = endpoints[(x2,y2)]
        cost = math.hypot(x2 - x1, y2 - y1)
        adj[i].append((j, cost))
        adj[j].append((i, cost))

    return nodes, adj

def find_closest_node(nodes, px, py):
    """
    Retorna idx do nó (endpoints) + dist
    que está mais próximo de (px, py).
    """
    best_dist = float('inf')
    best_idx  = None
    for i, (nx, ny) in enumerate(nodes):
        dist = math.hypot(px - nx, py - ny)
        if dist < best_dist:
            best_dist = dist
            best_idx  = i
    return best_idx, best_dist

def astar(nodes, adj, start_idx, goal_idx):
    """
    A* no grafo 'adj' para achar caminho entre start_idx e goal_idx.
    Retorna lista de idx se houver. Se não, []
    """
    # heuristica => dist eucl ao goal
    def heuristic(i):
        gx, gy = nodes[goal_idx]
        ix, iy = nodes[i]
        return math.hypot(gx - ix, gy - iy)

    open_set = []
    heapq.heappush(open_set, (0, start_idx))  # (f, node)
    came_from = { start_idx: None }
    gscore = { i: float('inf') for i in range(len(nodes)) }
    gscore[start_idx] = 0
    fscore = { i: float('inf') for i in range(len(nodes)) }
    fscore[start_idx] = heuristic(start_idx)

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal_idx:
            return reconstruct_path(came_from, current)

        for (nbr, cost) in adj[current]:
            tentative = gscore[current] + cost
            if tentative < gscore[nbr]:
                came_from[nbr] = current
                gscore[nbr]    = tentative
                fscore[nbr]    = tentative + heuristic(nbr)
                # push if not in open
                heapq.heappush(open_set, (fscore[nbr], nbr))

    return []

def reconstruct_path(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

###############################################################################
# 4) PROGRAMA PRINCIPAL: Exemplo
###############################################################################

def main():
    """
    Exemplo de como usar:
     1) Carregar rover (SkidSteerRoverModel)
     2) Definir control e alocador
     3) Achar e ir ao segmento mais próximo
     4) Construir grafo e rodar A*
     5) Seguir cada aresta do caminho
    """
    # Parâmetros do rover
    m=10; I=3; L=0.5; r=0.5; b=0.3; C_d=0.01; C_r=0.01; max_torque=80
    # Motores
    from roverclass import MotorModel
    resistance=1.00; inductance=0.2; kt=0.08; ke=0.05; friction=0.01; inertia=0.01
    pwm_min=-100; pwm_max=100; current_limit=50; damping_factor=0.05

    motor_FL = MotorModel(max_torque, resistance, inductance, kt, ke, friction, inertia,
                          pwm_min, pwm_max, current_limit, damping_factor)
    motor_FR = MotorModel(max_torque, resistance, inductance, kt, ke, friction, inertia,
                          pwm_min, pwm_max, current_limit, damping_factor)
    motor_RL = MotorModel(max_torque, resistance, inductance, kt, ke, friction, inertia,
                          pwm_min, pwm_max, current_limit, damping_factor)
    motor_RR = MotorModel(max_torque, resistance, inductance, kt, ke, friction, inertia,
                          pwm_min, pwm_max, current_limit, damping_factor)

    from roverclass import SkidSteerRoverModel
    rover = SkidSteerRoverModel(m,I,L,r,b,C_d,C_r, motor_FL,motor_FR,motor_RL,motor_RR)

    # Cria controlador
    ctrl = RoverController(kp_heading=2.0, kd_heading=0.1,
                           kp_speed=1.0, kd_speed=0.05,
                           pwm_max=100, wheelbase=L)

    G = SegmentUtils.load_graph("graph.pkl")
    segments = SegmentUtils.load_segments("segments.json")



    rover_state = np.array([segments[0][0],segments[0][1],0,0,0 ]) +  np.array([-2, -1, 0, 0, 0])  # x=2, y=-1, theta=0, v=0, omega=0
    dt=0.1
    battery_voltage=48

    # 1) Ir até o segmento mais próximo
    idx_seg, dist_seg = closest_segment(rover_state[0], rover_state[1], segments)
    print("Segmento mais próximo:", idx_seg, "dist=", dist_seg)
    # Implementar "ir até" => uma simulação simples de heading e speed control
    # Exemplo: viramos heading pro ponto mais próximo do segmento e andamos
    # (código abreviado)

    # 2) A* do endpoint do seg mais próximo ao endpoint do seg do goal
    # ex. goal: (4,4) => achamos qual seg e qual endpoint
    goal_x, goal_y = (segments[2000][0],segments[2000][1])
    # Montar grafo
    nodes, adj = build_graph_from_segments(segments)
    # idx do node mais próximo do rover (após chegar no seg)
    start_node, dist_s = find_closest_node(nodes, rover_state[0], rover_state[1])
    goal_node, dist_g  = find_closest_node(nodes, goal_x, goal_y)
    path = astar(nodes, adj, start_node, goal_node)
    if not path:
        print("Caminho não encontrado!")
        return
    print("Caminho A* (índices):", path)

    # 3) Seguir cada aresta do path
    # ex. i-> i+1
    # A cada iteração, calculamos heading_error, speed_error => ctrl => pwm
    # Atualiza rover_state via rover.dynamics
    # até chegar no final

    # Exemplo de loop abreviado:
    for i in range(len(path)-1):
        # extrair (xA,yA), (xB,yB)
        idxA = path[i]
        idxB = path[i+1]
        (xA,yA) = nodes[idxA]
        (xB,yB) = nodes[idxB]
        # Mover rover do A->B
        # defina speed_d = 1 m/s, heading_d = ...
        # ...
        # Simular
        done = False
        while not done:
            # 1) calc heading_d (direção do (xB,yB))
            dx = xB - rover_state[0]
            dy = yB - rover_state[1]
            distAB = math.hypot(dx, dy)
            if distAB<0.1:
                done=True
                break
            heading_d = math.atan2(dy, dx)
            heading_error = (heading_d - rover_state[2])
            # normalizar erro [-pi, pi] se quiser
            # speed_d = 1.0
            # speed_error = speed_d - rover_state[3]

            # deriv (simplificado) => zero
            heading_err_dot = 0
            speed_err_dot   = 0

            u_heading, u_speed = ctrl.compute_control(heading_error, heading_err_dot,
                                                      1.0 - rover_state[3], speed_err_dot)
            # converter => (v_des, w_des)
            # ex: v_des = rover_state[3] + u_speed, w_des = u_heading
            # simplificação
            v_des = 1.0 + u_speed
            w_des = u_heading
            pwm_left, pwm_right = ctrl.pwm_allocator(v_des, w_des, battery_voltage)
            # 2) 4 motores => ex: pwm_FL = pwm_left, ...
            pwm_inputs = np.array([pwm_left, pwm_right, pwm_left, pwm_right])
            # 3) update rover
            rover_state = rover.dynamics(rover_state, pwm_inputs, dt, battery_voltage)
        print(f"Completou aresta de {idxA} -> {idxB}")

    print("Chegou no goal ou parou a sim.")

if __name__ == "__main__":
    main()
