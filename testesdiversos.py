import networkx as nx
import heapq
from itertools import permutations

class MultiGraphPlanner:
    def __init__(self, positions_graph, mission_graph, robots_graphs, mission_positions, mission_times):
        """
        Inicializa o planejador de múltiplos grafos.
        :param positions_graph: Grafo das posições (G_p)
        :param mission_graph: Grafo das missões (G_m)
        :param robots_graphs: Dicionário de grafos individuais dos robôs {R1: G_r1, R2: G_r2}
        :param mission_positions: Dicionário ligando missões a posições {"Missão A": "P3", ...}
        :param mission_times: Dicionário com tempos de execução de cada missão {"Missão A": 5, "Missão B": 10, ...}
        """
        self.G_p = positions_graph
        self.G_m = mission_graph
        self.G_r = robots_graphs
        self.mission_positions = mission_positions
        self.mission_times = mission_times  # Tempo necessário para executar cada missão

    def heuristic(self, a, b):
        """ Heurística baseada no tempo mínimo necessário para ir de a -> b """
        try:
            return nx.shortest_path_length(self.G_p, source=a, target=b, weight="weight")
        except nx.NetworkXNoPath:
            return float("inf")

    def a_star(self, start, goal):
        """ Algoritmo A* para encontrar o melhor caminho minimizando o tempo total """
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {node: float("inf") for node in self.G_p.nodes}
        g_score[start] = 0

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path, g_score[goal]  # Retorna caminho e tempo total

            for neighbor in self.G_p.neighbors(current):
                weight = self.G_p[current][neighbor].get("weight", 1)  # Tempo de deslocamento
                tentative_g_score = g_score[current] + weight

                if tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    came_from[neighbor] = current

        return None, float("inf")  # Nenhum caminho encontrado

    def get_mission_execution_order(self):
        """ Retorna a ordem correta de execução das missões considerando dependências """
        return list(nx.topological_sort(self.G_m))

    def find_minimum_mission_time_plan(self, robots_positions):
        """ Encontra a melhor alocação de missões minimizando o tempo total de execução """
        best_plan = None
        best_total_time = float("inf")

        mission_order = self.get_mission_execution_order()  # Respeita dependências
        mission_positions_sorted = [(m, self.mission_positions[m]) for m in mission_order]

        for perm in permutations(mission_positions_sorted, len(mission_positions_sorted)):
            robot_time = {robot: 0 for robot in robots_positions.keys()}  # Tempo acumulado de cada robô
            plan = {robot: [] for robot in robots_positions.keys()}
            assigned_missions = set()
            temp_robot_positions = robots_positions.copy()  # Mantém a posição correta de cada robô

            for mission, position in perm:
                best_robot = None
                best_path = None
                best_time_cost = float("inf")

                for robot, start_position in temp_robot_positions.items():
                    if mission in assigned_missions:
                        continue

                    path, travel_time = self.a_star(start_position, position)
                    execution_time = self.mission_times[mission]
                    total_time = robot_time[robot] + travel_time + execution_time

                    if path and total_time < best_time_cost:
                        best_robot = robot
                        best_path = path
                        best_time_cost = total_time

                if best_robot:
                    plan[best_robot].append({
                        "mission": mission,
                        "path": best_path,
                        "travel_time": travel_time,
                        "execution_time": self.mission_times[mission]
                    })
                    temp_robot_positions[best_robot] = position  # Atualiza posição do robô
                    robot_time[best_robot] = best_time_cost  # Atualiza tempo total do robô
                    assigned_missions.add(mission)

            max_robot_time = max(robot_time.values())  # O tempo total é o tempo do robô que termina por último
            if len(assigned_missions) == len(self.mission_positions) and max_robot_time < best_total_time:
                best_total_time = max_robot_time
                best_plan = plan

        return best_plan, best_total_time

    def execute_plan(self, planned_paths, total_time):
        """ Simula a execução dos planos dos robôs """
        print(f"🔹 Tempo mínimo total para concluir todas as missões: {total_time}")
        for robot, missions in planned_paths.items():
            for mission_data in missions:
                print(f"🚀 {robot} executará {mission_data['mission']} em {mission_data['path'][-1]} "
                      f"(Deslocamento: {mission_data['travel_time']}s, Execução: {mission_data['execution_time']}s) "
                      f"seguindo o caminho: {mission_data['path']}")

# 📍 Criando os Grafos

G_p = nx.Graph()  # Grafo das posições
G_m = nx.DiGraph()  # Grafo de missões
G_r = {"R1": nx.Graph(), "R2": nx.Graph()}  # Grafos individuais dos robôs

# 📌 Adicionando nós ao grafo das posições
positions = {
    "P1": (0, 0), "P2": (1, 0), "P3": (2, 0),
    "P4": (0, 1), "P5": (1, 1), "P6": (2, 1)
}

for pos, coords in positions.items():
    G_p.add_node(pos, pos=coords)

# 📌 Adicionando arestas ao grafo de posições com tempos de deslocamento
edges = [
    ("P1", "P2", 4), ("P2", "P3", 2),
    ("P1", "P4", 3), ("P2", "P5", 6), ("P3", "P6", 3),
    ("P4", "P5", 2), ("P5", "P6", 4)
]

for u, v, w in edges:
    G_p.add_edge(u, v, weight=w)

# 📌 Criando o grafo de missões (dependências)
G_m.add_edges_from([("Missão A", "Missão B"), ("Missão B", "Missão C")])

# 📌 Ligando missões às posições no mapa
mission_positions = {
    "Missão A": "P3",
    "Missão B": "P6",
    "Missão C": "P5"
}

# 📌 Tempo de execução de cada missão (em segundos)
mission_times = {
    "Missão A": 5,
    "Missão B": 10,
    "Missão C": 7
}

# 📌 Definindo as posições iniciais dos robôs
robots_positions = {"R1": "P1", "R2": "P4"}

# 🔍 Criando o planejador
planner = MultiGraphPlanner(G_p, G_m, G_r, mission_positions, mission_times)

# 🚀 Encontrando o melhor plano
optimal_plan, min_time = planner.find_minimum_mission_time_plan(robots_positions)

# 📌 Executando o melhor plano encontrado
planner.execute_plan(optimal_plan, min_time)
