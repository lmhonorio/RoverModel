import networkx as nx
import heapq
from itertools import permutations
import matplotlib.pyplot as plt


class MultiGraphPlanner:
    """
    Classe responsável por planejar e otimizar a execução de missões por múltiplos robôs
    em um ambiente representado por um grafo, minimizando o tempo total de execução.

    Métodos:
        - heuristic(a, b): Calcula a heurística baseada no tempo mínimo entre dois nós.
        - a_star(start, goal): Algoritmo A* para encontrar o melhor caminho minimizando o tempo total.
        - get_mission_execution_order(): Retorna a ordem correta das missões considerando dependências.
        - find_minimum_mission_time_plan(robots_positions): Aloca missões minimizando o tempo total.
        - execute_plan(planned_paths, total_time, schedule): Exibe a execução planejada e gera um gráfico de Gantt.
        - plot_gantt(schedule): Gera um gráfico de Gantt representando a execução do plano.
    """

    def __init__(self, grafo_mapa, mission_graph, robots_graphs, mission_positions, mission_times, mission_execution):
        """
        Inicializa o planejador com os grafos e informações de missões e robôs.

        Parâmetros:
            - grafo_mapa: Dicionário contendo os estados e transições do ambiente.
            - mission_graph: Grafo direcionado representando dependências entre missões.
            - robots_graphs: Dicionário com os grafos de cada robô.
            - mission_positions: Dicionário ligando missões às posições no mapa.
            - mission_times: Dicionário contendo tempos de execução de cada missão.
            - mission_execution: Dicionário definindo quais robôs podem executar cada missão.
        """
        self.grafo_mapa = grafo_mapa
        self.G_m = mission_graph
        self.G_r = robots_graphs
        self.mission_positions = mission_positions
        self.mission_times = mission_times
        self.mission_execution = mission_execution

    def heuristic(self, a, b):
        """Calcula a heurística baseada no tempo mínimo necessário para ir de a -> b."""
        transition = self.grafo_mapa['transitions'].get((a, b), float("inf"))
        return transition[0] if isinstance(transition, tuple) else transition

    def a_star(self, start, goal):
        """Implementação do algoritmo A* para encontrar o melhor caminho minimizando o tempo total."""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {node: float("inf") for node in self.grafo_mapa['states']}
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
                return path, g_score[goal]

            for neighbor in self.grafo_mapa['states']:
                if (current, neighbor) in self.grafo_mapa['transitions']:
                    weight = self.grafo_mapa['transitions'][(current, neighbor)][0]
                    tentative_g_score = g_score[current] + weight

                    if tentative_g_score < g_score[neighbor]:
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self.heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, neighbor))
                        came_from[neighbor] = current

        return None, float("inf")  # Nenhum caminho encontrado

    def get_mission_execution_order(self):
        """Retorna a ordem correta de execução das missões considerando dependências."""
        return list(nx.topological_sort(self.G_m))

    def find_minimum_mission_time_plan(self, robots_positions):
        """Encontra a melhor alocação de missões minimizando o tempo total."""
        best_plan = None
        best_total_time = float("inf")
        best_schedule = None

        mission_order = self.get_mission_execution_order()
        mission_positions_sorted = [(m, self.mission_positions[m]) for m in mission_order]

        for perm in permutations(mission_positions_sorted, len(mission_positions_sorted)):
            robot_time = {robot: 0 for robot in robots_positions.keys()}
            plan = {robot: [] for robot in robots_positions.keys()}
            schedule = []
            assigned_missions = set()
            temp_robot_positions = robots_positions.copy()

            for mission, position in perm:
                best_robot = None
                best_path = None
                best_time_cost = float("inf")

                available_robots = self.mission_execution[mission]
                available_robots.sort(key=lambda r: robot_time[r])

                for robot in available_robots:
                    start_position = temp_robot_positions[robot]
                    if mission in assigned_missions:
                        continue

                    path, travel_time = self.a_star(start_position, position)
                    execution_time = self.mission_times[mission]
                    start_time = robot_time[robot] + travel_time
                    end_time = start_time + execution_time
                    total_time = end_time

                    if path and total_time < best_time_cost:
                        best_robot = robot
                        best_path = path
                        best_time_cost = total_time

                if best_robot:
                    plan[best_robot].append({
                        "mission": mission,
                        "path": best_path,
                        "travel_time": travel_time,
                        "execution_time": self.mission_times[mission],
                        "start_time": robot_time[best_robot] + travel_time,
                        "end_time": best_time_cost
                    })
                    schedule.append({
                        "robot": best_robot,
                        "mission": mission,
                        "start": robot_time[best_robot] + travel_time,
                        "end": best_time_cost
                    })
                    temp_robot_positions[best_robot] = position
                    robot_time[best_robot] = best_time_cost
                    assigned_missions.add(mission)

            max_robot_time = max(robot_time.values())
            if len(assigned_missions) == len(self.mission_positions) and max_robot_time < best_total_time:
                best_total_time = max_robot_time
                best_plan = plan
                best_schedule = schedule

        return best_plan, best_total_time, best_schedule

    def execute_plan(self, planned_paths, total_time, schedule):
        """ Simula a execução dos planos dos robôs e plota o gráfico de Gantt """
        print(f"🔹 Tempo mínimo total para concluir todas as missões: {total_time}")
        for robot, missions in planned_paths.items():
            for mission_data in missions:
                print(f"🚀 {robot} executará {mission_data['mission']} em {mission_data['path'][-1]} "
                      f"(Deslocamento: {mission_data['travel_time']}s, Execução: {mission_data['execution_time']}s) "
                      f"seguindo o caminho: {mission_data['path']}")

        self.plot_gantt(schedule)

    def plot_gantt(self, schedule):
        """ Plota um gráfico de Gantt do cronograma de execução """
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, task in enumerate(schedule):
            ax.barh(task["robot"], task["end"] - task["start"], left=task["start"], color='skyblue')
            ax.text(task["start"] + (task["end"] - task["start"]) / 2, i, task["mission"],
                    ha='center', va='center', color='black', fontsize=12, fontweight='bold')

        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Robôs")
        ax.set_title("Cronograma de Execução (Gráfico de Gantt)")
        plt.show()
