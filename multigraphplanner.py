
import networkx as nx
import heapq
from itertools import permutations
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib.patches import Patch
import json
from concurrent.futures import ThreadPoolExecutor



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

    def find_minimum_mission_time_plan_par(self, robots_positions):
        """
        Versão paralelizada: permite deslocamentos em paralelo e avalia robôs concorrentes
        para cada missão usando ThreadPoolExecutor.
        """
        best_plan = {robot: [] for robot in robots_positions.keys()}
        schedule = []
        robot_time = {robot: 0.0 for robot in robots_positions.keys()}
        robot_position = robots_positions.copy()
        mission_finish_times = {}
        mission_order = self.get_mission_execution_order()

        for mission in mission_order:
            available_robots = self.mission_execution[mission]
            predecessor_end = 0  # Dependências desconsideradas

            def evaluate_robot(robot):
                earliest_start = max(robot_time[robot], predecessor_end)
                path, travel_time = self.a_star(robot_position[robot], self.mission_positions[mission][0])
                if path is None:
                    return None
                mission_start_time = earliest_start + travel_time
                return (mission_start_time, travel_time, path, robot)

            # 🔄 Avaliar robôs concorrentes em paralelo
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(evaluate_robot, available_robots))

            # 🏆 Selecionar melhor robô
            best_result = None
            for res in results:
                if res is None:
                    continue
                if best_result is None or res[0] < best_result[0]:
                    best_result = res

            if best_result:
                mission_start_time, travel_time, path, best_robot = best_result
                execution_time = self.mission_times[mission]
                end_time = mission_start_time + execution_time

                best_plan[best_robot].append({
                    "mission": mission,
                    "path": path,
                    "travel_time": travel_time,
                    "execution_time": execution_time,
                    "start_time": mission_start_time - travel_time,
                    "end_time": end_time
                })

                robot_position[best_robot] = self.mission_positions[mission][0]
                robot_time[best_robot] = end_time
                mission_finish_times[mission] = end_time

        total_time = max(mission_finish_times.values()) if mission_finish_times else 0
        return best_plan, total_time, schedule

    def find_minimum_mission_time_plan(self, robots_positions):
        """
        Versão otimizada que permite deslocamentos em paralelo com outras vistorias,
        mantendo apenas as restrições de precedência entre as próprias vistorias.
        """
        best_plan = {robot: [] for robot in robots_positions.keys()}
        schedule = []
        robot_time = {robot: 0.0 for robot in robots_positions.keys()}
        robot_position = robots_positions.copy()

        # Armazena o tempo de término de cada missão para controle de dependências
        mission_finish_times = {}

        # Ordem topológica das missões (considerando apenas dependências entre vistorias)
        mission_order = self.get_mission_execution_order()

        for mission in mission_order:
            best_robot = None
            best_path = None
            best_start_time = float("inf")
            best_travel_time = 0.0

            available_robots = self.mission_execution[mission]

            for robot in available_robots:
                # Verifica restrições de precedência (vistorias anteriores devem ter terminado)
                # predecessors = list(self.G_m.predecessors(mission))
                # predecessor_end = max([mission_finish_times[p] for p in predecessors], default=0)
                predecessor_end = 0  # Ignora dependências

                # Tempo mais cedo que o robô pode começar a se deslocar
                earliest_start = max(robot_time[robot], predecessor_end)

                # Calcula caminho e tempo de deslocamento
                path, travel_time = self.a_star(robot_position[robot], self.mission_positions[mission][0])

                if path is None:
                    continue

                # Tempo de início da vistoria (após deslocamento)
                mission_start_time = earliest_start + travel_time

                if mission_start_time < best_start_time:
                    best_robot = robot
                    best_path = path
                    best_start_time = mission_start_time
                    best_travel_time = travel_time

            if best_robot:
                execution_time = self.mission_times[mission]
                end_time = best_start_time + execution_time

                # Registra no plano
                best_plan[best_robot].append({
                    "mission": mission,
                    "path": best_path,
                    "travel_time": best_travel_time,
                    "execution_time": execution_time,
                    "start_time": best_start_time - best_travel_time,  # início do deslocamento
                    "end_time": end_time
                })

                # Atualiza estado do robô
                robot_position[best_robot] = self.mission_positions[mission][0]
                mission_finish_times[mission] = end_time

                # O robô pode começar novo deslocamento imediatamente após terminar a vistoria
                robot_time[best_robot] = end_time

        total_time = max(mission_finish_times.values()) if mission_finish_times else 0
        return best_plan, total_time, schedule

    def execute_plan(self, planned_paths, total_time, schedule):
        """
        Exibe as missões executadas por cada robô no formato:
        🚀 R1 executará M_B em TPC2_4 (Deslocamento: 16.6s, Execução: 3s) seguindo o caminho: [...]
        """
        print(f"🔹 Tempo mínimo total para concluir todas as missões: {total_time}")
        for robot, missions in planned_paths.items():
            for mission_data in missions:
                mission_name = mission_data["mission"]
                path = mission_data.get("path", [])
                travel_t = mission_data.get("travel_time", 0)
                exec_t = mission_data.get("execution_time", 0)
                final_node = path[-1] if path else "???"

                print(
                    f"🚀 {robot} executará {mission_name} em {final_node} "
                    f"(Deslocamento: {travel_t}s, Execução: {exec_t}s) "
                    f"seguindo o caminho: {path}"
                )

        self.plot_gantt(planned_paths)


    def plot_gantt(self, planned_paths):
        travel_color = "#7FB3D5"  # Azul
        exec_color = "#82E0AA"  # Verde
        wait_color = "#F7DC6F"  # Amarelo (se tiver espera)

        fig, ax = plt.subplots(figsize=(12, 6))
        # Mapear robôs em Y
        all_robots = sorted(planned_paths.keys())
        y_map = {r: i for i, r in enumerate(all_robots)}

        for robot in all_robots:
            y = y_map[robot]
            for task_data in planned_paths[robot]:
                mission = task_data["mission"]
                st = task_data.get("start_time", 0)
                travel_t = task_data.get("travel_time", 0)
                exec_t = task_data.get("execution_time", 0)
                arr = st + travel_t
                end = arr + exec_t

                # Deslocamento => [st, arr]
                ax.barh(y, travel_t, left=st, color=travel_color, edgecolor="black")
                ax.text((st + arr) / 2, y, "Desloc.", ha="center", va="center", fontsize=8)

                # Execução => [arr, end]
                ax.barh(y, exec_t, left=arr, color=exec_color, edgecolor="black")
                ax.text((arr + end) / 2, y, mission, ha="center", va="center", fontsize=8)

        # Ajustar Y
        ax.set_yticks([y_map[r] for r in all_robots])
        ax.set_yticklabels(all_robots)

        # Legendas
        patches = [
            Patch(facecolor=travel_color, label="Deslocamento"),
            Patch(facecolor=exec_color, label="Execução")
        ]
        ax.legend(handles=patches)

        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Robô")
        ax.set_title("Gantt - Deslocamento e Execução")
        plt.show()

    @staticmethod
    def gerar_mission_positions_from_json(observacao_por_obstaculo, obstaculos):
        """
        Gera um dicionário de posições de missões a partir dos dados JSON.
        Retorna um dicionário onde cada chave é o nome da missão e o valor é uma lista de posições.
        """
        mission_positions = {}
        for obs in obstaculos:
            pontos = observacao_por_obstaculo.get(obs, [])
            labels = [
                ponto.get("label")
                for ponto in pontos
                if ponto.get("label", "").startswith(obs + ".")
            ]
            if labels:  # Só adiciona se houver pontos de observação
                mission_positions[obs] = labels
        return mission_positions

    def convert_plan_to_dict(self, optimal_plan):
        """
        Converte o plano ótimo em um dicionário no formato:
        {
            "R1": [("ID1", "D"), ("ID2", "M"), ...],
            ...
        }
        Apenas marca com 'D' os pontos que o robô passou e que não são missão.
        Marca com 'M' os pontos de missão definidos em self.mission_positions.
        """
        result = {}
        # Missões podem ser agrupadas como todos os pontos de missão possíveis
        mission_points = set(self.mission_positions.keys())

        for robot, tasks in optimal_plan.items():
            result[robot] = []
            for task in tasks:
                path = task.get("path", [])
                for node in path:
                    if node in mission_points:
                        result[robot].append((node, "M"))
                    else:
                        result[robot].append((node, "D"))
        return result

    @staticmethod
    def save_plan_dict_to_json(plan_dict, file_path):
        import json

        """
        Salva o dicionário gerado por convert_plan_to_dict em um arquivo JSON.

        Parâmetros:
            - plan_dict: dicionário no formato {"R1": [("P1", "D"), ("P2", "M")], ...}
            - file_path: caminho do arquivo de saída (ex: "./jsons/robo_plan.json")
        """
        # Converter as tuplas para listas (JSON não suporta tuplas diretamente)
        serializable = {robot: [[point, label] for point, label in steps] for robot, steps in plan_dict.items()}
        with open(file_path, "w") as f:
            json.dump(serializable, f, indent=4)





