import heapq
from itertools import permutations, islice
from tqdm import tqdm
from queue import PriorityQueue
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
import os
import networkx as nx
from networkx.algorithms.approximation import traveling_salesman_problem as tsp


class FixedTaskPlanner:
    """ Planejador de tarefas fixas e distribuídas para múltiplos robôs com otimização TSP. """

    def __init__(self, grafo_mapa, mission_positions, mission_times, mission_execution):
        self.grafo_mapa = grafo_mapa # Grafo completo do mapa
        self.mission_positions = mission_positions # Nó do grafo onde a missão deve ser executada
        self.mission_times = mission_times # Tempo de execução da missão
        self.mission_execution = mission_execution # Quais robôs podem executar a missão


    @staticmethod
    def otimizar_rota_tsp(G, ponto_inicial, pontos_clusterizados):
        """
        G: Grafo original com pesos nas arestas (Greduced_map)
        ponto_inicial: Label inicial do robô
        pontos_clusterizados: Lista de labels dos pontos atribuídos ao robô
        """

        # Subgrafo contendo o ponto inicial e pontos atribuídos ao robô
        nodes_subgrafo = [ponto_inicial] + pontos_clusterizados
        subgrafo = G.subgraph(nodes_subgrafo).copy()

        # Garantir que todas as conexões estejam presentes com pesos reais do grafo original
        for u in subgrafo.nodes():
            for v in subgrafo.nodes():
                if u != v and not subgrafo.has_edge(u, v):
                    try:
                        length = nx.shortest_path_length(G, u, v, weight='weight')
                        subgrafo.add_edge(u, v, weight=length)
                    except nx.NetworkXNoPath:
                        continue  # Ignorar se não houver caminho

        # Encontrar a solução aproximada TSP
        rota = tsp(subgrafo, weight='weight', cycle=False)

        return rota


    @staticmethod
    def tsp_nearest_neighbor(G, ponto_inicial, pontos):
        """
        Implementa o algoritmo do vizinho mais próximo para o TSP.
        """
        nao_visitados = set(pontos)
        atual = ponto_inicial
        rota = [atual]

        while nao_visitados:
            proximo_ponto = min(
                nao_visitados, key=lambda x: nx.shortest_path_length(G, atual, x, weight='weight')
            )
            nao_visitados.remove(proximo_ponto)
            rota.append(proximo_ponto)
            atual = proximo_ponto

        return rota


    @staticmethod
    def clusterizar_pontos_balanceado(G, pontos, robots_positions, mission_execution):
        """
        G: Grafo completo do mapa
        pontos: Dicionário de pontos a serem distribuídos {label: (lat, lon)}
        robots_positions: Posições iniciais dos robôs {robot_id: label}
        mission_execution: Quais robôs podem executar cada missão {label: [robot_ids]}
        """
        os.environ["OMP_NUM_THREADS"] = "1"
        # Obter coordenadas dos pontos
        labels_pontos = list(pontos.keys())
        coords_pontos = np.array([G.nodes[label]['pos'] for label in labels_pontos])

        # Obter coordenadas dos robôs
        labels_robos = list(robots_positions.keys())
        coords_robos = np.array([G.nodes[robots_positions[r]]['pos'] for r in labels_robos])

        # Inicializar KMeans com centróides iniciais (posições dos robôs)
        kmeans = KMeans(n_clusters=len(robots_positions), init=coords_robos, n_init=1, random_state=42)

        # Clusterizar pontos
        labels_clusters = kmeans.fit_predict(coords_pontos)

        # Distribuir pontos respeitando restrições de execução
        pontos_por_robo = defaultdict(list)

        for ponto_label, cluster_label in zip(labels_pontos, labels_clusters):
            robo_atual = labels_robos[cluster_label]
            # Se o robô atribuído pode executar a missão, atribua diretamente
            if robo_atual in mission_execution[ponto_label]:
                pontos_por_robo[robo_atual].append(ponto_label)
            else:
                # Caso não possa, atribua ao outro robô disponível
                possiveis_robos = mission_execution[ponto_label]
                for robo in possiveis_robos:
                    if robo != robo_atual:
                        pontos_por_robo[robo].append(ponto_label)
                        break

        return pontos_por_robo

    def a_star(self, start, goal):
        """
        Implementa o algoritmo A* para encontrar o caminho mais curto entre dois pontos.
        """
        if isinstance(start, list): start = start[0]
        if isinstance(goal, list): goal = goal[0]

        if start not in self.grafo_mapa['states'] or goal not in self.grafo_mapa['states']:
            return None, float("inf")

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {node: float("inf") for node in self.grafo_mapa['states']}
        g_score[start] = 0

        def heuristic(a, b):
            transition = self.grafo_mapa['transitions'].get((a, b), float("inf"))
            return transition[0] if isinstance(transition, tuple) else transition

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

            for (src, neighbor), (weight, _) in self.grafo_mapa['transitions'].items():
                if src != current:
                    continue

                tentative_g_score = g_score[current] + weight
                if tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    came_from[neighbor] = current

        return None, float("inf")

    def distribute_and_schedule_tasks(self, robots_positions, fixed_tasks_per_robot, distributed_tasks,
                                      method="permutation"):
        """
        Distribui tarefas fixas e distribuídas entre múltiplos robôs e otimiza a rota usando TSP.
        robots_positions: Posições iniciais dos robôs {robot_id: label}
        fixed_tasks_per_robot: Tarefas fixas por robô {robot_id: [labels]}
        distributed_tasks: Tarefas distribuídas a serem alocadas [labels]
        method: Método de planejamento ("permutation" ou "nearest")
        """
        if method == "nearest":
            return self.nearest_neighbor_planner(robots_positions, fixed_tasks_per_robot, distributed_tasks)


    def nearest_neighbor_planner(self, robots_positions, fixed_tasks_per_robot, distributed_tasks):
        """
        Planejador baseado no algoritmo do vizinho mais próximo com otimização TSP.
        """
        robot_time = {robot: 0 for robot in robots_positions}
        plan = {robot: [] for robot in robots_positions}
        schedule = []
        assigned = set()
        temp_robot_positions = robots_positions.copy()

        for robot, tasks in fixed_tasks_per_robot.items():
            for task in tasks:
                pos = self.mission_positions[task]
                path, travel_time = self.a_star(temp_robot_positions[robot], pos)
                if path is None:
                    continue
                exec_time = self.mission_times[task]
                start_time = robot_time[robot] + travel_time
                end_time = start_time + exec_time

                plan[robot].append({
                    "mission": task,
                    "path": path,
                    "travel_time": travel_time,
                    "execution_time": exec_time,
                    "start_time": start_time,
                    "end_time": end_time
                })
                schedule.append({
                    "robot": robot,
                    "mission": task,
                    "start": start_time,
                    "end": end_time
                })
                temp_robot_positions[robot] = pos
                robot_time[robot] = end_time
                assigned.add(task)

        remaining_tasks = set(distributed_tasks) - assigned

        while remaining_tasks:
            best_robot = None
            best_task = None
            best_path = None
            best_cost = float("inf")
            best_travel = None

            for task in remaining_tasks:
                for robot in self.mission_execution[task]:
                    start = temp_robot_positions[robot]
                    pos = self.mission_positions[task]
                    path, travel_time = self.a_star(start, pos)
                    if path is None:
                        continue
                    exec_time = self.mission_times[task]
                    end_time = robot_time[robot] + travel_time + exec_time

                    if end_time < best_cost:
                        best_cost = end_time
                        best_robot = robot
                        best_task = task
                        best_path = path
                        best_travel = travel_time

            if best_robot is None or best_path is None:
                print(f"⚠️ Nenhum caminho viável para tarefas restantes: {remaining_tasks}")
                break

            exec_time = self.mission_times[best_task]
            start_time = robot_time[best_robot] + best_travel
            end_time = start_time + exec_time

            plan[best_robot].append({
                "mission": best_task,
                "path": best_path,
                "travel_time": best_travel,
                "execution_time": exec_time,
                "start_time": start_time,
                "end_time": end_time
            })
            schedule.append({
                "robot": best_robot,
                "mission": best_task,
                "start": start_time,
                "end": end_time
            })

            temp_robot_positions[best_robot] = self.mission_positions[best_task]
            robot_time[best_robot] = end_time
            assigned.add(best_task)
            remaining_tasks.remove(best_task)

        # 🔁 Otimização TSP por robô após alocação
        for robot, tasks in plan.items():
            if len(tasks) <= 2:
                continue
            start_node = robots_positions[robot]
            task_nodes = [t["mission"] for t in tasks]
            best_order = None
            best_cost = float("inf")

            for perm in permutations(task_nodes):
                cost = 0
                current = start_node
                valid = True
                for p in perm:
                    path, travel = self.a_star(current, self.mission_positions[p])
                    if path is None:
                        valid = False
                        break
                    cost += travel + self.mission_times[p]
                    current = self.mission_positions[p]
                if valid and cost < best_cost:
                    best_cost = cost
                    best_order = perm

            if best_order:
                robot_time[robot] = 0
                temp_robot_positions[robot] = start_node
                plan[robot] = []
                schedule = [s for s in schedule if s["robot"] != robot]

                for task_id in best_order:
                    pos = self.mission_positions[task_id]
                    path, travel_time = self.a_star(temp_robot_positions[robot], pos)
                    exec_time = self.mission_times[task_id]
                    start_time = robot_time[robot] + travel_time
                    end_time = start_time + exec_time

                    plan[robot].append({
                        "mission": task_id,
                        "path": path,
                        "travel_time": travel_time,
                        "execution_time": exec_time,
                        "start_time": start_time,
                        "end_time": end_time
                    })
                    schedule.append({
                        "robot": robot,
                        "mission": task_id,
                        "start": start_time,
                        "end": end_time
                    })
                    temp_robot_positions[robot] = pos
                    robot_time[robot] = end_time

        total_time = max(robot_time.values()) if plan else float("inf")
        return plan, total_time, schedule