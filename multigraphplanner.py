import networkx as nx
import heapq
from itertools import permutations
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib.patches import Patch


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
        """
        Encontra a melhor alocação de missões, garantindo que cada missão
        só inicie após a finalização total de suas predecessoras.
        """

        best_plan = {robot: [] for robot in robots_positions.keys()}
        schedule = []
        robot_time = {robot: 0.0 for robot in robots_positions.keys()}
        temp_robot_positions = robots_positions.copy()

        # Armazena o tempo de término de cada missão
        mission_finish_times = {}

        # Obter ordem topológica das missões
        mission_order = self.get_mission_execution_order()

        for mission in mission_order:
            best_robot = None
            best_path = None
            best_end_time = float("inf")
            best_travel_time = 0.0

            available_robots = self.mission_execution[mission]
            available_robots = sorted(available_robots, key=lambda r: robot_time[r])

            for robot in available_robots:
                start_position = temp_robot_positions[robot]
                path, travel_time = self.a_star(start_position, self.mission_positions[mission])

                if path is None:
                    continue

                # 🛑 Espera pela missão predecessora antes de iniciar 🛑
                predecessors = list(self.G_m.predecessors(mission))
                max_predecessor_end = max(
                    (mission_finish_times[p] for p in predecessors if p in mission_finish_times),
                    default=0  # Se não houver predecessores, inicia no tempo 0
                )

                # Ajusta o tempo de início da missão considerando o robô e suas dependências
                start_time = max(robot_time[robot], max_predecessor_end)
                execution_time = self.mission_times[mission]
                end_time = start_time + travel_time + execution_time

                if end_time < best_end_time:
                    best_robot = robot
                    best_path = path
                    best_end_time = end_time
                    best_travel_time = travel_time

            if best_robot:
                # Registra o tempo de finalização da missão para futuras dependências
                mission_finish_times[mission] = best_end_time

                # Atualiza plan e schedule
                best_plan[best_robot].append({
                    "mission": mission,
                    "path": best_path,
                    "travel_time": best_travel_time,
                    "execution_time": execution_time,
                    "start_time": start_time,
                    "end_time": best_end_time
                })
                schedule.append({
                    "robot": best_robot,
                    "mission": mission,
                    "start": start_time,
                    "end": best_end_time,
                    "travel_time": best_travel_time,
                    "execution_time": execution_time
                })

                # Atualiza posição do robô e tempo de disponibilidade
                temp_robot_positions[best_robot] = self.mission_positions[mission]
                robot_time[best_robot] = best_end_time

        total_time = max(robot_time.values())
        return best_plan, total_time, schedule

    def find_minimum_mission_time_plan2(self, robots_positions):
        planned_paths = {r: [] for r in robots_positions}
        schedule = []
        pos = dict(robots_positions)
        time_robot = {r: 0.0 for r in robots_positions}
        done_time = {}

        # Para armazenar info de deslocamento:
        travel_info = {}  # key: (robot, task) -> { "path": [...], "travel_time": X, "travel_start": T }

        all_tasks = list(self.get_mission_execution_order())
        pred_count = {}
        for t in all_tasks:
            preds = list(self.G_m.predecessors(t))
            pred_count[t] = len(preds)

        events = []

        def add_event(t, etype, r, task):
            heapq.heappush(events, (t, etype, r, task))

        # Tarefas sem predecessor -> disparar travel_begin
        available_tasks = [t for t in all_tasks if pred_count[t] == 0]
        for task in available_tasks:
            r = min(self.mission_execution[task], key=lambda x: time_robot[x])
            add_event(time_robot[r], "travel_begin", r, task)

        while events:
            t_current, etype, r, task = heapq.heappop(events)
            time_robot[r] = t_current

            if etype == "travel_begin":
                start_pos = pos[r]
                path, travel_t = self.a_star(start_pos, self.mission_positions[task])
                if path is None:
                    continue
                travel_info[(r, task)] = {
                    "path": path,
                    "travel_time": travel_t,
                    "travel_start": t_current
                }
                arrival_time = t_current + travel_t
                add_event(arrival_time, "travel_end", r, task)

            elif etype == "travel_end":
                pos[r] = self.mission_positions[task]
                # Se ainda tiver predecessores pendentes, não inicia exec
                if pred_count[task] > 0:
                    # Fica aguardando. Qdo predecessor terminar, liberamos a exec
                    pass
                else:
                    # Pode iniciar execução
                    add_event(t_current, "exec_begin", r, task)

            elif etype == "exec_begin":
                # Inicia execução
                e_time = self.mission_times[task]
                exec_end = t_current + e_time
                add_event(exec_end, "exec_end", r, task)

            elif etype == "exec_end":
                done_time[task] = t_current
                # Armazena nos planned_paths
                tinfo = travel_info.get((r, task), {})
                # travel_time e path
                t_travel = tinfo.get("travel_time", 0)
                p_path = tinfo.get("path", [])

                planned_paths[r].append({
                    "mission": task,
                    "path": p_path,
                    "travel_time": t_travel,
                    "execution_time": self.mission_times[task],
                    "start_time": t_current - self.mission_times[task],  # approx
                    "end_time": t_current
                })

                # Libera sucessoras
                for suc in self.G_m.successors(task):
                    pred_count[suc] -= 1
                    if pred_count[suc] == 0:
                        r_best = min(self.mission_execution[suc], key=lambda x: time_robot[x])
                        add_event(time_robot[r_best], "travel_begin", r_best, suc)

        total_time = max(done_time.values()) if done_time else 0
        return planned_paths, total_time, schedule




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
                # Se quiser mostrar o nó final como "em <nó final do caminho>"
                # Basta usar path[-1] (ou algo como self.mission_positions[mission_name], a seu critério)
                final_node = path[-1] if path else "???"

                print(
                    f"🚀 {robot} executará {mission_name} em {final_node} "
                    f"(Deslocamento: {travel_t}s, Execução: {exec_t}s) "
                    f"seguindo o caminho: {path}"
                )


        # Se quiser exibir 'schedule' como log de eventos:
        # for ev in schedule:
        #     print(ev)
        self.plot_gantt(planned_paths)



    def plot_gantt(self, planned_paths):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

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
                arr = task_data.get("arrival_time", st)
                exec_start = task_data.get("execution_start", arr)
                end = task_data.get("end_time", exec_start)
                path = task_data.get("path", [])

                # Desloc. => [st, arr]
                ax.barh(y, arr - st, left=st, color=travel_color, edgecolor="black")
                ax.text((st + arr) / 2, y, "Desloc.", ha="center", va="center", fontsize=8)

                # Exec. => [exec_start, end]
                ax.barh(y, end - exec_start, left=exec_start, color=exec_color, edgecolor="black")
                ax.text((exec_start + end) / 2, y, mission, ha="center", va="center", fontsize=8)

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





