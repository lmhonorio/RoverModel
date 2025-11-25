import random
import numpy as np
import networkx as nx
from multigraphplanner import MultiGraphPlanner



class Solution:
    def __init__(self, grafo, robots, tasks, cache_astar, allocations=None, id=None):
        self.grafo = grafo
        self.robots = robots
        self.tasks = tasks
        self.cache_astar = cache_astar
        self.allocations = allocations
        self.iteration = 0
        self.id = None
        self.distance = None
        self.time = None
        self.balance_load = None
        self.metrics = [0, 0, 0]
        self.dominated_count = 0
        self.dominated_solutions = []
        self.crowding_distance = 0  # Este será usado no próximo passo
        self.crowding_distance = 0  # Este será usado no próximo passo
        self.rank = None  # Adicione aqui para tornar explícito
        self.strength = 0
        self.fitness = 0

    def objectives(self):
        """
        Retorna os objetivos como uma lista, no formato esperado para cálculo de hipervolume.
        """
        return [self.distance, self.time, self.balance_load]


    # LÓGICA: A FUNÇÃO VAI ALOCANDO, PARA CADA ROBÕ ALTERNADAMENTE, A TASK MAIS PRÓXIMA A ELE RESPEITANDO SUA CAPACIDADE    
    
############################################ SOLUÇÃO GULOSA ####################################################
    
    def greedy_allocate_tasks_graph_based(self, robots, tasks):
        remaining_tasks = tasks[:]
        caminho_por_robo = {robot.id: [] for robot in robots}
        allocations = [[] for _ in robots]

        while remaining_tasks:
            for robot_idx, robot in enumerate(robots):
                if not remaining_tasks:
                    break

                # Posição atual baseada na última alocação
                if allocations[robot_idx]:
                    pos_atual = allocations[robot_idx][-1].exit_point["label"]
                else:
                    pos_atual = robot.initial_position

                melhor_task = None
                menor_distancia = float("inf")
                melhor_caminho = []

                for task in remaining_tasks:
                    destino = task.entry_point["label"]
                    # print(f"🔍 Calculando distância de {pos_atual} até {destino}...")

                    try:
                        caminho, distancia = self.cache_astar.get_path(pos_atual, destino)
                        # print(f"➡️  Distância: {distancia}")
                    except Exception as e:
                        print(f"[⚠️ Erro no A*] {e}")
                        continue

                    if distancia < menor_distancia:
                        melhor_task = task
                        menor_distancia = distancia
                        melhor_caminho = caminho

                if melhor_task and robot.can_allocate(melhor_task):
                    # print(f"✅ Alocando {melhor_task.id} para Robô {robot.id}")
                    allocations[robot_idx].append(melhor_task)
                    remaining_tasks.remove(melhor_task)

                    # Caminho até a task
                    caminho_por_robo[robot.id].extend(melhor_caminho)
                    # Caminho interno da task
                    rota_labels = [p["label"] for p in melhor_task.rota]
                    caminho_por_robo[robot.id].extend(rota_labels)

                    # Atualiza posição do robô (opcional, se quiser armazenar)
                    robot.current_position = melhor_task.exit_point["label"]

            if not any(robot.can_allocate(task) for task in remaining_tasks):
                break

        for i, robot in enumerate(robots):
            robot.allocations = allocations[i]
            robot.path = caminho_por_robo[robot.id]
            # print(f"🛣️ Caminho final do Robô {robot.id}: {robot.path}")

        self.allocations = allocations
        return allocations


####################################### SOLUÇÃO RANDOMICA ##################################################

    def generate_random_solution_graph_based(self, robots, tasks, G_mapa):
        remaining_tasks = tasks[:]
        caminho_por_robo = {robot.id: [] for robot in robots}
        # print(f"caminho por robo: {caminho_por_robo}")
        allocations = [[] for _ in robots]

        while remaining_tasks:
            for robot_idx, robot in enumerate(robots):
                if not remaining_tasks:
                    break

                # Filtra tasks que o robô pode alocar
                candidate_tasks = [task for task in remaining_tasks if robot.can_allocate(task)]
                if not candidate_tasks:
                    continue

                # Escolhe uma aleatoriamente
                chosen_task = random.choice(candidate_tasks)
                destino = chosen_task.entry_point["label"]

                # Posição atual
                if allocations[robot_idx]:
                    pos_atual = allocations[robot_idx][-1].exit_point["label"]
                else:
                    pos_atual = robot.initial_position

                try:
                    # print(f"pos atual: {pos_atual}, destino: {destino}")
                    caminho, distancia = self.cache_astar.get_path(pos_atual, destino)
                    # print(f"caminho: {caminho}")
                except Exception as e:
                    print(f"[⚠️ Erro no A*] {e}")
                    continue

                # Aloca
                allocations[robot_idx].append(chosen_task)
                remaining_tasks.remove(chosen_task)
                caminho_por_robo[robot.id].extend(caminho)
                rota_labels = [p["label"] for p in chosen_task.rota]
                caminho_por_robo[robot.id].extend(rota_labels)

                robot.current_position = chosen_task.exit_point["label"]

            if not any(robot.can_allocate(task) for task in remaining_tasks):
                break

        for i, robot in enumerate(robots):
            robot.allocations = allocations[i]
            robot.path = caminho_por_robo[robot.id]
            # print(f"🛣️ Caminho aleatório do Robô {robot.id}: {robot.path}")

        self.allocations = allocations
        return allocations
    
    def greedy_randomized_allocate_tasks_graph_based(self, robots, tasks, G_mapa, alpha=0.3):
        remaining_tasks = tasks[:]
        caminho_por_robo = {robot.id: [] for robot in robots}
        allocations = [[] for _ in robots]

        while remaining_tasks:
            for robot_idx, robot in enumerate(robots):
                if not remaining_tasks:
                    break

                # Filtra tarefas viáveis
                candidate_tasks = [task for task in remaining_tasks if robot.can_allocate(task)]
                if not candidate_tasks:
                    continue

                # Define posição atual do robô
                if allocations[robot_idx]:
                    pos_atual = allocations[robot_idx][-1].exit_point["label"]
                else:
                    pos_atual = robot.initial_position

                # Calcula distância até cada task candidata
                distancias = []
                for task in candidate_tasks:
                    try:
                        _, dist = self.cache_astar.get_path(pos_atual, task.entry_point["label"])
                        distancias.append((task, dist))
                    except Exception as e:
                        print(f"[⚠️ Erro no A*] {e}")
                        continue

                # Ordena pelas mais próximas (mais guloso)
                distancias.sort(key=lambda x: x[1])
                if not distancias:
                    continue

                # Define limite com alpha
                limite = max(1, int(len(distancias) * alpha))
                escolhida, _ = random.choice(distancias[:limite])

                # Alocação
                try:
                    caminho, _ = self.cache_astar.get_path(pos_atual, escolhida.entry_point["label"])
                except Exception as e:
                    print(f"[⚠️ Erro no A*] {e}")
                    continue

                allocations[robot_idx].append(escolhida)
                remaining_tasks.remove(escolhida)

                caminho_por_robo[robot.id].extend(caminho)
                rota_labels = [p["label"] for p in escolhida.rota]
                caminho_por_robo[robot.id].extend(rota_labels)
                robot.current_position = escolhida.exit_point["label"]

            if not any(robot.can_allocate(task) for task in remaining_tasks):
                break

        for i, robot in enumerate(robots):
            robot.allocations = allocations[i]
            robot.path = caminho_por_robo[robot.id]
            # print(f"🎲 Caminho final do Robô {robot.id}: {robot.path}")

        self.allocations = allocations
        return allocations



    
    
########################################## CALCULO DAS METRICAS ###########################################################
    
    def calculate_robot_execution_distance(self, initial_position_label, battery_time, task_list):
        travel_distance = 0

        if task_list:
            # Distância do ponto inicial até a primeira tarefa
            entry_label = task_list[0].entry_point["label"]
            _, dist = self.cache_astar.get_path(initial_position_label, entry_label)
            travel_distance += dist

            # Distâncias entre tarefas consecutivas (pelo label de saída/entrada)
            for i in range(len(task_list) - 1):
                origem = task_list[i].exit_point["label"]
                destino = task_list[i + 1].entry_point["label"]
                _, dist = self.cache_astar.get_path(origem, destino)
                travel_distance += dist

            # Distância da última tarefa de volta ao ponto inicial (opcional)
            last_exit = task_list[-1].exit_point["label"]
            _, dist = self.cache_astar.get_path(last_exit, initial_position_label)
            travel_distance += dist

        if travel_distance > battery_time:
            return float("inf")

        return travel_distance

    
    def calculate_robot_execution_time(self, initial_position_label, robot_tasks):
        inspection_time = sum(task.inspection_time for task in robot_tasks)
        travel_time = 0

        if robot_tasks:
            entry_label = robot_tasks[0].entry_point["label"]
            _, dist = self.cache_astar.get_path(initial_position_label, entry_label)
            travel_time += dist

            for i in range(len(robot_tasks) - 1):
                origem = robot_tasks[i].exit_point["label"]
                destino = robot_tasks[i + 1].entry_point["label"]
                _, dist = self.cache_astar.get_path(origem, destino)
                travel_time += dist

        return inspection_time + travel_time

    
    def calculate_robot_remaining_energy(self, initial_battery_time, initial_position_label, robot_tasks):
        inspection_distance = sum(task.inspection_distance for task in robot_tasks)
        travel_distance = 0

        if robot_tasks:
            entry_label = robot_tasks[0].entry_point["label"]
            _, dist = self.cache_astar.get_path(initial_position_label, entry_label)
            travel_distance += dist

            for i in range(len(robot_tasks) - 1):
                origem = robot_tasks[i].exit_point["label"]
                destino = robot_tasks[i + 1].entry_point["label"]
                _, dist = self.cache_astar.get_path(origem, destino)
                travel_distance += dist

        return initial_battery_time - (inspection_distance + travel_distance)
    

    def calculate_execution_distance(self):
        total_distance = 0
        for robot_idx, task_list in enumerate(self.allocations):
            robot = self.robots[robot_idx]
            total_distance += self.calculate_robot_execution_distance(
                robot.initial_position, robot.initial_battery_time, task_list
            )
        self.distance = total_distance
        self.metrics[0] = total_distance
        return total_distance

    def calculate_execution_time(self):
        max_time = max(
            self.calculate_robot_execution_time(
                robot.initial_position, task_list)
            for robot, task_list in zip(self.robots, self.allocations)
        )
        self.time = max_time
        self.metrics[1] = max_time
        return max_time

    def calculate_balance_load(self):
        remaining = [
            self.calculate_robot_remaining_energy(
                robot.initial_battery_time, robot.initial_position, task_list)
            for robot, task_list in zip(self.robots, self.allocations) if task_list
        ]
        self.balance_load = np.std(remaining) if remaining else 0
        self.metrics[2] = self.balance_load
        return self.balance_load

    def calculate_metrics(self):
        self.calculate_execution_distance()
        self.calculate_execution_time()
        self.calculate_balance_load()


################################################################################

 
    
    # Exibe métricas da solução
    def print_solution_metrics(self, label="Solution"):
        print(f"{label} distance: {self.distance}")
        print(f"{label} time: {self.time}")
        print(f"{label} balance load: {self.balance_load}")
        print("---------------------------------------------------------")

    def to_dict(self):
        return {
            "distance": self.distance,
            "time": self.time,
            "balance_load": self.balance_load,
            "robots": [robot.to_dict() for robot in self.robots],
            "tasks": [task.to_dict() for task in self.tasks] if self.tasks else None,
        }
    
    def shallow_copy(self):
        new_robots = [robot.shallow_copy() for robot in self.robots]
        return Solution(new_robots, self.tasks)
    
    def copy(self):
        """
        Cria uma cópia da solução atual.

        Returns:
            Solution: Uma nova instância da solução com os mesmos dados.
        """
        # Cria cópias profundas dos atributos que precisam ser independentes
        copied_allocations = [list(robot_tasks) for robot_tasks in self.allocations]
        return Solution(self.robots, self.tasks, copied_allocations, self.cache_astar)
    
    def get_improvement_metric(self):
        """
        Calcula a métrica de melhoria para a solução atual.

        Returns:
            float: Valor da métrica de melhoria.
        """
        # Exemplo: Combinação ponderada de distância, tempo e balanceamento
        weight_distance = 0.5
        weight_time = 0.3
        balance_load = 0.2

        # Inverter métricas para minimizar
        improvement_metric = (
            self.distance * weight_distance +
            self.time * weight_time +
            self.balance_load * balance_load
        )

        return improvement_metric
        

    ########################################## GERAR POPULAÇÃO #################################################
    

    
def generate_hybrid_population(robots, tasks, pop_size, G_mapa, cache_astar, alpha=0.3):
    print("🔁 Gerando população híbrida")
    population = []

    for i in range(pop_size // 2):
        sol_random = Solution(G_mapa, robots, tasks, cache_astar)
        sol_random.allocations = sol_random.generate_random_solution_graph_based(
            robots, tasks, G_mapa
        )
        sol_random.calculate_metrics()
        population.append(sol_random)
        #print(i)

    for j in range(pop_size // 2 - 1):
        sol_greedy = Solution(G_mapa, robots, tasks, cache_astar)
        sol_greedy.allocations = sol_greedy.greedy_randomized_allocate_tasks_graph_based(
            robots, tasks, G_mapa, alpha=alpha
        )
        sol_greedy.calculate_metrics()
        population.append(sol_greedy)
        #print(j)
    sol_greedy = Solution(G_mapa, robots, tasks, cache_astar)
    sol_greedy.allocations = sol_greedy.greedy_allocate_tasks_graph_based(
        robots, tasks)
    sol_greedy.calculate_metrics()
    population.append(sol_greedy)
    return population


def generate_random_population(robots, tasks, pop_size, G_mapa, cache_astar):
    """
    Gera uma população inicial com soluções aleatórias baseadas em grafo e A*.
    """
    print("🔁 Gerando população aleatória baseada em grafo")
    population = []

    for _ in range(pop_size):
        sol = Solution(G_mapa, robots, tasks, cache_astar)
        sol.allocations = sol.generate_random_solution_graph_based(robots, tasks, G_mapa)
        sol.calculate_metrics()
        population.append(sol)

    return population


def calcula_metricas(best_solution):
    # Dado um objeto Solution chamado "solution" já instanciado corretamente
    # Primeiro passo: criar um dict das rotas com ids numéricos dos robôs
    rotas_por_robo = {}
    distancias_por_robo = {}
    tempo_por_robo = {}
    balances = {}

    for idx, tasks_do_robo in enumerate(best_solution.allocations):
        robo_id = f"robot_{idx+1}"
        posicao_inicial_robo = best_solution.robots[idx].start_node_label
        coord_inicial_robo = best_solution.robots[idx].start_node_coord_abs
        balance_robo = 1000000

        rota_nos = [posicao_inicial_robo]
        ponto_anterior = coord_inicial_robo
        distancia_total_robo = 0
        tempo_total_por_robo = 0


        for i, task in enumerate(tasks_do_robo):
            # Define o ponto seguinte (entrada da próxima task)
            if i < len(tasks_do_robo) - 1:
                proxima_task = tasks_do_robo[i+1]
                ponto_seguinte = proxima_task.entry_point['coord_abs']
            # else:
                # ponto_seguinte = coord_inicial_robo  # ou ponto final do robô

            # Calcula melhor rota de observação para a task
            nome_estrategia, custo, rota_otima = task.melhor_rota_para_task(ponto_anterior, ponto_seguinte)

            # Adiciona os pontos da rota ótima
            rota_nos.extend(p['label'] for p in rota_otima)

            # Acumula distância
            distancia_total_robo += custo

            #Acumula tempo
            tempo_total_por_robo += custo
            tempo_total_por_robo += 15 * len(task.observation_points)

            #Atualiza Balanceamento
            balance_robo -= custo

            # Atualiza o ponto anterior
            ponto_anterior = rota_otima[-1]['coord_abs']

        rotas_por_robo[robo_id] = rota_nos
        distancias_por_robo[robo_id] = distancia_total_robo
        tempo_por_robo[robo_id] = tempo_total_por_robo
        balances[robo_id] = balance_robo

    print(f"MÉTRICAS MOVNS")

    print("\n📏 Distâncias totais percorridas por robô:")
    for robo_id, dist in distancias_por_robo.items():
        print(f"{robo_id}: {dist:.2f} metros")

    print("\n📏 Tempos totais gastos por robô:")
    for robo_id, tempo in tempo_por_robo.items():
        print(f"{robo_id}: {tempo:.2f} unidades")

    balances = list(balances.values())
    balance_load = np.std(balances)

    print("\n📏 Balanceamento de carga dos robôs:")
    print(f"{balance_load}")

    print("###################################\n")

    return rotas_por_robo

def calcular_custos_totais_solucao(rotas_por_robo, cache_astar):
    custos_totais = {}
    distancias_totais = {}
    balances = {}
    pontos = 0
    for robo, rota in rotas_por_robo.items():
        custo_total = 0
        distancia_total = 0
        balance_robo = 1000000
        for i in range(len(rota) - 1):
            origem = rota[i]
            destino = rota[i + 1]

            _, custo = cache_astar.get_path(origem, destino)
            # print(f"Custo {origem} -> {destino} = {custo}")
            custo_total += custo
            custo_total += 15
            distancia_total += custo
            balance_robo -= custo
            pontos += 1


        custos_totais[robo] = custo_total
        distancias_totais[robo] = distancia_total
        balances[robo] = balance_robo

    balances = list(balances.values())
    balance_load = np.std(balances)

    print("MÉTRICAS ABORDAGEM CLUSTER")

    print("\n📏 Distâncias totais percorridas por robô:")
    for robo_id, dist in distancias_totais.items():
        print(f"{robo_id}: {dist:.2f} metros")

    print("\n📏 Tempos totais gastos por robô:")
    for robo_id, tempo in custos_totais.items():
        print(f"{robo_id}: {tempo:.2f} unidades")
    
    print("\n📏 Balanceamento de carga dos robôs:")
    print(f"{balance_load}")

    

    return custos_totais, distancias_totais, balances, pontos