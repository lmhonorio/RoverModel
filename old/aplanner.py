import heapq
from old.baseclasses import *







class AStarPlanner:
    def __init__(self, initial_state, goal_predicates_func, operators, robots, grafo_mapa, get_goal_state_predicates):
        self.initial_state = initial_state
        self.goal_predicates_func = goal_predicates_func
        self.operators = operators
        self.robots = robots
        self.grafo_mapa = grafo_mapa
        self.get_goal_state_predicates = get_goal_state_predicates

    def heuristic(self, state):
        """Heurística baseada na distância até o objetivo"""
        goal_predicates = self.goal_predicates_func()
        return len(goal_predicates - state.predicates)

    def is_goal(self, state):
        """Verifica se o estado atende às condições de objetivo"""
        goal_predicates = self.goal_predicates_func()
        goal_state_predicates = self.get_goal_state_predicates(state)
        return goal_predicates.issubset(goal_state_predicates)

    def extract_robot_locations(self, state):
        """Extrai as localizações de R1 e R2 a partir dos predicados do estado"""
        r1_location, r2_location = None, None
        for predicate in state.predicates:
            predicate_str = str(predicate)
            if predicate_str.startswith("Em(") and "," in predicate_str:
                parts = predicate_str.strip("Em()").split(",")  # Exemplo: "Em(PR11_2, R1)"
                if len(parts) == 2:
                    location, robot = parts[0].strip(), parts[1].strip()
                    if robot == "R1":
                        r1_location = location
                    elif robot == "R2":
                        r2_location = location
        return r1_location, r2_location

    def plan_graph_based(self):
        """
        Busca primeiro o caminho ótimo no grafo e depois verifica se ele é executável.
        Agora considera os caminhos para R1 e R2 separadamente.
        """

        # Criar grafo NetworkX
        G = nx.Graph()

        # Adicionar estados (nós)
        for node in self.grafo_mapa["states"]:
            G.add_node(node)

        # Adicionar transições (arestas com peso)
        for (u, v), weight in self.grafo_mapa["transitions"].items():
            if isinstance(weight, tuple):  # Se já é uma tupla (distância, bateria)
                distancia, bateria = weight
            else:  # Se for um float, converte para (distância, bateria)
                distancia, bateria = weight, 1.0  # Assume bateria padrão

            G.add_edge(u, v, distance=distancia, battery=bateria)

        # Identificar localização dos robôs no estado inicial
        r1_location, r2_location = self.extract_robot_locations(self.initial_state)

        if not r1_location or not r2_location:
            print("🚨 ERRO: Não foi possível encontrar as posições iniciais dos robôs!")
            return None

        # 🔥 Agora pegamos somente as LOCALIZAÇÕES, sem o predicado
        goal_locations = {str(p).split("(")[-1].split(")")[0] for p in self.goal_predicates_func()}

        print(f"📍 R1 em {r1_location}, R2 em {r2_location}")
        print(f"🎯 Buscando caminhos até os objetivos: {goal_locations}")

        # Criamos dicionários para armazenar os caminhos de R1 e R2
        best_paths = {"R1": None, "R2": None}

        for goal in goal_locations:
            if goal not in G:
                continue  # Pula se o objetivo não existe no grafo

            # 🔥 Encontrar caminho para R1
            if best_paths["R1"] is None:
                try:
                    path_r1 = nx.shortest_path(G, source=r1_location, target=goal, weight="distance")
                    print(f"✅ Melhor caminho encontrado para R1: {path_r1}")

                    # Verifica se é possível seguir esse caminho segundo as regras do sistema
                    if self.validate_path(path_r1):
                        best_paths["R1"] = path_r1  # 🔥 Armazena o caminho para R1

                except nx.NetworkXNoPath:
                    print(f"❌ Sem caminho de {r1_location} até {goal}")

            # 🔥 Encontrar caminho para R2
            if best_paths["R2"] is None:
                try:
                    path_r2 = nx.shortest_path(G, source=r2_location, target=goal, weight="distance")
                    print(f"✅ Melhor caminho encontrado para R2: {path_r2}")

                    # Verifica se é possível seguir esse caminho segundo as regras do sistema
                    if self.validate_path(path_r2):
                        best_paths["R2"] = path_r2  # 🔥 Armazena o caminho para R2

                except nx.NetworkXNoPath:
                    print(f"❌ Sem caminho de {r2_location} até {goal}")

        if best_paths["R1"] and best_paths["R2"]:
            print("🚀 Caminhos válidos encontrados para ambos os robôs!")
            return best_paths  # Retorna os caminhos de ambos os robôs

        print("🚨 Nenhum plano encontrado para um ou ambos os robôs!")
        return None

    def validate_path(self, path):
        """
        Verifica se um caminho encontrado no grafo pode ser seguido pelos robôs.
        Testa aplicando as regras dos operadores e mostra o motivo de falha.
        """
        current_state = self.initial_state

        for i in range(len(path) - 1):
            from_location = path[i]
            to_location = Instance("Place", path[i + 1])  # 🔥 Converte string para objeto Instance

            move_possible = False
            failure_reason = None  # Variável para armazenar o motivo da falha

            for operator in self.operators:
                for robot in self.robots:
                    isapplicable, failure_reason = operator.is_applicable(current_state, robot, to_location,
                                                                          self.grafo_mapa)

                    if isapplicable:
                        move_possible = True
                        break  # Achou um operador válido, pode seguir

                if move_possible:
                    break  # Sai do loop de operadores

            if not move_possible:
                print(f"❌ Movimento inválido de {from_location} para {to_location.name}!")
                if failure_reason:
                    print(f"⚠ Motivo da falha: {failure_reason}")  # 🔥 Mostra o motivo da falha
                return True  # Se algum movimento for impossível, rejeita o caminho

        return True  # Se passou por todos os testes, o caminho é válido

    def plan(self):
        open_set = []
        heapq.heappush(open_set, (0, self.initial_state))
        came_from = {}
        g_score = {self.initial_state: 0}
        visited = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            #print(f"🟢 Expandindo: {current}")

            if self.is_goal(current):
                print("🎯 Objetivo atingido!")
                return self.reconstruct_path(came_from, current)

            visited.add(current)


            neighbor_locations = [Instance("Place", i) for i in sorted(self.grafo_mapa['states'], key=str)]

            for operator in self.operators:
                for robot in self.robots:
                    for location in neighbor_locations:
                        isapplicable, peso = operator.is_applicable(current, robot, location, self.grafo_mapa)
                        if not isapplicable:
                            continue

                        next_state = current.apply(operator, robot, location, state=current, state_map=self.grafo_mapa)
                        if next_state is None:
                            print(f"⚠ Erro ao aplicar {operator} com {robot} em {location}")
                            continue

                        tentative_g_score = g_score[current] + peso

                        if next_state not in g_score or tentative_g_score < g_score[next_state]:
                            g_score[next_state] = tentative_g_score
                            f_score = tentative_g_score + self.heuristic(next_state)
                            heapq.heappush(open_set, (f_score, next_state))
                            came_from[next_state] = (current, operator, robot, location)

        print("🚨 Nenhum plano encontrado.")
        return None  # No plan found

    def reconstruct_path(self, came_from, current):
        path = []
        final = current
        while current in came_from:
            current, operator, robot, location = came_from[current]
            path.insert(0, (operator, robot, location))
        return path, final
