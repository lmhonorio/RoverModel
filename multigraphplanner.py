
import networkx as nx
import heapq
from itertools import permutations
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib.patches import Patch
import json
from concurrent.futures import ThreadPoolExecutor
import math
import re
from itertools import product
from multiprocessing import Pool
from tqdm import tqdm


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

    @staticmethod
    def process_label_for_astar(args):
        label, current, automata = args
        new_state = list(current)
        total_weight = 0
        valid = True

        for i, aut in enumerate(automata):
            succs = [
                (v, data) for _, v, k, data in aut.out_edges(current[i], keys=True, data=True)
                if data.get("label") == label
            ]
            if succs:
                v, data = succs[0]
                new_state[i] = v
                total_weight += data.get("weight", 1)
            else:
                # Label existe no autômato, mas não está habilitado no estado atual
                labels_aut = {data.get("label") for _, _, _, data in aut.edges(data=True, keys=True)}
                if label in labels_aut:
                    valid = False
                    break

        if not valid:
            return None
        return tuple(new_state), total_weight, label


    @staticmethod
    def find_optimal_route_through_all_states(G):
        """
        Retorna a rota de menor custo que passa por todos os estados do grafo (TSP exato).
        Considera os pesos das transições.

        Parâmetros:
            G: nx.Graph ou nx.DiGraph com pesos nas arestas.

        Retorna:
            caminho_ideal: lista com a sequência de estados visitados.
            custo_total: soma dos pesos do caminho.
        """
        if G.number_of_nodes() <= 1:
            return list(G.nodes), 0

        nodes = list(G.nodes)
        best_path = None
        min_cost = float('inf')

        for perm in permutations(nodes):
            custo = 0
            valido = True
            for i in range(len(perm) - 1):
                try:
                    peso = nx.dijkstra_path_length(G, source=perm[i], target=perm[i + 1], weight='weight')
                    custo += peso
                except nx.NetworkXNoPath:
                    valido = False
                    break
            if valido and custo < min_cost:
                best_path = perm
                min_cost = custo

        return list(best_path), min_cost

    @staticmethod
    def parallel_composition_multiple_astar(automata, condition=lambda states: True, heuristic=lambda s: 0):
        G = nx.MultiDiGraph()

        start_tuple = tuple(a.graph['start'] for a in automata)
        start_name = ",".join(start_tuple)
        G.add_node(start_name)
        G.graph['start'] = start_name
        G.nodes[start_name]['style'] = 'filled'
        G.nodes[start_name]['color'] = 'lightgreen'

        visited = set()
        heap = []
        heapq.heappush(heap, (0 + heuristic(start_tuple), 0, start_tuple))  # (f, g, state)

        progress_bar = tqdm(total=0, desc="A* paralelizado (até aceitar)", unit="estado", dynamic_ncols=True)

        while heap:
            f, g, current = heapq.heappop(heap)
            current_name = ",".join(current)

            if current in visited:
                continue
            visited.add(current)
            progress_bar.total = len(visited) + len(heap)
            progress_bar.update(1)

            if all(automata[i].nodes[current[i]].get('accepting_state', False) for i in range(len(automata))):
                G.nodes[current_name]['shape'] = 'doublecircle'
                G.nodes[current_name]['color'] = 'orange'
                G.nodes[current_name]['accepting_state'] = True
                progress_bar.set_postfix_str(f"\u2714 Aceito: {current_name}")
                break

            all_labels = set(
                data.get("label")
                for aut in automata
                for _, _, _, data in aut.edges(data=True, keys=True)
                if "label" in data
            )

            args_list = [(label, current, automata) for label in all_labels]

            with Pool() as pool:
                results = pool.map(process_label_for_astar, args_list)

            for result in results:
                if result is None:
                    continue
                new_state_tuple, total_weight, label = result

                if not condition(new_state_tuple):
                    continue

                new_name = ",".join(new_state_tuple)

                if new_name not in G:
                    G.add_node(new_name)
                    if all(automata[i].nodes[new_state_tuple[i]].get('accepting_state', False)
                           for i in range(len(automata))):
                        G.nodes[new_name]['shape'] = 'doublecircle'
                        G.nodes[new_name]['color'] = 'orange'
                        G.nodes[new_name]['accepting_state'] = True

                G.add_edge(current_name, new_name, key=label, label=label, weight=total_weight)

                if new_state_tuple not in visited:
                    g_new = g + total_weight
                    f_new = g_new + heuristic(new_state_tuple)
                    heapq.heappush(heap, (f_new, g_new, new_state_tuple))

        progress_bar.close()
        return G


    @staticmethod
    def get_shortest_path_to_accepting(G):
        """
        Retorna o menor caminho do estado inicial até o primeiro estado de aceite,
        incluindo os estados visitados e os rótulos (transições) disparados.

        Parâmetros:
        -----------
        G : nx.MultiDiGraph
            Grafo com:
            - G.graph['start']: estado inicial
            - Atributo 'accepting_state' nos nós
            - Arestas com 'label' e 'weight'

        Retorna:
        --------
        caminho_estados : list[str]
            Sequência de estados visitados.
        transicoes : list[str]
            Sequência de transições (labels) disparadas.
        custo_total : float
            Custo total acumulado do caminho.
        """
        start = G.graph.get("start")
        if start is None:
            raise ValueError("Grafo não possui estado inicial definido (G.graph['start']).")

        finais = [n for n, d in G.nodes(data=True) if d.get('accepting_state')]
        if not finais:
            raise ValueError("Nenhum estado de aceite encontrado no grafo.")

        menor_caminho = None
        menor_transicoes = None
        menor_custo = float('inf')

        for fim in finais:
            try:
                caminho = nx.shortest_path(G, source=start, target=fim, weight="weight")
                custo = nx.path_weight(G, caminho, weight="weight")

                transicoes = []
                for i in range(len(caminho) - 1):
                    u = caminho[i]
                    v = caminho[i + 1]

                    # Busca a aresta com menor peso entre u e v
                    menor_label = None
                    menor_peso = float('inf')
                    for key, data in G[u][v].items():
                        if data.get("weight", 1) < menor_peso:
                            menor_peso = data["weight"]
                            menor_label = data.get("label", "")

                    transicoes.append(menor_label)

                if custo < menor_custo:
                    menor_caminho = caminho
                    menor_transicoes = transicoes
                    menor_custo = custo

            except nx.NetworkXNoPath:
                continue

        if menor_caminho is None:
            raise ValueError("Não existe caminho do estado inicial para nenhum estado de aceite.")

        print("🧭 Menor caminho de execução:")
        for i, estado in enumerate(menor_caminho):
            print(f"  {estado}")
            if i < len(menor_transicoes):
                print(f"    └──[{menor_transicoes[i]}]→")

        print(f"✔ Custo total: {menor_custo}")

        return menor_caminho, menor_transicoes, menor_custo


    @staticmethod
    def get_path_from_label(g, path_labels):
        """
        Dado um grafo NX cujo nome dos nós é diretamente o label desejado,
        retorna as coordenadas (x,y) correspondentes a cada label da lista.

        Parâmetros
        ----------
        g : nx.Graph
            Grafo onde o nome do nó (node ID) é o próprio label.
            Cada nó deve ter atributo 'pos' com (x,y).
        path_labels : list[str]
            Lista de labels para buscar no grafo.

        Retorna
        -------
        coords : list[tuple]
            Lista de coordenadas (x,y) associadas aos labels fornecidos.
            Se um label não for encontrado ou não tiver 'pos', é emitido um aviso.
        """
        coords = []
        for label in path_labels:
            if label in g.nodes:
                node_data = g.nodes[label]
                pos = node_data.get("pos")
                if pos:
                    coords.append(pos)
                else:
                    print(f"⚠️ Nó encontrado, mas sem coordenadas (pos): {label}")
            else:
                print(f"⚠️ Label não encontrado no grafo: {label}")
        return coords

    @staticmethod
    def AdicionaEdge(grafo, origem, destino, label):
        grafo.add_edge(origem, destino, label=label)


    @staticmethod
    def parallel_composition_multiple_astar(automata, condition=lambda states: True, heuristic=lambda s: 0):
        """
        Composição paralela de múltiplos autômatos usando busca A*,
        respeitando transições sincronizadas e parciais.

        Parâmetros:
        -----------
        automata : list[nx.MultiDiGraph]
            Lista de autômatos (com 'label' e 'weight' nas transições).
        condition : function
            Função opcional para restringir combinações de estados.
        heuristic : function
            Heurística A* baseada no estado composto atual.

        Retorna:
        --------
        G : nx.MultiDiGraph
            Autômato composto, interrompido na primeira aceitação.
        """
        G = nx.MultiDiGraph()

        start_tuple = tuple(a.graph['start'] for a in automata)
        start_name = ",".join(start_tuple)
        G.add_node(start_name)
        G.graph['start'] = start_name
        G.nodes[start_name]['style'] = 'filled'
        G.nodes[start_name]['color'] = 'lightgreen'

        visited = set()
        heap = []
        heapq.heappush(heap, (0 + heuristic(start_tuple), 0, start_tuple))  # (f, g, state)

        progress_bar = tqdm(total=0, desc="A* (até aceitar)", unit="estado", dynamic_ncols=True)

        while heap:
            f, g, current = heapq.heappop(heap)
            current_name = ",".join(current)

            if current in visited:
                continue
            visited.add(current)
            progress_bar.total = len(visited) + len(heap)
            progress_bar.update(1)

            # Verifica se estado composto é de aceite
            if all(automata[i].nodes[current[i]].get('accepting_state', False) for i in range(len(automata))):
                G.nodes[current_name]['shape'] = 'doublecircle'
                G.nodes[current_name]['color'] = 'orange'
                G.nodes[current_name]['accepting_state'] = True
                progress_bar.set_postfix_str(f"✔ Aceito: {current_name}")
                break

            # Extrai todos os labels possíveis
            all_labels = set(
                data.get("label")
                for aut in automata
                for _, _, _, data in aut.edges(data=True, keys=True)
                if "label" in data
            )

            # Para cada label possível (ação)
            for label in all_labels:
                new_state = list(current)
                total_weight = 0
                transicao_valida = False

                for i, aut in enumerate(automata):
                    transicoes = [
                        (v, data) for _, v, k, data in aut.out_edges(current[i], keys=True, data=True)
                        if data.get("label") == label
                    ]
                    if transicoes:
                        v, data = transicoes[0]  # pega a primeira transição válida
                        new_state[i] = v
                        total_weight += data.get("weight", 1)
                        transicao_valida = True
                    else:
                        # Transição não disponível, mantém estado atual
                        new_state[i] = current[i]

                new_state_tuple = tuple(new_state)
                if not transicao_valida or not condition(new_state_tuple):
                    continue

                new_name = ",".join(new_state_tuple)

                if new_name not in G:
                    G.add_node(new_name)
                    if all(automata[i].nodes[new_state_tuple[i]].get('accepting_state', False)
                           for i in range(len(automata))):
                        G.nodes[new_name]['shape'] = 'doublecircle'
                        G.nodes[new_name]['color'] = 'orange'
                        G.nodes[new_name]['accepting_state'] = True

                G.add_edge(current_name, new_name, key=label, label=label, weight=total_weight)

                if new_state_tuple not in visited:
                    g_new = g + total_weight
                    f_new = g_new + heuristic(new_state_tuple)
                    heapq.heappush(heap, (f_new, g_new, new_state_tuple))

        progress_bar.close()
        return G


    @staticmethod
    def parallel_composition_multiple_astar_old(automata, condition=lambda states: True, heuristic=lambda s: 0):
        """
        Composição paralela de múltiplos autômatos usando busca A*.

        - Só expande os estados compostos de menor custo acumulado (g + h).
        - Permite transições sincronizadas ou parciais.

        Parâmetros:
        -----------
        automata : list of nx.MultiDiGraph
            Lista de autômatos.
        condition : function
            Função de restrição que recebe uma tupla de estados e retorna True/False.
        heuristic : function
            Função heurística que recebe um estado composto (tuple) e retorna um valor estimado de custo.

        Retorna:
        --------
        G : nx.MultiDiGraph
            Autômato composto explorado via A*.
        """
        G = nx.MultiDiGraph()

        # Estado inicial
        start_tuple = tuple(a.graph['start'] for a in automata)
        start_name = ",".join(start_tuple)
        G.add_node(start_name)
        G.graph['start'] = start_name
        G.nodes[start_name]['style'] = 'filled'
        G.nodes[start_name]['color'] = 'lightgreen'

        if all(automata[i].nodes[state].get('accepting_state', False) for i, state in enumerate(start_tuple)):
            G.nodes[start_name]['shape'] = 'doublecircle'
            G.nodes[start_name]['color'] = 'orange'
            G.nodes[start_name]['accepting_state'] = True

        visited = set()
        heap = []
        heapq.heappush(heap, (0 + heuristic(start_tuple), 0, start_tuple))  # (f, g, state)

        while heap:
            f, g, current = heapq.heappop(heap)
            current_name = ",".join(current)

            if current in visited:
                continue
            visited.add(current)

            # Expandir todas transições possíveis (eventos)
            for i, aut in enumerate(automata):
                for _, v, k, data in aut.out_edges(current[i], keys=True, data=True):
                    label = data.get("label")
                    weight = data.get("weight", 1)

                    # Novo estado composto: só o autômato i avança
                    new_state = list(current)
                    new_state[i] = v
                    new_state = tuple(new_state)
                    new_name = ",".join(new_state)

                    if not condition(new_state):
                        continue

                    # Criação do nó novo
                    if new_name not in G:
                        G.add_node(new_name)

                        if all(automata[j].nodes[new_state[j]].get('accepting_state', False) for j in
                               range(len(automata))):
                            G.nodes[new_name]['shape'] = 'doublecircle'
                            G.nodes[new_name]['color'] = 'orange'
                            G.nodes[new_name]['accepting_state'] = True

                    # Aresta parcial
                    G.add_edge(current_name, new_name, key=label, label=label, weight=weight)

                    if new_state not in visited:
                        g_new = g + weight
                        f_new = g_new + heuristic(new_state)
                        heapq.heappush(heap, (f_new, g_new, new_state))

        return G



    @staticmethod
    def info_automato(G):
        """
        Retorna o número de estados (nós) e transições (arestas) de um autômato.

        Parâmetros:
            G : nx.Graph, nx.DiGraph ou nx.MultiDiGraph

        Retorna:
            dict com:
                - 'num_estados': número de nós
                - 'num_transicoes': número de transições (arestas)
        """
        return {
            'num_estados': G.number_of_nodes(),
            'num_transicoes': G.number_of_edges()
        }

    @staticmethod
    def parallel_composition(automaton_A, automaton_B, condition=lambda state_A, state_B: True):
        import networkx as nx

        parallel_automaton = nx.MultiDiGraph()

        # 1. Criação dos estados compostos
        for state_A in automaton_A.nodes:
            for state_B in automaton_B.nodes:
                if not condition(state_A, state_B):
                    continue

                parallel_state = f'{state_A},{state_B}'
                parallel_automaton.add_node(parallel_state)

                if state_A == automaton_A.graph['start'] and state_B == automaton_B.graph['start']:
                    parallel_automaton.graph['start'] = parallel_state
                    parallel_automaton.nodes[parallel_state]['color'] = 'lightgreen'
                    parallel_automaton.nodes[parallel_state]['style'] = 'filled'

                if automaton_A.nodes[state_A].get('accepting_state') and automaton_B.nodes[state_B].get(
                        'accepting_state'):
                    parallel_automaton.nodes[parallel_state]['shape'] = 'doublecircle'
                    parallel_automaton.nodes[parallel_state]['color'] = 'orange'
                    parallel_automaton.nodes[parallel_state]['accepting_state'] = True

        # 2. Geração dos alfabetos
        sigmaA = set([data['label'] for _, _, data in automaton_A.edges(data=True)])
        sigmaB = set([data['label'] for _, _, data in automaton_B.edges(data=True)])
        all_labels = sigmaA.union(sigmaB)

        # 3. Transições compostas
        for state_A in automaton_A.nodes:
            for state_B in automaton_B.nodes:
                if not condition(state_A, state_B):
                    continue

                for label in all_labels:
                    current_state = f'{state_A},{state_B}'

                    # Tentamos obter as transições em cada autômato individualmente
                    successors_A = [
                        (v, data) for u, v, data in automaton_A.edges(state_A, data=True)
                        if data.get('label') == label
                    ]
                    successors_B = [
                        (v, data) for u, v, data in automaton_B.edges(state_B, data=True)
                        if data.get('label') == label
                    ]

                    # Casos possíveis:
                    # 1. Ambos têm a transição → sincroniza
                    if successors_A and successors_B:
                        for (v_A, _) in successors_A:
                            for (v_B, _) in successors_B:
                                to_state = f'{v_A},{v_B}'
                                parallel_automaton.add_edge(current_state, to_state, key=label, label=label)

                    # 2. Só A tem a transição → B mantém o estado
                    elif successors_A:
                        for (v_A, _) in successors_A:
                            to_state = f'{v_A},{state_B}'
                            parallel_automaton.add_edge(current_state, to_state, key=label, label=label)

                    # 3. Só B tem a transição → A mantém o estado
                    elif successors_B:
                        for (v_B, _) in successors_B:
                            to_state = f'{state_A},{v_B}'
                            parallel_automaton.add_edge(current_state, to_state, key=label, label=label)

                    # 4. Nenhum tem a transição: nada a fazer

        return parallel_automaton

    @staticmethod
    def xml_to_graph(graphxml):
        # Criar um grafo direcionado (DiGraph)
        G = nx.MultiDiGraph(format='png', engine='dot')

        # Adicionar estados
        G.add_nodes_from(graphxml['states'])

        # Adicionar transições
        for transition, target_state in graphxml['transitions'].items():
            current_state, symbol = transition
            target_state, weight = target_state
            G.add_edge(current_state, target_state, key=symbol, label=symbol, weight=weight)


        # Definir os estados finais
        for state in graphxml['accepting_states']:
            G.nodes[state]['accepting_state'] = True
            G.nodes[state]['shape'] = 'doublecircle'
            G.nodes[state]['color'] = 'orange'

        # Aplicar propriedades aos estados diferenciados
        if 'diferenciados' in graphxml:
            estados_diferenciados, propriedades = graphxml['diferenciados']
            for state in estados_diferenciados:
                if state in graphxml['states']:  # Garantir que o estado existe
                    for key, value in propriedades.items():
                        G.nodes[state][key] = value

        G.graph['start'] = graphxml['start']
        G.nodes[G.graph['start']]['style'] = 'filled'
        G.nodes[graphxml['start']]['color'] = 'lightgreen'

        return G

    # Função que será paralelizada para processar estados compostos
    @staticmethod
    def process_composite_state(args):
        composite_state, automata, events = args
        source_state_name = ",".join(composite_state)
        transitions = []

        for event in events:
            next_state = list(composite_state)
            valid_transition = True

            for i, automaton in enumerate(automata):
                # Verificar se o evento existe no autômato atual
                possible_transitions = [v for u, v, label in automaton.edges(data="label") if
                                        u == composite_state[i] and label == event]

                if possible_transitions:
                    next_state[i] = possible_transitions[0]
                elif event in set(label for _, _, label in automaton.edges(data="label")):
                    # Evento existe no autômato mas não é válido para o estado atual
                    valid_transition = False
                    break

            if valid_transition:
                next_state_name = ",".join(next_state)
                transitions.append((source_state_name, next_state_name, event))

        return transitions

    # Função principal de composição paralela
    @staticmethod
    def parallel_composition_multiple_fast(automata, condition=lambda states: True):
        """
        Composição paralela de múltiplos autômatos com suporte a paralelismo.

        Args:
            automata (list of nx.MultiDiGraph): Lista de autômatos a serem compostos.
            condition (callable): Função que recebe uma tupla de estados
                                  e retorna True se os estados puderem ser combinados.

        Returns:
            nx.MultiDiGraph: O autômato resultante da composição paralela.
        """
        parallel_automaton = nx.MultiDiGraph()

        # Obter todos os estados iniciais dos autômatos
        start_states = [automaton.graph['start'] for automaton in automata]

        # Criar estados compostos a partir do produto cartesiano dos estados de cada autômato
        all_states = list(product(*[automaton.nodes for automaton in automata]))
        events = set(label for automaton in automata for (_, _, label) in automaton.edges(data="label"))

        for composite_state in all_states:
            # Verificar se os estados compostos satisfazem a condição
            if not condition(composite_state):
                continue

            # Criar o estado composto
            composite_state_name = ",".join(composite_state)
            parallel_automaton.add_node(composite_state_name)

            # Definir o estado inicial
            if composite_state == tuple(start_states):
                parallel_automaton.graph['start'] = composite_state_name
                parallel_automaton.nodes[composite_state_name]['color'] = 'lightgreen'
                parallel_automaton.nodes[composite_state_name]['style'] = 'filled'

            # Definir estados de aceitação
            if all(automata[i].nodes[state].get('accepting_state', False) for i, state in enumerate(composite_state)):
                parallel_automaton.nodes[composite_state_name]['shape'] = 'doublecircle'
                parallel_automaton.nodes[composite_state_name]['color'] = 'orange'
                parallel_automaton.nodes[composite_state_name]['accepting_state'] = True

        # Preparar argumentos para processamento paralelo
        args_list = [(composite_state, automata, events) for composite_state in all_states if
                     condition(composite_state)]

        # Processar transições em paralelo
        with Pool() as pool:
            results = pool.map(MultiGraphPlanner.process_composite_state, args_list)

        # Adicionar as transições ao autômato
        for transitions in results:
            for u, v, label in transitions:
                MultiGraphPlanner.AdicionaEdge(parallel_automaton, u, v, label)

        return parallel_automaton




    @staticmethod
    def get_start_and_accepting_states(G):
        """
        Retorna o estado inicial e o conjunto de estados de aceite de um grafo G.

        Parâmetros:
            G : nx.Graph, nx.DiGraph ou nx.MultiDiGraph
                O grafo que representa um autômato, com:
                - G.graph['start']: estado inicial (opcional)
                - Cada nó pode ter atributo 'accepting_state': True

        Retorna:
            (start_state, accepting_states)
            start_state: str ou None
            accepting_states: set de estados com 'accepting_state' = True
        """
        start_state = G.graph.get('start', None)

        accepting_states = {
            node for node, data in G.nodes(data=True)
            if data.get('accepting_state') == True
        }

        return start_state, accepting_states

    @staticmethod
    def imprimir_multidigraph(grafo):
        """
        Imprime os nós, arestas e atributos de um MultiDiGraph.

        Args:
            grafo (nx.MultiDiGraph): O grafo a ser impresso.
        """
        print("Nós do grafo:")
        for no, atributos in grafo.nodes(data=True):
            print(f"  {no}")

        print("\nArestas do grafo:")
        for origem, destino, atributos in grafo.edges(data=True):
            print(f"  {origem} -> {destino} : {atributos['label']}")
            #print(f" path = {atributos['path']}")


    @staticmethod
    def gerar_grafos_execucao(pontos_por_robo):
        """
        Para cada ponto cria um grafo com:
            - Nó inicial: 'n-<num_ponto>'
            - Nó final: 'Rx-n-<num_ponto>'
            - Transições: 'R1-n-<num_ponto>' ou 'R2-n-<num_ponto>' dependendo do robô.
            - Se um ponto estiver listado para mais de um robô, múltiplas transições saem do nó inicial.

        Retorna:
            Lista de grafos NX, um para cada ponto.
        """


        pontos_unicos = set()
        for pontos in pontos_por_robo.values():
            pontos_unicos.update(pontos)

        lista_grafos = []

        for ponto in pontos_unicos:
            # Extrai a parte numérica final (ex.: b_busip4.2502 -> 2502)
            num_ponto = ponto.rsplit('.', 1)[-1]

            # Cria grafo direcionado (DiGraph)
            G_ponto = nx.DiGraph()

            # Cria os nós inicial e final
            no_inicial = f"n-{num_ponto}"
            no_final = f"V-{num_ponto}"

            dfa = {
                'alphabet': set(),
                'states': {no_inicial, no_final},
                'start': no_inicial,
                'accepting_states': {no_final},
                'transitions': {}
            }

            # Para cada robô que contém esse ponto, cria uma transição
            for robo, pontos_robo in pontos_por_robo.items():
                if ponto in pontos_robo:
                    label_transicao = f"{robo}-{num_ponto}"
                    # Adiciona aresta inicial → final com label da transição
                    dfa['transitions'][(no_inicial, label_transicao)] = (no_final, 0)
                    dfa['alphabet'].add(label_transicao)

            lista_grafos.append(dfa)

        return lista_grafos


    @staticmethod
    def parse_numeric_suffix(node_name):
        """ Extrai a parte após o último '.' para usar como sufixo numérico. """
        if '.' in node_name:
            return node_name.rsplit('.', 1)[-1]
        return node_name  # se não houver '.', devolve a string inteira

    @staticmethod
    def graph_to_dfa_bidirectional(robot, dG, dstart=None, daccepting=None):
        """
        Transforma o grafo G em um dicionário no formato de DFA, com transições bidirecionais,
        mas em vez de (from, label) -> to, cada transição fica (from, label) -> (to, peso).

        Formato resultante do dfa:
        {
          'alphabet': set([...]),
          'states': set([...]),
          'start': <estado_inicial ou None>,
          'accepting_states': set([...]),
          'transitions': {
              (from, label): (to, peso),
              ...
          }
        }

        Regras de criação de transição:
          - Para cada aresta (u, v) em G, se o G[u][v] tiver 'weight', pegamos esse valor como peso.
            Caso contrário, usamos 1.0.
          - Criamos 2 transições:
            (u, "su_sv") => (v, w)
            (v, "robot_param_sv_su") => (u, w)
            onde su é a parte numérica do 'u', sv a parte numérica do 'v'.

        Parâmetros
        ----------
        G : networkx.Graph
            Grafo cujos nós podem ser strings como "ls_ip4.345" etc.
            Se G[u][v] tiver data["weight"], usaremos como peso.
        robot_param : str
            Prefixo para o label da transição "volta".
            Ex: "R1_667_345".
        start : opcional
            Nome de estado inicial.
        accepting : iterável opcional
            Conjunto de estados finais.

        Retorna
        -------
        dfa : dict
            Dicionário com chaves: 'alphabet', 'states', 'start', 'accepting_states', 'transitions'.
            Em 'transitions', a chave é (estado, label), e o valor é (destino, peso).
        """
        G = dG[robot]
        start = dstart[robot]
        accepting = daccepting[robot]

        dfa = {
            'alphabet': set(),
            'states': set([robot+":"+node for node in G.nodes()]),
            'start': robot+":"+start,
            'accepting_states': set([robot+":"+node for node in accepting]) if accepting else set(),
            'transitions': {}
        }

        # Percorre as arestas para criar transições
        # Se G for Graph, edges(data=True) retorna (u, v, data)
        for u, v, data in G.edges(data=True):
            # Extrair sufixos numéricos
            su = MultiGraphPlanner.parse_numeric_suffix(str(u))
            sv = MultiGraphPlanner.parse_numeric_suffix(str(v))

            # Construir labels
            label_uv = f"{robot}:{su}_{sv}"  # ex: "345_667"
            label_vu = f"{robot}:{sv}_{su}"  # ex: "R1_667_345"

            # Acha peso da aresta
            w = data.get("weight", 1.0)

            # Transição de u -> v
            dfa['transitions'][(robot+":"+u, label_uv)] = (robot+":"+v, w)
            dfa['alphabet'].add(label_uv)

            # Transição de v -> u
            dfa['transitions'][(robot+":"+v, label_vu)] = (robot+":"+u, w)
            dfa['alphabet'].add(label_vu)

        #adiciona a execucao das atividades da missao a custo zero
        for state in accepting:
            num_ponto = state.rsplit('.', 1)[-1]
            label_uv = f"{robot}-{num_ponto}"  # ex: "345_667"
            dfa['transitions'][(robot+":"+state, label_uv)] = (robot+":"+state, 0)
            dfa['alphabet'].add(label_uv)


        return dfa

    @staticmethod
    def update_automaton_graph(G, start_state=None, final_states=None,
                               transitions=None, diferenciados=None):
        """
        Recebe um grafo (G) já existente (ex: MultiDiGraph) e atualiza:
          - Estado inicial (start_state),
          - Estados finais (final_states),
          - Transições (transitions), no estilo (orig, label, peso opcional) => destino,
          - Estados diferenciados (diferenciados).

        Parâmetros:
        -----------
        G : Graph ou DiGraph ou MultiDiGraph (já criado)
        start_state : str (opcional)
            Nome do estado inicial. Se fornecido, G.graph['start'] = start_state
            e esse nó recebe alguns atributos (color, style, etc).
        final_states : iterável de strings (opcional)
            Lista ou conjunto de estados finais. Cada um receberá shape='doublecircle', color='orange', etc.
        transitions : lista ou dict (opcional)
            Se for lista, cada item é ((orig, symbol) ou (orig, symbol, weight), dest).
            Se for dict, chaves são (orig, symbol) ou (orig, symbol, weight), valor é o estado destino.
        diferenciados : tuple (opcional)
            (lista_estados, dict_atributos). Aplica esse dict_atributos a cada estado em lista_estados.

        Retorna:
        --------
        G : o próprio grafo, após as modificações.
        """

        # Se 'transitions' não for None, adiciona/atualiza arestas
        if transitions is not None:
            if isinstance(transitions, dict):
                # converte dict => lista de ((orig, symbol, [weight]), destino)
                transitions_items = list(transitions.items())
            else:
                # se já for lista, assumimos que seja [((orig, symbol, [weight]), dest), ...]
                transitions_items = transitions

            for key, target_state in transitions_items:
                if len(key) == 2:
                    (current_state, symbol) = key
                    nweight = 1
                elif len(key) == 3:
                    (current_state, symbol, nweight) = key
                else:
                    raise ValueError(f"Transição inválida ou formato inesperado: {key}")

                # Caso algum nó não exista, adicionamos silenciosamente
                if current_state not in G:
                    G.add_node(current_state)
                if target_state not in G:
                    G.add_node(target_state)

                # Adiciona aresta no estilo MultiDiGraph: (orig, dest, key=symbol, label=symbol, weight=...)
                # Se seu G for um Graph/DiGraph simples, esse 'key=symbol' não faz diferença.
                G.add_edge(current_state, target_state, key=symbol, label=symbol, weight=nweight)

        # Se final_states for fornecido, marcar tais estados
        if final_states is not None:
            for st in final_states:
                if st in G.nodes():
                    G.nodes[st]['accepting_state'] = True
                    G.nodes[st]['shape'] = 'doublecircle'
                    G.nodes[st]['color'] = 'orange'

        # Se diferenciados for fornecido, aplicar atributos
        if diferenciados is not None:
            # Exemplo de diferenciados: (["q2"], {"shape":"box","color":"red"})
            estados_dif, props = diferenciados
            for st in estados_dif:
                if st in G.nodes():
                    for k, v in props.items():
                        G.nodes[st][k] = v

        # Se start_state for fornecido, marcar no G.graph['start'] e ajustar atributos
        if start_state is not None:
            G.graph['start'] = start_state
            if start_state not in G:
                G.add_node(start_state)  # garante que o nó exista
            G.nodes[start_state]['style'] = 'filled'
            G.nodes[start_state]['color'] = 'lightgreen'
            G.nodes[start_state]['arrowhead'] = 'vee'

        return G

    @staticmethod
    def dict_to_nx_graph(grafo_mapa):
        """
        Converte um dicionário 'grafo_mapa' (states + transitions)
        em um grafo NetworkX não-direcionado, com os pesos em 'weight'.
        """
        G = nx.Graph()
        for state in grafo_mapa["states"]:
            G.add_node(state)
        for (orig, dst), (dist, _) in grafo_mapa["transitions"].items():
            G.add_edge(orig, dst, weight=dist)
        return G

    @staticmethod
    def find_nearest_node(G_loaded, tx, ty, metric: str = "L2"):
        import math
        """
        Encontra o nó mais próximo de (tx, ty) com base em G.nodes[n]['pos'] = (x, y).
        Retorna (nó, distancia).
        Se nenhum nó tiver 'pos', retornará (None, float('inf')).
        """
        # G_loaded pode ser (a) um dicionário { 'states', 'transitions'} ou (b) um nx.Graph
        if isinstance(G_loaded, dict) and "states" in G_loaded and "transitions" in G_loaded:
            # então converter para nx.Graph
            G = MultiGraphPlanner.dict_to_nx_graph(G_loaded)
        elif isinstance(G_loaded, nx.Graph):
            # já é grafo Nx
            G = G_loaded
        else:
            raise ValueError("Formato de graph7.json inesperado. Verifique seu pipeline.")

        # 2) Para cada nó, parsear o label (que é o nome do nó)
        #    Exemplo: se for "(-165.9766, -77.6645)" iremos extrair x=-165.9766, y=-77.6645
        for node in G.nodes():
            coords = MultiGraphPlanner.parse_label_to_xy(str(node))  # 'node' em string
            if coords is not None:
                G.nodes[node]["pos"] = coords

        nearest = None
        min_dist = float('inf')
        for node in G.nodes:
            if "pos" not in G.nodes[node]:
                continue  # Ignora nós sem atributo pos

            x_node, y_node = G.nodes[node]["pos"]
            dx = x_node - tx
            dy = y_node - ty

            if metric == "L1":
                d = abs(dx) + abs(dy)
            elif metric == "L2":
                d = math.hypot(dx, dy)
            elif metric == "Linf":
                d = max(abs(dx), abs(dy))
            else:
                raise ValueError(f"Métrica desconhecida: {metric}")

            if d < min_dist:
                min_dist = d
                nearest = node
                # best_pos = (nx, ny)
            # dist = math.hypot(x_node - tx, y_node - ty)
            # if dist < min_dist:
            #     min_dist = dist
            #     nearest = node

        label = G.nodes[nearest].get("label", str(nearest))
        return label, nearest, min_dist

    @staticmethod
    def parse_label_to_xy(label):
        """
        Tenta parsear o label do nó no formato:
          "(-165.9766, -77.6645)"
        e retornar (x, y) como floats. Retorna None se não conseguir.
        """
        pattern = r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)"
        match = re.match(pattern, label)
        if match:
            x_str = match.group(1)
            y_str = match.group(2)
            return float(x_str), float(y_str)
        else:
            return None

    @staticmethod
    def build_inspection_graph(start_point, inspection_points, grafo_mapa):
        """
        Cria um grafo NetworkX (G_robot) com nós = {start_point} U {inspection_points}.
        Para cada par de nós (u, v), calcula o menor caminho no grafo_mapa.
        Se esse caminho não tiver nenhum outro nó de {start_point} + inspection_points
        no meio (ou seja, excluindo u e v), então cria aresta (u, v) em G_robot,
        com peso igual à soma das distâncias do menor caminho.
        """
        # 1) Converter o grafo_mapa (dicionário) para um grafo NetworkX
        G_env = grafo_mapa

        # 2) Conjunto de todos os "estados" que nos interessam
        states_of_interest = set(inspection_points)

        if len(start_point)>0:
            for sp in start_point:
                states_of_interest.add(sp)

        # 3) Criar um grafo vazio para retornar
        G_robot = nx.DiGraph()

        # Adiciona as posições das missões (pontos de observação)
        for mission_point in states_of_interest:
            if mission_point in grafo_mapa and 'pos' in grafo_mapa.nodes[mission_point]:
                G_robot.add_node(mission_point, pos=grafo_mapa.nodes[mission_point]['pos'])
            else:
                print(f"[Aviso] Ponto de missão '{mission_point}' sem 'pos'. Definindo padrão (0,0).")
                G_robot.add_node(mission_point, pos=(0.0, 0.0))


        # Adicionar nós
        # for st in states_of_interest:
        #     G_robot.add_node(st)
        #     G_robot[st]['pos']=grafo_mapa[st]['pos']

        # 4) Vamos testar pares (u, v) usando combinações
        from itertools import combinations
        for u, v in combinations(states_of_interest, 2):
            # Tenta achar caminho mais curto no G_env
            try:
                path = nx.shortest_path(G_env, source=u, target=v, weight="weight")
                dist = nx.shortest_path_length(G_env, source=u, target=v, weight="weight")
            except nx.NetworkXNoPath:
                # Não existe caminho
                continue

            # Verifica se existe algum outro estado de interesse no meio do caminho
            # path[1:-1] = nós intermediários
            intermediarios = set(path[1:-1])
            if intermediarios.intersection(states_of_interest):
                continue

            # Caso não tenha nenhum estado de interesse no meio,
            # adicionamos a aresta com o peso (dist)
            G_robot.add_edge(u, v, weight=dist)
            G_robot.add_edge(v, u, weight=dist)
            su = MultiGraphPlanner.parse_numeric_suffix(str(u))
            sv = MultiGraphPlanner.parse_numeric_suffix(str(v))

            # (Opcional) se quiser guardar o caminho completo no atributo:
            G_robot[u][v]['label'] = f"{su}_{sv}"
            G_robot[u][v]['path'] = path

            G_robot[v][u]['label'] = f"{sv}_{su}"
            G_robot[v][u]['path'] = path

        return G_robot




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

    def save_optimal_plan_to_json(self, optimal_plan, file_path):
        """
        Salva o optimal_plan no formato:

        "R1": [
          {
            "mission": "b_busip4",
            "tasks": [
              {
                "point": "b_busip4.173.3.2478",
                "path": [...],
                "travel_time": ...,
                "execution_time": ...,
                "start_time": ...,
                "end_time": ...
              },
              {
                "point": "b_busip4.173.18.2493",
                "path": [...],
                ...
              }
            ]
          },
          {
            "mission": "b_busip7",
            "tasks": [
              ...
            ]
          }
          ...
        ],
        "R2": [...]

        Observações:
          - 'mission' = prefixo extraído antes do primeiro '.'.
          - 'point'   = o nome completo da missão.
          - 'path'    = lista de strings (nós percorridos).
          - tempos e path são float e lista, tudo pronto pra JSON.
        """

        # Dicionário de resultado final
        final_data = {}

        for robot, tasks in optimal_plan.items():
            # Agrupador interno: prefixo -> lista de sub-tarefas
            prefix_groups = {}

            for task in tasks:
                mission_full = task["mission"]  # ex: "b_busip4.173.3.2478"
                # Extrai prefixo antes do primeiro ponto
                if "." in mission_full:
                    mission_prefix = mission_full.split(".", 1)[0]
                else:
                    # Se não houver ponto, assume tudo como prefixo
                    mission_prefix = mission_full

                # Monta objeto do sub-item
                sub_item = {
                    "point": mission_full,
                    "path": task["path"],
                    "travel_time": float(task["travel_time"]),
                    "execution_time": float(task["execution_time"]),
                    "start_time": float(task["start_time"]),
                    "end_time": float(task["end_time"])
                }

                # Adiciona no grupo correspondente
                if mission_prefix not in prefix_groups:
                    prefix_groups[mission_prefix] = []
                prefix_groups[mission_prefix].append(sub_item)

            # Agora convertemos prefix_groups em uma lista:
            # [
            #   {
            #     "mission": "b_busip4",
            #     "tasks": [ { "point":..., "path":..., ... }, ... ]
            #   },
            #   ...
            # ]
            robot_missions = []
            for prefix, sub_items in prefix_groups.items():
                robot_missions.append({
                    "mission": prefix,
                    "tasks": sub_items
                })

            final_data[robot] = robot_missions

        # Grava em JSON
        with open(file_path, "w") as f:
            json.dump(final_data, f, indent=4)

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





