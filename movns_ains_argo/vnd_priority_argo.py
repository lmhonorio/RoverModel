import random
import numpy as np
from movns_ains_argo import solution_priority_argo as solution_priority


def dominates_pareto_with_tolerance(solution_metrics, other_metrics, tolerances=1.02, improvement_factor=1.02):
    """
    Verifica se uma solução domina outra no sentido de Pareto, considerando tolerâncias para piora leve em objetivos.

    Args:
        solution_metrics (list or np.ndarray): Métricas da solução candidata.
        other_metrics (list or np.ndarray): Métricas da solução a ser comparada.
        tolerances (list or np.ndarray, optional): Valores de tolerância para piora em cada objetivo. 
                                                   Se None, considera tolerância zero.
        improvement_factor (float, optional): Fator pelo qual um objetivo deve ser significativamente melhor.

    Returns:
        bool: True se `solution_metrics` domina `other_metrics` com tolerância, False caso contrário.
    """
    solution_metrics = np.array(solution_metrics)
    other_metrics = np.array(other_metrics)
    if tolerances is None:
        tolerances = np.zeros_like(solution_metrics)

    # Verifica se é "não pior" considerando tolerâncias
    is_not_worse = np.all(solution_metrics <= other_metrics*tolerances)

    # Verifica se é significativamente melhor em pelo menos um objetivo
    is_significantly_better = np.any(solution_metrics < other_metrics / improvement_factor)

    # Se for não pior considerando tolerâncias e significativamente melhor em pelo menos um, então domina
    return is_not_worse and is_significantly_better



def swap_intra_robot_random(solution, robots, num_robots, tolerance=0.98):
    """
    Realiza um swap aleatório em duas tarefas alocadas ao mesmo robô.
    Verifica se a solução após o swap melhora em relação à atual.

    Args:
        allocations (list): Vetores de alocações (vetor de vetores).
        robots (list): Lista de robôs.
        tolerance (float): Tolerância para aceitar uma piora.

    Returns:
        allocations, bool: Vetores de alocações atualizados e flag indicando se houve melhoria.
    """
    # Selecionar um robô aleatoriamente
    robot_idx = random.randint(0, num_robots - 1)
    # robot_tasks = allocations[robot_idx]
    robot_tasks = solution.allocations[robot_idx]  # Trabalhar diretamente com as alocações do robô

    # Se o robô não tiver ao menos duas tarefas, não há o que trocar
    if len(robot_tasks) < 2:
        return solution, False

    # Selecionar duas tarefas aleatórias para troca
    i, j = random.sample(range(len(robot_tasks)), 2)

    # Função auxiliar para obter a distância com bordas
    def get_distance(before_task, current_task, after_task):
        distance = 0
        if before_task is not None:
            _, cost = solution.cache_astar.get_path(before_task.exit_point["label"], current_task.entry_point["label"])
            distance += cost
        else:
            _, cost = solution.cache_astar.get_path(robots[robot_idx].initial_position, current_task.entry_point["label"])
            distance += cost
        if after_task is not None:
            _, cost = solution.cache_astar.get_path(current_task.exit_point["label"], after_task.entry_point["label"])
            distance += cost
        else:
            _, cost = solution.cache_astar.get_path(robots[robot_idx].initial_position, current_task.exit_point["label"])
            distance += cost
        return distance

    # Tarefas antes e depois
    task_before_i = robot_tasks[i - 1] if i > 0 else None
    task_after_i = robot_tasks[i + 1] if i < len(robot_tasks) - 1 else None
    task_before_j = robot_tasks[j - 1] if j > 0 else None
    task_after_j = robot_tasks[j + 1] if j < len(robot_tasks) - 1 else None

    # Distância antes do swap
    current_distance = (
        get_distance(task_before_i, robot_tasks[i], task_after_i) +
        get_distance(task_before_j, robot_tasks[j], task_after_j)
    )

    # Fazer o swap
    robot_tasks[i], robot_tasks[j] = robot_tasks[j], robot_tasks[i]

    # Tarefas antes e depois para o estado pós-swap
    new_task_before_i = robot_tasks[i - 1] if i > 0 else None
    new_task_after_i = robot_tasks[i + 1] if i < len(robot_tasks) - 1 else None
    new_task_before_j = robot_tasks[j - 1] if j > 0 else None
    new_task_after_j = robot_tasks[j + 1] if j < len(robot_tasks) - 1 else None

    # Distância após o swap
    new_distance = (
        get_distance(new_task_before_i, robot_tasks[i], new_task_after_i) +
        get_distance(new_task_before_j, robot_tasks[j], new_task_after_j)
    )


    # Avaliar impacto
    distance_gain = current_distance - new_distance


    if (distance_gain > 0):
        # Atualizar alocação na solução se a troca for benéfica
        solution.allocations[robot_idx] = robot_tasks[:]
        solution.calculate_metrics()
        return solution, True
    

    # Atualizar as alocações
    robot_tasks[i], robot_tasks[j] = robot_tasks[j], robot_tasks[i]
    return solution, False


def swap_with_closest_task_single_robot(solution, robots, num_robots, tolerance=0.98):
    """
    Percorre todas as tarefas de um único robô e realiza swaps com as tarefas mais próximas.

    Args:
        solution (Solution): Objeto principal da solução contendo alocações.
        robots (list): Lista de robôs.
        distance_matrix (np.ndarray): Matriz de distâncias entre tarefas.
        tolerance (float): Critério de aceitação para uma piora.

    Returns:
        Solution, bool: Solução modificada e flag indicando se houve melhoria.
    """

    # Selecionar um robô aleatoriamente
    robot_idx = random.randint(0, num_robots - 1)
    # robot = robots[robot_idx]
    robot_tasks = solution.allocations[robot_idx]  # Trabalhar diretamente com as alocações do robô
    # Se o robô não tiver ao menos duas tarefas, não há o que trocar
    if len(robot_tasks) < 2:
        return solution, False
    # Função auxiliar para obter a distância com bordas
    def get_distance(before_task, current_task, after_task):
        distance = 0
        if before_task is not None:
            _, cost = solution.cache_astar.get_path(before_task.exit_point["label"], current_task.entry_point["label"])
            distance += cost
        else:
            _, cost = solution.cache_astar.get_path(robots[robot_idx].initial_position, current_task.entry_point["label"])
            distance += cost
        if after_task is not None:
            _, cost = solution.cache_astar.get_path(current_task.exit_point["label"], after_task.entry_point["label"])
            distance += cost
        else:
            _, cost = solution.cache_astar.get_path(robots[robot_idx].initial_position, current_task.exit_point["label"])
            distance += cost
        return distance

    # Iterar sobre todas as tarefas do robô
    for task_idx, selected_task in enumerate(robot_tasks):
        closest_task_idx = None
        min_distance = float('inf')

        for i, task in enumerate(robot_tasks):
            if i != task_idx:
                _, distance = solution.cache_astar.get_path(selected_task.exit_point["label"], task.entry_point["label"])
                if distance < min_distance:
                    min_distance = distance
                    closest_task_idx = i

        if closest_task_idx is None:
            continue

        # Tarefas antes e depois do índice atual e do mais próximo
        task_before_current = robot_tasks[task_idx - 1] if task_idx > 0 else None
        task_after_current = robot_tasks[task_idx + 1] if task_idx < len(robot_tasks) - 1 else None
        task_before_closest = robot_tasks[closest_task_idx - 1] if closest_task_idx > 0 else None
        task_after_closest = robot_tasks[closest_task_idx + 1] if closest_task_idx < len(robot_tasks) - 1 else None

        # Distância antes do swap
        current_distance = (
            get_distance(task_before_current, robot_tasks[task_idx], task_after_current) +
            get_distance(task_before_closest, robot_tasks[closest_task_idx], task_after_closest)
        )

        # Fazer o swap
        robot_tasks[task_idx], robot_tasks[closest_task_idx] = robot_tasks[closest_task_idx], robot_tasks[task_idx]

        # Tarefas antes e depois do índice atual e do mais próximo
        new_task_before_current = robot_tasks[task_idx - 1] if task_idx > 0 else None
        new_task_after_current = robot_tasks[task_idx + 1] if task_idx < len(robot_tasks) - 1 else None
        new_task_before_closest = robot_tasks[closest_task_idx - 1] if closest_task_idx > 0 else None
        new_task_after_closest = robot_tasks[closest_task_idx + 1] if closest_task_idx < len(robot_tasks) - 1 else None

        # Distância após o swap
        new_distance = (
            get_distance(new_task_before_current, robot_tasks[task_idx], new_task_after_current) +
            get_distance(new_task_before_closest, robot_tasks[closest_task_idx], new_task_after_closest)
        )


        

        # Avaliar impacto
        distance_gain = current_distance - new_distance


        if (distance_gain > 0):
            # Atualizar alocação na solução se a troca for benéfica
            solution.allocations[robot_idx] = robot_tasks[:]
            solution.calculate_metrics()
            return solution, True

        # Reverter o swap caso não melhore
        robot_tasks[task_idx], robot_tasks[closest_task_idx] = robot_tasks[closest_task_idx], robot_tasks[task_idx]

    return solution, False

def swap_with_farthest_task_single_robot(solution, robots, num_robots, tolerance=0.98):
    """
    Percorre todas as tarefas de um único robô e realiza swaps com as tarefas mais distantes.

    Args:
        solution (Solution): Objeto principal da solução contendo alocações.
        num_robots (int): Número total de robôs.
        distance_matrix (np.ndarray): Matriz de distâncias entre tarefas.
        tolerance (float): Critério de aceitação para uma piora.

    Returns:
        Solution, bool: Solução modificada e flag indicando se houve melhoria.
    """
    # Selecionar um robô aleatoriamente
    robot_idx = random.randint(0, num_robots - 1)
    robot_tasks = solution.allocations[robot_idx]  # Trabalhar diretamente com as alocações do robô

    # Se o robô não tiver ao menos duas tarefas, não há o que trocar
    if len(robot_tasks) < 2:
        return solution, False

    # Função auxiliar para obter a distância com bordas
    def get_distance(before_task, current_task, after_task):
        distance = 0
        if before_task is not None:
            _, cost = solution.cache_astar.get_path(before_task.exit_point["label"], current_task.entry_point["label"])
            distance += cost
        else:
            _, cost = solution.cache_astar.get_path(robots[robot_idx].initial_position, current_task.entry_point["label"])
            distance += cost
        if after_task is not None:
            _, cost = solution.cache_astar.get_path(current_task.exit_point["label"], after_task.entry_point["label"])
            distance += cost
        else:
            _, cost = solution.cache_astar.get_path(robots[robot_idx].initial_position, current_task.exit_point["label"])
            distance += cost
        return distance

    # Iterar sobre todas as tarefas do robô
    for task_idx, selected_task in enumerate(robot_tasks):
        farthest_task_idx = None
        max_distance = float('-inf')

        # Encontrar a tarefa mais distante da tarefa selecionada
        for i, task in enumerate(robot_tasks):
            if i != task_idx:
                _, distance = solution.cache_astar.get_path(selected_task.exit_point["label"], task.entry_point["label"])
                if distance > max_distance:
                    max_distance = distance
                    farthest_task_idx = i

        if farthest_task_idx is None:
            continue

        # Tarefas antes e depois do índice atual e do mais distante
        task_before_current = robot_tasks[task_idx - 1] if task_idx > 0 else None
        task_after_current = robot_tasks[task_idx + 1] if task_idx < len(robot_tasks) - 1 else None
        task_before_farthest = robot_tasks[farthest_task_idx - 1] if farthest_task_idx > 0 else None
        task_after_farthest = robot_tasks[farthest_task_idx + 1] if farthest_task_idx < len(robot_tasks) - 1 else None

        # Distância antes do swap
        current_distance = (
            get_distance(task_before_current, robot_tasks[task_idx], task_after_current) +
            get_distance(task_before_farthest, robot_tasks[farthest_task_idx], task_after_farthest)
        )

        # Fazer o swap
        robot_tasks[task_idx], robot_tasks[farthest_task_idx] = robot_tasks[farthest_task_idx], robot_tasks[task_idx]

        # Tarefas antes e depois do índice atual e do mais distante após o swap
        new_task_before_current = robot_tasks[task_idx - 1] if task_idx > 0 else None
        new_task_after_current = robot_tasks[task_idx + 1] if task_idx < len(robot_tasks) - 1 else None
        new_task_before_farthest = robot_tasks[farthest_task_idx - 1] if farthest_task_idx > 0 else None
        new_task_after_farthest = robot_tasks[farthest_task_idx + 1] if farthest_task_idx < len(robot_tasks) - 1 else None

        # Distância após o swap
        new_distance = (
            get_distance(new_task_before_current, robot_tasks[task_idx], new_task_after_current) +
            get_distance(new_task_before_farthest, robot_tasks[farthest_task_idx], new_task_after_farthest)
        )

        # Avaliar impacto
        distance_gain = current_distance - new_distance


        if (distance_gain > 0):
            # Atualizar alocação na solução se a troca for benéfica
            solution.allocations[robot_idx] = robot_tasks[:]
            solution.calculate_metrics()
            return solution, True

        # Reverter o swap caso não melhore
        robot_tasks[task_idx], robot_tasks[farthest_task_idx] = robot_tasks[farthest_task_idx], robot_tasks[task_idx]

    return solution, False

def swap_inter_robot(solution, num_robots, tolerance=1.02, improvement_threshold=0.02):
    """
    Realiza um swap inter-robôs entre tarefas baseando-se na proximidade dos clusters.

    Args:
        solution (Solution): Objeto principal da solução contendo alocações.
        num_robots (int): Número total de robôs.
        cache_astar: Objeto com método get_path para consulta de caminhos.
        tolerance (float): Tolerância para aceitar uma piora.
        improvement_threshold (float): Percentual mínimo de melhora em pelo menos um objetivo para aceitar piora em outro.

    Returns:
        Solution, bool: Solução modificada e flag indicando se houve melhoria.
    """
    # Selecionar dois robôs aleatoriamente
    robot_1_idx, robot_2_idx = random.sample(range(num_robots), 2)
    robot_1_tasks = solution.allocations[robot_1_idx]
    robot_2_tasks = solution.allocations[robot_2_idx]

    if not robot_1_tasks or not robot_2_tasks:
        return solution, False

    # Tarefa de robot_1 mais próxima de qualquer tarefa de robot_2
    task_1 = min(robot_1_tasks, key=lambda t1: min(
        solution.cache_astar.get_path(t1.exit_point["label"], t2.entry_point["label"])[1] for t2 in robot_2_tasks
    ))

    # Tarefa de robot_2 mais próxima de qualquer tarefa de robot_1
    task_2 = min(robot_2_tasks, key=lambda t2: min(
        solution.cache_astar.get_path(t2.exit_point["label"], t1.entry_point["label"])[1] for t1 in robot_1_tasks
    ))

    index_1 = robot_1_tasks.index(task_1)
    index_2 = robot_2_tasks.index(task_2)

    # Salvar métricas anteriores
    previous_metrics = np.array(solution.metrics)

    # Fazer o swap
    robot_1_tasks[index_1], robot_2_tasks[index_2] = robot_2_tasks[index_2], robot_1_tasks[index_1]

    # Recalcular métricas com base no cache_astar
    solution.calculate_metrics()

    current_metrics = np.array(solution.metrics)

    if dominates_pareto_with_tolerance(current_metrics, previous_metrics):
        return solution, True

    # Reverter se não melhorar
    robot_1_tasks[index_1], robot_2_tasks[index_2] = robot_2_tasks[index_2], robot_1_tasks[index_1]
    solution.calculate_metrics()
    return solution, False


def move_task_between_robots(solution, num_robots, tolerance=0.99):
    """
    Move uma tarefa de um robô para outro, considerando proximidade de clusters com base no cache A*.

    Args:
        solution (Solution): Objeto principal da solução contendo alocações.
        num_robots (int): Número total de robôs.
        cache_astar: Objeto com método get_path para consulta de caminhos.
        tolerance (float): Critério de aceitação para uma piora.

    Returns:
        Solution, bool: Solução modificada e flag indicando se houve melhoria.
    """
    robot_from_idx, robot_to_idx = random.sample(range(num_robots), 2)
    robot_from_tasks = solution.allocations[robot_from_idx]
    robot_to_tasks = solution.allocations[robot_to_idx]

    if not robot_from_tasks:
        return solution, False

    # Seleciona a task a ser movida — aquela mais próxima de alguma task do robô de destino
    task_to_move = min(robot_from_tasks, key=lambda task_from: 
                       min(solution.cache_astar.get_path(task_from.exit_point["label"], task_to.entry_point["label"])[1] 
                           for task_to in robot_to_tasks)) \
        if robot_to_tasks else robot_from_tasks[0]

    original_index_from = robot_from_tasks.index(task_to_move)

    # Determinar onde inserir a tarefa no robô de destino
    if robot_to_tasks:
        closest_task_to = min(robot_to_tasks, key=lambda task_to: 
                              solution.cache_astar.get_path(task_to.exit_point["label"], task_to_move.entry_point["label"])[1])
        insert_index = robot_to_tasks.index(closest_task_to) + 1
    else:
        insert_index = 0

    # Realiza o movimento
    robot_from_tasks.remove(task_to_move)
    robot_to_tasks.insert(insert_index, task_to_move)

    # Atualiza as métricas
    previous_metrics = np.array(solution.metrics)
    solution.calculate_metrics()
    current_metrics = np.array(solution.metrics)

    if np.all(current_metrics <= previous_metrics * tolerance):
        # Movimento aceito
        solution.allocations[robot_from_idx] = robot_from_tasks
        solution.allocations[robot_to_idx] = robot_to_tasks
        return solution, True

    # Reverter caso não tenha melhorado
    robot_to_tasks.pop(insert_index)
    robot_from_tasks.insert(original_index_from, task_to_move)
    solution.allocations[robot_from_idx] = robot_from_tasks
    solution.allocations[robot_to_idx] = robot_to_tasks
    solution.calculate_metrics()
    return solution, False

def bring_closer_task_forward(solution, robots, num_robots):
    """
    Identifica o trajeto mais longo entre tarefas consecutivas (ou trajeto inicial)
    e tenta trazer uma tarefa posterior (mais próxima) para frente para reduzir o trajeto longo.

    Args:
        solution (Solution): Solução atual com alocações.
        num_robots (int): Número de robôs.

    Returns:
        Solution, bool: Solução modificada e flag indicando se houve melhoria.
    """
    for robot_idx in range(num_robots):
        robot_tasks = solution.allocations[robot_idx]

        if len(robot_tasks) < 2:
            continue

        max_distance = -1
        max_idx = None
        trajeto_inicial = False

        initial_pos = robots[robot_idx].initial_position
        _, dist_inicial = solution.cache_astar.get_path(initial_pos, robot_tasks[0].entry_point["label"])

        if dist_inicial > max_distance:
            max_distance = dist_inicial
            max_idx = -1
            trajeto_inicial = True

        for i in range(len(robot_tasks) - 1):
            from_task = robot_tasks[i]
            to_task = robot_tasks[i + 1]
            _, dist = solution.cache_astar.get_path(from_task.exit_point["label"], to_task.entry_point["label"])
            if dist > max_distance:
                max_distance = dist
                max_idx = i
                trajeto_inicial = False

        best_improvement = 0
        best_j = None
        reference_label = initial_pos if trajeto_inicial else robot_tasks[max_idx].exit_point["label"]
        start_search_idx = 1 if trajeto_inicial else (max_idx + 2)

        for j in range(start_search_idx, len(robot_tasks)):
            candidate_task = robot_tasks[j]
            _, new_dist = solution.cache_astar.get_path(reference_label, candidate_task.entry_point["label"])
            improvement = max_distance - new_dist
            if improvement > best_improvement:
                best_improvement = improvement
                best_j = j

        if best_j is not None:
            previous_metrics = np.array(solution.metrics)

            candidate_task = robot_tasks.pop(best_j)
            insert_idx = 0 if trajeto_inicial else (max_idx + 1)
            robot_tasks.insert(insert_idx, candidate_task)

            solution.calculate_metrics()
            current_metrics = np.array(solution.metrics)

            if dominates_pareto_with_tolerance(current_metrics, previous_metrics):
                return solution, True

            # Reverter
            robot_tasks.pop(insert_idx)
            robot_tasks.insert(best_j, candidate_task)
            solution.calculate_metrics()

    return solution, False

def push_farther_task_backwards(solution, robots, num_robots):
    """
    Identifica o trajeto mais curto entre duas tarefas consecutivas de um robô
    e tenta empurrar uma tarefa mais distante (posterior) para ser executada logo após,
    aumentando a eficiência ao evitar agrupamentos muito apertados de tarefas.

    Args:
        solution (Solution): Solução atual com alocações.
        num_robots (int): Número de robôs.

    Returns:
        Solution, bool: Solução modificada e flag indicando se houve melhoria.
    """
    improved = False

    for robot_idx in range(num_robots):
        robot_tasks = solution.allocations[robot_idx]
        if len(robot_tasks) < 3:
            continue  # Precisa de ao menos 3 tarefas para essa operação fazer sentido

        min_distance = float('inf')
        min_idx = None

        # Encontra o menor trajeto entre duas tarefas consecutivas (ou início -> primeira tarefa)
        for i in range(-1, len(robot_tasks) - 1):
            if i == -1:
                from_label = robots[robot_idx].initial_position
                to_task = robot_tasks[0]
            else:
                from_label = robot_tasks[i].exit_point["label"]
                to_task = robot_tasks[i + 1]
            _, dist = solution.cache_astar.get_path(from_label, to_task.entry_point["label"])
            if dist < min_distance:
                min_distance = dist
                min_idx = i

        if min_idx is None:
            continue

        from_label = robots[robot_idx].initial_position if min_idx == -1 else robot_tasks[min_idx].exit_point["label"]

        best_improvement = 0
        best_j = None

        for j in range(min_idx + 2, len(robot_tasks)):
            candidate_task = robot_tasks[j]
            _, new_dist = solution.cache_astar.get_path(from_label, candidate_task.entry_point["label"])
            improvement = new_dist - min_distance

            if improvement > best_improvement:
                best_improvement = improvement
                best_j = j

        if best_j is not None:
            previous_metrics = np.array(solution.metrics)

            candidate_task = robot_tasks.pop(best_j)
            insert_position = min_idx + 1
            robot_tasks.insert(insert_position, candidate_task)

            solution.calculate_metrics()
            current_metrics = np.array(solution.metrics)

            if dominates_pareto_with_tolerance(current_metrics, previous_metrics):
                improved = True
                return solution, True
            else:
                # Reverter se não melhorar
                robot_tasks.pop(insert_position)
                robot_tasks.insert(best_j, candidate_task)
                solution.calculate_metrics()

    return solution, improved

def fill_long_gap_with_external_task(solution, robots, num_robots):
    for robot_idx in range(num_robots):
        robot_tasks = solution.allocations[robot_idx]
        if len(robot_tasks) < 2:
            continue

        # 1. Identificar maior lacuna (incluindo inicial)
        max_distance = -1
        max_idx = None
        initial_pos = robots[robot_idx].initial_position
        _, dist_inicial = solution.cache_astar.get_path(initial_pos, robot_tasks[0].entry_point["label"])
        if dist_inicial > max_distance:
            max_distance = dist_inicial
            max_idx = -1

        for i in range(len(robot_tasks) - 1):
            _, dist = solution.cache_astar.get_path(robot_tasks[i].exit_point["label"], robot_tasks[i + 1].entry_point["label"])
            if dist > max_distance:
                max_distance = dist
                max_idx = i

        # 2. Buscar tarefa de outro robô próxima dessa lacuna
        if max_idx == -1:
            reference_label = initial_pos
        else:
            reference_label = robot_tasks[max_idx].exit_point["label"]

        best_task = None
        best_from_idx = None
        best_robot_j = None
        min_dist = float("inf")

        for other_robot_idx in range(num_robots):
            if other_robot_idx == robot_idx:
                continue

            other_tasks = solution.allocations[other_robot_idx]
            for task in other_tasks:
                _, dist = solution.cache_astar.get_path(reference_label, task.entry_point["label"])
                if dist < min_dist:
                    min_dist = dist
                    best_task = task
                    best_from_idx = other_tasks.index(task)
                    best_robot_j = other_robot_idx

        # 3. Tentar mover e validar com dominates_pareto_with_tolerance
        if best_task is not None:
            prev_metrics = np.array(solution.metrics)

            # Remover do outro robô
            solution.allocations[best_robot_j].pop(best_from_idx)
            # Inserir no robô com a lacuna
            insert_idx = 0 if max_idx == -1 else max_idx + 1
            solution.allocations[robot_idx].insert(insert_idx, best_task)

            solution.calculate_metrics()
            current_metrics = np.array(solution.metrics)

            if dominates_pareto_with_tolerance(current_metrics, prev_metrics):
                return solution, True

            # Reverter
            solution.allocations[robot_idx].pop(insert_idx)
            solution.allocations[best_robot_j].insert(best_from_idx, best_task)
            solution.calculate_metrics()

    return solution, False




def apply_vnd(solution, robots, cache_astar):
    """
    Aplica o VND diretamente no objeto Solution.

    Args:
        solution (Solution): Objeto principal da solução.
        robots (list): Lista de robôs.
        distance_matrix (np.ndarray): Matriz de distâncias entre tarefas.

    Returns:
        Solution: Solução atualizada após aplicar o VND.
    """
    improved = True
    tolerance = 0.98
    initial_position = (150, 150)

    while improved:
        improved = False
        for operation in ['swap_with_closest_task_single_robot', 'swap_intra_robot_random', 'swap_with_farthest_task_single_robot', 'swap_inter_robot', 'move_task_between_robots']:
        # for operation in ['swap_with_closest_task_single_robot', 'swap_with_farthest_task_single_robot', 'swap_intra_robot_random']:
            if operation == 'swap_with_closest_task_single_robot':
                solution, improvement = swap_with_closest_task_single_robot(solution, robots, initial_position, tolerance)
                if improvement:
                    improved = True
                    break
            elif operation == 'swap_with_farthest_task_single_robot':
                solution, improvement = swap_with_farthest_task_single_robot(solution, robots, initial_position, tolerance)
                if improvement:
                    improved = True
                    break
            elif operation == 'swap_intra_robot_random':
                solution, improvement = swap_intra_robot_random(solution, robots, initial_position, tolerance)
                if improvement:
                    improved = True
                    break
            elif operation == 'swap_inter_robot':
                solution, improvement = swap_inter_robot(solution, robots, initial_position, tolerance)
                if improvement:
                    improved = True
                    break
            elif operation == 'move_task_between_robots':
                solution, improvement = move_task_between_robots(solution, robots, initial_position, tolerance)
                if improvement:
                    improved = True
                    break

    return solution

def apply_vnd_movns(solution, robots, neighborhood_stats=None):
    """
    Aplica o VND diretamente no objeto Solution.

    Args:
        solution (Solution): Objeto principal da solução.
        robots (list): Lista de robôs.
        distance_matrix (np.ndarray): Matriz de distâncias entre tarefas.

    Returns:
        Solution: Solução atualizada após aplicar o VND.
    """
    improved = True
    improvements = 0
    tolerance = 0.99
    num_robots = len(robots)

    while improved:
        # print(f"Antes: Melhor distância: {solution.distance}, Melhor tempo: {solution.time}, Melhor max_priority_time: {solution.max_priority_time}")
        improved = False
        for operation in ['swap_with_closest_task_single_robot', 'swap_with_farthest_task_single_robot', 'swap_inter_robot', 'bring_closer_task_forward', 'fill_long_gap_with_external_task']:
            if neighborhood_stats is not None:
                neighborhood_stats[operation]['calls'] += 1
        # for operation in ['swap_with_closest_task_single_robot', 'swap_with_farthest_task_single_robot', 'swap_intra_robot_random']:
            if operation == 'swap_with_closest_task_single_robot':
                solution, improvement = swap_with_closest_task_single_robot(solution, robots, num_robots)
                if improvement:
                    improved = True
                    improvements += 1
                    if neighborhood_stats is not None:
                        neighborhood_stats[operation]['successes'] += 1
                    break
            elif operation == 'swap_with_farthest_task_single_robot':
                solution, improvement = swap_with_farthest_task_single_robot(solution, robots, num_robots)
                if improvement:
                    improved = True
                    improvements += 1
                    if neighborhood_stats is not None:
                        neighborhood_stats[operation]['successes'] += 1
                    break
            elif operation == 'push_farther_task_backwards':
                solution, improvement = push_farther_task_backwards(solution, robots, num_robots)
                if improvement:
                    improved = True
                    improvements += 1
                    if neighborhood_stats is not None:
                        neighborhood_stats[operation]['successes'] += 1
                    break
            elif operation == 'swap_inter_robot':
                solution, improvement = swap_inter_robot(solution, num_robots, tolerance)
                if improvement:
                    improved = True
                    improvements += 1
                    if neighborhood_stats is not None:
                        neighborhood_stats[operation]['successes'] += 1
                    break
            elif operation == 'bring_closer_task_forward':
                solution, improvement = bring_closer_task_forward(solution, robots, num_robots)
                if improvement:
                    improved = True
                    improvements += 1
                    if neighborhood_stats is not None:
                        neighborhood_stats[operation]['successes'] += 1
                    break
            elif operation == 'fill_long_gap_with_external_task':
                solution, improvement = fill_long_gap_with_external_task(solution, robots, num_robots)
                if improvement:
                    improved = True
                    improvements += 1
                    if neighborhood_stats is not None:
                        neighborhood_stats[operation]['successes'] += 1
                    break

    # print(f"Depois: Melhor distância: {solution.distance}, Melhor tempo: {solution.time}, Melhor max_priority_time: {solution.max_priority_time}")
    return solution, improvements


def apply_vnd_movns_ains(solution, robots, neighborhood_stats=None):
    """
    Aplica vizinhanças de forma aleatória e adaptativa. Escolhe uma vizinhança aleatoriamente
    e aplica até que não haja mais melhorias. Atualiza estatísticas de uso das vizinhanças.

    Args:
        solution (Solution): Solução inicial.
        robots (list): Lista de robôs.
        initial_positions (list): Lista das posições iniciais dos robôs.
        neighborhood_stats (dict): Dicionário para registrar chamadas e sucessos.

    Returns:
        Solution: Solução atualizada após o processo.
        int: Número de melhorias realizadas.
    """
    improvements = 0
    tolerance = 0.99
    num_robots = len(robots)

    neighborhoods = [
        'swap_with_closest_task_single_robot',
        'swap_with_farthest_task_single_robot',
        'swap_inter_robot',
        'bring_closer_task_forward',
        'fill_long_gap_with_external_task'
    ]

    while True:
        operation = random.choice(neighborhoods)
        improvement = False

        if neighborhood_stats is not None:
            neighborhood_stats[operation]['calls'] += 1

        if operation == 'swap_with_closest_task_single_robot':
            solution, improvement = swap_with_closest_task_single_robot(solution, robots, num_robots)
        elif operation == 'swap_with_farthest_task_single_robot':
            solution, improvement = swap_with_farthest_task_single_robot(solution, robots, num_robots)
        elif operation == 'swap_inter_robot':
            solution, improvement = swap_inter_robot(solution, num_robots, tolerance)
        elif operation == 'move_task_between_robots':
            solution, improvement = move_task_between_robots(solution, num_robots, tolerance)
        elif operation == 'bring_closer_task_forward':
            solution, improvement = bring_closer_task_forward(solution, robots, num_robots)
        elif operation == 'push_farther_task_backwards':
            solution, improvement = push_farther_task_backwards(solution, robots, num_robots)
        elif operation == 'fill_long_gap_with_external_task':
            solution, improvement = fill_long_gap_with_external_task(solution, robots, num_robots)

        if improvement:
            improvements += 1
            if neighborhood_stats is not None:
                neighborhood_stats[operation]['successes'] += 1
        else:
            break

    return solution, improvements





def apply_best_neighborhood(solution, robots, neighborhood_stats):
    # print("rodei best_neighborhood")
    """
    Aplica a vizinhança que causa a maior melhoria em cada iteração.

    Args:
        solution (Solution): Solução inicial.
        robots (list): Lista de robôs.
        distance_matrix (np.ndarray): Matriz de distâncias entre tarefas.

    Returns:
        Solution: Solução final após aplicar a estratégia de Best Neighborhood Descent.
        int: Número de melhorias realizadas.
    """
    improvements = 0
    tolerance = 0.99
    num_robots = len(robots)

    # Lista de operações de vizinhança disponíveis
    neighborhoods = [
        'swap_with_closest_task_single_robot',
        'swap_with_farthest_task_single_robot',
        'swap_inter_robot',
        'bring_closer_task_forward',
        'fill_long_gap_with_external_task'
    ]

    improved = True
    while improved:
        improved = False
        best_solution = None
        best_improvement = float('inf')
        best_operation = None

        # Avaliar todas as vizinhanças
        for operation in neighborhoods:
            temp_solution = solution.copy()  # Cria uma cópia da solução atual
            improvement = False

            if neighborhood_stats is not None:
                neighborhood_stats[operation]['calls'] += 1

            # Aplica a vizinhança
            if operation == 'swap_with_closest_task_single_robot':
                temp_solution, improvement = swap_with_closest_task_single_robot(temp_solution, robots, num_robots)

            elif operation == 'swap_with_farthest_task_single_robot':
                temp_solution, improvement = swap_with_farthest_task_single_robot(temp_solution, robots, num_robots)

            elif operation == 'swap_intra_robot_random':
                temp_solution, improvement = swap_intra_robot_random(temp_solution, robots, num_robots)

            elif operation == 'swap_inter_robot':
                temp_solution, improvement = swap_inter_robot(temp_solution, num_robots)

            elif operation == 'move_task_between_robots':
                temp_solution, improvement = move_task_between_robots(temp_solution, num_robots)

            elif operation == 'push_farther_task_backwards':
                temp_solution, improvement = push_farther_task_backwards(temp_solution, robots, num_robots)

            elif operation == 'fill_long_gap_with_external_task':
                temp_solution, improvement = fill_long_gap_with_external_task(temp_solution, robots, num_robots)


            # Verifica se houve melhoria e se é a melhor até agora
            if improvement and temp_solution.get_improvement_metric() < best_improvement:
                best_solution = solution_priority.Solution(temp_solution.robots, temp_solution.tasks, temp_solution.allocations)
                # best_solution = temp_solution
                best_improvement = temp_solution.get_improvement_metric()
                best_operation = operation

        # Se encontrou uma melhoria, aplica a melhor solução e continua
        if best_solution:
            solution = solution_priority.Solution(best_solution.robots, best_solution.tasks, best_solution.allocations)
            solution.calculate_metrics()
            # solution = best_solution
            improved = True
            improvements += 1
            if neighborhood_stats is not None:
                neighborhood_stats[operation]['successes'] += 1

    return solution, improvements 

