import os
import random
import time
from matplotlib import pyplot as plt
import numpy as np
from pymoo.indicators.hv import HV
from scipy.spatial.distance import pdist, squareform
from movns_ains_argo.pareto_front import Pareto_Front
import os
from movns_ains_argo import plot_graphs
from movns_ains_argo.robot import Robot

from movns_ains_argo import save_results_argo as save_results
from movns_ains_argo import vnd_priority_argo as vnd_priority
from movns_ains_argo import solution_priority_argo as solution_priority
from movns_ains_argo import task_priority_argo as task_priority





def calculate_hypervolume(population, ref_point, metrics_log=None):
    """
    Calcula o hipervolume usando o Pymoo e captura as métricas de cada geração.

    Args:
        population (list): População atual contendo soluções.
        ref_point (list): Ponto de referência para cálculo do hipervolume.
        metrics_log (list, optional): Lista para armazenar métricas de cada geração.

    Returns:
        float: Valor do hipervolume.
    """
    metrics = np.array([sol.metrics for sol in population.solutions])
    
    # Logar métricas, se necessário
    if metrics_log is not None:
        metrics_log.append(metrics)
    
    hv = HV(ref_point)
    return hv.do(metrics)

def calculate_spacing(pareto_front):
    """
    Calcula o spacing da frente de Pareto.

    Args:
        pareto_front (np.ndarray): Matriz de soluções na frente de Pareto (N x M),
                                   onde N é o número de soluções e M é o número de objetivos.

    Returns:
        float: Valor do spacing ou None se não for possível calcular.
    """
    # Verificar se há soluções suficientes na frente de Pareto
    if len(pareto_front) <= 1:
        # Não é possível calcular spacing com 1 ou 0 soluções
        return None

    # Distâncias euclidianas entre todas as soluções
    distances = squareform(pdist(pareto_front, metric='euclidean'))

    # Adicionar np.inf na diagonal para ignorar distâncias de si mesmo
    np.fill_diagonal(distances, np.inf)

    # Para cada solução, encontre a menor distância para outra solução
    min_distances = np.min(distances, axis=1)

    # Média das distâncias
    mean_distance = np.mean(min_distances)

    # Spacing: Desvio médio absoluto das distâncias mínimas em relação à média
    spacing = np.mean(np.abs(min_distances - mean_distance))
    
    return spacing

def shake_solution(solution, num_tasks_to_shake):
    """
    Aplica uma perturbação na solução atual alterando aleatoriamente uma quantidade de tarefas.

    Args:
        solution (Solution): A solução original a ser sacudida.
        num_tasks_to_shake (int): Número de tarefas a serem alteradas na solução.

    Returns:
        Solution: Uma nova solução após o shake.
    """
    # Coleta todas as tarefas das alocações
    all_tasks = [task for robot_tasks in solution.allocations for task in robot_tasks]

    # Seleciona aleatoriamente as tarefas para "sacudir"
    tasks_to_shake = random.sample(all_tasks, min(num_tasks_to_shake, len(all_tasks)))

    # Remove as tarefas selecionadas da alocação original
    remaining_tasks = [task for task in all_tasks if task not in tasks_to_shake]

    # Reorganizar as tarefas sacudidas aleatoriamente com as restantes
    new_task_order = remaining_tasks[:]
    for task in tasks_to_shake:
        idx = random.randint(0, len(new_task_order))
        new_task_order.insert(idx, task)

    # Repartir as tarefas de volta nos robôs com base nas alocações originais
    robot_task_counts = [len(robot_tasks) for robot_tasks in solution.allocations]
    new_allocations = []
    pointer = 0
    for count in robot_task_counts:
        new_allocations.append(new_task_order[pointer:pointer + count])
        pointer += count

    # Atualizar as alocações na solução
    child = solution_priority.Solution(solution.grafo, solution.robots, solution.tasks, solution.cache_astar, new_allocations)

    # Recalcula as métricas da solução
    child.calculate_metrics()

    return child

def movns_vnd(robots, tasks, pop_size, time_limit, population, local=None, pop_kind=None, run=None):
    start_time = time.time()  # Início da medição do tempo

    # population = solution_priority.generate_hybrid_population(robots, tasks, pop_size)
    # population = solution_priority.generate_random_population(robots, tasks, pop_size)
    
    generation = 0
    hv_values = []  # Armazena valores de hipervolume
    ref_point = [30000, 8000, 8000]
    metrics_log = []  # Lista para armazenar métricas de cada geração

    pareto_front = Pareto_Front()

    for sol in population:
        sol.calculate_metrics()
        pareto_front.add_solution(sol)
        # sol.print_solution_metrics()

    while time.time() - start_time < time_limit:
        # Seleciona uma solução da frente de Pareto
        # print(f"len pareto solutions: {len(pareto_front.solutions)}")
        candidate_solution = random.choice(pareto_front.solutions)

        # Aplica o "shake" à solução antes do VND
        num_tasks_to_shake = random.randint(1, len(tasks) // 10)  # Por exemplo, sacudir 10% das tarefas
        shaken_solution = shake_solution(candidate_solution, num_tasks_to_shake)
        # print(f"Antes: Geração {generation}: Melhor distância: {shaken_solution.distance}, Melhor tempo: {shaken_solution.time}, Melhor max_priority_time: {shaken_solution.max_priority_time}")
        # Aplica o VND na solução sacudida
        better_solution, improvements = vnd_priority.apply_vnd_movns(shaken_solution, len(robots))
        # print(f"Depois: Geração {generation}: Melhor distância: {better_solution.distance}, Melhor tempo: {better_solution.time}, Melhor max_priority_time: {better_solution.max_priority_time}")
        # Atualiza a frente de Pareto se encontrar uma solução melhor
        if improvements > 0:
            updated = pareto_front.update_pareto_front(better_solution)
            # print(f"Gerei: Geração {generation}: Melhor distância: {better_solution.distance}, Melhor tempo: {better_solution.time}, Melhor max_priority_time: {better_solution.max_priority_time}, updated: {updated}")

            generation += 1
            hv = calculate_hypervolume(pareto_front, ref_point, metrics_log=metrics_log)
            hv_values.append(hv)

    """ for robot_tasks in pareto_front.solutions[0].allocations:
        for task in robot_tasks:
            print(task.id)
        print("-----------------------------------") """


    # Após o loop, analisa e plota as métricas
    evolutions = plot_graphs.analyze_metrics_evolution(metrics_log)
    evolutions = save_results.analyze_metrics_evolution(metrics_log)

    save_results.save_results_to_file(evolutions, filename=f"evolutions_{local}_{pop_kind}_{len(robots)}_{len(tasks)}_{run}.json")
    save_results.save_results_to_file(hv_values, filename=f"hv_{local}_{pop_kind}_{len(robots)}_{len(tasks)}_{run}.json")

    # Após o loop, analisa e plota as métricas
    evolutions = plot_graphs.analyze_metrics_evolution(metrics_log)
    # plot_graphs.plot_metrics_evolution(evolutions)

    # Plot da evolução do hipervolume
    plt.plot(hv_values)
    plt.title("Evolução do Hipervolume")
    plt.xlabel("Gerações")
    plt.ylabel("Hipervolume")
    # plt.show()

    return pareto_front.solutions, hv, metrics_log, generation


def movns_bnd(robots, tasks, pop_size, time_limit, population, local=None, pop_kind=None, run=None):
    start_time = time.time()  # Início da medição do tempo

    # population = solution_priority.generate_hybrid_population(robots, tasks, pop_size)
    # population = solution_priority.generate_random_population(robots, tasks, pop_size)
    
    generation = 0
    hv_values = []  # Armazena valores de hipervolume
    ref_point = [30000, 8000, 8000]
    metrics_log = []  # Lista para armazenar métricas de cada geração

    pareto_front = Pareto_Front()

    for sol in population:
        sol.calculate_metrics()
        pareto_front.add_solution(sol)
        # sol.print_solution_metrics()

    while time.time() - start_time < time_limit:
        # Seleciona uma solução da frente de Pareto
        # print(f"len pareto solutions: {len(pareto_front.solutions)}")
        candidate_solution = random.choice(pareto_front.solutions)

        # Aplica o "shake" à solução antes do VND
        num_tasks_to_shake = random.randint(1, len(tasks) // 15)  # Por exemplo, sacudir 10% das tarefas
        shaken_solution = shake_solution(candidate_solution, num_tasks_to_shake)
        # print(f"Antes: Geração {generation}: Melhor distância: {shaken_solution.distance}, Melhor tempo: {shaken_solution.time}, Melhor max_priority_time: {shaken_solution.max_priority_time}")
        # Aplica o VND na solução sacudida
        better_solution, improvements = vnd_priority.apply_best_neighborhood(shaken_solution, len(robots))
        # print(f"Depois: Geração {generation}: Melhor distância: {better_solution.distance}, Melhor tempo: {better_solution.time}, Melhor max_priority_time: {better_solution.max_priority_time}")
        # Atualiza a frente de Pareto se encontrar uma solução melhor
        if improvements > 0:
            updated = pareto_front.update_pareto_front(better_solution)
            # print(f"Gerei: Geração {generation}: Melhor distância: {better_solution.distance}, Melhor tempo: {better_solution.time}, Melhor max_priority_time: {better_solution.max_priority_time}, updated: {updated}")

            generation += 1
            hv = calculate_hypervolume(pareto_front, ref_point, metrics_log=metrics_log)
            hv_values.append(hv)

    """ for robot_tasks in pareto_front.solutions[0].allocations:
        for task in robot_tasks:
            print(task.id)
        print("-----------------------------------") """
    
    # Após o loop, analisa e plota as métricas
    evolutions = plot_graphs.analyze_metrics_evolution(metrics_log)
    evolutions = save_results.analyze_metrics_evolution(metrics_log)

    save_results.save_results_to_file(evolutions, filename=f"evolutions_{local}_{pop_kind}_{len(robots)}_{len(tasks)}_{run}.json")
    save_results.save_results_to_file(hv_values, filename=f"hv_{local}_{pop_kind}_{len(robots)}_{len(tasks)}_{run}.json")

    # plot_graphs.plot_metrics_evolution(evolutions)

    # Plot da evolução do hipervolume
    plt.plot(hv_values)
    plt.title("Evolução do Hipervolume")
    plt.xlabel("Gerações")
    plt.ylabel("Hipervolume")
    # plt.show()

    return pareto_front.solutions, hv, metrics_log, generation

def report_neighborhood_stats(stats):
    print("\n=== Estatísticas das Vizinhanças ===")
    for name, s in stats.items():
        calls = s['calls']
        success = s['successes']
        success_rate = success / calls if calls > 0 else 0
        print(f"{name}: chamadas = {calls}, sucessos = {success}, taxa = {success_rate:.2%}")

    
def movns_ains(robots, tasks, pop_size, time_limit, population, local=None, pop_kind=None, run=None):
    start_time = time.time()  # Início da medição do tempo

    # population = solution_priority.generate_hybrid_population(robots, tasks, pop_size)
    # population = solution_priority.generate_random_population(robots, tasks, pop_size)
    
    generation = 0
    hv_values = []  # Armazena valores de hipervolume
    ref_point = [30000, 8000, 8000]
    metrics_log = []  # Lista para armazenar métricas de cada geração

    pareto_front = Pareto_Front()

    neighborhood_stats = {
    'swap_with_closest_task_single_robot': {'calls': 0, 'successes': 0},
    'swap_with_farthest_task_single_robot': {'calls': 0, 'successes': 0},
    'swap_inter_robot': {'calls': 0, 'successes': 0},
    'bring_closer_task_forward': {'calls': 0, 'successes': 0},
    'fill_long_gap_with_external_task': {'calls': 0, 'successes': 0}
}


    for sol in population:
        sol.calculate_metrics()
        pareto_front.add_solution(sol)
        # sol.print_solution_metrics()

    while time.time() - start_time < time_limit:
        # Seleciona uma solução da frente de Pareto
        # print(f"len pareto solutions: {len(pareto_front.solutions)}")
        candidate_solution = random.choice(pareto_front.solutions)
        # Aplica o "shake" à solução antes do VND
        num_tasks_to_shake = random.randint(1, len(tasks) // 3)  # Por exemplo, sacudir 10% das tarefas
        shaken_solution = shake_solution(candidate_solution, num_tasks_to_shake)
        # print(f"Antes: Geração {generation}: Melhor distância: {shaken_solution.distance}, Melhor tempo: {shaken_solution.time}, Melhor max_priority_time: {shaken_solution.balance_load}")
        # Aplica o VND na solução sacudida
        # better_solution, improvements = vnd_priority.apply_vnd_movns_ains(shaken_solution, robots, neighborhood_stats)
        better_solution, improvements = vnd_priority.apply_vnd_movns(shaken_solution, robots, neighborhood_stats)
        # better_solution, improvements = vnd_priority.apply_best_neighborhood(shaken_solution, robots, neighborhood_stats)
        # print(f"Depois: Geração {generation}: Melhor distância: {better_solution.distance}, Melhor tempo: {better_solution.time}, Melhor max_priority_time: {better_solution.max_priority_time}")
        # Atualiza a frente de Pareto se encontrar uma solução melhor
        if improvements > 0:
            updated = pareto_front.update_pareto_front(better_solution)
            # print(f"Gerei: Geração {generation}: Melhor distância: {better_solution.distance}, Melhor tempo: {better_solution.time}, Melhor max_priority_time: {better_solution.max_priority_time}, updated: {updated}")

            generation += 1
            hv = calculate_hypervolume(pareto_front, ref_point, metrics_log=metrics_log)
            hv_values.append(hv)

    report_neighborhood_stats(neighborhood_stats)
    # Após o loop, analisa e plota as métricas
    evolutions = plot_graphs.analyze_metrics_evolution(metrics_log)
    evolutions = save_results.analyze_metrics_evolution(metrics_log)

    save_results.save_results_to_file(evolutions, filename=f"evolutions_{local}_{pop_kind}_{len(robots)}_{len(tasks)}_{run}.json")
    save_results.save_results_to_file(hv_values, filename=f"hv_{local}_{pop_kind}_{len(robots)}_{len(tasks)}_{run}.json")

    # plot_graphs.plot_metrics_evolution(evolutions)

    # Plot da evolução do hipervolume
    """ plt.plot(hv_values)
    plt.title("Evolução do Hipervolume")
    plt.xlabel("Gerações")
    plt.ylabel("Hipervolume") """
    # plt.show()

    return pareto_front.solutions, hv, metrics_log, generation



pop_size = 50

def run_movns(robots, tasks, G_p, cache_astar, time_limit):
    population = solution_priority.generate_hybrid_population(robots, tasks, 50, G_p, cache_astar)
    print("Rodando MOVNS")
    final_population, hv, metrics_log, total_generations = movns_ains(
                    robots=robots,
                    tasks=tasks,
                    pop_size=int(pop_size),
                    time_limit=time_limit,
                    population=population,
                    local= "ains",
                    pop_kind= "hybrid",
                    run= 1
                )
    return final_population


################################## ABORDAGEM DINÂMICA COM FALHA DE ROBÔ ##################################################

def find_current_task_with_graph(robot, solution, interrupt_time, cache_astar, robots):
    elapsed_time = 0
    current_label = robot.initial_position
    total_energy_consumed = 0

    robot_idx = robots.index(robot)

    for idx, task in enumerate(solution.allocations[robot_idx]):
        _, travel_time = cache_astar.get_path(current_label, task.entry_point["label"])
        task_time = travel_time + task.inspection_time
        total_energy_consumed += (travel_time + task.inspection_distance)

        if elapsed_time + task_time > interrupt_time:
            remaining_tasks = solution.allocations[robot_idx][idx+1:]
            return task, elapsed_time, current_label, remaining_tasks, total_energy_consumed

        elapsed_time += task_time
        current_label = task.exit_point["label"]

    return None, elapsed_time, current_label, [], total_energy_consumed




def greedy_reallocate_failed_robot_with_graph(robots, best_solution, failed_robot_id, cache_astar):
    """
    Reatribui de forma gulosa as tarefas de um robô que falhou, usando o cache A* e estrutura baseada em grafo.

    Args:
        robots (list): Lista de objetos Robot (com .id e .initial_position).
        best_solution (Solution): Solução base para redistribuição.
        failed_robot_id (int): ID do robô que falhou.
        cache_astar (CacheAStar): Cache para cálculo de caminhos no grafo.

    Returns:
        new_solution (Solution): Solução com as tarefas redistribuídas.
        all_remaining_tasks (list): Lista de todas as tarefas remanescentes consideradas.
    """
    interrupt_time = best_solution.time / 3

    all_remaining_tasks = []
    failed_tasks = []
    num_robots = len(robots)

    assigned_tasks = [[] for _ in range(num_robots)]
    robots_final_labels = {}


    active_robots = [robot for robot in robots if robot.id != failed_robot_id]


    for robot in robots:
        current_task, _, current_label, remaining_tasks, _ = find_current_task_with_graph(
            robot, best_solution, interrupt_time, cache_astar, robots
        )

        robot_idx = robots.index(robot)

        if robot.id == failed_robot_id:
            if current_task:
                failed_tasks.append(current_task)
                all_remaining_tasks.append(current_task)
            failed_tasks.extend(remaining_tasks)
            all_remaining_tasks.extend(remaining_tasks)
        else:
            assigned_tasks[robot_idx] = remaining_tasks.copy()
            all_remaining_tasks.extend(remaining_tasks)
            last_task = best_solution.allocations[robot_idx][-1] if best_solution.allocations[robot_idx] else None
            robots_final_labels[robot_idx] = last_task.exit_point["label"] if last_task else robot.initial_position

    # Reatribuir tarefas do robô que falhou
    while failed_tasks:
        for robot in active_robots:
            if not failed_tasks:
                break

            robot_idx = robots.index(robot)
            last_label = robots_final_labels[robot_idx]

            next_task = min(
                failed_tasks,
                key=lambda task: cache_astar.get_path(last_label, task.entry_point["label"])[1]
            )

            assigned_tasks[robot_idx].append(next_task)
            robots_final_labels[robot_idx] = next_task.exit_point["label"]
            failed_tasks.remove(next_task)

    assigned_tasks_filtered = [assigned_tasks[robots.index(robot)] for robot in active_robots]


    greedy_failure_solution = solution_priority.Solution(
        grafo=best_solution.grafo,
        robots=active_robots,
        tasks=all_remaining_tasks,
        cache_astar=cache_astar,
        allocations=assigned_tasks_filtered
    )

    greedy_failure_solution.calculate_metrics()

    print("⚙️ Replanejamento com abordagem gulosa após falha:")
    greedy_failure_solution.print_solution_metrics()

    return greedy_failure_solution, all_remaining_tasks


def dynamic_movns(final_population, robots, new_tasks):
    global distance_matrix
    """ best_time_solution = min(final_population, key=lambda sol: sol.time)
    #print(f"best_time_solution: {best_time_solution.time}")

    interrupt_time = best_time_solution.time / 3
    all_remaining_tasks = []
    for robot in robots:
        current_task, elapsed_time, last_position, remaining_tasks, energy_consumed = find_current_task(robot, best_time_solution, interrupt_time, distance_matrix)
        all_remaining_tasks.extend(remaining_tasks)

        robot.current_position = current_task.coordinates  # ou current_task.exit_point
        robot.remaining_battery = robot.initial_battery_time - energy_consumed
        # print(f"robot {robot.id} current position: {robot.current_position}, remaining tasks: {remaining_tasks}")

    all_remaining_tasks.extend(new_tasks)
    #print(len(all_remaining_tasks))

    new_population = solution_priority.generate_hybrid_population(robots, all_remaining_tasks, pop_size) """
    all_remaining_tasks = []
    for alloc in final_population[0].allocations:
        all_remaining_tasks.extend(alloc)
    solutions, hv, metrics_log, generation = movns_ains(
        robots=robots,
        tasks=all_remaining_tasks,
        pop_size=pop_size,
        time_limit=100,  # tempo restante da missão
        population=final_population
    )

    best_time_solution = min(solutions, key=lambda sol: sol.time)
    print(f"movns allocations: {len(best_time_solution.allocations[0])}")
    print("MOVNS-AINS Time: ")
    best_time_solution.print_solution_metrics()

    best_distance_solution = min(solutions, key=lambda sol: sol.distance)
    print(f"movns allocations: {len(best_distance_solution.allocations[0])}")
    print("MOVNS-AINS Distance: ")
    best_distance_solution.print_solution_metrics()

    best_balance_load_solution = min(solutions, key=lambda sol: sol.balance_load)
    print(f"movns allocations: {len(best_balance_load_solution.allocations[0])}")
    print("MOVNS-AINS Balance Load: ")
    best_balance_load_solution.print_solution_metrics()