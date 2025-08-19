from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import pdist


max_x = 600
max_y = 250

def plot_cluster_weights(num_clusters, cluster_weights):
    plt.figure()
    plt.bar(range(1, num_clusters+1), cluster_weights)
    plt.xlabel('Cluster ID')
    plt.ylabel('Total Weight (Inspection Time)')
    plt.title('Weight of Each Cluster')
    plt.show()

def plot_clusters(clusters):
    for cluster_id, cluster in enumerate(clusters):
        if cluster:
            cluster_positions = np.array([[task.x, task.y] for task in cluster])  # Itera sobre as tarefas dentro de cada cluster
            plt.scatter(cluster_positions[:, 0], cluster_positions[:, 1], label=f'Cluster {cluster_id+1}')
            plt.xlim(0, max_x)
    plt.ylim(0, max_y)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Task Allocation')
    plt.legend()
    plt.show()

def plot_robot_paths(solution, title=None, filename=None, max_x=500, max_y=350):
    """
    Plota os caminhos dos robôs com base nas alocações e adiciona números nos vértices para indicar a sequência.
    """
    colors = ['red', 'blue', 'green', 'orange', 'purple']  # Cores diferentes para cada robô
    plt.figure(figsize=(10, 6))
    
    for robot_index, robot in enumerate(solution.robots):
        if robot.allocations:
            task_positions = np.array([[task.x, task.y] for task in robot.allocations])
            entry_points = np.array([task.coordinates for task in robot.allocations])
            coordinates = np.array([task.coordinates for task in robot.allocations])
            
            # Plotar os pontos de entrada das tarefas alocadas ao robô
            plt.scatter(task_positions[:, 0], task_positions[:, 1], 
                        color=colors[robot_index % len(colors)], 
                        label=f'Robot {robot.id}', 
                        s=300,  # Aumenta o tamanho dos pontos
                        alpha=0.3)  # Diminui a opacidade
            
            # Adicionar números nos vértices para indicar a ordem das tarefas
            for i, (x, y) in enumerate(task_positions):
                plt.text(x, y, str(i+1), fontsize=12, color='black', ha='center', va='center')  # Adiciona o número da tarefa

            # Conectar os pontos das tarefas alocadas
            for i in range(len(task_positions) - 1):
                plt.plot([task_positions[i][0], task_positions[i + 1][0]], 
                         [task_positions[i][1], task_positions[i + 1][1]], 
                         color=colors[robot_index % len(colors)], linestyle='-')

    plt.xlim(0, max_x)
    plt.ylim(0, max_y)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title if title else 'Robot Task Allocation and Paths with Task Order')
    plt.legend()
    plt.grid(True)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')  # Salvar o gráfico
    # plt.show()


# Função para plotar a evolução das métricas
def plot_evolution(iterations, distances, times, balance_loads):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plotar a evolução da distância
    axs[0].plot(iterations, distances, label='Distance')
    axs[0].set_title('Evolution of Execution Distance')
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel('Distance')
    
    # Plotar a evolução do tempo
    axs[1].plot(iterations, times, label='Execution Time', color='orange')
    axs[1].set_title('Evolution of Execution Time')
    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel('Time')
    
    # Plotar a evolução do balanceamento de carga
    axs[2].plot(iterations, balance_loads, label='Balance Load', color='green')
    axs[2].set_title('Evolution of Balance Load')
    axs[2].set_xlabel('Iterations')
    axs[2].set_ylabel('Balance Load')
    
    plt.tight_layout()
    # plt.show()

############################# PLOTAGENS ###############################################

def plot_pareto_solutions(pareto_archive, title=None, filename=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Acesse diretamente os atributos calculados de cada solução
    distances = [sol.distance for sol in pareto_archive if sol.distance is not None]
    times = [sol.time for sol in pareto_archive if sol.time is not None]
    balance = [sol.balance_load for sol in pareto_archive if sol.balance_load is not None]

    # Verificar se há valores suficientes para plotar
    if not distances or not times or not balance:
        print("No valid solutions to plot.")
        return

    ax.scatter(distances, times, balance, c='r', marker='o')

    ax.set_xlabel('Execution Distance')
    ax.set_ylabel('Execution Time')
    ax.set_zlabel('Balance Load')
    ax.set_title(title if title else 'Pareto Front Solutions')

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')  # Salvar o gráfico
    # plt.show()




def plot_robot_paths_with_boxes(solution, box_size=20, max_x=600, max_y=250):
    """
    Plota os caminhos dos robôs com base nas alocações, mostrando as regiões como boxes com pontos de entrada e saída.
    """
    colors = ['red', 'blue', 'green', 'orange', 'purple']  # Cores diferentes para cada robô
    plt.figure(figsize=(10, 6))
    
    for robot_index, robot in enumerate(solution.robots):
        if robot.allocations:
            for task in robot.allocations:
                # Posição do box (representando a task)
                box_bottom_left = (task.x - box_size / 2, task.y - box_size / 2)
                
                # Plotar o retângulo (caixa) representando a tarefa
                plt.gca().add_patch(plt.Rectangle(box_bottom_left, box_size, box_size, fill=False, edgecolor=colors[robot_index % len(colors)], linestyle='--'))

                # Plotar os pontos de entrada e saída
                plt.plot(task.coordinates[0], task.coordinates[1], marker='o', markersize=5, color='black', label='Entry Point' if robot_index == 0 else "")
                plt.plot(task.coordinates[0], task.coordinates[1], marker='x', markersize=5, color='black', label='Exit Point' if robot_index == 0 else "")
                
                # Desenha uma linha entre o ponto de saída da tarefa anterior e o ponto de entrada da próxima tarefa
                for i in range(len(robot.allocations) - 1):
                    current_task = robot.allocations[i]
                    next_task = robot.allocations[i + 1]
                    
                    plt.plot([current_task.coordinates[0], next_task.coordinates[0]],
                             [current_task.coordinates[1], next_task.coordinates[1]],
                             color=colors[robot_index % len(colors)], linestyle='-', marker='o')
    
    # Adicionar detalhes ao gráfico
    plt.xlim(0, max_x)
    plt.ylim(0, max_y)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Robot Paths with Task Boxes')
    plt.legend(['Entry Point', 'Exit Point'])
    plt.grid(True)
    # plt.show()

def filtered_plot(pareto_front, title=None, filename=None):
    # Suavização com média móvel
    window_size = 500  # Tamanho da janela de suavização
    smooth_execution_distance = uniform_filter1d(pareto_front.distances, size=window_size)
    smooth_execution_time = uniform_filter1d(pareto_front.times, size=window_size)
    smooth_balance_load = uniform_filter1d(pareto_front.balance_loads, size=window_size)

    # Plotagem dos gráficos suavizados
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Evolução da Distância
    axs[0].plot(pareto_front.iterations, pareto_front.distances, color='blue', alpha=0.2, label="Raw Data")  # Dados brutos com opacidade
    axs[0].plot(pareto_front.iterations, smooth_execution_distance, color='blue', label="Smoothed Data")  # Dados suavizados
    axs[0].set_title('Evolution of Execution Distance')
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel('Distance')
    axs[0].legend()

    # Evolução do Tempo
    axs[1].plot(pareto_front.iterations, pareto_front.times, color='orange', alpha=0.2, label="Raw Data")  # Dados brutos com opacidade
    axs[1].plot(pareto_front.iterations, smooth_execution_time, color='orange', label="Smoothed Data")  # Dados suavizados
    axs[1].set_title('Evolution of Execution Time')
    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel('Time')
    axs[1].legend()

    # Evolução do Balanceamento de Carga
    axs[2].plot(pareto_front.iterations, pareto_front.balance_loads, color='green', alpha=0.2, label="Raw Data")  # Dados brutos com opacidade
    axs[2].plot(pareto_front.iterations, smooth_balance_load, color='green', label="Smoothed Data")  # Dados suavizados
    axs[2].set_title('Evolution of Balance Load')
    axs[2].set_xlabel('Iterations')
    axs[2].set_ylabel('Balance Load')
    axs[2].legend()

    plt.tight_layout()
    if title:
        fig.suptitle(title, fontsize=16, y=1.02)  # Título geral do gráfico
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')  # Salvar o gráfico
    # plt.show()

def analyze_metrics_evolution(metrics_log, normalize=False):
    """
    Analisa a evolução das métricas ao longo das gerações, com opção de normalização.

    Args:
        metrics_log (list): Lista de arrays contendo métricas de cada geração.
        normalize (bool): Se True, normaliza as métricas entre 0 e 1.

    Returns:
        dict: Contém as evoluções das melhores, piores, médias e diversidade das métricas.
    """
    evolutions = {
        "best_distance": [],
        "worst_distance": [],
        "mean_distance": [],
        "best_time": [],
        "worst_time": [],
        "mean_time": [],
        "best_priority_time": [],
        "worst_priority_time": [],
        "mean_priority_time": [],
        "diversity": []
    }

    # Min e Max de cada métrica para normalização, se necessário
    if normalize:
        all_metrics = np.vstack(metrics_log)
        min_values = all_metrics.min(axis=0)
        max_values = all_metrics.max(axis=0)
    
    for metrics in metrics_log:
        if normalize:
            metrics = (metrics - min_values) / (max_values - min_values)

        # Melhor, pior e média valores para cada métrica
        evolutions["best_distance"].append(metrics[:, 0].min())
        evolutions["worst_distance"].append(metrics[:, 0].max())
        evolutions["mean_distance"].append(metrics[:, 0].mean())
        evolutions["best_time"].append(metrics[:, 1].min())
        evolutions["worst_time"].append(metrics[:, 1].max())
        evolutions["mean_time"].append(metrics[:, 1].mean())
        evolutions["best_priority_time"].append(metrics[:, 2].min())
        evolutions["worst_priority_time"].append(metrics[:, 2].max())
        evolutions["mean_priority_time"].append(metrics[:, 2].mean())

        # Diversidade (média das distâncias pareadas)
        evolutions["diversity"].append(np.mean(pdist(metrics)))

    return evolutions

def plot_metrics_evolution(evolutions, normalize=False):
    """
    Plota a evolução das métricas ao longo das gerações, com opção de normalização.

    Args:
        evolutions (dict): Dicionário contendo as evoluções das métricas.
        normalize (bool): Se True, indica que os valores são normalizados.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 12))

    title_suffix = " (Normalizado)" if normalize else ""

    # Distância
    plt.subplot(3, 1, 1)
    plt.plot(evolutions["best_distance"], label="Melhor Distância", linestyle="--")
    plt.plot(evolutions["worst_distance"], label="Pior Distância", linestyle="--")
    plt.plot(evolutions["mean_distance"], label="Média Distância", linestyle="-")
    plt.xlabel("Geração")
    plt.ylabel("Distância")
    plt.title(f"Evolução da Distância{title_suffix}")
    plt.legend()
    plt.grid()

    # Tempo Total
    plt.subplot(3, 1, 2)
    plt.plot(evolutions["best_time"], label="Melhor Tempo", linestyle="--")
    plt.plot(evolutions["worst_time"], label="Pior Tempo", linestyle="--")
    plt.plot(evolutions["mean_time"], label="Média Tempo", linestyle="-")
    plt.xlabel("Geração")
    plt.ylabel("Tempo Total")
    plt.title(f"Evolução do Tempo Total{title_suffix}")
    plt.legend()
    plt.grid()

    # Tempo de Conclusão das Tarefas Prioritárias
    plt.subplot(3, 1, 3)
    plt.plot(evolutions["best_priority_time"], label="Melhor Tempo Prioritário", linestyle="--")
    plt.plot(evolutions["worst_priority_time"], label="Pior Tempo Prioritário", linestyle="--")
    plt.plot(evolutions["mean_priority_time"], label="Média Tempo Prioritário", linestyle="-")
    plt.xlabel("Geração")
    plt.ylabel("Tempo Prioritário")
    plt.title(f"Evolução do Tempo das Tarefas Prioritárias{title_suffix}")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_metrics_evolution_comparison(evolutions, normalize=False):
    """
    Plota a evolução das métricas ao longo das gerações, com opção de normalização.

    Args:
        evolutions (dict): Dicionário contendo as evoluções das métricas.
        normalize (bool): Se True, indica que os valores são normalizados.
    """

    plt.figure(figsize=(14, 12))

    title_suffix = " (Normalizado)" if normalize else ""

    # Distância
    plt.subplot(3, 1, 1)
    plt.plot(evolutions["mean_distance"], label="Média Distância", linestyle="-")
    plt.xlabel("Geração")
    plt.ylabel("Distância")
    plt.title(f"Evolução da Distância{title_suffix}")
    plt.legend()
    plt.grid()

    # Tempo Total
    plt.subplot(3, 1, 2)
    plt.plot(evolutions["mean_time"], label="Média Tempo", linestyle="-")
    plt.xlabel("Geração")
    plt.ylabel("Tempo Total")
    plt.title(f"Evolução do Tempo Total{title_suffix}")
    plt.legend()
    plt.grid()

    # Tempo de Conclusão das Tarefas Prioritárias
    plt.subplot(3, 1, 3)
    plt.plot(evolutions["mean_priority_time"], label="Média Tempo Prioritário", linestyle="-")
    plt.xlabel("Geração")
    plt.ylabel("Tempo Prioritário")
    plt.title(f"Evolução do Tempo das Tarefas Prioritárias{title_suffix}")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def read_evolutions_from_file(filepath):
    """
    Reads a JSON file containing metric evolutions.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: Dictionary containing the evolution of metrics.
    """
    with open(filepath, "r") as f:
        evolutions = json.load(f)
    return evolutions

def interpolate_metrics(methods_data, total_time=100):
    """
    Interpolates the data for each method to align metrics over time.

    Args:
        methods_data (dict): Dictionary containing data for each method.
        total_time (int): Total time (in seconds) for which the data will be interpolated.

    Returns:
        dict: Dictionary with interpolated metrics over time.
    """
    interpolated_data = {}
    for method, data in methods_data.items():
        generations = np.arange(len(data["mean_distance"]))
        time = np.linspace(0, total_time, len(generations))  # Maps iterations to time

        # Create interpolation functions for each metric
        interp_distance = interp1d(time, data["mean_distance"], kind='linear', fill_value="extrapolate")
        interp_time = interp1d(time, data["mean_time"], kind='linear', fill_value="extrapolate")
        interp_priority_time = interp1d(time, data["mean_priority_time"], kind='linear', fill_value="extrapolate")

        # Interpolate to uniform time
        uniform_time = np.linspace(0, total_time, total_time + 1)
        interpolated_data[method] = {
            "time": uniform_time,
            "mean_distance": interp_distance(uniform_time),
            "mean_time": interp_time(uniform_time),
            "mean_priority_time": interp_priority_time(uniform_time),
        }
    return interpolated_data

def plot_all_metrics_together(methods_data, total_time=100):
    """
    Plots the evolution of metrics over time, with all methods in the same graph.

    Args:
        methods_data (dict): Dictionary containing interpolated data for each method.
        total_time (int): Total time (in seconds) for which the data has been interpolated.
    """
    plt.figure(figsize=(14, 10))

    # Distance
    plt.subplot(3, 1, 1)
    for method, data in methods_data.items():
        plt.plot(data["time"], data["mean_distance"], label=f"{method}", linestyle="-")
    plt.xlabel("Time (s)", fontsize=18)
    plt.ylabel("Mean Distance", fontsize=18)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.title("Comparison of Distance Evolution", fontsize=18)
    plt.legend(fontsize=16)
    plt.grid()

    # Total Time
    plt.subplot(3, 1, 2)
    for method, data in methods_data.items():
        plt.plot(data["time"], data["mean_time"], label=f"{method}", linestyle="-")
    plt.xlabel("Time (s)", fontsize=18)
    plt.ylabel("Mean Total Time", fontsize=18)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.title("Comparison of MIssion Time Evolution", fontsize=18)
    plt.legend(fontsize=16)
    plt.grid()

    # Priority Task Completion Time
    plt.subplot(3, 1, 3)
    for method, data in methods_data.items():
        plt.plot(data["time"], data["mean_priority_time"], label=f"{method}", linestyle="-")
    plt.xlabel("Time (s)", fontsize=18)
    plt.ylabel("Mean Priority Time", fontsize=18)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.title("Comparison of Balance Load Evolution", fontsize=18)
    plt.legend(fontsize=16)
    plt.grid()

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Paths to JSON files for different methods
    filepaths = {
        "MOVNS-AINS": "/home/milena/catkin_ws/src/ardupilot_gazebo/experiment_results_balance/evolutions_ains_hybrid_3_50_0.json",
        "MOVNS-VND": "/home/milena/catkin_ws/src/ardupilot_gazebo/experiment_results_balance/evolutions_vnd_random_3_50_0.json",
        "NSGA-II": "/home/milena/catkin_ws/src/ardupilot_gazebo/experiment_results_balance/evolutions_standard_random_3_100_0.json"
    }

    # Read data for each method
    methods_data = {method: read_evolutions_from_file(filepath) for method, filepath in filepaths.items()}

    # Interpolate data to uniform time
    interpolated_data = interpolate_metrics(methods_data, total_time=100)

    # Plot comparisons in a single graph for each metric
    plot_all_metrics_together(interpolated_data, total_time=100)
