import json
import numpy as np
from scipy.spatial.distance import pdist

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

def save_results_to_file(evolutions, filename):
    filename = f'/home/milena/catkin_ws/src/ardupilot_gazebo/experiment_results_balance/{filename}'
    """
    Salva as evoluções das métricas em um arquivo JSON.

    Args:
        evolutions (dict): Dicionário contendo as evoluções das métricas.
        filename (str): Nome do arquivo para salvar os dados.
    """
    with open(filename, "w") as f:
        json.dump(evolutions, f)
    # print(f"Evoluções salvas em {filename}")