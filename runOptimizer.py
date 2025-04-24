from geneticoptimizator import GeneticRoverParameterIdentifier
from roverclass import EvaluateRoverParameters
from deap import base, creator, tools, algorithms
import pandas as pd
import matplotlib.pyplot as plt
from pymavlog import MavLog
import numpy as np
import cma


def imprimir_parametros(resultado, nomes, limites):
    print("Valores dos parâmetros encontrados:\n")
    for valor, nome, limite in zip(resultado, nomes, limites):
        limite_str = f"[{limite[0]}, {limite[1]}]"
        print(f"{nome:<25}: {valor:.4f} (limite: {limite_str})")


# xlsx_path = "./planilhas/sequencia_2_1.xlsx"


#Melhores parâmetros encontrados: [4.516266503497172, 10.03200165091387, 0.16581728638199877, 0.553173282487138, 0.3731352193678904, 0.1944778492320581, 0.15167720958952693, 0.08944725412845686, 0.9280263620618783, 0.44688651189838446, 2.1153459605371685, 2.5362292106515216, 12.358466287491023, 9.96642799981114, 0.2616938533447877, 0.35, 0.3167776514566347, 0.3021156806543137]
# Melhores parâmetros encontrados: [5.9835229681682724, 5.017001233794611, 0.4435001825912026, 0.4954477383499165, 0.5193475054703981, 0.23784683532200385, 0.10877830904301412, 0.09113025347956612, 0.6775985150347202, 3.3208777106157226, 2.1748164153897993, 1.398284813386942, 6.950858972981971, 6.906902566881234, 0.25260486525204096, 0.3270591057901752, 0.288496963608336, 0.42221998818227907]

param_names = [
    "I - momento de inércia", "m - massa", "r - raio da roda", "L - entre-eixos",
    "kt", "ktarget_velocity", "time_constant", "torque_scale",
    "linear_force_scale", "angular_force_scale", "rwheel", "lwheel",
    "C_r", "C_omega", "r_FR", "r_FL", "r_RL", "r_RR"
]

param_bounds = [
    (1, 8),          # I - momento de inércia
    (5, 10),         # m - massa
    (0.1, 0.6),      # r - raio da roda
    (0.1, 1.0),      # L - entre-eixos
    (0.01, 1.80),    # kt
    (0.01, 0.7),    # ktarget_velocity
    (0.01, 2),       # time_constant
    (0.01, 0.5),     # torque_scale
    (0.01, 2),       # linear_force_scale
    (0.1, 4.0),     # angular_force_scale
    (1.0, 3.05),      # rwheel
    (1.0, 3.05),      # lwheel
    (0.01, 8.0),     # C_r
    (0.01, 8.0),     # C_omega
    (0.24, 0.26),      # r_FR
    (0.24, 0.26),      # r_FL
    (0.24, 0.26),      # r_RL
    (0.24, 0.26)       # r_RR
]



if __name__ == '__main__':

    xlsx_path = "./planilhas/sequencia_1_1.xlsx"
    sheet_name = "Sheet1"


    # identificador = GeneticRoverParameterIdentifier(
    #     excel_file=xlsx_path,
    #     sheet_name=sheet_name,
    #     population_size=600,
    #     generations=10,
    #     param_bounds=param_bounds
    # )
    #
    # melhores_parametros = identificador.run_genetic_algorithm()
    #
    # imprimir_parametros(melhores_parametros,param_names,param_bounds)
    #
    # identificador.plot_results()


    #### OTIMIZANDO COM CMA

    lower_bounds = [b[0] for b in param_bounds]
    upper_bounds = [b[1] for b in param_bounds]
    bounds = [lower_bounds, upper_bounds]

    rover = EvaluateRoverParameters(xlsx_path,sheet_name)


    def evaluate(individual):
        return rover.evaluate(individual)

    x0 = [5.9835229681682724, 5.017001233794611, 0.4435001825912026, 0.4954477383499165, 0.5193475054703981, 0.23784683532200385, 0.10877830904301412, 0.09113025347956612, 0.6775985150347202, 3.3208777106157226, 2.1748164153897993, 1.398284813386942, 6.950858972981971, 6.906902566881234, 0.25260486525204096, 0.25260486525204096,0.25260486525204096,0.25260486525204096]
    sigma0 = 0.2


    res = cma.fmin(evaluate, x0, sigma0, {
        'bounds': bounds,
        'popsize': 50,
        'maxiter': 150
    })

    print(res[0])
    print(res[1])


