from geneticoptimizator import GeneticRoverParameterIdentifier
import pandas as pd
import matplotlib.pyplot as plt
from pymavlog import MavLog
import numpy as np

xlsx_path = "./planilhas/sequencia_1_1.xlsx"
# xlsx_path = "./planilhas/sequencia_2_1.xlsx"

# Rover parameters - parametros para serem otimizados
I = 2.0  # Moment of inertia [kg*m²] - used in angular dynamics
kt = 0.08  # Torque constant [Nm/A] - scales PWM to torque
time_constant = 0.2  # Motor response time [s] - first-order dynamics
torque_scale = 0.1  # Empirical scaling factor for torque output
ktarget_velocity = 0.035
# Rover dynamics parameters
time_constant_linear = 0.3  # Linear velocity response time [s]
time_constant_angular = 0.3  # Angular velocity response time [s]

m = 12.0  # Mass [kg] - directly used in dynamics calculations
L = 0.25  # Wheelbase [m] - critical for turn calculations
r = 0.1  # Wheel radius [m] - converts angular to linear velocity

# Motor control parameters (actually used in simplified model)
pwm_min = -100  # Minimum PWM value (full reverse)
pwm_max = 100  # Maximum PWM value (full forward)

# Rover parameters - parametros para serem otimizados
I = 3.0  # Moment of inertia [kg*m²] - used in angular dynamics
kt = 0.08  # Torque constant [Nm/A] - scales PWM to torque
time_constant = 0.1  # Motor response time [s] - first-order dynamics
torque_scale = 0.2  # Empirical scaling factor for torque output

# Rover dynamics parameters
time_constant_linear = 0.3  # Linear velocity response time [s]
time_constant_angular = 0.3  # Angular velocity response time [s]

# Create motor instances with only used parameters
#I, m, r, L, kt, ktarget_velocity, time_constant, torque_scale, linear_force_scale, angular_force_scale, rwheel, lwheel, C_r, C_omega =
# caso 3 - individual [2.0, 14.272440605296403, 0.12249219879247844, 0.35383118900042243, 1.6511575897250539, 0.1378961515171513, 3.616126592138924, 0.1715126720922374, 1.6630449672673004, 1.0175374734140008, 0.5834198957273958, 1.4317556579505484, 7.037854382049224, 12.0, 0.21951136153695738, 0.3, 0.22569910378652452, 0.3]

param_bounds = [
    (1, 5),          # I - momento de inércia
    (5, 20),         # m - massa
    (0.1, 0.3),     # r - raio da roda
    (0.1, 0.8),      # L - entre-eixos
    (0.01, 2.50),  # kt
    (0.01, 0.7),    # ktarget_velocity
    (0.01, 6),       # time_constant
    (0.01, 0.5),     # torque_scale
    (0.01, 8),       # linear_force_scale
    (0.1, 2.0),     # angular_force_scale
    (1.0, 1.05),      # rwheel
    (1.0, 1.05),      # lwheel
    (0.01, 16.0),     # C_r
    (0.01, 14.0),     # C_omega
    (0.21, 0.35),      # r_FR
    (0.21, 0.35),      # r_FL
    (0.21, 0.35),      # r_RL
    (0.21, 0.35)       # r_RR
]

# param_bounds = [
#     (1, 8),  # I
#     (0.0001, 2.50),  # kt
#     (0.001, 0.3),  # ktarget_velocity
#     (0.01, 5),  # time_constant
#     (0.01, 0.5),  # torque_scale
#     (0.01, 5),  # time_constant_linear
#     (0.01, 0.50),  # time_constant_angular
#     (-2.2, 2.2),  # right wheel resistance
#     (-2.2, 2.2),  # left  wheel resistance
#     (0.02, 0.3)  # turning_gain
# ]
if __name__ == '__main__':
    identificador = GeneticRoverParameterIdentifier(
        excel_file=xlsx_path,
        sheet_name="Sheet1",
        population_size=40,
        generations=50,
        param_bounds=param_bounds
    )

    melhores_parametros = identificador.run_genetic_algorithm()
    identificador.plot_results()