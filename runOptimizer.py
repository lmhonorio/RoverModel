from geneticoptimizator import GeneticRoverParameterIdentifier
import pandas as pd
import matplotlib.pyplot as plt
from pymavlog import MavLog
import numpy as np

xlsx_path = "./planilhas/sequenciaa1.xlsx"

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
L = 0.5  # Wheelbase [m] - critical for turn calculations
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

param_bounds = [
    (1, 20),  # I
    (0.0001, 0.50),  # kt
    (0.001, 0.3),  # ktarget_velocity
    (0.01, 5),  # time_constant
    (0.01, 0.5),  # torque_scale
    (0.01, 5),  # time_constant_linear
    (0.01, 0.20),  # time_constant_angular
    (0.02, 2),  # right wheel resistance
    (0.02, 2)  # left  wheel resistance
]

identificador = GeneticRoverParameterIdentifier(
    excel_file=xlsx_path,
    sheet_name="Sheet1",
    population_size=10,
    generations=100,
    param_bounds=param_bounds
)

melhores_parametros = identificador.run_genetic_algorithm()
identificador.plot_results()