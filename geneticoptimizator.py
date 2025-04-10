import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from roverclass import MotorModel, SkidSteerRoverModel
from deap import base, creator, tools, algorithms
import multiprocessing

class GeneticRoverParameterIdentifier:
    def __init__(self, excel_file, sheet_name, population_size=50, generations=20,
                 param_bounds=None, constants = None):
        self.excel_file = excel_file
        self.sheet_name = sheet_name

        if excel_file.endswith(".csv"):
            self.data_real = pd.read_csv(excel_file)
        else:
            self.data_real = pd.read_excel(excel_file, sheet_name=sheet_name)

        print("Colunas disponíveis:", self.data_real.columns.tolist())

        self.population_size = population_size
        self.generations = generations



        self.param_bounds = param_bounds or [
        (2, 4),        # I
        (0.0001,0.1),  # kt
        (0.001,0.045), # ktarget_velocity
        (0.2, 0.5),    # time_constant
        (0.1, 0.3),    # torque_scale
        (0.1, 0.4),    # time_constant_linear
        (0.1, 0.4),    # time_constant_angular
        (0.5,1.5),     # right wheel resistance
        (0.5, 1.5) ,    # left  wheel resistance
        (0.01, 0.3)  # turning_gain
        ]

        # # Rover physical parameters (essential ones)
        # m = 15.0  # Mass [kg] - directly used in dynamics calculations
        # L = 0.5  # Wheelbase [m] - critical for turn calculations
        # r = 0.1  # Wheel radius [m] - converts angular to linear velocity
        #
        # # Motor control parameters (actually used in simplified model)
        # pwm_min = -100  # Minimum PWM value (full reverse)
        # pwm_max = 100  # Maximum PWM value (full forward)

        self.constants = constants or {
            "mass" : 15, # mass
            "L" : 0.3, # Wheelbase [m] - critical for turn calculations
            "r" : 0.1,  # Wheel radius [m] - converts angular to linear velocity
            "pwm_min": -100,  # Minimum PWM value (full reverse)
            "pwm_max": 100  # Maximum PWM value (full forward)
        }

        self.last_sim_data = []

    def scale_pwm(self, pwm, motor_id):
        return  (pwm - 1500) * (100 / 400)

    def evaluate(self, individual):
        try:

            # # Create motor instances with only used parameters
            # motor_FR = MotorModel(kt=kt, ktarget_velocity=ktarget_velocity, pwm_min=pwm_min, pwm_max=pwm_max,
            #                       time_constant=time_constant, torque_scale=torque_scale, orientation=-1,
            #                       wheel_radius=r)
            #
            # # Initialize rover model with all active parameters
            # rover_model = SkidSteerRoverModel(
            #     m=m,  # Mass
            #     I=I,  # Moment of inertia
            #     L=L,  # Wheelbase
            #     r=r,  # Wheel radius
            #     motor_FL=motor_FL,  # Front left motor
            #     motor_FR=motor_FR,  # Front right motor
            #     motor_RL=motor_RL,  # Rear left motor
            #     motor_RR=motor_RR,  # Rear right motor
            #     rleft=rwheel,
            #     rright=lwheel,
            #     C_r=cr,
            #     C_omega=comega,
            #     linear_force_scale=time_constant_linear,
            #     angular_force_scale=time_constant_angular
            # )
            I, m, r, L, kt, ktarget_velocity, time_constant, torque_scale, linear_force_scale, angular_force_scale, rwheel, lwheel, C_r, C_omega, rFR, rFL, rRL, rRR = individual

            R = 0.2

            motor_FR = MotorModel(kt=kt, ktarget_velocity= ktarget_velocity,  pwm_min=self.constants["pwm_min"], pwm_max=self.constants["pwm_max"], time_constant=time_constant, torque_scale=torque_scale, orientation=-1, wheel_radius = rFR)
            motor_FL = MotorModel(kt=kt, ktarget_velocity= ktarget_velocity,  pwm_min=self.constants["pwm_min"], pwm_max=self.constants["pwm_max"], time_constant=time_constant, torque_scale=torque_scale, orientation=1, wheel_radius = rFL)
            motor_RL = MotorModel(kt=kt, ktarget_velocity= ktarget_velocity,  pwm_min=self.constants["pwm_min"], pwm_max=self.constants["pwm_max"], time_constant=time_constant, torque_scale=torque_scale, orientation=1, wheel_radius = rRL)
            motor_RR = MotorModel(kt=kt, ktarget_velocity= ktarget_velocity,  pwm_min=self.constants["pwm_min"], pwm_max=self.constants["pwm_max"], time_constant=time_constant, torque_scale=torque_scale, orientation=-1, wheel_radius = rRR)

            rover = SkidSteerRoverModel(
                m=m,
                I=I,
                L=L,
                r=r,
                motor_FL=motor_FL,
                motor_FR=motor_FR,
                motor_RL=motor_RL,
                motor_RR=motor_RR,
                rright= rwheel,
                rleft= lwheel,
                C_r=C_r,
                C_omega=C_omega,
                linear_force_scale=linear_force_scale,
                angular_force_scale=angular_force_scale
            )

            state = np.array([0, 0, 0, 0, 0])
            dt = 0.1
            battery_voltage = 48
            error_total = 0
            sim_data = []

            for _, row in self.data_real.iterrows():
                pwm_inputs = np.array([
                    self.scale_pwm(row['RCOU.C1'], 1),
                    self.scale_pwm(row['RCOU.C2'], 2),
                    self.scale_pwm(row['RCOU.C3'], 3),
                    self.scale_pwm(row['RCOU.C4'], 4)
                ])

                state = rover.dynamics(state, pwm_inputs, dt, battery_voltage)

                linear_real = row['GPS[0].Spd']
                angular_real = row['IMU[0].GyrZ']
                linear_sim = state[3]
                angular_sim = state[4]

                if np.isnan(linear_sim) or np.isnan(angular_sim):
                    return (1e6,)

                error = abs(linear_real - linear_sim) + abs(angular_real - angular_sim)
                error_total += error

                sim_data.append({
                    "time": row['timestamp(ms)'] / 1000.0,
                    "linear_real": linear_real,
                    "linear_sim": linear_sim,
                    "angular_real": angular_real,
                    "angular_sim": angular_sim
                })

            self.last_sim_data = sim_data
            return error_total,

        except Exception:
            return (1e6,)

    def plot_results(self):
        if not self.last_sim_data:
            print("Nenhum dado para plotar.")
            return

        time = [d['time'] for d in self.last_sim_data]
        linear_real = [d['linear_real'] for d in self.last_sim_data]
        linear_sim = [d['linear_sim'] for d in self.last_sim_data]
        angular_real = [d['angular_real'] for d in self.last_sim_data]
        angular_sim = [d['angular_sim'] for d in self.last_sim_data]

        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(time, linear_real, label='Vel. Linear Real')
        plt.plot(time, linear_sim, label='Vel. Linear Simulada')
        plt.ylabel("Velocidade Linear [m/s]")
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(time, angular_real, label='Vel. Angular Real')
        plt.plot(time, angular_sim, label='Vel. Angular Simulada')
        plt.ylabel("Velocidade Angular [rad/s]")
        plt.xlabel("Tempo [s]")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def clip_individual(individual, bounds):
        for i, (min_val, max_val) in enumerate(bounds):
            individual[i] = np.clip(individual[i], min_val, max_val)
        return individual

    def run_genetic_algorithm(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        for i, bounds in enumerate(self.param_bounds):
            toolbox.register(f"attr_float_{i}", np.random.uniform, bounds[0], bounds[1])

        toolbox.register("individual", tools.initCycle, creator.Individual,
                         tuple(getattr(toolbox, f"attr_float_{i}") for i in range(len(self.param_bounds))), n=1)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutGaussian,
                         mu=[(a + b) / 2 for a, b in self.param_bounds],
                         sigma=[(b - a) / 5 for a, b in self.param_bounds],
                         indpb=0.2)


        toolbox.register("select", tools.selTournament, tournsize=3)

        # pool = multiprocessing.Pool()
        # toolbox.register("map", pool.map)

        population = toolbox.population(n=self.population_size)

        for gen in range(self.generations):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.8, mutpb=0.1)

            for i, ind in enumerate(offspring):
                offspring[i] = GeneticRoverParameterIdentifier.clip_individual(ind, self.param_bounds)
            # fits = toolbox.map(toolbox.evaluate, offspring)
            fits = list(map(toolbox.evaluate, offspring))
            for ind, fit in zip(offspring, fits):
                ind.fitness.values = fit

            population = toolbox.select(offspring, k=len(population))
            top = tools.selBest([ind for ind in population if not np.isnan(ind.fitness.values[0])], 1)[0]
            print(f"Geração {gen+1}: Erro do melhor indivíduo = {top.fitness.values[0]:.4f}")

        best_individual = tools.selBest([ind for ind in population if not np.isnan(ind.fitness.values[0])], 1)[0]
        print("Melhores parâmetros encontrados:", best_individual)

        self.evaluate(best_individual)
        self.plot_results()
        return best_individual