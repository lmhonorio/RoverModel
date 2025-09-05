import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from roverdynamics import SkidSteerRoverModel
from deap import base, creator, tools, algorithms
import multiprocessing

class GeneticRoverParameterIdentifier:
    def __init__(self, excel_file, sheet_name, population_size=50, generations=20, isLinear = True,
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
        self.isLinear = isLinear



        self.param_bounds = param_bounds or [
    (0.01, 0.05),    # scale: desacoplamento da rotação
    (0.01, 4.0),     # gain_L
    (0.01, 2.0),     # tau_L
    (0.01, 10.0)    #  zeta_L
]



        self.constants = constants or {
            "mass" : 15, # mass
            "L" : 0.3, # Wheelbase [m] - critical for turn calculations
            "r" : 0.1,  # Wheel radius [m] - converts angular to linear velocity
            "pwm_min": -100,  # Minimum PWM value (full reverse)
            "pwm_max": 100  # Maximum PWM value (full forward)
        }

        self.last_sim_data = []

    # def scale_pwm(self, pwm, motor_id):
    #     return  (pwm - 1500) * (100 / 400)
    def scale_pwm(self, pwm, motor_id):
        return  pwm

    def evaluate(self, individual):
        try:

            vscale, vgain, vtau, vzeta, ascale, again, atau, azeta = individual

            rover = SkidSteerRoverModel(
                v_scale=vscale,
                vgain_L=vgain,
                vtau_L=vtau,
                vzeta_L=vzeta,
                angular_force_scale=ascale,
                again_L=again,
                atau_L=atau,
                azeta_L=azeta
            )



            state = np.array([0, 0, 0, 0, 0])
            dt = 0.05
            battery_voltage = 48
            error_total = 0
            sim_data = []


            for idx, row in self.data_real.iterrows():


                pwm_inputs = np.array([
                    self.scale_pwm(row['RCOU.C1'], 1),
                    self.scale_pwm(row['RCOU.C2'], 2),
                    self.scale_pwm(row['RCOU.C3'], 3),
                    self.scale_pwm(row['RCOU.C4'], 4)
                ])

                state = rover.dynamics(pwm_inputs)

                linear_real = row['GPS[0].Spd']
                angular_real = row['IMU[0].GyrZ']
                linear_sim = state[3]
                angular_sim = state[4]


                if np.isnan(linear_sim) or np.isnan(angular_sim):
                    return (1e6,)

                if self.isLinear:
                    error = abs(linear_real - linear_sim) + abs(angular_real - angular_sim)
                else:
                    error = abs(angular_real - angular_sim)


                error_total += error

                sim_data.append({
                    "time": row['timestamp(ms)'] / 1000.0,
                    "linear_real": linear_real,
                    "linear_sim": linear_sim,
                    "angular_real": angular_real,
                    "angular_sim": angular_sim,
                    "pwm1": pwm_inputs[0],
                    "pwm2": pwm_inputs[1],
                    "pwm3": pwm_inputs[2],
                    "pwm4": pwm_inputs[3]
                })

            self.last_sim_data = sim_data
            return error_total,

        except Exception:
            return (1e6,)


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


        toolbox.register("select", tools.selTournament, tournsize=1)

        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

        population = toolbox.population(n=self.population_size)

        for gen in range(self.generations):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.1)

            for i, ind in enumerate(offspring):
                offspring[i] = GeneticRoverParameterIdentifier.clip_individual(ind, self.param_bounds)
            fits = toolbox.map(toolbox.evaluate, offspring)
            # fits = list(map(toolbox.evaluate, offspring))
            for ind, fit in zip(offspring, fits):
                ind.fitness.values = fit

            elite_size = 1
            elite = tools.selBest(population, elite_size)  # os melhores da geração atual
            population = toolbox.select(offspring, k=len(population) - elite_size)
            population.extend(elite)

            valid = [ind for ind in population if ind.fitness.valid and not np.isnan(ind.fitness.values[0])]
            if valid:
                top = tools.selBest(valid, 1)[0]


            print(f"Geração {gen+1}: Erro do melhor indivíduo = {top.fitness.values[0]:.4f}")

        best_individual = tools.selBest([ind for ind in population if not np.isnan(ind.fitness.values[0])], 1)[0]
        print("Melhores parâmetros encontrados:", best_individual)

        self.evaluate(best_individual)
        return best_individual



    def plot_results(self):
        if not self.last_sim_data:
            print("Nenhum dado para plotar.")
            return

        time = [d['time'] for d in self.last_sim_data]
        linear_real = [d['linear_real'] for d in self.last_sim_data]
        linear_sim = [d['linear_sim'] for d in self.last_sim_data]
        angular_real = [d['angular_real'] for d in self.last_sim_data]
        angular_sim = [d['angular_sim'] for d in self.last_sim_data]

        pwm1 = [d.get('pwm1', 0) for d in self.last_sim_data]
        pwm2 = [d.get('pwm2', 0) for d in self.last_sim_data]
        pwm3 = [d.get('pwm3', 0) for d in self.last_sim_data]
        pwm4 = [d.get('pwm4', 0) for d in self.last_sim_data]

        plt.figure(figsize=(12, 8))

        # Velocidade linear
        plt.subplot(3, 1, 1)
        plt.plot(time, linear_real, label='Vel. Linear Real')
        plt.plot(time, linear_sim, label='Vel. Linear Simulada')
        plt.ylabel("Velocidade Linear [m/s]")
        plt.legend()
        plt.grid(True)

        # Velocidade angular
        plt.subplot(3, 1, 2)
        plt.plot(time, angular_real, label='Vel. Angular Real')
        plt.plot(time, angular_sim, label='Vel. Angular Simulada')
        plt.ylabel("Velocidade Angular [rad/s]")
        plt.xlabel("Tempo [s]")
        plt.legend()
        plt.grid(True)

        # PWM
        plt.subplot(3, 1, 3)
        plt.plot(time, pwm1, label="PWM FL")
        plt.plot(time, pwm2, label="PWM FR")
        plt.plot(time, pwm3, label="PWM RL")
        plt.plot(time, pwm4, label="PWM RR")
        plt.ylabel("PWM [-100, 100]")
        plt.xlabel("Tempo [s]")
        plt.title("Sinais PWM por Motor")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    @staticmethod
    def separar_parametros_rover_a(individual):
        """
        Separa os parâmetros do vetor individual em dois dicionários:
        - angular_params: para modelagem da velocidade angular
        - linear_params: para modelagem da velocidade linear
        """
        (a_scale, v_scale,
         again_L, atau_L, azeta_L,
         again_R, atau_R, azeta_R,
         vgain_L, vtau_L, vzeta_L,
         vgain_R, vtau_R, vzeta_R) = individual

        angular_params = {
            "angular_force_scale": a_scale,
            "gain_L": again_L,
            "tau_L": atau_L,
            "zeta_L": azeta_L,
            "gain_R": again_R,
            "tau_R": atau_R,
            "zeta_R": azeta_R
        }

        linear_params = {
            "v_scale": v_scale,
            "gain_L": vgain_L,
            "tau_L": vtau_L,
            "zeta_L": vzeta_L,
            "gain_R": vgain_R,
            "tau_R": vtau_R,
            "zeta_R": vzeta_R
        }

        return angular_params, linear_params

    @staticmethod
    def clip_individual(individual, bounds):
        for i, (min_val, max_val) in enumerate(bounds):
            individual[i] = np.clip(individual[i], min_val, max_val)
        return individual