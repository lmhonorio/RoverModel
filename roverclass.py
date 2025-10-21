import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Classe para carregar obstáculos a partir da planilha testeststewtsat
class ObstacleLoader:
    def __init__(self, file_path, sheet_name):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.obstacles = []
        self.load_obstacles()

    def load_obstacles(self):
        df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)

        self.obstacles = [
            {
                "pos": (row["Px"], row["Py"]),
                "size": (row["Vx_largura"], row["Vy_altura"]),
                "color": (255, 0, 0),
                "label": row["ID"]
            }
            for _, row in df.iterrows()
        ]

    def get_obstacles(self):
        return self.obstacles


class MotorModel:
    def __init__(self, kt, ktarget_velocity, pwm_min, pwm_max,
                 time_constant=0.2, torque_scale=0.1, orientation=1, wheel_radius=0.1, max_torque = 5):
        self.kt = kt
        self.ktarget_velocity = ktarget_velocity
        self.pwm_min = pwm_min
        self.pwm_max = pwm_max
        self.time_constant = time_constant
        self.torque_scale = torque_scale
        self.angular_velocity = 0
        self.orientation = orientation
        self.wheel_radius = wheel_radius
        self.max_torque = max_torque

    def update_force(self, pwm, dt, battery_voltage):
        normalized_pwm = np.clip(pwm / self.pwm_max, -1, 1)
        target_velocity = normalized_pwm * (battery_voltage / self.kt) * self.ktarget_velocity
        self.angular_velocity += (target_velocity - self.angular_velocity) * (dt / self.time_constant)

        torque = self.kt * self.angular_velocity * self.torque_scale

        # torque = np.clip(torque,0, self.max_torque)

        force = torque / self.wheel_radius

        return self.orientation * force, self.orientation * torque, self.orientation * self.angular_velocity



class EvaluateRoverParameters:
    def __init__(self, excel_file, sheet_name):

        self.data_real = pd.read_excel(excel_file, sheet_name=sheet_name)
        self.last_sim_data = []

        return


    def imprimir_resultado(self, resultado, nomes):
        print("Valores dos parâmetros encontrados:\n")
        for valor, nome in zip(resultado, nomes):
            print(f"{nome:<25}: {valor:.4f} ")

    def plot_results(self):

        last_sim_data = self.last_sim_data

        if not last_sim_data:
            print("Nenhum dado para plotar.")
            return

        time = [d['time'] for d in last_sim_data]
        linear_real = [d['linear_real'] for d in last_sim_data]
        linear_sim = [d['linear_sim'] for d in last_sim_data]
        angular_real = [d['angular_real'] for d in last_sim_data]
        angular_sim = [d['angular_sim'] for d in last_sim_data]

        pwm1 = [d.get('pwm1', 0) for d in last_sim_data]
        pwm2 = [d.get('pwm2', 0) for d in last_sim_data]
        pwm3 = [d.get('pwm3', 0) for d in last_sim_data]
        pwm4 = [d.get('pwm4', 0) for d in last_sim_data]

        torque = [d['torque'] for d in last_sim_data]
        forca = [d['forca'] for d in last_sim_data]

        plt.figure(figsize=(12, 8))

        # Velocidade linear
        plt.subplot(4, 1, 1)
        plt.plot(time, linear_real, label='Vel. Linear Real')
        plt.plot(time, linear_sim, label='Vel. Linear Simulada')
        plt.ylabel("Velocidade Linear [m/s]")
        plt.legend()
        plt.grid(True)

        # Velocidade angular
        plt.subplot(4, 1, 2)
        plt.plot(time, angular_real, label='Vel. Angular Real')
        plt.plot(time, angular_sim, label='Vel. Angular Simulada')
        plt.ylabel("Velocidade Angular [rad/s]")
        plt.xlabel("Tempo [s]")
        plt.legend()
        plt.grid(True)

        # PWM
        plt.subplot(4, 1, 3)
        plt.plot(time, pwm1, label="PWM FR", linewidth=12)
        plt.plot(time, pwm2, label="PWM FL", linewidth=8)
        plt.plot(time, pwm3, label="PWM RL", linewidth=5)
        plt.plot(time, pwm4, label="PWM RR", linewidth=2)
        plt.ylabel("PWM [-100, 100]")
        plt.xlabel("Tempo [s]")
        plt.title("Sinais PWM por Motor")
        plt.legend()
        plt.grid(True)

        # PWM
        plt.subplot(4, 1, 4)
        plt.plot(time, forca, label='forca')
        plt.plot(time, torque, label='torque')
        plt.ylabel("forca e torque")
        plt.xlabel("Tempo [s]")
        plt.title("forcas atuantes")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def evaluate(self, individual):
        try:

            I, m, r, L, kt, ktarget_velocity, time_constant, torque_scale, linear_force_scale, angular_force_scale, rwheel, lwheel, C_r, C_omega, rFR, rFL, rRL, rRR = individual

            motor_FR = MotorModel(kt=kt, ktarget_velocity= ktarget_velocity,  pwm_min=-100, pwm_max=100, time_constant=time_constant, torque_scale=torque_scale, orientation=-1, wheel_radius = rFR)
            motor_FL = MotorModel(kt=kt, ktarget_velocity= ktarget_velocity,  pwm_min=-100, pwm_max=100, time_constant=time_constant, torque_scale=torque_scale, orientation=1, wheel_radius = rFL)
            motor_RL = MotorModel(kt=kt, ktarget_velocity= ktarget_velocity,  pwm_min=-100, pwm_max=100, time_constant=time_constant, torque_scale=torque_scale, orientation=1, wheel_radius = rRL)
            motor_RR = MotorModel(kt=kt, ktarget_velocity= ktarget_velocity,  pwm_min=-100, pwm_max=100, time_constant=time_constant, torque_scale=torque_scale, orientation=-1, wheel_radius = rRR)

            rover = SkidSteerRoverModel(
                m=m,
                I=I,
                L=L,
                r=r,
                motor_FL=motor_FL,
                motor_FR=motor_FR,
                motor_RL=motor_RL,
                motor_RR=motor_RR,
                rright=rwheel,
                rleft=lwheel,
                C_r=C_r,
                C_omega=C_omega,
                linear_force_scale=linear_force_scale,
                angular_force_scale=angular_force_scale
            )

            state = np.array([0, 0, 0, 0, 0])

            battery_voltage = 48
            error_total = 0
            sim_data = []


            for idx, row in self.data_real.iterrows():

                dt = 0.02

                pwm_inputs = np.array([
                    row['RCOU.C1'],
                    row['RCOU.C2'],
                    row['RCOU.C3'],
                    row['RCOU.C4']
                ])

                pwm_FR, pwm_FL, pwm_RL, pwm_RR = pwm_inputs

                # Entradas médias de cada lado
                input_L = (pwm_FL + pwm_RL) / 2
                input_R = (pwm_FR + pwm_RR) / 2

                torque = input_L - input_R
                forca = input_L + input_R

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
                    "angular_sim": angular_sim,
                    "pwm1": pwm_inputs[0],
                    "pwm2": pwm_inputs[1],
                    "pwm3": pwm_inputs[2],
                    "pwm4": pwm_inputs[3],
                    "forca": forca,
                    "torque": torque
                })

            self.last_sim_data = sim_data
            return error_total

        except Exception:
            return (1e6,)


class SkidSteerRoverModel:
    def __init__(self, m, I, L, r,
                 motor_FL, motor_FR, motor_RL, motor_RR,
                 rleft=1, rright=1,C_r = 2, C_omega = 2, linear_force_scale = 4, angular_force_scale = 2):
        self.m = m
        self.I = I
        self.L = L
        self.r = r
        self.rleft = rleft
        self.rright = rright

        self.motor_FL = motor_FL
        self.motor_FR = motor_FR
        self.motor_RL = motor_RL
        self.motor_RR = motor_RR

        self.linear_force_scale = linear_force_scale
        self.angular_force_scale = angular_force_scale


        self.C_r = C_r
        self.C_omega = C_omega

    def dynamics(self, state, pwm_inputs, dt, battery_voltage):
        x, y, theta, v, omega = state
        pwm_FR, pwm_FL, pwm_RL, pwm_RR = pwm_inputs

        f_FL, _, _ = self.motor_FL.update_force(pwm_FL, dt, battery_voltage)
        f_FR, _, _ = self.motor_FR.update_force(pwm_FR, dt, battery_voltage)
        f_RL, _, _ = self.motor_RL.update_force(pwm_RL, dt, battery_voltage)
        f_RR, _, _ = self.motor_RR.update_force(pwm_RR, dt, battery_voltage)

        f_left = self.rleft * (f_FL + f_RL)
        f_right = self.rright * (f_FR + f_RR)

        # Resistências
        F_resist = self.C_r * v
        Tau_resist = self.C_omega * omega



        # Dinâmica translacional e rotacional com resistência
        a_linear = self.linear_force_scale * ((f_left + f_right) - F_resist) / self.m
        # alpha = self.angular_force_scale * (((f_right - f_left) * self.r) - self.C_omega * Tau_resist) / self.I
        alpha = self.angular_force_scale * ((-f_right + f_left) * self.r - Tau_resist) / self.I

        # # Dinâmica translacional e rotacional
        # a_linear = (f_left + f_right) / self.m
        # alpha = (f_left - f_right) * self.r / self.I

        v += a_linear * dt
        omega += alpha * dt


        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt

        return np.array([x, y, theta, v, omega])
