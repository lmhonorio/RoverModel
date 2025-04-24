import math
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

class ThirdOrderSystem:
    def __init__(self, K, tau1, tau2, tau3, dt):
        self.K = K
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau3 = tau3
        self.dt = dt
        self.reset()
        self.compute_discrete_coefficients()

    def reset(self):
        self.u = [0, 0, 0, 0]  # entrada atual e 3 anteriores
        self.y = [0, 0, 0]     # saída atual e 2 anteriores

    def compute_discrete_coefficients(self):
        T = self.dt
        s1 = 1 / self.tau1
        s2 = 1 / self.tau2
        s3 = 1 / self.tau3

        # Aproximação de Tustin para (s + s1)(s + s2)(s + s3)
        a0 = 1 + T*(s1 + s2 + s3) + (T**2)*(s1*s2 + s1*s3 + s2*s3) + (T**3)*(s1*s2*s3)
        self.b0 = self.K * (T**3 * s1*s2*s3) / a0
        self.b1 = 3 * self.b0
        self.b2 = 3 * self.b0
        self.b3 = self.b0

        # Coeficientes do denominador (simplificação para estabilidade)
        self.a1 = (3 + 2*T*(s1 + s2 + s3) + T**2*(s1*s2 + s1*s3 + s2*s3)) / a0
        self.a2 = (3 + T*(s1 + s2 + s3)) / a0
        self.a3 = 1 / a0

    def update(self, u_current):
        self.u = [u_current] + self.u[:3]
        y_new = (
            self.b0 * self.u[0] +
            self.b1 * self.u[1] +
            self.b2 * self.u[2] +
            self.b3 * self.u[3] -
            self.a1 * self.y[0] -
            self.a2 * self.y[1] -
            self.a3 * self.y[2]
        )
        self.y = [y_new] + self.y[:2]
        return y_new

class SecondOrderSystem:
    def __init__(self, K, tau, zeta, dt):
        self.K = K
        self.tau = tau
        self.zeta = zeta
        self.dt = dt
        self.reset()
        self.compute_discrete_coefficients()

    def reset(self):
        self.u = [0, 0, 0]
        self.y = [0, 0]

    def compute_discrete_coefficients(self):
        # Bilinear transform (Tustin's method)
        T = self.dt
        wn = 1.0 / self.tau
        a0 = T**2 * wn**2 + 2 * self.zeta * wn * T + 1
        self.b0 = self.K * T**2 * wn**2 / a0
        self.b1 = 2 * self.b0
        self.b2 = self.b0
        self.a1 = (2 * (T**2 * wn**2 - 1)) / a0
        self.a2 = (T**2 * wn**2 - 2 * self.zeta * wn * T + 1) / a0

    def update(self, u_current):
        self.u = [u_current] + self.u[:2]  # Shift inputs
        y_new = (
            self.b0 * self.u[0] +
            self.b1 * self.u[1] +
            self.b2 * self.u[2] -
            self.a1 * self.y[0] -
            self.a2 * self.y[1]
        )
        self.y = [y_new] + self.y[:1]  # Shift outputs
        return y_new



class EvaluateRoverParameters:
    def __init__(self, excel_file, sheet_name, dt=0.05):

        self.data_real = pd.read_excel(excel_file, sheet_name=sheet_name)
        self.last_sim_data = []
        self.dt = dt

        return


    def evaluate(self, individual):
        try:

            vscale, vgain, vtau, vzeta, ascale, again, atau, azeta, again2, atau2, azeta2   = individual

            rover = SkidSteerRoverModel(
                dt=self.dt,
                v_scale=vscale,
                vgain=vgain,
                vtau=vtau,
                vzeta=vzeta,
                a_scale=ascale,
                again=again,
                atau=atau,
                azeta = azeta,
                again2=again2,
                atau2=atau2,
                azeta2=azeta2
            )

            error_total = 0
            sim_data = []

            for idx, row in self.data_real.iterrows():
                pwm_inputs = np.array([
                    row['RCOU.C1'],
                    row['RCOU.C2'],
                    row['RCOU.C3'],
                    row['RCOU.C4']
                ])

                pwm_FR, pwm_FL, pwm_RL, pwm_RR = pwm_inputs * [-1, 1, 1, -1]

                # Entradas médias de cada lado
                input_L = (pwm_FL + pwm_RL) / 2
                input_R = (pwm_FR + pwm_RR) / 2

                torque = input_L - input_R
                forca = input_L + input_R

                state = rover.dynamics(pwm_inputs)

                linear_real = row['GPS[0].Spd']
                angular_real = row['IMU[0].GyrZ']
                linear_sim = state[3]
                angular_sim = state[4]

                error = abs(angular_real - angular_sim) + abs(linear_real - linear_sim)

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

        except:
            return 10^6


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



class SkidSteerRoverModel:
    def __init__(self,  dt = 0.05,
                 a_scale = 0.01,
                 v_scale=0.01,
                 again=1.0, atau=0.5, azeta=0.7,
                 again2=1.0, atau2=0.5, azeta2=0.7,
                 vgain=1.0, vtau=0.5, vzeta=0.7):

        self.a_scale = a_scale
        self.dt = dt
        self.time = 0

        self.state = np.array([0, 0, 0, 0, 0])  # [x, y, theta, v, omega]
        self.v_scale = v_scale

        self.vsys_L = SecondOrderSystem(K=vgain, tau=vtau, zeta=vzeta, dt=dt)
        self.asys_L = SecondOrderSystem(K=again, tau=atau, zeta=azeta, dt=dt)
        self.asys_R = SecondOrderSystem(K=again2, tau=atau2, zeta=azeta2, dt=dt)



    def dynamics(self, pwm_inputs):
        pwm_FR, pwm_FL, pwm_RL, pwm_RR = pwm_inputs * [-1,1,1,-1]

        # Entradas médias de cada lado
        input_L = (pwm_FL + pwm_RL)
        input_R = (pwm_FR + pwm_RR)

        torque = input_L - input_R
        forca = input_L+input_R

        # Atualizar resposta dos sistemas de 2ª ordem
        omega_L = self.asys_L.update(input_L )
        omega_R = self.asys_L.update(-input_R )
        v_LR = self.vsys_L.update(forca)

        omega = self.a_scale * (omega_L+omega_R)
        v = self.v_scale * (v_LR)

        x, y, theta, _, _ = self.state

        x = 0
        y = 0
        theta = 0
        self.time+= self.dt

        self.state = np.array([x, y, theta, v, omega])
        return self.state

