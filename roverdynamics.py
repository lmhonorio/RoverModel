import math

import numpy as np

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




class SkidSteerRoverModel:
    def __init__(self,  dt = 0.05,
                  angular_force_scale = 0.01,
                 v_scale=0.01,
                 again_L=1.0, atau_L=0.5, azeta_L=0.7,
                 vgain_L=1.0, vtau_L=0.5, vzeta_L=0.7):

        self.a_scale = angular_force_scale
        self.dt = dt

        self.state = np.array([0, 0, 0, 0, 0])  # [x, y, theta, v, omega]
        self.v_scale = v_scale

        self.vsys_L = SecondOrderSystem(K=vgain_L, tau=vtau_L, zeta=vzeta_L, dt=dt)
        self.asys_L = SecondOrderSystem(K=again_L, tau=atau_L, zeta=azeta_L, dt=dt)



    def dynamics(self, pwm_inputs):
        pwm_FR, pwm_FL, pwm_RL, pwm_RR = pwm_inputs * [-1,1,1,-1]

        # Entradas médias de cada lado
        input_L = (pwm_FL + pwm_RL) / 2
        input_R = (pwm_FR + pwm_RR) / 2

        # Atualizar resposta dos sistemas de 2ª ordem
        omega_LR = self.asys_L.update(-input_L + input_R)
        v_LR = self.vsys_L.update(input_L+input_R)

        omega = self.a_scale * (omega_LR)
        v = self.v_scale * (v_LR)

        x, y, theta, _, _ = self.state

        x = 0
        y = 0
        theta = 0

        self.state = np.array([x, y, theta, v, omega])
        return self.state

