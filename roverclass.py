import numpy as np
import pandas as pd

class MotorModel:
    def __init__(self, max_torque, resistance, inductance, kt, ke, friction, inertia, pwm_min, pwm_max, current_limit, damping_factor):
        self.max_torque = max_torque
        self.resistance = resistance  # Resistência do motor [Ohm]
        self.inductance = inductance  # Indutância do motor [H]
        self.kt = kt  # Constante de torque [Nm/A] (Aumentada)
        self.ke = ke  # Constante de força contra eletromotriz [V/(rad/s)]
        self.friction = friction  # Atrito interno (Reduzido)
        self.inertia = inertia  # Inércia do motor [kg*m²] (Reduzida para resposta mais rápida)
        self.pwm_min = pwm_min
        self.pwm_max = pwm_max
        self.current = 0  # Corrente inicial [A]
        self.torque = 0  # Torque inicial [Nm]
        self.angular_velocity = 0  # Velocidade angular inicial [rad/s]
        self.current_limit = current_limit  # Limite de corrente para evitar valores irreais
        self.damping_factor = damping_factor  # Fator de amortecimento (Reduzido)

    def update_torque(self, pwm, dt, battery_voltage):
        voltage = (pwm / self.pwm_max) * battery_voltage  # Convertendo PWM para tensão aplicada
        voltage_induced = self.ke * self.angular_velocity  # Tensão induzida pela rotação do motor
        voltage_net = voltage - voltage_induced  # Tensão efetiva no circuito

        di_dt = (voltage_net - self.current * self.resistance) / self.inductance  # Lei de Ohm e circuito RL
        self.current += di_dt * dt  # Atualiza corrente com suavização
        self.current = np.clip(self.current, -self.current_limit, self.current_limit)  # Limitando corrente

        self.torque = self.kt * self.current - self.friction * self.angular_velocity  # Torque gerado pelo motor

        # Atualização da velocidade angular considerando a inércia do motor e amortecimento
        angular_acceleration = (self.torque - self.damping_factor * self.angular_velocity) / self.inertia
        self.angular_velocity += angular_acceleration * dt

        return self.torque, self.current, self.angular_velocity


class SkidSteerRoverModel:
    def __init__(self, m, I, L, r, b, C_d, C_r, motor_FL, motor_FR, motor_RL, motor_RR):
        self.m = m
        self.I = I
        self.L = L
        self.r = r
        self.b = b
        self.C_d = C_d
        self.C_r = C_r
        self.motor_FL = motor_FL
        self.motor_FR = motor_FR
        self.motor_RL = motor_RL
        self.motor_RR = motor_RR

    def dynamics(self, state, pwm_inputs, dt, battery_voltage):
        x, y, theta, v, omega = state
        pwm_FL, pwm_FR, pwm_RL, pwm_RR = pwm_inputs

        tau_FL, _, _ = self.motor_FL.update_torque(pwm_FL, dt, battery_voltage)
        tau_FR, _, _ = self.motor_FR.update_torque(pwm_FR, dt, battery_voltage)
        tau_RL, _, _ = self.motor_RL.update_torque(pwm_RL, dt, battery_voltage)
        tau_RR, _, _ = self.motor_RR.update_torque(pwm_RR, dt, battery_voltage)

        # Convertendo a velocidade angular dos motores para a velocidade das rodas
        # Torque total em cada lado (esquerda e direita)
        tau_L = (tau_FL + tau_RL) / 2
        tau_R = (tau_FR + tau_RR) / 2

        # Velocidades das rodas
        omega_L = tau_L / (self.m * self.r)
        omega_R = tau_R / (self.m * self.r)

        # Velocidade linear e angular do veículo
        v_new = (self.r / 2) * (omega_R + omega_L)
        omega_new = (self.r / self.L) * (omega_R - omega_L)

        # Modelagem da resistência ao movimento
        F_drag = self.C_d * v**2
        F_rolling = self.C_r * v
        F_resistance = F_drag + F_rolling

        # Atualização das velocidades considerando resistências
        v = v_new - (F_resistance / self.m) * dt
        omega = omega_new - (self.b / self.I) * omega * dt

        # Atualização da posição
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt

        return np.array([x, y, theta, v, omega])



class RoverController:
    """
    Controlador de heading e velocidade (PI/PID simples)
    e alocação para PWM (modelo de equações simplificadas).
    """
    def __init__(self, kp_heading=2.0, kd_heading=0.0,
                 kp_speed=1.0, kd_speed=0.0,
                 pwm_max=100, wheelbase=0.5):
        self.kp_heading = kp_heading
        self.kd_heading = kd_heading
        self.kp_speed   = kp_speed
        self.kd_speed   = kd_speed
        self.pwm_max    = pwm_max
        self.L          = wheelbase  # Distância entre rodas

        # Estados para derivadas
        self.prev_heading_error = 0.0
        self.prev_speed_error   = 0.0

    def compute_control(self, heading_error, heading_error_dot,
                        speed_error, speed_error_dot):
        """
        Retorna (u_heading, u_speed) => sinal de controle para heading e speed.
        """
        # PID/PI simplificado, sem integral por exemplo.
        # heading
        ctrl_heading = self.kp_heading * heading_error + self.kd_heading * heading_error_dot
        # speed
        ctrl_speed   = self.kp_speed   * speed_error   + self.kd_speed   * speed_error_dot
        return ctrl_heading, ctrl_speed

    def pwm_allocator(self, v_des, w_des, battery_voltage):
        """
        Converte (v_des, w_des) -> (pwm_left, pwm_right).
        Modelo simples de skid-steer:
            v_des = (wL + wR)/2 * R
            w_des = (wR - wL)/L * R
        em que R = raio da roda => assumido 1 para simplificar
        e 'pwm' ~ w(angular)
        """
        # Para simplificar, assumimos:
        # w_left  = (v_des - w_des * L/2)/R
        # w_right = (v_des + w_des * L/2)/R
        # E pwm ~ w*(pwm_max / w_max). Faltam escalas exatas, mas é um ex.
        # Se a bateria estiver com 48 V, apenas normalizamos por pwm_max.

        R = 1.0  # Raio fictício
        w_left  = (v_des - w_des*(self.L/2)) / R
        w_right = (v_des + w_des*(self.L/2)) / R

        # Limit w_(left/right) para a pwm
        # Exemplo: pwm ~ w. Ajuste se quiser um mapeamento real.
        pwm_left  = np.clip(w_left,  -self.pwm_max, self.pwm_max)
        pwm_right = np.clip(w_right, -self.pwm_max, self.pwm_max)
        return 90*pwm_left, 90*pwm_right



# Classe para carregar obstáculos a partir da planilha
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