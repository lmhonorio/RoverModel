import numpy as np
import matplotlib.pyplot as plt


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


# Parâmetros do rover
m = 10  # Massa [kg]
I = 3  # Momento de inércia [kg*m²]
L = 0.5  # Distância entre rodas [m]
r = 0.1  # Raio das rodas [m]
b = 0.3  # Resistência ao giro
C_d = 0.01  # Coeficiente de arrasto
C_r = 0.01  # Coeficiente de resistência ao rolamento
max_torque = 80  # Torque máximo de cada motor [Nm]

resistance = 1.00
inductance = 0.2
kt = 0.08
ke = 0.05
friction = 0.01
inertia = 0.01
pwm_min = -100
pwm_max = 100
current_limit = 50
damping_factor = 0.05

# Criando os motores
motor_FL = MotorModel(max_torque, resistance, inductance, kt, ke, friction, inertia, pwm_min, pwm_max, current_limit, damping_factor)
motor_FR = MotorModel(max_torque, resistance, inductance, kt, ke, friction, inertia, pwm_min, pwm_max, current_limit, damping_factor)
motor_RL = MotorModel(max_torque, resistance, inductance, kt, ke, friction, inertia, pwm_min, pwm_max, current_limit, damping_factor)
motor_RR = MotorModel(max_torque, resistance, inductance, kt, ke, friction, inertia, pwm_min, pwm_max, current_limit, damping_factor)

# Criando o modelo do rover
rover_model = SkidSteerRoverModel(m, I, L, r, b, C_d, C_r, motor_FL, motor_FR, motor_RL, motor_RR)

# Estado inicial do rover
state = np.array([0, 0, 0, 0, 0])  # [x, y, theta, v, omega]

# Configuração dos sinais PWM e da tensão da bateria ao longo do tempo
battery_voltage = 48  # Inicialmente a bateria está com 24V
pwm_inputs = np.array([80, 100, 80, 100])  # Sinais PWM para os motores

# pwm_inputs = np.array([pwm_FL, pwm_FR, pwm_RL, pwm_RR])
pwm_sequences = [
    (np.array([80, 100, 80, 100]), 3.0),
    (np.array([100, 80, 100, 80]), 2.0),
    (np.array([50, 50, 50, 50]), 4.0),
    (np.array([50, -50, 50, -50]), 6.0)
]

dt = 0.1  # Passo de tempo
num_steps = 20  # Número de passos da simulação

trajectory = []
velocities = []
battery_voltages = []
time_global = []

t= 0

trajectory.append(state[:3])
velocities.append(state[3:])
battery_voltages.append(battery_voltage)
time_global.append(t)
t += dt


for pwm_inputs, duration in pwm_sequences:
    num_steps = int(duration / dt)
    for _ in range(num_steps):
        state = rover_model.dynamics(state, pwm_inputs, dt, battery_voltage)
        trajectory.append(state[:3])
        velocities.append(state[3:])
        battery_voltages.append(battery_voltage)
        time_global.append(t)
        t += dt

trajectory = np.array(trajectory)
velocities = np.array(velocities)
time_global = np.array(time_global)

# Gráfico da trajetória
plt.figure(figsize=(8, 6))
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label="Trajetória do Rover")
plt.scatter(trajectory[0, 0], trajectory[0, 1], color='g', marker='o', label="Posição Inicial")
plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='r', marker='x', label="Posição Final")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Simulação do Rover Skid-Steering")
plt.legend()
plt.grid(True)
plt.show()

# Gráfico das velocidades
plt.figure(figsize=(8, 6))
plt.plot(time_global, velocities[:, 0], 'r-', label="Velocidade Linear (v)")
plt.plot(time_global, velocities[:, 1], 'g-', label="Velocidade Angular (ω)")
plt.xlabel("Tempo [s]")
plt.ylabel("Velocidade")
plt.title("Evolução das Velocidades do Rover")
plt.legend()
plt.grid(True)
plt.show()

# Gráfico da tensão da bateria ao longo do tempo
# plt.figure(figsize=(8, 6))
# plt.plot(np.arange(num_steps) * dt, battery_voltages, 'b-', label="Tensão da Bateria (V)")
# plt.xlabel("Tempo [s]")
# plt.ylabel("Tensão [V]")
# plt.title("Variação da Tensão da Bateria Durante a Simulação")
# plt.legend()
# plt.grid(True)
# plt.show()