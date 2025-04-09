import numpy as np
import matplotlib.pyplot as plt
from roverclass import MotorModel, SkidSteerRoverModel


#I, kt, ktarget_velocity, time_constant, torque_scale, time_constant_linear, time_constant_angular, rwheel, lwheel = individual = [11.362369375359686, 0.07370867052404233, 0.03077704492650191, 2.3980967708841745, 3.377603602633413, 2.673203620155811, 0.507326963775808, 0.957779291276004, 1.0367148165604743]
# Rover parameters - parametros conhecidos
# Rover physical parameters (essential ones)
m = 15.0    # Mass [kg] - directly used in dynamics calculations
L = 0.5     # Wheelbase [m] - critical for turn calculations
r = 0.1     # Wheel radius [m] - converts angular to linear velocity

# Motor control parameters (actually used in simplified model)
pwm_min = -100   # Minimum PWM value (full reverse)
pwm_max = 100    # Maximum PWM value (full forward)

# Rover parameters - parametros para serem otimizados
I = 3.0     # Moment of inertia [kg*m²] - used in angular dynamics
kt = 0.08       # Torque constant [Nm/A] - scales PWM to torque
ktarget_velocity = 0.035
time_constant = 0.1   # Motor response time [s] - first-order dynamics
torque_scale = 0.2    # Empirical scaling factor for torque output


# Rover dynamics parameters
time_constant_linear = 0.3   # Linear velocity response time [s]
time_constant_angular = 0.3  # Angular velocity response time [s]

rwheel = 1.3
lwheel = 0.8

# I, kt, ktarget_velocity, time_constant, torque_scale, time_constant_linear, time_constant_angular, rwheel, lwheel = [11.362369375359686, 0.07370867052404233, 0.03077704492650191, 2.3980967708841745, 3.377603602633413, 2.673203620155811, 0.507326963775808, 0.957779291276004, 1.0367148165604743]

# Create motor instances with only used parameters
motor_FR = MotorModel(kt=kt, ktarget_velocity=ktarget_velocity,  pwm_min=pwm_min, pwm_max=pwm_max, time_constant=time_constant, torque_scale=torque_scale,  orientation= -1)
motor_FL = MotorModel(kt=kt, ktarget_velocity=ktarget_velocity,  pwm_min=pwm_min, pwm_max=pwm_max, time_constant=time_constant, torque_scale=torque_scale,  orientation= 1)
motor_RL = MotorModel(kt=kt, ktarget_velocity=ktarget_velocity,  pwm_min=pwm_min, pwm_max=pwm_max, time_constant=time_constant, torque_scale=torque_scale,  orientation= 1)
motor_RR = MotorModel(kt=kt, ktarget_velocity=ktarget_velocity,  pwm_min=pwm_min, pwm_max=pwm_max, time_constant=time_constant, torque_scale=torque_scale,  orientation= -1)

# Initialize rover model with all active parameters
rover_model = SkidSteerRoverModel(
    m=m,                  # Mass
    I=I,                  # Moment of inertia
    L=L,                  # Wheelbase
    r=r,                  # Wheel radius
    motor_FL=motor_FL,    # Front left motor
    motor_FR=motor_FR,    # Front right motor
    motor_RL=motor_RL,    # Rear left motor
    motor_RR=motor_RR,    # Rear right motor
    rleft= rwheel,
    rright= lwheel,
    time_constant_linear=time_constant_linear,    # Linear response
    time_constant_angular=time_constant_angular   # Angular response
)

# Estado inicial do rover
state = np.array([0, 0, 0, 0, 0])  # [x, y, theta, v, omega]

# Configuração da bateria
battery_voltage = 48  # Tensão da bateria [V]

# Sequência de comandos PWM
#np.array([pwm_FR, pwm_FL, pwm_RL, pwm_RR], tempo)
#orientacao = np.array([-1, 1, 1, -1])
pwm_sequences = [
    (np.array([-80, 100, 100, -80]), 1.8),  # Curva suave para a direita
    (np.array([60, -40, -40, 60]), 1.7),  # Ré com curva leve
    (np.array([-90, 90, 90, -90]), 2.0),  # Movimento reto rápido
    (np.array([30, 40, 40, 30]), 1.6),  # Movimento misto
    (np.array([-50, 50, 50, -50]), 1.9),  # Movimento reto devagar
    (np.array([100, -80, -80, 100]), 2.0),  # Ré rápida curva para esquerda
    (np.array([-70, 70, 70, -70]), 1.7),  # Movimento reto médio
    (np.array([90, -90, -90, 90]), 1.5),  # Ré em alta velocidade
    (np.array([-60, 80, 80, -60]), 1.8),  # Curva controlada para a direita
    (np.array([50, -50, -50, 50]), 1.6)  # Ré lenta
]

#mask = np.array([-1, 1, 1, -1])  # Máscara para inverter sinais se necessário

dt = 0.1  # Passo de tempo [s]
trajectory = []
velocities = []
time_global = []

# Estado inicial
trajectory.append(state[:3])
velocities.append(state[3:])
time_global.append(0)

t = 0  # Tempo inicial

for original_pwm_inputs, duration in pwm_sequences:
    pwm_inputs = original_pwm_inputs
    num_steps = int(duration / dt)

    for _ in range(num_steps):
        state = rover_model.dynamics(state, pwm_inputs, dt, battery_voltage)
        linear_sim = state[3]
        angular_sim = state[4]
        trajectory.append(state[:3])
        velocities.append(state[3:])
        time_global.append(t)
        t += dt

# Convertendo para arrays numpy
trajectory = np.array(trajectory)
velocities = np.array(velocities)
time_global = np.array(time_global)

# Gráfico das velocidades
plt.figure(figsize=(8, 6))
plt.plot(time_global, velocities[:, 0], 'r-', label="Velocidade Linear (v)")
plt.plot(time_global, velocities[:, 1], 'g-', label="Velocidade Angular (ω)")
plt.xlabel("Tempo [s]")
plt.ylabel("Velocidade")
plt.title("Evolução das Velocidades do Rover (Modelo Simplificado)")
plt.legend()
plt.grid(True)
plt.show()