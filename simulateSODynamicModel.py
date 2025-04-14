import numpy as np
import matplotlib.pyplot as plt
from roverdynamics import SkidSteerRoverModel


#I, kt, ktarget_velocity, time_constant, torque_scale, time_constant_linear, time_constant_angular, rwheel, lwheel = individual = [11.362369375359686, 0.07370867052404233, 0.03077704492650191, 2.3980967708841745, 3.377603602633413, 2.673203620155811, 0.507326963775808, 0.957779291276004, 1.0367148165604743]
# Rover parameters - parametros conhecidos
# Rover physical parameters (essential ones)
m = 10.0    # Mass [kg] - directly used in dynamics calculations
L = 0.3     # Wheelbase [m] - critical for turn calculations
r = 0.2     # Wheel radius [m] - converts angular to linear velocity

# Motor control parameters (actually used in simplified model)
pwm_min = -100   # Minimum PWM value (full reverse)
pwm_max = 100    # Maximum PWM value (full forward)

# Rover parameters - parametros para serem otimizados
I = 2.0     # Moment of inertia [kg*m²] - used in angular dynamics
kt = 9.8       # Torque constant [Nm/A] - scales PWM to torque
ktarget_velocity = 0.1535
time_constant = 0.1   # Motor response time [s] - first-order dynamics
torque_scale = .05    # Empirical scaling factor for torque output


# Rover dynamics parameters
time_constant_angular = 1.01  # Angular velocity response time [s]

rwheel = 1.0
lwheel = 1.0



cr = 10
comega = 1.90

# I, kt, ktarget_velocity, time_constant, torque_scale, time_constant_linear, time_constant_angular, rwheel, lwheel = [11.362369375359686, 0.07370867052404233, 0.03077704492650191, 2.3980967708841745, 3.377603602633413, 2.673203620155811, 0.507326963775808, 0.957779291276004, 1.0367148165604743]

rFR = 0.2
rFL = 0.2
rRL = 0.2
rRR = 0.2

# individual = [7.755988242159815, 36.54332723498386, 0.1960132675789864, 0.28762870914277927, 1.0291899747122495,
#               0.03215144323168917, 3.541950513828812, 0.4846157118012735, 6.0, 0.37670415436957455, 1.3656705950706196,
#               1.2736602351503212, 7.874448702118817, 6.944908597346093, 0.21060876231648473, 0.23112138256489415, 0.25,
#               0.25]
#
# I, m, r, L, kt, ktarget_velocity, time_constant, torque_scale, linear_force_scale, angular_force_scale, rwheel, lwheel, C_r, C_omega, rFR, rFL, rRL, rRR = individual

# Create motor instances with only used parameters
# class SkidSteerRoverModel:
#     def __init__(self, m, I, L, r, dt = 0.05,
#                  rleft=1, rright=1, angular_force_scale = 0.01,
#                  gain_L=1.0, tau_L=0.5, zeta_L=0.7,
#                  gain_R=1.0, tau_R=0.5, zeta_R=0.7,
#                  v_scale=0.01):
# Initialize rover model with all active parameters

angular_force_scale, again_L,atau_L, azeta_L =  [0.03315811690034011, 0.26720455706332963, 5.0, 5.0]

rover_model = SkidSteerRoverModel(
    v_scale=0.01,
    vgain_L=0.01, vtau_L=0.5, vzeta_L=0.7,
    angular_force_scale=angular_force_scale,
    again_L=again_L, atau_L=atau_L, azeta_L=azeta_L
)

# Estado inicial do rover
state = np.array([0, 0, 0, 0, 0])  # [x, y, theta, v, omega]

# Configuração da bateria
battery_voltage = 48  # Tensão da bateria [V]

