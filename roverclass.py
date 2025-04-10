import numpy as np

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
        F_resist = -self.C_r * v
        Tau_resist = -self.C_omega * omega

        # Dinâmica translacional e rotacional com resistência
        a_linear = self.linear_force_scale * ((f_left + f_right) + F_resist) / self.m
        alpha = self.angular_force_scale * (((f_left - f_right) * self.r) + Tau_resist) / self.I

        # # Dinâmica translacional e rotacional
        # a_linear = (f_left + f_right) / self.m
        # alpha = (f_left - f_right) * self.r / self.I

        v += a_linear * dt
        omega += alpha * dt


        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt

        return np.array([x, y, theta, v, omega])
