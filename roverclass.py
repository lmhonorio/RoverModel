import numpy as np

class MotorModel:
    def __init__(self, kt, ktarget_velocity, pwm_min, pwm_max,
                 time_constant=0.2, torque_scale=0.1,  orientation = 1):
        self.kt = kt  # Torque constant [Nm/A]
        self.pwm_min = pwm_min
        self.pwm_max = pwm_max
        self.time_constant = time_constant
        self.torque_scale = torque_scale
        self.angular_velocity = 0  # Initial angular velocity
        self.orientation = orientation
        self.ktarget_velocity = ktarget_velocity

    def update_torque(self, pwm, dt, battery_voltage):
        normalized_pwm = np.clip(pwm / self.pwm_max, -1, 1)
        target_velocity = normalized_pwm * (battery_voltage / self.kt) * self.ktarget_velocity
        self.angular_velocity += (target_velocity - self.angular_velocity) * (dt / self.time_constant)
        self.torque = self.kt * normalized_pwm * battery_voltage * self.torque_scale
        return self.orientation* self.torque, 0, self.orientation * self.angular_velocity


class SkidSteerRoverModel:
    def __init__(self, m, I, L, r,
                 motor_FL, motor_FR, motor_RL, motor_RR,
                 rleft=1, rright=1,
                 time_constant_linear=0.3, time_constant_angular=0.3, turning_gain=0.1):
        self.m = m
        self.I = I
        self.L = L
        self.r = r
        self.time_constant_linear = time_constant_linear
        self.time_constant_angular = time_constant_angular
        self.turning_gain = turning_gain
        self.rleft = rleft
        self.rright = rright

        self.motor_FL = motor_FL
        self.motor_FR = motor_FR
        self.motor_RL = motor_RL
        self.motor_RR = motor_RR

    def dynamics(self, state, pwm_inputs, dt, battery_voltage):
        x, y, theta, v, omega = state
        pwm_FR, pwm_FL, pwm_RL, pwm_RR = pwm_inputs

        _, _, w_FL = self.motor_FL.update_torque(pwm_FL, dt, battery_voltage)
        _, _, w_FR = self.motor_FR.update_torque(pwm_FR, dt, battery_voltage)
        _, _, w_RL = self.motor_RL.update_torque(pwm_RL, dt, battery_voltage)
        _, _, w_RR = self.motor_RR.update_torque(pwm_RR, dt, battery_voltage)

        w_left = self.rleft * (w_FL + w_RL) / 2
        w_right = self.rright *(w_FR + w_RR) / 2

        target_v = self.r * (w_right + w_left) / 2
        target_omega = self.turning_gain * self.r * (w_left - w_right) / self.L

        v += (target_v - v) * (dt / self.time_constant_linear)
        omega += (target_omega - omega) * (dt / self.time_constant_angular)

        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt

        return np.array([x, y, theta, v, omega])
