import numpy as np

from task.player import Player


class PID:
    def __init__(
        self,
        P: float,
        I: float,
        D: float,
        saturation_max=np.inf,
        saturation_min=-np.inf,
    ):
        # PID gains
        self.P = P
        self.I = I
        self.D = D

        # output saturation
        self.saturation_max = saturation_max
        self.saturation_min = saturation_min

        # internal state
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_call = True

        # anti-windup limits for integral term (derived from saturation if possible)
        # If I == 0, integral clamping not used.
        if abs(self.I) > 1e-12:
            # keep integral such that I * integral roughly fits in saturation range
            self._integral_min = self.saturation_min / self.I
            self._integral_max = self.saturation_max / self.I
        else:
            self._integral_min = -np.inf
            self._integral_max = np.inf

    def compute(self, error: float, dt: float):
        """
        Standard PID:
            u = P*error + I*integral + D*derivative

        - error: by convention, use (setpoint - measurement) or (measurement - setpoint)
                 consistently across your calls. The caller in pid_player uses both signs.
        - dt: time step (seconds)
        """

        # protect dt
        if dt <= 0:
            derivative = 0.0
        else:
            if self.first_call:
                derivative = 0.0
                self.first_call = False
            else:
                derivative = (error - self.prev_error) / dt

        # integrate
        self.integral += error * dt

        # anti-windup: clamp integral to avoid runaway when I != 0
        self.integral = np.clip(self.integral, self._integral_min, self._integral_max)

        # PID formula
        output = self.P * error + self.I * self.integral + self.D * derivative

        # apply saturation (and return)
        output = np.clip(output, self.saturation_min, self.saturation_max)

        # store previous error
        self.prev_error = error

        return output


class PIDPlayer(Player):
    def __init__(self):
        self.name = "PID"
        self.alpha = 200
        self.anim_id = 1
        super().__init__()

        self.thruster_amplitude *= 3
        self.diff_amplitude *= 3

        # cascade 1 (Y)
        self.x_pid = PID(0.2 / 3, 0, 0.2, 25, -25)
        self.angle_pid = PID(0.02 / 3, 0, 0.01 / 3, 1, -1)
        # cascade 2 (X)
        self.y_pid = PID(2.0, 0.5, 1.5, 100, -100)
        self.y_speed_pid = PID(-1 / 3, 0, 0, 1, -1)

    def act(self, obs):
        thruster_left = self.thruster_mean
        thruster_right = self.thruster_mean

        (
            x_target,
            y_target,
            x_drone,
            y_drone,
            _,
            y_drone_speed,
            drone_angle,
            _,
        ) = obs

        # -------------------------
        # Cascade 1: control Y (altitude) -> sum thrust
        # (y - y_setpoint) -> PID -> y_speed_setpoint
        # (y_speed - y_speed_setpoint) -> PID -> T_sum
        # -------------------------
        # error_y defined as (current - desired) per description: (y - y_setpoint)
        error_y = y_target - y_drone
        # first PID produces setpoint for vertical speed
        y_drone_speed_setpoint = self.y_pid.compute(error_y, self.dt)

        # error for speed loop: (y_speed - y_speed_setpoint)
        error_y_drone_speed = y_drone_speed_setpoint - y_drone_speed

        # note: existing code expects negative passed into y_speed_pid:
        # thrust_0 = self.y_speed_pid.compute(-error_y_drone_speed, self.dt)
        # that makes the inner PID see (setpoint - measurement) convention.
        thrust_0 = self.y_speed_pid.compute(error_y_drone_speed, self.dt)

        # -------------------------
        # Cascade 2: control X -> angle -> thrust difference
        # (x - x_setpoint) -> PID -> angle_setpoint
        # (angle - angle_setpoint) -> PID -> T_diff
        # -------------------------
        error_x = x_drone - x_target
        angle_setpoint = self.x_pid.compute(error_x, self.dt)

        error_angle = drone_angle - angle_setpoint

        # same sign convention as above: pass negative so inner PID sees (setpoint - measurement)
        thrust_1 = self.angle_pid.compute(-error_angle, self.dt)

        # calculating motor thrusts
        thruster_left += thrust_0 * self.thruster_amplitude
        thruster_right += thrust_0 * self.thruster_amplitude

        thruster_left -= thrust_1 * self.diff_amplitude
        thruster_right += thrust_1 * self.diff_amplitude

        return thruster_left, thruster_right