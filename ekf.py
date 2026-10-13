"""Extended Kalman filter for the robot's pose [x, y, heading].

Prediction: distance from the wheel encoders and heading change from the
gyro, which (unlike the encoders) is not fooled by wheel slip. Update: noisy
x, y position fixes. Noise levels are fixed; the Adaptive EKF will tune them.
"""

import math

import numpy as np

DISTANCE_NOISE = .02  # encoder distance error, fraction of distance travelled
SLIP_NOISE = 1e-4  # m, unmodelled sideways/along-track slip per prediction
SAMPLE_PERIOD = .05  # s between predictions (robot.SAMPLE_PERIOD)


class EKF:
    def __init__(self, x, y, heading, fix_noise, gyro_noise, dt=SAMPLE_PERIOD):
        """Start at a known pose with fix noise σ (m) and gyro noise σ (rad/s)."""
        self.state = np.array([x, y, heading], dtype=float)
        self.P = np.diag([1e-6, 1e-6, 1e-6])
        self.fix_noise = max(fix_noise, 1e-3)
        self.turn_variance = (gyro_noise*dt)**2 + 1e-8

    @property
    def x(self):
        return self.state[0]

    @property
    def y(self):
        return self.state[1]

    @property
    def heading(self):
        return self.state[2]

    def predict(self, distance, turn):
        """Move by distance (m) along the midpoint heading, turning by turn (rad)."""
        middle = self.heading + turn/2
        c, s = math.cos(middle), math.sin(middle)
        self.state += [distance*c, distance*s, turn]
        self.state[2] = math.remainder(self.state[2], math.tau)
        # Jacobians of the motion model with respect to the state and to [distance, turn].
        F = np.array([[1, 0, -distance*s], [0, 1, distance*c], [0, 0, 1]])
        G = np.array([[c, -distance*s/2], [s, distance*c/2], [0, 1]])
        M = np.diag([(DISTANCE_NOISE*distance)**2, self.turn_variance])
        Q = G @ M @ G.T + np.diag([SLIP_NOISE**2, SLIP_NOISE**2, 0])
        self.P = F @ self.P @ F.T + Q

    def update_position(self, fix_x, fix_y):
        """Correct with a position fix (m)."""
        H = np.array([[1., 0, 0], [0, 1, 0]])
        R = np.eye(2) * self.fix_noise**2
        innovation = np.array([fix_x, fix_y]) - H @ self.state
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state += K @ innovation
        self.state[2] = math.remainder(self.state[2], math.tau)
        # Joseph form keeps P symmetric and positive definite.
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
