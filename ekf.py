"""Extended Kalman filter for the robot's pose [x, y, heading].

Prediction: distance from the wheel encoders and heading change from the
gyro, which (unlike the encoders) is not fooled by wheel slip. Update: noisy
x, y position fixes. The noise levels it assumes (EKFSettings) are tuning
parameters; they need not match the real sensors. AdaptiveEKF starts from
them, then estimates the fix σ (R) and, optionally, the gyro σ (heading part
of Q) itself. With gate on, a fix whose normalised innovation squared fails a
χ² test is rejected as an outlier.
"""

from collections import deque
from dataclasses import astuple, dataclass
import math

import numpy as np

SAMPLE_PERIOD = .05  # s between predictions (robot.SAMPLE_PERIOD)
MEMORY = 20  # s of recent readings the Adaptive EKF's noise estimates are based on
H = np.array([[1., 0, 0], [0, 1, 0]])  # a position fix measures x and y
GATE = 9.21  # χ² with 2 degrees of freedom, 99%: larger NIS means "reject the fix"


@dataclass(frozen=True)
class EKFSettings:
    fix_noise: float = .1  # m, assumed position-fix σ → measurement noise R
    gyro_noise: float = .01  # rad/s, assumed gyro σ → heading part of Q
    distance_noise: float = .02  # encoder distance σ as a fraction of distance → Q
    slip_noise: float = 1e-4  # m per prediction, unmodelled slip → Q
    initial_sigma: float = 1e-3  # m and rad, starting σ of x, y and heading → P₀
    gate: bool = False  # reject fixes that fail the χ² outlier test

    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for v in astuple(self)):
            raise ValueError('EKF settings must be finite and nonnegative')


class EKF:
    def __init__(self, x, y, heading, settings=EKFSettings(), dt=SAMPLE_PERIOD):
        """Start at a known pose (m, m, rad)."""
        self.state = np.array([x, y, heading], dtype=float)
        self.settings, self.dt = settings, dt
        self.P = np.eye(3) * max(settings.initial_sigma, 1e-6)**2
        self.nis = None  # normalised innovation squared of the latest fix

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
        self.P = F @ self.P @ F.T + self.process_noise(G, distance)

    def process_noise(self, G, distance):
        """Q for one prediction, from the encoder, gyro and slip noise."""
        cfg = self.settings
        M = np.diag([(cfg.distance_noise*distance)**2, (self.gyro_sigma*self.dt)**2 + 1e-8])
        return G @ M @ G.T + np.diag([cfg.slip_noise**2, cfg.slip_noise**2, 0])

    @property
    def gyro_sigma(self):
        return self.settings.gyro_noise

    def measurement_noise(self):
        return np.eye(2) * max(self.settings.fix_noise, 1e-3)**2

    def update_position(self, fix_x, fix_y):
        """Correct with a position fix (m). Returns False if the gate rejected it."""
        R = self.measurement_noise()
        innovation = np.array([fix_x, fix_y]) - H @ self.state
        S_inv = np.linalg.inv(H @ self.P @ H.T + R)
        # If the filter is right about its uncertainty, NIS follows a χ² distribution (2 dof).
        self.nis = float(innovation @ S_inv @ innovation)
        if self.settings.gate and self.nis > GATE:
            return False
        K = self.P @ H.T @ S_inv
        self.state += K @ innovation
        self.state[2] = math.remainder(self.state[2], math.tau)
        # Joseph form keeps P symmetric and positive definite.
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return True


class AdaptiveEKF(EKF):
    """EKF that estimates the fix and gyro noise instead of assuming them.

    Fix σ (R): residual-based covariance matching (Akhlaghi et al., 2017).
    After each fix, R ← αR + (1−α)(εεᵀ + H P Hᵀ), with post-fit residual
    ε = z − Hx and α = exp(−Δt/memory) for the time Δt since the last fix.

    Gyro σ (heading part of Q): from the gyro readings themselves. The true
    yaw rate changes smoothly, so the second difference g[k+1] − 2g[k] + g[k−1]
    is almost pure noise, with σ√6. A median over the last `memory` seconds
    ignores the brief jumps where the robot starts or stops turning.

    The settings' fix and gyro σ are only starting guesses. With
    adapt_gyro=False only R adapts and the gyro σ stays as assumed.
    """

    MIN_GYRO_SAMPLES = 20  # use the assumed gyro σ until this many second differences

    def __init__(self, x, y, heading, settings=EKFSettings(), memory=MEMORY, adapt_gyro=True, dt=SAMPLE_PERIOD):
        super().__init__(x, y, heading, settings, dt)
        self.adapt_gyro = adapt_gyro
        self.R = super().measurement_noise()
        self.gyro_readings = deque(maxlen=3)
        self.gyro_changes = deque()
        self.since_fix = 0.0
        self.memory = memory

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, memory):
        self._memory = max(float(memory), 1.)
        window = max(round(self._memory/self.dt), self.MIN_GYRO_SAMPLES)
        self.gyro_changes = deque(self.gyro_changes, maxlen=window)

    @property
    def fix_sigma(self):
        """Estimated position-fix σ (m)."""
        return math.sqrt(np.trace(self.R)/2)

    @property
    def gyro_sigma(self):
        """Estimated gyro σ (rad/s)."""
        if not self.adapt_gyro or len(self.gyro_changes) < self.MIN_GYRO_SAMPLES:
            return self.settings.gyro_noise
        # 1.4826 × median absolute value is the σ of Gaussian samples.
        return 1.4826*float(np.median(self.gyro_changes))/math.sqrt(6)

    def observe_gyro(self, rate):
        """Record a gyro reading (rad/s); call once per sample."""
        self.gyro_readings.append(rate)
        if len(self.gyro_readings) == 3:
            before, now, after = self.gyro_readings
            self.gyro_changes.append(abs(after - 2*now + before))

    def predict(self, distance, turn):
        super().predict(distance, turn)
        self.since_fix += self.dt

    def measurement_noise(self):
        return self.R

    def update_position(self, fix_x, fix_y):
        if not super().update_position(fix_x, fix_y):
            return False
        alpha = math.exp(-self.since_fix/self.memory)
        self.since_fix = 0.0
        residual = np.array([fix_x, fix_y]) - H @ self.state
        R = alpha*self.R + (1 - alpha)*(np.outer(residual, residual) + H @ self.P @ H.T)
        self.R = (R + R.T)/2
        return True
