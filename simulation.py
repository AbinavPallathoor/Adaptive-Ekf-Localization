"""Deterministic physics and sensor sampling shared by the desktop app."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import platform

import mujoco
import numpy as np

MODEL = Path(__file__).parent / 'models' / 'differential_drive.xml'
FIELDS = (
    'sim_time', 'wall_time', 'x', 'y', 'z', 'theta',
    'qw', 'qx', 'qy', 'qz', 'position_x', 'position_y',
    'left_angle', 'right_angle', 'left_velocity', 'right_velocity',
    'left_encoder', 'right_encoder',
    'true_gyro_x', 'true_gyro_y', 'true_gyro_z', 'gyro_x', 'gyro_y', 'gyro_z',
    'accel_x', 'accel_y', 'accel_z',
    'left_control', 'right_control', 'left_target', 'right_target',
)


@dataclass(frozen=True)
class Settings:
    drive_speed: float = 4.0
    turn_speed: float = 3.0
    ramp: float = 6.0
    log_hz: int = 20
    position_noise: float = 0.0
    encoder_noise: float = 0.0
    gyro_noise: float = 0.0
    seed: int = 42

    def __post_init__(self):
        values = (self.drive_speed, self.turn_speed, self.ramp,
                  self.position_noise, self.encoder_noise, self.gyro_noise)
        if not all(np.isfinite(v) and v >= 0 for v in values):
            raise ValueError('Speeds and noise must be finite and nonnegative')
        if not 0 < self.ramp <= 50 or max(self.drive_speed, self.turn_speed) > 12:
            raise ValueError('Wheel speed limit is 12 rad/s; ramp must be in (0, 50]')
        if self.log_hz not in (10, 20, 50):
            raise ValueError('Sampling rate must be 10, 20, or 50 Hz')
        if not isinstance(self.seed, int) or not 0 <= self.seed <= 2**31-1:
            raise ValueError('Seed must be a nonnegative 32-bit integer')


class Simulation:
    def __init__(self, settings=None):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.data = mujoco.MjData(self.model)
        self.target = np.zeros(2)
        self.configure(settings or Settings())
        self.reset()

    def configure(self, settings):
        self.settings = settings
        self.rng = np.random.default_rng(settings.seed)
        self.sample_steps = round(1 / settings.log_hz / self.model.opt.timestep)
        self.steps = 0
        self.target[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.latest = self.sample()

    def drive(self, direction):
        forward, turn = self.settings.drive_speed, self.settings.turn_speed
        commands = {'forward': (forward, forward), 'reverse': (-forward, -forward),
                    'left': (-turn, turn), 'right': (turn, -turn), 'stop': (0, 0)}
        if direction not in commands:
            raise ValueError('Unknown direction')
        self.target[:] = commands[direction]

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.target[:] = 0
        self.steps = 0
        self.rng = np.random.default_rng(self.settings.seed)
        mujoco.mj_forward(self.model, self.data)
        self.latest = self.sample()

    def advance(self, duration=.02):
        samples = []
        for _ in range(round(duration / self.model.opt.timestep)):
            limit = self.settings.ramp * self.model.opt.timestep
            self.data.ctrl[:] += np.clip(self.target - self.data.ctrl, -limit, limit)
            mujoco.mj_step(self.model, self.data)
            self.steps += 1
            if self.steps % self.sample_steps == 0:
                mujoco.mj_forward(self.model, self.data)
                self.latest = self.sample()
                samples.append(self.latest)
        return samples

    def sample(self):
        d, cfg = self.data, self.settings
        x, y, z = d.qpos[:3]
        qw, qx, qy, qz = d.qpos[3:7]
        theta = np.arctan2(2*(qw*qz + qx*qy), 1-2*(qy*qy + qz*qz))
        position = np.array([x, y]) + self.rng.normal(0, cfg.position_noise, 2)
        velocity = np.array([d.sensor('left_encoder').data[0], d.sensor('right_encoder').data[0]])
        encoders = velocity + self.rng.normal(0, cfg.encoder_noise, 2)
        true_gyro = d.sensor('gyro').data.copy()
        gyro = true_gyro + self.rng.normal(0, cfg.gyro_noise, 3)
        values = [float(d.time), datetime.now(timezone.utc).isoformat(),
                  x, y, z, theta, qw, qx, qy, qz, *position,
                  d.joint('left_wheel_joint').qpos[0], d.joint('right_wheel_joint').qpos[0],
                  *velocity, *encoders, *true_gyro, *gyro,
                  *d.sensor('accelerometer').data, *d.ctrl, *self.target]
        return {name: value if name == 'wall_time' else float(value)
                for name, value in zip(FIELDS, values, strict=True)}

    def metadata(self):
        return {'settings': asdict(self.settings), 'model_xml': MODEL.read_text(),
                'mujoco_version': mujoco.__version__, 'numpy_version': np.__version__,
                'python_version': platform.python_version(), 'platform': platform.platform(),
                'initial_state': {'sim_time': float(self.data.time),
                                  'qpos': self.data.qpos.tolist(),
                                  'qvel': self.data.qvel.tolist(),
                                  'ctrl': self.data.ctrl.tolist(),
                                  'target': self.target.tolist(),
                                  'rng_state': self.rng.bit_generator.state}}
