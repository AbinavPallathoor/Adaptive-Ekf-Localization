"""Headless robot simulation: physics stepping, routes, readings and sensors.

Shared by viewer.py (which adds the MuJoCo window and the link to the app)
and the tests. Poses are of the axle midpoint, the point the robot turns about.
"""

import json
import math
from pathlib import Path

import mujoco
import numpy as np

MODEL = Path(__file__).resolve().parent / 'models' / 'differential_drive.xml'
WHEEL_RADIUS = .06
TRACK = .30
SAMPLE_PERIOD = .05  # s between readings sent to the app (20 Hz)
STEPS_PER_SAMPLE = 25  # physics steps of 2 ms per reading
RAMP = 6  # rad/s², limit on how fast motor commands change
SPEEDS = (1, 2, 5, 10)
FIX_RATES = (.5, 1, 2, 5)  # Hz
# Each route is a repeating list of segments, starting from the robot's start pose.
ROUTES = {
    'square': [('straight', 1.0), ('turn', math.pi/2)] * 4,
    'hexagon': [('straight', .6), ('turn', math.pi/3)] * 6,
    'figure8': [('arc', .4, math.tau), ('arc', .4, -math.tau)],
}
FIELDS = ('time', 'x', 'y', 'heading', 'left_angle', 'right_angle', 'gyro_z', 'fix_x', 'fix_y')

DRIVE_SPEED = 4  # rad/s wheel speed on straights and arcs (0.24 m/s)
TURN_SPEED = 3  # rad/s wheel speed when turning on the spot


def load():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def true_pose(data):
    """Ground-truth (x, y, heading) of the axle midpoint."""
    qw, qx, qy, qz = data.body('robot').xquat
    heading = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    x, y = data.site('axle').xpos[:2]
    return float(x), float(y), heading


def readings(data):
    """Ground truth plus noise-free wheel encoder angles and gyro yaw rate."""
    x, y, heading = true_pose(data)
    return {'time': float(data.time), 'x': x, 'y': y, 'heading': heading,
            'left_angle': float(data.joint('left_wheel_joint').qpos[0]),
            'right_angle': float(data.joint('right_wheel_joint').qpos[0]),
            'gyro_z': float(data.sensor('gyro').data[2])}


class Sensors:
    """Adds seeded Gaussian noise to the gyro and produces noisy position fixes."""

    def __init__(self, seed, gyro_noise, fix_noise, fix_rate):
        self.rng = np.random.default_rng(seed)
        self.gyro_noise, self.fix_noise = gyro_noise, fix_noise
        self.fix_every = round(1 / (fix_rate * SAMPLE_PERIOD))
        self.count = 0

    def sample(self, truth):
        gyro = truth['gyro_z'] + self.rng.normal(0, self.gyro_noise)
        fix_x = fix_y = None
        if self.count % self.fix_every == 0:
            fix_x, fix_y = (float(v) for v in np.array([truth['x'], truth['y']])
                            + self.rng.normal(0, self.fix_noise, 2))
        self.count += 1
        return {**truth, 'gyro_z': float(gyro), 'fix_x': fix_x, 'fix_y': fix_y}


def wrap(angle):
    return math.remainder(angle, math.tau)


class RouteFollower:
    """Autopilot that drives a route exactly using the true pose.

    Segment end points and headings are precomputed from the start pose, so
    errors never accumulate from lap to lap.
    """

    def __init__(self, route, start):
        x, y, heading = start
        self.segments = []
        for segment in ROUTES[route]:
            begin = (x, y, heading)
            if segment[0] == 'straight':
                x += segment[1]*math.cos(heading)
                y += segment[1]*math.sin(heading)
            elif segment[0] == 'turn':
                heading += segment[1]
            else:
                _, radius, angle = segment
                side = math.copysign(1, angle)
                cx, cy = x - side*radius*math.sin(heading), y + side*radius*math.cos(heading)
                heading += angle
                x, y = cx + side*radius*math.sin(heading), cy - side*radius*math.cos(heading)
            self.segments.append((segment, begin, (x, y, heading)))
        self.index = 0
        self.laps = 0
        self.swept = 0.0
        self.last_bearing = None

    def advance(self):
        self.index += 1
        self.swept, self.last_bearing = 0.0, None
        if self.index == len(self.segments):
            self.index = 0
            self.laps += 1

    def target(self, data):
        """Wheel-speed targets (left, right) in rad/s for the current segment."""
        x, y, heading = true_pose(data)
        settled = np.abs(data.ctrl).max() < .3
        segment, (x0, y0, h0), (x1, y1, h1) = self.segments[self.index]
        kind = segment[0]
        if kind == 'straight':
            ux, uy = math.cos(h0), math.sin(h0)
            remaining = (x1 - x)*ux + (y1 - y)*uy
            cross = -(x - x0)*uy + (y - y0)*ux
            if remaining < .002 and settled:
                self.advance()
                return self.target(data)
            # Speed profile that can stop within the ramp limit at the corner.
            speed = min(DRIVE_SPEED, .8*math.sqrt(2*RAMP*max(remaining, 0)/WHEEL_RADIUS))
            turn = 4*wrap(h0 - heading) - 10*cross
            return self.wheels(speed*WHEEL_RADIUS, turn * min(1, speed/DRIVE_SPEED))
        if kind == 'turn':
            error = wrap(h1 - heading)
            if abs(error) < .003 and settled:
                self.advance()
                return self.target(data)
            rate_limit = 2*WHEEL_RADIUS*TURN_SPEED/TRACK
            rate = min(rate_limit, .8*math.sqrt(2*RAMP*2*WHEEL_RADIUS/TRACK*abs(error)))
            return self.wheels(0, math.copysign(rate, error))
        _, radius, angle = segment
        side = math.copysign(1, angle)
        cx, cy = x0 - side*radius*math.sin(h0), y0 + side*radius*math.cos(h0)
        bearing = math.atan2(y - cy, x - cx)
        if self.last_bearing is None:
            # Measure progress from the ideal start, so lateness never carries over.
            self.last_bearing = math.atan2(y0 - cy, x0 - cx)
        self.swept += wrap(bearing - self.last_bearing)
        self.last_bearing = bearing
        if side*self.swept >= abs(angle) - 1e-3:
            self.advance()
            return self.target(data)
        speed = DRIVE_SPEED*WHEEL_RADIUS
        radial = math.hypot(x - cx, y - cy) - radius
        tangent = bearing + side*math.pi/2
        turn = side*speed/radius + side*8*radial + 3*wrap(tangent - heading)
        return self.wheels(speed, turn)

    @staticmethod
    def wheels(speed, turn):
        """Wheel speeds (rad/s) for forward speed (m/s) and turn rate (rad/s)."""
        spin = turn*TRACK/2
        return ((speed - spin)/WHEEL_RADIUS, (speed + spin)/WHEEL_RADIUS)


def apply_target(model, data, target, nstep):
    """Step the physics, ramping motor commands toward the wheel-speed target."""
    limit = RAMP * model.opt.timestep
    target = np.clip(target, -12, 12)
    for _ in range(nstep):
        data.ctrl[:] += np.clip(target - data.ctrl, -limit, limit)
        mujoco.mj_step(model, data)


def parse_command(line):
    """Return the speed-up factor from a 'speed N' line, or None if invalid."""
    match line.split():
        case ['speed', value] if value.isdigit() and int(value) in SPEEDS:
            return int(value)
    return None


def encode_sample(sample):
    return json.dumps(sample)


def decode_sample(line):
    """Parse one sample; position fixes are None when no fix was taken."""
    sample = json.loads(line)
    result = {field: float(sample[field]) for field in FIELDS if field not in ('fix_x', 'fix_y')}
    fix = sample['fix_x'], sample['fix_y']
    result['fix_x'], result['fix_y'] = (None, None) if None in fix else (float(fix[0]), float(fix[1]))
    return {field: result[field] for field in FIELDS}
