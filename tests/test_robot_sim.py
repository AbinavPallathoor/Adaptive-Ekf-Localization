from pathlib import Path
import math
import sys

import mujoco
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import robot


def run_laps(route, laps=1):
    model, data = robot.load()
    start = robot.true_pose(data)
    follower = robot.RouteFollower(route, start)
    poses, lap_ends = [], []
    while follower.laps < laps and data.time < 200:
        robot.apply_target(model, data, follower.target(data), robot.STEPS_PER_SAMPLE)
        poses.append(robot.true_pose(data))
        if follower.laps > len(lap_ends):
            lap_ends.append(poses[-1])
    return start, poses, lap_ends


# The figure 8 switches turn direction at the lap point, so its heading lags there.
@pytest.mark.parametrize('route,heading_tolerance', [('square', 2), ('hexagon', 2), ('figure8', 5)])
def test_routes_close_after_one_lap(route, heading_tolerance):
    start, _, (end,) = run_laps(route)
    assert math.dist(start[:2], end[:2]) < .03
    assert abs(robot.wrap(end[2] - start[2])) < math.radians(heading_tolerance)


def test_figure8_does_not_drift_over_laps():
    start, _, ends = run_laps('figure8', laps=3)
    distances = [math.dist(start[:2], end[:2]) for end in ends]
    assert max(distances) - min(distances) < .005


@pytest.mark.parametrize('route,size', [('square', 1.0), ('hexagon', 1.2)])
def test_polygons_have_the_expected_size(route, size):
    _, poses, _ = run_laps(route)
    xs = [p[0] for p in poses]
    # Square: 1 m sides. Hexagon: 0.6 m sides, 1.2 m between opposite corners.
    assert max(xs) - min(xs) == pytest.approx(size, abs=.03)


def test_square_turns_on_the_spot():
    _, poses, _ = run_laps('square')
    # The axle midpoint never strays from the square's edges.
    start_x, start_y = poses[0][:2]
    for x, y, _ in poses:
        on_edge = min(abs(x - start_x), abs(x - start_x - 1), abs(y - start_y), abs(y - start_y - 1))
        assert on_edge < .02


def test_sensors_are_repeatable_and_rate_limited():
    def fixes(seed):
        model, data = robot.load()
        sensors = robot.Sensors(seed=seed, gyro_noise=.01, fix_noise=.1, fix_rate=1)
        return [sensors.sample(robot.readings(data)) for _ in range(40)]
    first, again, other = fixes(1), fixes(1), fixes(2)
    assert first == again
    assert first != other
    with_fix = [s for s in first if s['fix_x'] is not None]
    assert len(with_fix) == 2  # 40 samples at 20 Hz is 2 s, at 1 fix per second


def test_zero_noise_sensors_match_truth():
    model, data = robot.load()
    sensors = robot.Sensors(seed=0, gyro_noise=0, fix_noise=0, fix_rate=5)
    truth = robot.readings(data)
    sample = sensors.sample(truth)
    assert sample['fix_x'] == truth['x'] and sample['fix_y'] == truth['y']
    assert sample['gyro_z'] == truth['gyro_z']


def test_commands():
    assert robot.parse_command('speed 5') == ('speed', 5)
    assert robot.parse_command('speed 0') is None
    assert robot.parse_command('fix_noise 0.25') == ('fix_noise', .25)
    assert robot.parse_command('gyro_noise 0') == ('gyro_noise', 0)
    for bad in ('fix_noise -0.1', 'fix_noise nan', 'gyro_noise 5', 'fix_noise x', 'nonsense'):
        assert robot.parse_command(bad) is None


def test_samples_report_the_noise_in_force():
    model, data = robot.load()
    sensors = robot.Sensors(seed=0, gyro_noise=.01, fix_noise=.1, fix_rate=1)
    assert sensors.sample(robot.readings(data))['fix_sigma'] == .1
    sensors.set_noise(fix_noise=.4)
    sample = sensors.sample(robot.readings(data))
    assert (sample['fix_sigma'], sample['gyro_sigma']) == (.4, .01)


def test_noise_step_happens_once_at_its_time():
    sensors = robot.Sensors(seed=0, gyro_noise=.01, fix_noise=.1, fix_rate=1)
    step = robot.NoiseStep(30, fix_noise=.3)
    step.apply(sensors, 29.95)
    assert sensors.fix_noise == .1
    step.apply(sensors, 30)
    assert (sensors.fix_noise, sensors.gyro_noise) == (.3, .01)
    sensors.set_noise(fix_noise=.2)
    step.apply(sensors, 40)
    assert sensors.fix_noise == .2
    robot.NoiseStep(None).apply(sensors, 100)
    assert sensors.fix_noise == .2
