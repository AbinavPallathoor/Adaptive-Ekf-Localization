from pathlib import Path
import math
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dead_reckoning import DeadReckoning, wheel_motion
from ekf import EKF
from localization import Tracker
import robot


def test_wheel_motion():
    assert wheel_motion(.1, .1) == pytest.approx((.1, 0))
    assert wheel_motion(-.15, .15) == pytest.approx((0, 1))


def test_prediction_alone_matches_dead_reckoning():
    odometry = DeadReckoning(0, 0, 0, 0, 0)
    ekf = EKF(0, 0, 0, fix_noise=.1, gyro_noise=.01)
    for step in range(1, 50):
        left, right = step*.2, step*.25
        before = odometry.left_angle, odometry.right_angle
        odometry.update(left, right)
        distance, turn = wheel_motion(.06*(left - before[0]), .06*(right - before[1]))
        ekf.predict(distance, turn)
    assert ekf.x == pytest.approx(odometry.x)
    assert ekf.y == pytest.approx(odometry.y)
    assert ekf.heading == pytest.approx(odometry.heading)


def test_uncertainty_grows_then_a_fix_pulls_the_estimate_in():
    ekf = EKF(0, 0, 0, fix_noise=.1, gyro_noise=.01)
    start = np.trace(ekf.P)
    for _ in range(40):
        ekf.predict(.012, 0)
    grown = np.trace(ekf.P)
    assert grown > start
    before = ekf.x
    ekf.update_position(before + .2, .1)
    assert before < ekf.x < before + .2
    assert ekf.y > 0
    assert np.trace(ekf.P) < grown
    assert np.allclose(ekf.P, ekf.P.T)


def test_heading_stays_wrapped():
    ekf = EKF(0, 0, 3.1, fix_noise=.1, gyro_noise=.01)
    ekf.predict(0, .2)
    assert -math.pi <= ekf.heading <= math.pi


def run_tracker(route, laps, fix_noise=.1, gyro_noise=.01, fix_rate=1):
    model, data = robot.load()
    follower = robot.RouteFollower(route, robot.true_pose(data))
    sensors = robot.Sensors(seed=42, gyro_noise=gyro_noise, fix_noise=fix_noise, fix_rate=fix_rate)
    tracker = Tracker(fix_noise=fix_noise, gyro_noise=gyro_noise)
    while follower.laps < laps:
        tracker.add(sensors.sample(robot.readings(data)))
        robot.apply_target(model, data, follower.target(data), robot.STEPS_PER_SAMPLE)
    return tracker


@pytest.mark.parametrize('route', ['square', 'figure8'])
def test_ekf_beats_dead_reckoning_on_a_real_run(route):
    tracker = run_tracker(route, laps=2)
    errors = tracker.errors()
    assert errors['ekf_rms'] < errors['dr_rms']
    # Better than trusting the 10 cm position fixes alone.
    assert errors['ekf_rms'] < .1
    assert len(tracker.fixes[0]) > 20


def test_tracker_errors_start_at_zero():
    tracker = Tracker(fix_noise=.1, gyro_noise=.01)
    model, data = robot.load()
    sample = robot.Sensors(0, 0, 0, 1).sample(robot.readings(data))
    tracker.add(sample)
    errors = tracker.errors()
    assert errors['dr_now'] == errors['ekf_now'] == 0
