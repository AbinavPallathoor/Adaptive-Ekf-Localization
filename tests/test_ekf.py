from pathlib import Path
import math
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dead_reckoning import DeadReckoning, wheel_motion
from ekf import EKF, AdaptiveEKF, EKFSettings
from localization import Tracker
import robot


def test_wheel_motion():
    assert wheel_motion(.1, .1) == pytest.approx((.1, 0))
    assert wheel_motion(-.15, .15) == pytest.approx((0, 1))


def test_prediction_alone_matches_dead_reckoning():
    odometry = DeadReckoning(0, 0, 0, 0, 0)
    ekf = EKF(0, 0, 0, EKFSettings(fix_noise=.1, gyro_noise=.01))
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
    ekf = EKF(0, 0, 0, EKFSettings(fix_noise=.1, gyro_noise=.01))
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
    ekf = EKF(0, 0, 3.1, EKFSettings(fix_noise=.1, gyro_noise=.01))
    ekf.predict(0, .2)
    assert -math.pi <= ekf.heading <= math.pi


def run_tracker(route, laps, fix_noise=.1, gyro_noise=.01, fix_rate=1, assumed=None, step=None, seed=42):
    """Run the route; the filters assume the true noise unless given assumed (fix σ, gyro σ)."""
    model, data = robot.load()
    follower = robot.RouteFollower(route, robot.true_pose(data))
    sensors = robot.Sensors(seed=seed, gyro_noise=gyro_noise, fix_noise=fix_noise, fix_rate=fix_rate)
    step = step or robot.NoiseStep(None)
    assumed_fix, assumed_gyro = assumed or (fix_noise, gyro_noise)
    tracker = Tracker(EKFSettings(fix_noise=assumed_fix, gyro_noise=assumed_gyro))
    while follower.laps < laps:
        step.apply(sensors, data.time)
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
    tracker = Tracker(EKFSettings(fix_noise=.1, gyro_noise=.01))
    model, data = robot.load()
    sample = robot.Sensors(0, 0, 0, 1).sample(robot.readings(data))
    tracker.add(sample)
    errors = tracker.errors()
    assert errors['dr_now'] == errors['ekf_now'] == 0


def moved_by_fix(settings):
    ekf = EKF(0, 0, 0, settings)
    for _ in range(20):
        ekf.predict(.012, 0)
    before = ekf.x
    ekf.update_position(before + .2, 0)
    return ekf.x - before


def test_trusting_fixes_less_means_smaller_corrections():
    trusting = moved_by_fix(EKFSettings(fix_noise=.05))
    doubtful = moved_by_fix(EKFSettings(fix_noise=.5))
    assert 0 < doubtful < trusting


def test_more_process_noise_means_bigger_corrections():
    confident = moved_by_fix(EKFSettings(distance_noise=.01, slip_noise=1e-4))
    unsure = moved_by_fix(EKFSettings(distance_noise=.2, slip_noise=.005))
    assert unsure > confident


def test_initial_uncertainty_sets_starting_covariance():
    ekf = EKF(0, 0, 0, EKFSettings(initial_sigma=.05))
    assert np.allclose(np.diag(ekf.P), .05**2)


def test_settings_reject_impossible_values():
    with pytest.raises(ValueError):
        EKFSettings(fix_noise=-1)


def test_tracker_retunes_running_filter():
    tracker = Tracker(EKFSettings(fix_noise=.1))
    model, data = robot.load()
    tracker.add(robot.Sensors(0, 0, 0, 1).sample(robot.readings(data)))
    tracker.tune(EKFSettings(fix_noise=.3, distance_noise=.05))
    assert tracker.estimators['ekf'].settings.fix_noise == .3
    assert tracker.settings.distance_noise == .05


def test_adaptive_ekf_is_an_ekf_until_it_has_data():
    plain, adaptive = EKF(0, 0, 0), AdaptiveEKF(0, 0, 0)
    for _ in range(10):
        for estimate in (plain, adaptive):
            estimate.predict(.012, .01)
    assert np.allclose(adaptive.state, plain.state)
    assert np.allclose(adaptive.P, plain.P)
    assert adaptive.fix_sigma == pytest.approx(EKFSettings().fix_noise)


def test_adaptive_ekf_estimates_fix_and_gyro_noise():
    rng = np.random.default_rng(0)
    aekf = AdaptiveEKF(0, 0, 0, EKFSettings(fix_noise=.5, gyro_noise=.001), memory=20)
    # Standing still at the origin while the gyro reads a steady turn rate.
    for step in range(4000):
        aekf.observe_gyro(.3 + rng.normal(0, .04))
        aekf.predict(0, 0)
        if step % 20 == 0:
            aekf.update_position(*rng.normal(0, .1, 2))
    assert aekf.fix_sigma == pytest.approx(.1, rel=.25)
    assert aekf.gyro_sigma == pytest.approx(.04, rel=.15)
    assert np.allclose(aekf.R, aekf.R.T)
    assert np.all(np.linalg.eigvalsh(aekf.R) > 0)


def test_gyro_estimate_ignores_the_start_of_a_turn():
    aekf = AdaptiveEKF(0, 0, 0, memory=5)
    for step in range(100):
        aekf.observe_gyro(0 if step < 50 else min(1, (step - 50)*.12))
    assert aekf.gyro_sigma < .001


@pytest.mark.parametrize('route,assumed', [('square', (.02, .01)), ('figure8', (.05, .01))])
def test_adaptive_ekf_beats_a_mistuned_ekf(route, assumed):
    # Square: fixes trusted far too much. Figure 8: gyro trusted far too much.
    gyro_noise = .1 if route == 'figure8' else .01
    # Averaged over a few seeds: one unlucky noise draw can narrow the gap on its own.
    ratios, gyro_sigmas = [], []
    for seed in (1, 2, 3):
        tracker = run_tracker(route, laps=2, fix_noise=assumed[0] if route == 'figure8' else .1,
                              gyro_noise=gyro_noise, fix_rate=5 if route == 'figure8' else 1, assumed=assumed,
                              seed=seed)
        errors = tracker.errors()
        ratios.append(errors['aekf_rms'] / errors['ekf_rms'])
        assert errors['aekf_rms'] < errors['ekf_rms']
        gyro_sigmas.append(tracker.estimators['aekf'].gyro_sigma)
    assert np.mean(ratios) < .6
    assert np.mean(gyro_sigmas) == pytest.approx(gyro_noise, rel=.2)


def test_adaptive_ekf_matches_a_well_tuned_ekf():
    errors = run_tracker('figure8', laps=2).errors()
    assert errors['aekf_rms'] < 1.2*errors['ekf_rms']
    assert errors['aekf_rms'] < .1*errors['dr_rms']


def test_adaptive_ekf_follows_a_noise_step():
    tracker = run_tracker('square', laps=2, fix_noise=.05, fix_rate=2, step=robot.NoiseStep(30, fix_noise=.3))
    history = tracker.noise_history
    before = [estimate for time, estimate in zip(history['time'], history['aekf_fix_sigma']) if 20 < time < 30]
    assert max(before) < .1
    assert history['fix_sigma'][-1] == .3
    assert tracker.estimators['aekf'].fix_sigma > .2
    # The plain EKF keeps assuming the old noise.
    assert history['ekf_fix_sigma'][-1] == .05


def test_tracker_retunes_adaptive_memory_only():
    tracker = Tracker(EKFSettings(fix_noise=.1), memory=20)
    model, data = robot.load()
    tracker.add(robot.Sensors(0, 0, 0, 1).sample(robot.readings(data)))
    tracker.tune(EKFSettings(fix_noise=.3), memory=5)
    assert tracker.estimators['aekf'].memory == 5
    # Its R is its own estimate, not the new assumption.
    assert tracker.estimators['aekf'].fix_sigma == pytest.approx(.1)
