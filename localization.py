"""Runs dead reckoning, the EKF and Adaptive EKFs side by side on the simulation's readings."""

import math

import numpy as np

from dead_reckoning import WHEEL_RADIUS, DeadReckoning, wheel_motion
from ekf import EKF, MEMORY, SAMPLE_PERIOD, AdaptiveEKF, EKFSettings

# Adaptive EKF variants: key → (name, adapts gyro σ as well as R).
ADAPTIVE = {'aekf': ('Adaptive EKF', True), 'aekf_r': ('Adaptive EKF (R only)', False)}
NAMES = {'dr': 'Dead reckoning', 'ekf': 'EKF', **{key: name for key, (name, _) in ADAPTIVE.items()}}
COVERAGE_LIMIT = 5.991  # χ² with 2 degrees of freedom, 95%: the filter's 95% position ellipse


class Tracker:
    """Feeds every sample to each estimator and keeps their paths, errors and statistics.

    adaptive lists the Adaptive EKF variants to run (keys of ADAPTIVE). All
    filters get exactly the same inputs. For the filters, it also checks how
    honest their uncertainty P is: how often the true position lies inside
    their 95% position ellipse (ideally 95% of the time).
    """

    def __init__(self, settings=EKFSettings(), memory=MEMORY, adaptive=('aekf',)):
        self.settings, self.memory, self.adaptive = settings, memory, tuple(adaptive)
        self.names = ('dr', 'ekf', *self.adaptive)
        self.estimators = {}
        self.last = None
        self.paths = {name: ([], []) for name in ('true', *self.names)}
        self.fixes = ([], [])
        self.outliers = ([], [])
        # At each fix: time, and the true, EKF-assumed and adaptive-estimated fix and gyro σ.
        keys = ['time', 'fix_sigma', 'gyro_sigma', 'ekf_fix_sigma', 'ekf_gyro_sigma']
        keys += [f'{name}_{kind}_sigma' for name in self.adaptive for kind in ('fix', 'gyro')]
        self.noise_history = {key: [] for key in keys}
        self.squared = dict.fromkeys(self.names, 0.0)
        self.inside = dict.fromkeys(self.filters(), 0)
        self.count = 0

    def filters(self):
        return self.names[1:]

    def tune(self, settings, memory=None):
        """Change the EKF's tuning and the adaptive filters' memory; applies from the next sample.

        The adaptive filters take the settings' noise σ only as starting guesses at the
        first sample; later changes reach them only for distance and slip σ and the gate.
        """
        self.settings = settings
        if memory is not None:
            self.memory = memory
        if self.estimators:
            self.estimators['ekf'].settings = settings
            for name in self.adaptive:
                estimator = self.estimators[name]
                estimator.settings, estimator.memory = settings, self.memory

    def start(self, sample):
        # All estimators start from the known initial pose.
        start = (sample['x'], sample['y'], sample['heading'])
        self.estimators['dr'] = DeadReckoning(*start, sample['left_angle'], sample['right_angle'])
        self.estimators['ekf'] = EKF(*start, self.settings)
        for name in self.adaptive:
            self.estimators[name] = AdaptiveEKF(*start, self.settings, self.memory, adapt_gyro=ADAPTIVE[name][1])

    def add(self, sample):
        """Process one sample of ground truth and sensor readings."""
        if self.last is None:
            self.start(sample)
        else:
            distance, _ = wheel_motion(WHEEL_RADIUS*(sample['left_angle'] - self.last['left_angle']),
                                       WHEEL_RADIUS*(sample['right_angle'] - self.last['right_angle']))
            self.estimators['dr'].update(sample['left_angle'], sample['right_angle'])
            turn = (self.last['gyro_z'] + sample['gyro_z'])/2 * SAMPLE_PERIOD
            for name in self.filters():
                estimator = self.estimators[name]
                estimator.predict(distance, turn)
                if sample['fix_x'] is not None:
                    estimator.update_position(sample['fix_x'], sample['fix_y'])
        for name in self.adaptive:
            self.estimators[name].observe_gyro(sample['gyro_z'])
        if sample['fix_x'] is not None:
            self.record_fix(sample)
        self.last = sample
        truth = np.array([sample['x'], sample['y']])
        for name, (xs, ys) in self.paths.items():
            x, y = truth if name == 'true' else (self.estimators[name].x, self.estimators[name].y)
            xs.append(x)
            ys.append(y)
        for name in self.names:
            self.squared[name] += self.position_error(name)**2
        for name in self.filters():
            error = np.array([self.estimators[name].x, self.estimators[name].y]) - truth
            if error @ np.linalg.solve(self.estimators[name].P[:2, :2], error) <= COVERAGE_LIMIT:
                self.inside[name] += 1
        self.count += 1

    def record_fix(self, sample):
        fixes = self.outliers if sample.get('fix_outlier') else self.fixes
        fixes[0].append(sample['fix_x'])
        fixes[1].append(sample['fix_y'])
        values = {'time': sample['time'], 'fix_sigma': sample['fix_sigma'], 'gyro_sigma': sample['gyro_sigma'],
                  'ekf_fix_sigma': self.settings.fix_noise, 'ekf_gyro_sigma': self.settings.gyro_noise}
        for name in self.adaptive:
            values[f'{name}_fix_sigma'] = self.estimators[name].fix_sigma
            values[f'{name}_gyro_sigma'] = self.estimators[name].gyro_sigma
        for key, value in values.items():
            self.noise_history[key].append(value)

    def position_error(self, name):
        estimate = self.estimators[name]
        return math.hypot(estimate.x - self.last['x'], estimate.y - self.last['y'])

    def errors(self):
        """Current and RMS position error (m) and heading error (rad) for each estimator,
        and for the filters the fraction of samples inside their 95% ellipse."""
        result = {}
        for name in self.names:
            estimate = self.estimators[name]
            result[f'{name}_now'] = self.position_error(name)
            result[f'{name}_rms'] = math.sqrt(self.squared[name] / self.count)
            result[f'{name}_heading'] = abs(math.remainder(estimate.heading - self.last['heading'], math.tau))
        for name in self.filters():
            result[f'{name}_coverage'] = self.inside[name] / self.count
        return result
