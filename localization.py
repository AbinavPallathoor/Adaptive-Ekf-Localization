"""Runs dead reckoning and the EKF side by side on the simulation's readings."""

import math

from dead_reckoning import WHEEL_RADIUS, DeadReckoning, wheel_motion
from ekf import EKF, SAMPLE_PERIOD


class Tracker:
    def __init__(self, fix_noise, gyro_noise):
        self.fix_noise, self.gyro_noise = fix_noise, gyro_noise
        self.odometry = self.ekf = self.last = None
        self.true_path, self.dr_path, self.ekf_path, self.fixes = ([], []), ([], []), ([], []), ([], [])
        self.squared = {'dr': 0.0, 'ekf': 0.0}
        self.count = 0

    def add(self, sample):
        """Process one sample of ground truth and sensor readings."""
        if self.last is None:
            # Both estimators start from the known initial pose.
            start = (sample['x'], sample['y'], sample['heading'])
            self.odometry = DeadReckoning(*start, sample['left_angle'], sample['right_angle'])
            self.ekf = EKF(*start, self.fix_noise, self.gyro_noise)
        else:
            distance, _ = wheel_motion(WHEEL_RADIUS*(sample['left_angle'] - self.last['left_angle']),
                                       WHEEL_RADIUS*(sample['right_angle'] - self.last['right_angle']))
            self.odometry.update(sample['left_angle'], sample['right_angle'])
            turn = (self.last['gyro_z'] + sample['gyro_z'])/2 * SAMPLE_PERIOD
            self.ekf.predict(distance, turn)
            if sample['fix_x'] is not None:
                self.ekf.update_position(sample['fix_x'], sample['fix_y'])
        if sample['fix_x'] is not None:
            self.fixes[0].append(sample['fix_x'])
            self.fixes[1].append(sample['fix_y'])
        self.last = sample
        for path, (x, y) in ((self.true_path, (sample['x'], sample['y'])),
                             (self.dr_path, (self.odometry.x, self.odometry.y)),
                             (self.ekf_path, (self.ekf.x, self.ekf.y))):
            path[0].append(x)
            path[1].append(y)
        self.squared['dr'] += self.position_error(self.odometry)**2
        self.squared['ekf'] += self.position_error(self.ekf)**2
        self.count += 1

    def position_error(self, estimate):
        return math.hypot(estimate.x - self.last['x'], estimate.y - self.last['y'])

    def errors(self):
        """Current and RMS position error (m) and heading error (rad) for each estimator."""
        result = {}
        for name, estimate in (('dr', self.odometry), ('ekf', self.ekf)):
            result[f'{name}_now'] = self.position_error(estimate)
            result[f'{name}_rms'] = math.sqrt(self.squared[name] / self.count)
            result[f'{name}_heading'] = abs(math.remainder(estimate.heading - self.last['heading'], math.tau))
        return result
