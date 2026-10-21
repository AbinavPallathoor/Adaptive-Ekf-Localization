"""Repeated, paired experiments without the MuJoCo window.

Each run drives one route under one scenario with its own random seed. All
estimators get exactly the same sensor readings, so each run gives one paired
observation: the RMSE of every estimator on the same data.

Command line (the app's Experiment page does the same):
    .venv/bin/python experiment.py --runs 20 --out output/experiment
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ekf import MEMORY, EKFSettings
from localization import ADAPTIVE, Tracker
import robot
from scenarios import BY_KEY, SCENARIOS

LAPS = 2
ESTIMATORS = ('dr', 'ekf', *ADAPTIVE)


def run(scenario_key, route, seed, gate=False, memory=MEMORY, keep_noise=False):
    """One run. Returns a dict of results; with keep_noise, also the raw sensor errors."""
    scenario = BY_KEY[scenario_key]
    model, data = robot.load()
    follower = robot.RouteFollower(route, robot.true_pose(data))
    sensors = robot.Sensors(seed, scenario.gyro_noise, scenario.fix_noise, scenario.fix_rate,
                            scenario.outlier_rate, scenario.outlier_noise, scenario.slip)
    step = robot.NoiseStep(scenario.step_time, scenario.step_gyro_noise, scenario.step_fix_noise)
    tracker = Tracker(replace(scenario.filter_settings(), gate=gate), memory, adaptive=tuple(ADAPTIVE))
    fix_errors, gyro_errors = [], []
    while follower.laps < LAPS:
        step.apply(sensors, data.time)
        truth = robot.readings(data)
        sample = sensors.sample(truth)
        tracker.add(sample)
        if keep_noise:
            gyro_errors.append(sample['gyro_z'] - truth['gyro_z'])
            if sample['fix_x'] is not None:
                fix_errors.append((sample['fix_x'] - truth['x'], sample['fix_y'] - truth['y']))
        robot.apply_target(model, data, follower.target(data), robot.STEPS_PER_SAMPLE)
    errors = tracker.errors()
    result = {'scenario': scenario_key, 'route': route, 'seed': seed, 'duration': data.time}
    for name in ESTIMATORS:
        result[f'{name}_rmse'] = errors[f'{name}_rms']
        if name != 'dr':
            result[f'{name}_coverage'] = errors[f'{name}_coverage']
    if keep_noise:
        result['fix_errors'] = np.array(fix_errors)
        result['gyro_errors'] = np.array(gyro_errors)
    return result


def seed(route, index):
    """Seed of the index-th run (from 1) on a route: different routes never share noise."""
    return 1000*(list(robot.ROUTES).index(route) + 1) + index


def jobs(scenarios, routes, runs, gate=False, memory=MEMORY):
    """Every (scenario, route, run) combination. Scenarios share seeds, so each sees the same noise."""
    return [(scenario, route, seed(route, index), gate, memory)
            for scenario in scenarios for route in routes for index in range(1, runs + 1)]


def _run(job):
    return run(*job)


def run_all(job_list, workers=None, progress=None):
    """Run the jobs in parallel; progress(done, total) is called after each. Returns a DataFrame."""
    workers = workers or max(1, min(len(job_list), (os.cpu_count() or 2) - 1))
    results = []
    if workers == 1:
        for job in job_list:
            results.append(_run(job))
            if progress:
                progress(len(results), len(job_list))
    else:
        with ProcessPoolExecutor(workers) as pool:
            for result in pool.map(_run, job_list):
                results.append(result)
                if progress:
                    progress(len(results), len(job_list))
    return pd.DataFrame(results)


def noise_sample(runs=5, route='square'):
    """Raw fix and gyro errors from a few steady runs, to check the noise is N(0, σ²)."""
    scenario = BY_KEY['steady']
    results = [run('steady', route, seed(route, index), keep_noise=True) for index in range(1, runs + 1)]
    return (np.concatenate([result['fix_errors'] for result in results]),
            np.concatenate([result['gyro_errors'] for result in results]), scenario.fix_noise, scenario.gyro_noise)


def main():
    import analysis
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=int, default=20, help='runs (seeds) per scenario and route')
    parser.add_argument('--scenarios', nargs='+', default=[s.key for s in SCENARIOS], choices=BY_KEY)
    parser.add_argument('--routes', nargs='+', default=list(robot.ROUTES), choices=robot.ROUTES)
    parser.add_argument('--gate', action='store_true', help='filters reject fixes that fail the χ² test')
    parser.add_argument('--memory', type=float, default=MEMORY, help='Adaptive EKF memory, s')
    parser.add_argument('--out', type=Path, default=Path('output/experiment'), help='folder for CSV and figures')
    args = parser.parse_args()
    job_list = jobs(args.scenarios, args.routes, args.runs, args.gate, args.memory)
    print(f'{len(job_list)} runs…')
    results = run_all(job_list, progress=lambda done, total: print(f'\r{done}/{total}', end='', flush=True))
    print()
    paths = analysis.export(results, args.out, *noise_sample())
    print(analysis.summary(results).to_string(index=False))
    print('\nSaved:', *paths, sep='\n  ')


if __name__ == '__main__':
    main()
