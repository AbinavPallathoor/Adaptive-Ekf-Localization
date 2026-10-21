import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analysis
from ekf import EKFSettings
import experiment
import robot
from scenarios import BY_KEY, SCENARIOS


def fake_results(scenarios=('steady', 'fix_jump'), runs=12, seed=0):
    """Runs where the adaptive filters beat the EKF clearly under fix_jump and tie under steady."""
    rng = np.random.default_rng(seed)
    rows = []
    for key in scenarios:
        for index in range(runs):
            ekf = .03 + rng.normal(0, .003)
            gain = .01 if key == 'fix_jump' else 0
            rows.append({'scenario': key, 'route': 'square', 'seed': index, 'duration': 60.0,
                         'dr_rmse': .3, 'ekf_rmse': ekf,
                         'aekf_rmse': ekf - gain + rng.normal(0, .001),
                         'aekf_r_rmse': ekf - gain/2 + rng.normal(0, .001),
                         'ekf_coverage': .9, 'aekf_coverage': .95, 'aekf_r_coverage': .93})
    return pd.DataFrame(rows)


# Scenarios

def test_scenarios_are_unique_and_described():
    assert len(BY_KEY) == len(SCENARIOS)
    assert all(scenario.description and scenario.short for scenario in SCENARIOS)


def test_filter_settings_assume_the_true_noise_unless_told_otherwise():
    assert BY_KEY['steady'].filter_settings() == EKFSettings(fix_noise=.1, gyro_noise=.01)
    wrong = BY_KEY['wrong'].filter_settings(EKFSettings(distance_noise=.05))
    assert (wrong.fix_noise, wrong.gyro_noise, wrong.distance_noise) == (.02, .002, .05)


# Statistics

def test_paired_test_finds_a_real_improvement():
    rng = np.random.default_rng(1)
    standard = .03 + rng.normal(0, .003, 20)
    result = analysis.paired_test(standard - .005 + rng.normal(0, .001, 20), standard)
    assert result['n'] == 20
    assert result['mean_d'] == pytest.approx(-.005, abs=.001)
    assert result['ci_low'] < result['mean_d'] < result['ci_high'] < 0
    assert result['p'] < 1e-6 and result['wilcoxon_p'] < 1e-3
    assert result['d_z'] < -2


def test_paired_test_without_enough_data():
    result = analysis.paired_test([.1], [.2])
    assert math.isnan(result['p']) and math.isnan(result['ci_low'])
    identical = analysis.paired_test([.1, .2, .3], [.1, .2, .3])
    assert math.isnan(identical['p']) and identical['mean_d'] == 0


def test_holm_adjustment():
    adjusted = analysis.holm([.01, .04, math.nan, .03])
    # Sorted .01, .03, .04 → ×3, ×2, ×1, kept monotone.
    assert adjusted[0] == pytest.approx(.03)
    assert adjusted[3] == pytest.approx(.06)
    assert adjusted[1] == pytest.approx(.06)
    assert math.isnan(adjusted[2])
    assert analysis.holm([.9, .8])[0] == 1.0


def test_verdicts():
    row = {'p_holm': .001, 'mean_d': -.002, 'change': -.2, 'ci_low': -.003}
    assert analysis.verdict(row) == 'Better: 20% lower error'
    assert analysis.verdict({**row, 'p_holm': .5, 'mean_d': .002, 'change': .1, 'ci_low': .001}) == \
        'Worse: 10% higher error'
    assert analysis.verdict({**row, 'p_holm': .5, 'ci_low': -.001}) == 'No clear difference'
    assert analysis.verdict({**row, 'p_holm': math.nan}) == 'Not enough runs'


def test_summary_and_headline():
    table = analysis.summary(fake_results())
    assert len(table) == 2 * 2  # scenarios × adaptive variants
    aekf = table[table['variant'] == 'aekf'].set_index('scenario')
    assert aekf.loc['fix_jump', 'verdict'].startswith('Better')
    assert aekf.loc['steady', 'verdict'] == 'No clear difference'
    assert aekf.loc['fix_jump', 'adaptive_coverage'] == pytest.approx(.95)
    headline = analysis.headline(table, 'aekf')
    assert '12 paired runs' in headline and 'Position fixes get noisier' in headline


def test_export_writes_tables_and_figures(tmp_path):
    results = fake_results()
    rng = np.random.default_rng(2)
    paths = analysis.export(results, tmp_path / 'out', rng.normal(0, .1, (50, 2)), rng.normal(0, .01, 500), .1, .01)
    names = sorted(path.name for path in paths)
    assert names == sorted(['runs.csv', 'summary.csv', 'calibration.png', 'noise_check.png', 'errors_aekf.png',
                            'difference_aekf.png', 'errors_aekf_r.png', 'difference_aekf_r.png'])
    assert all(path.stat().st_size > 0 for path in paths)
    assert len(pd.read_csv(tmp_path / 'out' / 'runs.csv')) == len(results)


# Experiment runs

def test_jobs_pair_scenarios_on_the_same_seeds():
    job_list = experiment.jobs(['steady', 'slip'], ['square', 'figure8'], 3, gate=True)
    assert len(job_list) == 2 * 2 * 3
    seeds = lambda scenario: [job[2] for job in job_list if job[0] == scenario]
    assert seeds('steady') == seeds('slip')
    assert len(set(seeds('steady'))) == 6  # different routes never share noise
    assert all(job[3] for job in job_list)


def test_one_run_gives_every_estimator(tmp_path):
    result = experiment.run('steady', 'square', experiment.seed('square', 1))
    assert result['duration'] > 10
    for name in experiment.ESTIMATORS:
        assert 0 < result[f'{name}_rmse'] < 1
    assert result['ekf_rmse'] < result['dr_rmse']
    assert 0 <= result['aekf_coverage'] <= 1
    # The same seed gives exactly the same run.
    again = experiment.run('steady', 'square', experiment.seed('square', 1))
    assert again['aekf_rmse'] == result['aekf_rmse']


def test_simulated_noise_has_the_set_sigma():
    fix_errors, gyro_errors, fix_sigma, gyro_sigma = experiment.noise_sample(runs=1)
    assert np.std(fix_errors) == pytest.approx(fix_sigma, rel=.25)
    assert np.std(gyro_errors) == pytest.approx(gyro_sigma, rel=.1)
    assert abs(np.mean(gyro_errors)) < gyro_sigma/5


# Experiment page

def page():
    from PySide6.QtWidgets import QApplication
    from experiment_page import ExperimentPage
    from ui import THEMES
    QApplication.instance() or QApplication([])
    return ExperimentPage(THEMES['light'])


def test_page_estimate_follows_the_choices():
    experiment_page = page()
    assert experiment_page.run_button.isEnabled()
    assert experiment_page.estimate.text().startswith(f'{len(SCENARIOS) * len(robot.ROUTES) * 20} runs')
    for box in experiment_page.route_boxes.values():
        box.setChecked(False)
    assert not experiment_page.run_button.isEnabled()
    assert 'at least one' in experiment_page.estimate.text()


def test_page_shows_and_saves_results(tmp_path):
    experiment_page = page()
    rng = np.random.default_rng(3)
    noise = (rng.normal(0, .1, (40, 2)), rng.normal(0, .01, 400), .1, .01)
    experiment_page.finished(fake_results(), noise)
    assert experiment_page.result_table.rowCount() == 2
    cells = [experiment_page.result_table.item(row, 5).text() for row in range(2)]
    assert any(cell.startswith('Better') for cell in cells)
    assert 'Adaptive EKF vs. EKF' in experiment_page.headline.text()
    experiment_page.variants.buttons()[1].click()
    assert 'Adaptive EKF (R only)' in experiment_page.headline.text()
    assert experiment_page.export_button.isEnabled()
    paths = experiment_page.export(tmp_path / 'saved')
    assert len(paths) == 8 and 'Saved 8 files' in experiment_page.saved.text()


def test_page_runs_a_small_experiment():
    from PySide6.QtTest import QTest
    experiment_page = page()
    for key, box in experiment_page.scenario_boxes.items():
        box.setChecked(key == 'steady')
    for key, box in experiment_page.route_boxes.items():
        box.setChecked(key == 'square')
    experiment_page.runs.setValue(2)
    experiment_page.workers = 2
    experiment_page.run_button.click()
    assert experiment_page.run_button.text() == 'Stop'
    assert not experiment_page.runs.isEnabled()
    for _ in range(1500):
        if experiment_page.table is not None or experiment_page.pool is None:
            break
        QTest.qWait(20)
    assert experiment_page.table is not None, experiment_page.headline.text()
    assert experiment_page.run_button.text() == 'Run experiment'
    assert len(experiment_page.results) == 2
    assert experiment_page.progress.value() == 2
