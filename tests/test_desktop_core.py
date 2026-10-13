import csv
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    path = ROOT / f'{name}.py'
    assert path.exists(), f'{name} has not been implemented'
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def test_sampling_and_zero_noise_match_ground_truth():
    sim = module('simulation')
    engine = sim.Simulation()
    engine.drive('forward')
    samples = []
    for _ in range(100):
        samples.extend(engine.advance())
    assert len(samples) == 40
    assert engine.data.qpos[0] > .3
    assert np.diff([s['sim_time'] for s in samples]) == pytest.approx([.05]*39)
    for s in samples:
        assert s['position_x'] == s['x']
        assert s['position_y'] == s['y']
        assert s['left_encoder'] == s['left_velocity']
        assert s['gyro_z'] == s['true_gyro_z']
    engine.reset()
    assert engine.data.time == 0
    assert engine.target == pytest.approx([0, 0])


def test_noise_reproducible_and_config_validated():
    sim = module('simulation')
    settings = sim.Settings(position_noise=.1, encoder_noise=.1, gyro_noise=.1, seed=42)
    a, b = sim.Simulation(settings), sim.Simulation(settings)
    left, right = a.advance(.1), b.advance(.1)
    for x, y in zip(left, right):
        for field in ('position_x', 'position_y', 'left_encoder', 'gyro_z'):
            assert x[field] == y[field]
    assert left[0]['position_x'] != left[0]['x']
    with pytest.raises(ValueError):
        sim.Settings(position_noise=-1)
    with pytest.raises(ValueError):
        sim.Settings(log_hz=500)
    a.drive('left')
    for _ in range(100): a.advance()
    assert a.data.ctrl == pytest.approx([-3, 3])
    a.drive('stop')
    for _ in range(100): a.advance()
    assert a.data.ctrl == pytest.approx([0, 0])


def test_database_persistence_isolation_export_and_metadata(tmp_path):
    sim, db = module('simulation'), module('database')
    engine = sim.Simulation()
    samples = engine.advance(.2)
    path = tmp_path / 'runs.sqlite3'
    store = db.RunStore(path)
    first = store.start_run('Baseline', engine.metadata())
    store.append(first, samples)
    store.finish_run(first)
    second = store.start_run('Another run', engine.metadata())
    store.append(second, samples[:1])
    store.finish_run(second)
    preview = store.read_samples(first, limit=2)
    assert len(preview) == 2
    assert [r['sequence'] for r in preview] == [2, 3]
    assert all(r['run_id'] == first for r in preview)
    exported = tmp_path / 'baseline.csv'
    assert store.export_csv(first, exported) == 4
    with exported.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert float(rows[-1]['sim_time']) == pytest.approx(.2)
    store.close()
    reopened = db.RunStore(path)
    runs = reopened.list_runs()
    assert [r['sample_count'] for r in runs] == [1, 4]
    assert json.loads(runs[1]['metadata_json'])['settings']['log_hz'] == 20
    assert 'model_xml' in json.loads(runs[1]['metadata_json'])
    reopened.close()
    with sqlite3.connect(path) as conn:
        assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'


def test_finished_run_rejects_more_samples(tmp_path):
    sim, db = module('simulation'), module('database')
    store = db.RunStore(tmp_path / 'runs.sqlite3')
    run = store.start_run('Closed', sim.Simulation().metadata())
    store.finish_run(run)
    with pytest.raises(ValueError):
        store.append(run, sim.Simulation().advance(.1))
    store.close()
