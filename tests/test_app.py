import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app
import robot

FAKE_VIEWER = """
import json, sys
open(sys.argv[1], 'w').write(json.dumps(sys.argv[2:]))
print('ready', flush=True)
base = {"heading": 0, "gyro_z": 0, "fix_x": None, "fix_y": None, "fix_sigma": 0.1, "gyro_sigma": 0.01,
        "fix_outlier": 0}
print(json.dumps({**base, "time": 1.2, "x": 0, "y": 0, "left_angle": 0, "right_angle": 0,
                  "fix_x": 0.05, "fix_y": -0.05}), flush=True)
print(json.dumps({**base, "time": 1.25, "x": 0.5, "y": -0.25,
                  "left_angle": 10, "right_angle": 10}), flush=True)
received = sys.stdin.read()
open(sys.argv[1] + '.stdin', 'w').write(received)
"""


def wait_for(condition):
    from PySide6.QtTest import QTest
    for _ in range(250):
        if condition(): return True
        QTest.qWait(20)
    return condition()


def make_window(tmp_path, script_text=FAKE_VIEWER):
    qt = app.QApplication.instance() or app.QApplication([])
    script, args = tmp_path / 'fake_viewer.py', tmp_path / 'args.json'
    script.write_text(script_text)
    window = app.MainWindow(viewer_command=[sys.executable, str(script), str(args)])
    return qt, window, window.live, args


def route_button(live, name):
    return next(button for button in live.routes.buttons() if button.text() == name)


def blank_sample():
    return {field: 0.0 for field in robot.FIELDS} | {'fix_x': None, 'fix_y': None}


def test_pages(tmp_path):
    qt, window, live, _ = make_window(tmp_path)
    names = [window.pages.tabText(i) for i in range(window.pages.count())]
    assert names == ['Live demo', 'Experiment', 'How it works']
    window.close()


def test_setup_is_passed_to_the_simulation_and_locked_while_running(tmp_path):
    qt, window, live, args = make_window(tmp_path)
    assert window.status.text() == 'Not running'
    assert not live.speed.isEnabled()
    route_button(live, 'Hexagon').click()
    live.fix_noise.setValue(.25)
    live.fix_rate.setCurrentText('2 Hz')
    live.gyro_noise.setValue(.02)
    live.launch_button.click()
    assert live.launch_button.text() == 'Stop'
    assert not live.setup.isEnabled()
    assert not any(button.isEnabled() for button in live.scenario_buttons.values())
    assert live.speed.isEnabled()
    assert wait_for(lambda: '1.2' in window.status.text())
    assert json.loads(args.read_text()) == [
        '--route', 'hexagon', '--speed', '1', '--seed', '42', '--gyro-noise', '0.02', '--fix-noise', '0.25',
        '--fix-rate', '2', '--outlier-rate', '0', '--outlier-noise', '1', '--slip', '0']
    live.speed.setCurrentText('5×')
    process = live.process
    live.launch_button.click()  # Stop
    assert wait_for(lambda: process.state() == app.QProcess.ProcessState.NotRunning)
    assert wait_for(lambda: live.launch_button.text() == 'Launch')
    assert live.setup.isEnabled()
    assert not live.speed.isEnabled()
    assert 'speed 5' in Path(str(args) + '.stdin').read_text()
    window.close()


def test_errors_and_paths_are_shown(tmp_path):
    qt, window, live, args = make_window(tmp_path)
    live.launch_button.click()
    assert wait_for(lambda: '1.2' in window.status.text() and live.tracker.count == 2)
    # Equal wheel angles of 10 rad drive every estimate 0.6 m straight ahead.
    assert live.errors['dr_now'].text() == '0.269'
    assert live.errors['adaptive_now'].text() != '–'
    live.plot.refresh(live.tracker, force=True)
    assert list(live.plot.lines['true'].get_xdata()) == [0, .5]
    assert list(live.plot.lines['dr'].get_xdata()) == pytest.approx([0, .6])
    assert list(live.plot.fix_dots.get_xdata()) == [.05]
    window.close()
    qt.processEvents()


def test_simulation_exit_resets_controls(tmp_path):
    qt, window, live, _ = make_window(tmp_path, "print('ready', flush=True)\n")
    live.launch_button.click()
    assert wait_for(lambda: live.launch_button.text() == 'Launch')
    assert window.status.text() == 'Simulation closed'
    assert live.setup.isEnabled()
    window.close()


def test_scenario_fills_the_settings_and_is_passed_on(tmp_path):
    qt, window, live, args = make_window(tmp_path)
    assert live.scenario_buttons['steady'].isChecked()
    live.scenario_buttons['outliers'].click()
    assert live.outlier_rate.value() == pytest.approx(10)
    assert 'wildly wrong' in live.description.text()
    live.scenario_buttons['wrong'].click()
    # Wrong assumptions: the filters no longer assume the true sensor noise.
    assert not live.match_noise.isChecked()
    assert live.tuning['fix_noise'].value() == pytest.approx(.02)
    assert live.fix_noise.value() == pytest.approx(.1)
    live.scenario_buttons['fix_jump'].click()
    assert live.match_noise.isChecked() and live.step.isChecked()
    assert live.step_fields['fix_noise'].value() == pytest.approx(.4)
    live.launch_button.click()
    assert wait_for(lambda: '1.2' in window.status.text())
    assert json.loads(args.read_text())[-6:] == [
        '--step-time', '30', '--step-fix-noise', '0.4', '--step-gyro-noise', '0.01']
    window.close()
    qt.processEvents()


def test_layout_stays_put_while_values_change(tmp_path):
    from PySide6.QtTest import QTest
    qt, window, live, _ = make_window(tmp_path)
    window.resize(1200, 720)
    window.show()
    QTest.qWait(100)
    sample = blank_sample()
    live.show_sample(sample)
    live.plot.draw()
    qt.processEvents()
    plot_geometry, axes_box = live.plot.geometry(), live.plot.axes.get_position().bounds
    for i, value in enumerate((-123.456, 0.5, -0.001, 98.7)):
        live.show_sample({**sample, 'time': i + 1.0, 'x': value, 'y': -value,
                          'left_angle': value, 'right_angle': -value, 'fix_x': value, 'fix_y': 1.0})
        live.plot.refresh(live.tracker, force=True)
        live.plot.draw()
        qt.processEvents()
        assert live.plot.geometry() == plot_geometry
        assert live.plot.axes.get_position().bounds == pytest.approx(axes_box)
    window.close()


def test_message_round_trip():
    sample = {field: 0.5 for field in robot.FIELDS} | {'fix_x': None, 'fix_y': None}
    assert robot.decode_sample(robot.encode_sample(sample)) == sample


def test_settings_are_hidden_match_sensors_and_apply_live(tmp_path):
    qt, window, live, args = make_window(tmp_path)
    assert not live.settings.isVisibleTo(live)
    live.settings_toggle.click()
    assert live.settings.isVisibleTo(live)
    # Matching: the filters' assumed noise follows the simulated sensors.
    assert live.match_noise.isChecked()
    live.fix_noise.setValue(.3)
    assert live.tuning['fix_noise'].value() == pytest.approx(.3)
    assert not live.tuning['fix_noise'].isEnabled()
    live.match_noise.setChecked(False)
    live.tuning['fix_noise'].setValue(.05)
    assert live.fix_noise.value() == pytest.approx(.3)
    live.tuning['initial_sigma'].setValue(.02)
    live.launch_button.click()
    assert wait_for(lambda: '1.2' in window.status.text())
    ekf = live.tracker.estimators['ekf']
    assert ekf.settings.fix_noise == pytest.approx(.05)
    # Started at .02²; the default start (.001²) would stay far below it.
    assert ekf.P[0, 0] >= .02**2
    assert not live.tuning['initial_sigma'].isEnabled()
    # Live retuning while running.
    live.tuning['distance_noise'].setValue(10)
    assert ekf.settings.distance_noise == pytest.approx(.10)
    live.gate.setChecked(True)
    assert ekf.settings.gate
    window.close()
    qt.processEvents()


def test_every_control_explains_itself(tmp_path):
    from PySide6.QtWidgets import QAbstractButton, QAbstractSpinBox, QComboBox
    qt, window, live, _ = make_window(tmp_path)
    controls = window.findChildren(QAbstractSpinBox) + window.findChildren(QComboBox)
    controls += [b for b in window.findChildren(QAbstractButton) if b.text()]
    missing = [c.text() if hasattr(c, 'text') else c for c in controls if not c.toolTip()]
    assert not missing
    assert all(label.toolTip() for label in live.errors.values())
    assert all(label.toolTip() for label in live.error_names.values())
    window.close()


def test_plot_fills_its_area(tmp_path):
    from PySide6.QtTest import QTest
    qt, window, live, _ = make_window(tmp_path)
    window.resize(1400, 700)
    window.show()
    QTest.qWait(100)
    live.show_sample(blank_sample())
    live.plot.refresh(live.tracker, force=True)
    live.plot.draw()
    box = live.plot.axes.get_window_extent()
    canvas = live.plot.figure.bbox
    # A wide window gets a wide plot (equal scale kept by widening the x range).
    assert box.width / canvas.width > .7
    x0, x1 = live.plot.axes.get_xlim()
    y0, y1 = live.plot.axes.get_ylim()
    assert (x1 - x0) / (y1 - y0) == pytest.approx(box.width / box.height, rel=.02)
    window.close()


def test_sensor_noise_changes_live_while_running(tmp_path):
    qt, window, live, args = make_window(tmp_path)
    live.launch_button.click()
    assert wait_for(lambda: '1.2' in window.status.text())
    assert live.fix_noise.isEnabled() and live.gyro_noise.isEnabled()
    live.fix_noise.setValue(.25)
    live.gyro_noise.setValue(.05)
    # The EKF keeps the assumptions it launched with; only the sensors change.
    assert live.tracker.estimators['ekf'].settings.fix_noise == pytest.approx(.1)
    process = live.process
    live.launch_button.click()  # Stop
    assert wait_for(lambda: process.state() == app.QProcess.ProcessState.NotRunning)
    received = Path(str(args) + '.stdin').read_text().splitlines()
    assert 'fix_noise 0.25' in received and 'gyro_noise 0.05' in received
    window.close()


def test_noise_step_is_passed_on_and_shown_when_it_happens(tmp_path):
    qt, window, live, args = make_window(tmp_path)
    assert not live.step_fields['time'].isEnabled()
    live.step.setChecked(True)
    live.step_fields['time'].setValue(0)
    live.step_fields['fix_noise'].setValue(.4)
    live.step_fields['gyro_noise'].setValue(.08)
    live.launch_button.click()
    assert wait_for(lambda: '1.2' in window.status.text())
    assert json.loads(args.read_text())[-6:] == [
        '--step-time', '0', '--step-fix-noise', '0.4', '--step-gyro-noise', '0.08']
    assert live.fix_noise.value() == pytest.approx(.4)
    assert live.gyro_noise.value() == pytest.approx(.08)
    window.close()
    qt.processEvents()


def test_adaptive_variant_and_memory(tmp_path):
    qt, window, live, args = make_window(tmp_path)
    live.memory.setValue(30)
    live.variant.setCurrentIndex(live.variant.findData('aekf_r'))
    assert live.error_names['adaptive'].text() == 'Adaptive EKF (R only)'
    live.launch_button.click()
    assert not live.variant.isEnabled()
    assert wait_for(lambda: '1.2' in window.status.text() and live.tracker.count == 2)
    assert live.tracker.adaptive == ('aekf_r',)
    assert live.tracker.estimators['aekf_r'].memory == 30
    assert set(live.plot.lines) == {'true', 'dr', 'ekf', 'aekf_r'}
    live.noise_plot.refresh(live.tracker, force=True)
    assert list(live.noise_plot.lines['fix_sigma'].get_ydata()) == [.1]
    assert len(live.noise_plot.lines['aekf_r_fix_sigma'].get_ydata()) == 1
    live.memory.setValue(10)  # applies live
    assert live.tracker.estimators['aekf_r'].memory == 10
    window.close()
    qt.processEvents()


def test_verdict_after_enough_time(tmp_path):
    qt, window, live, _ = make_window(tmp_path)
    sample = blank_sample()
    live.show_sample(sample)
    assert live.verdict.text().startswith('Launch')
    live.show_sample({**sample, 'time': app.VERDICT_AFTER, 'x': .1, 'left_angle': 2, 'right_angle': 2,
                      'fix_x': .1, 'fix_y': 0})
    assert "Adaptive EKF's RMS error is" in live.verdict.text()
    window.close()
