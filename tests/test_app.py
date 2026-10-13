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
base = {"heading": 0, "gyro_z": 0, "fix_x": None, "fix_y": None}
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
    return qt, app.MainWindow(viewer_command=[sys.executable, str(script), str(args)]), args


def test_setup_is_passed_to_the_simulation_and_locked_while_running(tmp_path):
    qt, window, args = make_window(tmp_path)
    assert window.status.text() == 'Not running'
    assert not window.speed.isEnabled()
    window.route.setCurrentText('Hexagon')
    window.fix_noise.setValue(.25)
    window.fix_rate.setCurrentText('2 Hz')
    window.gyro_noise.setValue(.02)
    window.launch_button.click()
    assert window.launch_button.text() == 'Stop'
    assert not window.setup.isEnabled()
    assert window.speed.isEnabled()
    assert wait_for(lambda: '1.2' in window.status.text())
    assert json.loads(args.read_text()) == [
        '--route', 'hexagon', '--speed', '1', '--seed', '42', '--gyro-noise', '0.02',
        '--fix-noise', '0.25', '--fix-rate', '2']
    window.speed.setCurrentText('5×')
    process = window.process
    window.launch_button.click()  # Stop
    assert wait_for(lambda: process.state() == app.QProcess.ProcessState.NotRunning)
    assert wait_for(lambda: window.launch_button.text() == 'Launch')
    assert window.setup.isEnabled()
    assert not window.speed.isEnabled()
    assert 'speed 5' in Path(str(args) + '.stdin').read_text()
    window.close()


def test_errors_and_paths_are_shown(tmp_path):
    qt, window, args = make_window(tmp_path)
    window.launch_button.click()
    assert wait_for(lambda: '1.2' in window.status.text())
    # Equal wheel angles of 10 rad drive both estimates 0.6 m straight ahead.
    assert window.errors['dr_now'].text() == '0.269'
    assert list(window.plot.true_line.get_xdata()) == [0, .5]
    assert list(window.plot.dr_line.get_xdata()) == pytest.approx([0, .6])
    assert list(window.plot.fix_dots.get_xdata()) == [.05]
    assert not window.sensor_panel.isVisible()
    window.sensor_toggle.click()
    assert window.sensor_panel.isVisibleTo(window)
    assert window.sensors['x'].text() == '0.500'
    window.close()
    qt.processEvents()


def test_simulation_exit_resets_controls(tmp_path):
    qt, window, _ = make_window(tmp_path, "print('ready', flush=True)\n")
    window.launch_button.click()
    assert wait_for(lambda: window.launch_button.text() == 'Launch')
    assert window.status.text() == 'Simulation closed'
    assert window.setup.isEnabled()
    window.close()


def test_layout_stays_put_while_values_change(tmp_path):
    from PySide6.QtTest import QTest
    qt, window, _ = make_window(tmp_path)
    window.resize(1200, 720)
    window.show()
    window.sensor_toggle.click()
    QTest.qWait(100)
    sample = {field: 0.0 for field in robot.FIELDS} | {'fix_x': None, 'fix_y': None}
    window.show_sample(sample)
    window.plot.draw()
    qt.processEvents()
    plot_geometry, axes_box = window.plot.geometry(), window.plot.axes.get_position().bounds
    for i, value in enumerate((-123.456, 0.5, -0.001, 98.7)):
        window.show_sample({**sample, 'time': i + 1.0, 'x': value, 'y': -value,
                            'left_angle': value, 'right_angle': -value, 'fix_x': value, 'fix_y': 1.0})
        window.plot.refresh(window.tracker, force=True)
        window.plot.draw()
        qt.processEvents()
        assert window.plot.geometry() == plot_geometry
        assert window.plot.axes.get_position().bounds == pytest.approx(axes_box)
    window.close()


def test_message_round_trip():
    sample = {field: 0.5 for field in robot.FIELDS} | {'fix_x': None, 'fix_y': None}
    assert robot.decode_sample(robot.encode_sample(sample)) == sample
