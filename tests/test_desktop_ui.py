import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app
import viewer

FAKE_VIEWER = """
import sys
print('ready', flush=True)
print('{"time": 1.25, "x": 0.5, "y": -0.25, "heading": 1.5707963, "speed": 0.2, "turn_rate": 0.1,'
      ' "left_angle": 3.0, "right_angle": 3.5, "left_speed": 4.0, "right_speed": 4.5,'
      ' "gyro_z": 0.1, "accel_x": 0.01, "accel_y": -0.02}', flush=True)
sys.stdin.read()
"""


def wait_for(condition, qt):
    from PySide6.QtTest import QTest
    for _ in range(250):
        if condition(): return True
        QTest.qWait(20)
    return condition()


def test_launch_links_and_close_stops_simulation(tmp_path):
    qt = app.QApplication.instance() or app.QApplication([])
    script = tmp_path / 'fake_viewer.py'
    script.write_text(FAKE_VIEWER)
    window = app.MainWindow(viewer_command=[sys.executable, str(script)])
    assert window.status.text() == 'Not running'
    window.launch_button.click()
    assert not window.launch_button.isEnabled()
    assert wait_for(lambda: window.status.text().startswith('Linked'), qt)
    assert wait_for(lambda: '1.2' in window.status.text(), qt)
    assert window.readings['x'].text() == '0.500'
    assert window.readings['y'].text() == '-0.250'
    assert window.readings['heading'].text() == '90.0'
    assert window.readings['speed'].text() == '0.200'
    assert window.readings['left_speed'].text() == '4.000'
    assert window.readings['gyro_z'].text() == '0.100'
    process = window.process
    window.close()
    qt.processEvents()
    assert process.state() == app.QProcess.ProcessState.NotRunning


def test_simulation_exit_resets_link(tmp_path):
    qt = app.QApplication.instance() or app.QApplication([])
    script = tmp_path / 'exits.py'
    script.write_text("print('ready', flush=True)\n")
    window = app.MainWindow(viewer_command=[sys.executable, str(script)])
    window.launch_button.click()
    assert wait_for(lambda: window.launch_button.isEnabled(), qt)
    assert window.status.text() == 'Simulation closed'
    window.close()


def test_readings_follow_the_physics():
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(viewer.MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    still = viewer.readings(data)
    assert set(still) == set(viewer.FIELDS)
    assert abs(still['speed']) < 1e-6 and abs(still['heading']) < 1e-6
    data.ctrl[:] = 4
    mujoco.mj_step(model, data, nstep=1000)
    moving = viewer.readings(data)
    assert moving['x'] > .1 and moving['speed'] > .1
    assert moving['left_speed'] > 3 and moving['left_angle'] > 3
    data.ctrl[:] = [-3, 3]
    mujoco.mj_step(model, data, nstep=500)
    turning = viewer.readings(data)
    assert turning['turn_rate'] > .5 and turning['gyro_z'] > .5


def test_state_message_round_trip():
    state = {field: 0.5 for field in viewer.FIELDS}
    assert viewer.decode_state(viewer.encode_state(state)) == state
