from pathlib import Path

import mujoco
import numpy as np
import pytest

MODEL = Path(__file__).resolve().parents[1] / 'models' / 'differential_drive.xml'


@pytest.mark.parametrize('wheels,mode', [([4, 4], 'forward'), ([-4, -4], 'reverse'), ([-3, 3], 'left')])
def test_robot_motion(wheels, mode):
    assert MODEL.exists(), 'Differential-drive model has not been created'
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    for _ in range(500):
        mujoco.mj_step(model, data)
    start = data.qpos[:2].copy()
    data.ctrl[:] = wheels
    for _ in range(750 if mode == 'left' else 1500):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert .06 < data.qpos[2] < .15
    w, x, y, z = data.qpos[3:7]
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    if mode == 'forward':
        assert data.qpos[0] - start[0] > .4
        assert abs(data.qpos[1] - start[1]) < .1
    elif mode == 'reverse':
        assert data.qpos[0] - start[0] < -.4
    else:
        assert yaw > .5
        assert np.linalg.norm(data.qpos[:2] - start) < .15
    assert data.sensor('gyro').data.shape == (3,)
    assert data.sensor('left_encoder').data[0] == pytest.approx(wheels[0], abs=.5)


def test_rear_wheels_and_front_support():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    assert model.body('left_wheel').pos[0] < -.05
    assert model.body('right_wheel').pos[0] < -.05
    assert model.geom('caster').pos[0] > .1


def test_direction_changes_stay_upright():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    peak_tilt = 0
    for command in ([0, 0], [4, 4], [-4, -4], [-3, 3], [3, -3], [0, 0]):
        data.ctrl[:] = command
        for _ in range(1000):
            mujoco.mj_step(model, data)
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
            peak_tilt = max(peak_tilt, np.arccos(np.clip(rotation[8], -1, 1)))
    assert np.degrees(peak_tilt) < 10
