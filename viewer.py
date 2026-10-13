"""MuJoCo simulation window launched by app.py. On macOS run with mjpython.

Link protocol over stdio: prints 'ready' once the window is open, then one
JSON object of readings (see FIELDS) per line at 10 Hz of sim time. Units:
m, m/s, rad, rad/s and m/s²; heading and turn rate are counterclockwise.
Exits when its window is closed or when stdin closes (the app quit).
"""

import json
import sys
import threading
import time

import mujoco
import numpy as np

from simulation import MODEL

STATE_PERIOD = .1
FIELDS = ('time', 'x', 'y', 'heading', 'speed', 'turn_rate',
          'left_angle', 'right_angle', 'left_speed', 'right_speed',
          'gyro_z', 'accel_x', 'accel_y')


def readings(data):
    """Ground-truth pose and motion plus the robot's sensor readings."""
    robot = data.body('robot')
    qw, qx, qy, qz = robot.xquat
    heading = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    # Free-joint linear velocity is in the world frame, angular in the body frame.
    velocity = data.joint('base').qvel
    turn_rate = (robot.xmat.reshape(3, 3) @ velocity[3:6])[2]
    gyro, accel = data.sensor('gyro').data, data.sensor('accelerometer').data
    values = (data.time, *robot.xpos[:2], heading,
              velocity[0]*np.cos(heading) + velocity[1]*np.sin(heading), turn_rate,
              data.joint('left_wheel_joint').qpos[0], data.joint('right_wheel_joint').qpos[0],
              data.sensor('left_encoder').data[0], data.sensor('right_encoder').data[0],
              gyro[2], accel[0], accel[1])
    return {field: float(value) for field, value in zip(FIELDS, values, strict=True)}


def encode_state(state):
    return json.dumps(state)


def decode_state(line):
    state = json.loads(line)
    return {field: float(state[field]) for field in FIELDS}


def main():
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    app_open = threading.Event()
    app_open.set()

    def watch_app():
        sys.stdin.read()
        app_open.clear()

    threading.Thread(target=watch_app, daemon=True).start()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0, 0, .05]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.6, 135, -35
        print('ready', flush=True)
        next_state = 0.0
        while viewer.is_running() and app_open.is_set():
            started = time.monotonic()
            with viewer.lock():
                mujoco.mj_step(model, data, nstep=10)
                if data.time >= next_state:
                    print(encode_state(readings(data)), flush=True)
                    next_state = data.time + STATE_PERIOD
            viewer.sync()
            time.sleep(max(0, 10*model.opt.timestep - (time.monotonic()-started)))


if __name__ == '__main__':
    main()
