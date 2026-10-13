"""Interactive differential-drive demo. On macOS: .venv/bin/mjpython simulate.py."""

from pathlib import Path
import threading
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np

MODEL = Path(__file__).parent / 'models' / 'differential_drive.xml'


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    target = np.zeros(2)
    command_lock = threading.Lock()
    commands = {
        glfw.KEY_UP: (4, 4), glfw.KEY_DOWN: (-4, -4),
        glfw.KEY_LEFT: (-3, 3), glfw.KEY_RIGHT: (3, -3),
        glfw.KEY_SPACE: (0, 0),
    }

    def on_key(key):
        if key in commands:
            with command_lock:
                target[:] = commands[key]

    print('Arrow keys: drive/turn | Space: stop. Commands persist until changed.', flush=True)
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0, 0, .05]
            viewer.cam.distance = 2.2
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -35
        print('MuJoCo viewer ready', flush=True)
        next_report = 0.0
        while viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
                with command_lock:
                    # Limit target acceleration to 6 rad/s² for gentle reversals.
                    limit = 6 * 10 * model.opt.timestep
                    data.ctrl[:] += np.clip(target - data.ctrl, -limit, limit)
                mujoco.mj_step(model, data, nstep=10)
                if data.time >= next_report:
                    w, x, y, z = data.qpos[3:7]
                    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
                    print(f't={data.time:6.1f}s  x={data.qpos[0]:+.3f}m  y={data.qpos[1]:+.3f}m  heading={np.degrees(yaw):+.1f}deg', flush=True)
                    next_report = data.time + 2
            viewer.sync()
            time.sleep(max(0, 10*model.opt.timestep - (time.monotonic()-started)))


if __name__ == '__main__':
    main()
