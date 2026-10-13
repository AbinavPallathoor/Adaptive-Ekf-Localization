# Adaptive EKF Localization

Simulation study comparing dead reckoning, standard EKF, and Adaptive EKF under changing sensor uncertainty.

## Environment

The project uses Python 3.14 and a local virtual environment.

```sh
source .venv/bin/activate
```

To recreate the installed environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

Start notebooks with `jupyter lab` and select **Adaptive EKF Localization** as the kernel.
On macOS, run scripts that use MuJoCo's passive viewer with `mjpython script.py`.

Dependencies: MuJoCo Python bindings, NumPy, Pandas, SciPy, Matplotlib, JupyterLab, ipykernel, and pytest.

## Desktop app

```sh
source .venv/bin/activate
python app.py
```

The PySide6 (Qt) app has one **Launch simulation** button. It opens the
robot in its own MuJoCo window (`viewer.py`, run under `mjpython` on macOS)
and links to it over the process's stdin/stdout: the window reports `ready`,
then streams readings at 10 Hz of simulation time. The app shows the link
status and:

- **Pose** (ground truth): x, y (m) and heading (°, counterclockwise from +x).
- **Motion** (ground truth): forward speed (m/s) and turn rate (°/s).
- **Wheel encoders:** total wheel angle (rad) and wheel speed (rad/s).
- **IMU:** gyro z, the yaw rate (rad/s), and accelerometer x, y (m/s², body frame).

Gyro x/y and accelerometer z are left out: on flat ground they stay near
0, 0 and 9.81 m/s². Closing the MuJoCo window resets the app to launch again;
closing the app closes the simulation window.

There are no drive controls or recordings in the app yet. `simulation.py` and
`database.py` (sensor noise and SQLite logging) remain for later steps.

## Drive the robot

```sh
source .venv/bin/activate
mjpython simulate.py
```

Click the 3D simulation view, then use **↑/↓** to move forward/reverse,
**←/→** to turn left/right, and **Space** to stop. Commands persist until
another key is pressed; wheel speeds ramp smoothly.
The orange stripe marks the front (+x).

The model has two independently driven 6 cm rear wheels, a 30 cm wheel track,
and a front low-friction ball support approximating a caster. Wheel velocity,
gyro, accelerometer, and ground-truth pose outputs are available without
added sensor noise. The Python runner prints pose every two seconds.

Run the headless motion checks with `python -m pytest -q`.
