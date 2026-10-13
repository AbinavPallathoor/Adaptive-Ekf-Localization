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

One window, three columns:

- **Setup** (left): route, position-fix noise σ, fix rate and gyro noise σ.
  Click **Launch** to open the robot in its own MuJoCo window; the button
  becomes **Stop**. Setup is locked while running. **Speed** (1×–10×) can be
  changed while running.
- **Plot** (middle): top-down true path, dead reckoning, EKF and the noisy
  position fixes, in metres.
- **Error vs. true position** (right): current and RMS position error (m) and
  heading error (°) for dead reckoning and the EKF. **Sensor readings**
  expands to show time, true pose, wheel angles, gyro and the last fix.

### Routes

The robot follows its route with a closed-loop autopilot using the true pose
(`robot.RouteFollower`), turning on the spot at corners. Corner points are
fixed from the start pose, so laps repeat without drifting.

- **Square:** 1 m sides, 90° turns.
- **Hexagon:** 0.6 m sides, 60° turns.
- **Figure 8:** two 0.4 m-radius circles, left then right.

### Sensors

Every 0.05 s (20 Hz of simulation time) the simulation sends ground truth and
sensor readings to the app over stdout (`robot.FIELDS`). Poses are of the axle
midpoint, the point the robot turns about.

- **Wheel encoders:** total wheel angles (rad), noise-free. Their error comes
  from wheel slip, mostly during turns.
- **Gyro:** yaw rate (rad/s) plus Gaussian noise σ.
- **Position fix:** true x, y plus Gaussian noise σ (m), at the fix rate
  (like a GPS or beacon).

Noise uses a fixed seed (42), so the same settings give the same run.

### Estimators

Both start from the known initial pose.

- **Dead reckoning** (`dead_reckoning.py`): wheel encoders only. Each step
  moves `d = r(ΔL+ΔR)/2` and turns `Δθ = r(ΔR−ΔL)/b` (wheel radius `r` =
  0.06 m, track `b` = 0.30 m), using the midpoint heading. Errors are never
  corrected.
- **EKF** (`ekf.py`), state `[x, y, θ]`:
  - *Predict:* distance `d` from the encoders and heading change from the
    gyro (trapezoid of consecutive readings × 0.05 s), so wheel slip in turns
    does not corrupt heading. Covariance `P = F P Fᵀ + G M Gᵀ + Q_slip`, with
    `M = diag((0.02·d)², (σ_gyro·Δt)²)`.
  - *Update* on each position fix: `H = [[1,0,0],[0,1,0]]`, `R = σ_fix² I`,
    gain `K = P Hᵀ (H P Hᵀ + R)⁻¹`, Joseph-form covariance update.
  - Noise levels are fixed and known; the Adaptive EKF will estimate them.

`localization.Tracker` runs both estimators on each sample and keeps the
paths and error statistics shown in the app.

## Robot model

`models/differential_drive.xml`: two independently driven 6 cm rear wheels,
a 30 cm wheel track, and a front low-friction ball support approximating a
caster. The orange stripe marks the front (+x); the orange dot between the
wheels is the axle midpoint used as the robot's pose.

`simulate.py` is an older standalone demo that drives the robot with the
arrow keys (`mjpython simulate.py`); the app does not use it.

Run the tests with `.venv/bin/python -m pytest -q`.
