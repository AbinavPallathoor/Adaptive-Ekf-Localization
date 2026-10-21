# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Simulation study comparing wheel-encoder dead reckoning, a standard EKF and an Adaptive EKF that estimates its own noise levels, for a MuJoCo differential-drive robot: a live demo plus a repeated, paired statistical experiment. README.md is the user-facing description of the app, routes, sensors and estimator maths; keep it in sync when behaviour changes.

## Commands

Python 3.14 in a local venv (`.venv/`). Install from `requirements-lock.txt` (pinned); `requirements.txt` lists only the top-level packages.

```sh
.venv/bin/python app.py                       # desktop app (PySide6 + Matplotlib)
.venv/bin/python -m pytest -q                 # all tests (~10 s)
.venv/bin/python experiment.py --runs 20      # headless paired experiment → output/experiment
.venv/bin/python -m pytest -q tests/test_ekf.py::test_name   # single test
```

There is no linter or build step. On macOS, anything that opens MuJoCo's passive viewer must run under `mjpython` (e.g. `mjpython viewer.py --route square`); `app.py` handles this itself.

## Architecture

`app.py`'s `MainWindow` holds three tabs: `LivePage` (live demo), `experiment_page.ExperimentPage` and a help page.

The live demo is two processes linked by a line-based stdio protocol:

- **`app.LivePage`** (Qt GUI) launches **`viewer.py`** as a `QProcess` (via `mjpython` on macOS, see `default_viewer_command`). Simulation settings (route, seed, noise σ, fix rate, bad fixes, slip, speed, optional `--step-*` noise step) are passed as CLI args.
- `viewer.py` → app (stdout): `ready` once the window is open, then one JSON sample per line every 0.05 s of sim time with the fields in `robot.FIELDS`. `fix_x`/`fix_y` are `null` except on position-fix samples; `fix_sigma`/`gyro_sigma`/`fix_outlier` are ground truth (noise in force, bad-fix flag), not for the estimators.
- app → `viewer.py` (stdin): `speed N`, `fix_noise σ`, `gyro_noise σ` lines (parsed by `robot.parse_command`, which returns `(name, value)`). Closing stdin (Stop button / app quit) makes the viewer exit.
- The scripted noise step (`robot.NoiseStep`) runs inside the sim on sim time so it is reproducible; the app only mirrors it into its sliders when a sample passes the step time.

Module roles:

- `robot.py` — headless simulation core shared by `viewer.py`, `experiment.py` and the tests: model loading, ground-truth pose (axle midpoint), `Sensors` (seeded noise on gyro and position fixes, bad fixes, encoder slip; one RNG stream per effect), `RouteFollower` (closed-loop autopilot on the *true* pose, segment endpoints precomputed from the start pose so laps don't drift), and the sample encode/decode for the stdio link.
- `dead_reckoning.py`, `ekf.py` — estimators. `EKFSettings` is a frozen dataclass of the filter's *assumed* noise levels (tuning), deliberately independent of the simulated sensor noise so the filter can be mistuned. `AdaptiveEKF(EKF)` overrides `measurement_noise` (adaptive R from post-fit residuals) and `gyro_sigma` (median of gyro second differences, fed via `observe_gyro`; off with `adapt_gyro=False`); both forget over `memory` seconds. Its settings are only starting guesses. `EKFSettings.gate` enables the χ² outlier gate.
- `localization.Tracker` — consumes decoded samples, runs dead reckoning, the EKF and the Adaptive EKF variants named in `adaptive` (keys of `localization.ADAPTIVE`: `aekf`, `aekf_r`) side by side in `tracker.estimators[name]`; keeps paths, RMS/current error stats, 95%-ellipse coverage and `noise_history` (true / EKF-assumed / adaptive-estimated σ at each fix). Filter predict uses encoder distance but **gyro** heading change (trapezoid of consecutive readings), so wheel slip in turns doesn't corrupt heading. `tune()` swaps EKF settings mid-run and changes the AEKF's memory, but never overwrites the AEKF's estimates.
- `scenarios.py` — the ready-made test conditions shared by the live demo and the experiment.
- `experiment.py` — headless runs (`run`, `jobs`, `run_all`, `noise_sample`); seeds are shared across scenarios so runs are paired. The Experiment page submits runs straight to a `ProcessPoolExecutor` and polls the futures with a `QTimer` on the GUI thread — a `QThread` worker deadlocked here, and progress callbacks from a worker thread would touch widgets off the GUI thread. Spawned workers re-import `__main__`, so any script that starts experiments needs an `if __name__ == '__main__'` guard. `analysis.py` — paired t-test, CI, Holm correction, verdicts, figures and `export` (CSV + PNG).
- `plots.py` (live path/noise plots, experiment figure canvas), `ui.py` (themes, stylesheet, `SliderField`) and `help_text.py` (all tooltip/help HTML) are kept out of `app.py`; put new UI copy in `help_text.py`.
- `models/differential_drive.xml` — the robot. `simulate.py` is an old keyboard-driven demo, unused by the app.

Constants duplicated across modules that must stay consistent with the XML model: `WHEEL_RADIUS`/`TRACK` (`robot.py`, `dead_reckoning.py`) and `SAMPLE_PERIOD` (`robot.py`, `ekf.py`).

## Testing notes

- `tests/test_app.py` sets `QT_QPA_PLATFORM=offscreen` and replaces the viewer with a fake script (`MainWindow(viewer_command=[...])`) that echoes its args and emits canned samples (they must include every field in `robot.FIELDS` or they are silently dropped) — use this pattern for app tests instead of launching MuJoCo. Live-demo widgets are on `window.live`.
- `tests/test_experiment.py` covers scenarios, the statistics on synthetic results, single headless runs and the Experiment page (one small real run through its process pool).
- `tests/test_robot_sim.py` runs real headless physics via `robot.py` (no viewer) to check route closure and lap drift.
- Noise uses a fixed seed (42 by default), so runs are reproducible. Performance assertions (Adaptive EKF vs. mistuned EKF) average over a few seeds, since one noise draw can be unlucky.

`.presentation-build/`, `.chart-data-*/` and `output/` are generated (presentation build, experiment output) and git-ignored.
