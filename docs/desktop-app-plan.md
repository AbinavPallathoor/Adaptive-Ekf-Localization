# Desktop localization app

Build one PySide6 window with an embedded MuJoCo render, drive controls,
simulation pause/reset, live pose and sensors, settings, and saved recordings.
Use the existing rear-wheel model. Keep the standalone viewer available.

## Design

- `simulation.py`: deterministic stepping, drive ramp, ground truth, and
  independently configurable Gaussian position/encoder/gyro noise.
- `database.py`: SQLite runs and timestamped samples, configuration and initial
  state snapshots, CSV export. Transactions and foreign keys preserve runs.
- `app.py`: Qt widgets, an embedded render, camera drag/zoom, drive buttons and
  arrows, settings, recording lifecycle, live readings, recordings table.
- Local database: `data/localization.sqlite3`, ignored by Git.
- Rendering and physics run on the Qt thread with a 20 ms timer. Rendering is
  capped at 25 Hz. Simulation time is authoritative; overloaded machines may
  run slower than wall time, which is also recorded.
- Settings are fixed during a recording. Reset finishes recording before
  rewinding simulation time. Pause writes no duplicate samples. Closing the
  app finishes the active recording and closes the database and renderer.
- Native light UI with slate text, blue controls, red recording/stop emphasis,
  and a large simulation view. No browser or server.

## Implementation and validation

1. Write failing physics/sensor and SQLite persistence/export tests.
2. Implement simulation and storage; pass tests using real MuJoCo and SQLite.
3. Add Qt UI and dependency; test controls, pause/reset, recording, saved-run
   selection/export, and shutdown with Qt's real event loop.
4. Render the actual window for visual inspection; launch it on macOS.
5. Document launching, sampling units, database location, and current scope.

EKF, wheel-slip experiments, outliers, and Monte Carlo automation remain later
research steps. This app logs the data currently available and configurable
sensor noise; it does not claim to implement those later methods.
