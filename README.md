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

Three pages: **Live demo**, **Experiment** and **How it works** (the maths).
Hover over any control or number for an explanation.

**Live demo** — a sidebar beside a large plot.

1. **Route:** Square, Hexagon or Figure 8.
2. **What happens to the sensors:** a ready-made scenario (`scenarios.py`,
   below). Picking one fills in every setting.
3. **Run:** **Launch** opens the robot in its own MuJoCo window and becomes
   **Stop**; **speed** (1×–10×) can be changed while running. Below it, the
   current and RMS position error (m) of dead reckoning, the EKF and the
   Adaptive EKF, and after 10 s a one-line verdict comparing the two filters.
4. **Settings** (collapsed):
   - *Sensor noise:* fix σ and gyro σ; also change the noise during a run.
   - *Simulation* (set at launch): fix rate, percentage and σ of bad fixes,
     wheel slip, an optional noise step at a set time, and the random seed.
   - *Filter assumptions:* fix σ (R), gyro, distance and slip σ (Q) and
     initial σ (P₀). **Assume the sensor noise above** sets the assumed fix and
     gyro σ to the simulated ones at launch; untick it to mistune the filters
     on purpose. **Reject bad fixes** turns on the χ² gate. Changes apply
     immediately during a run, except initial σ.
   - *Adaptive EKF:* which variant to run (R and gyro σ, or R only) and its
     memory (s).

The plot shows the true path, the estimates, each filter's 95% position
ellipse and the position fixes (bad ones as crosses), in metres at equal
scale. Below it, fix σ and gyro σ over time: true, assumed by the EKF and
estimated by the Adaptive EKF.

**Experiment** — repeated, paired runs without the 3-D window (see
*Experiment* below). Choose scenarios, routes, runs per combination and the
gate, then **Run experiment** (runs in parallel; **Stop** cancels). The page
shows a results table per Adaptive EKF variant, four figures (differences
with confidence intervals, RMSE of each run, uncertainty check, noise check)
and **Save results for the report…**, which writes CSVs and PNGs.

The interface follows the system's light or dark mode. Explanatory text lives
in `help_text.py`; colours, the stylesheet and the slider control in `ui.py`;
the plot widgets in `plots.py`.

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

- **Wheel encoders:** total wheel angles (rad). Besides the physical slip in
  turns, an optional *slip* σ makes each wheel's change per sample over-read
  by a fraction |N(0, slip²)|, so the encoders report more travel than happened.
- **Gyro:** yaw rate (rad/s) plus Gaussian noise σ.
- **Position fix:** true x, y plus Gaussian noise σ (m), at the fix rate
  (like a GPS or beacon). A fraction *outlier rate* of fixes are bad fixes,
  with σ *outlier noise* instead.

Each sample also carries the noise σ in force (`fix_sigma`, `gyro_sigma`) and
whether the fix was bad (`fix_outlier`), for comparison only. The app changes
the noise during a run by sending `fix_noise σ` / `gyro_noise σ` lines on the
simulation's stdin; a noise step
(`viewer.py --step-time T --step-fix-noise σ --step-gyro-noise σ`) is applied by
the simulation itself at simulation time T, so it is the same at every speed.

Noise is seeded (default 42), and gyro, fixes, bad fixes and slip each have
their own random stream, so the same settings give the same run (unless the
noise is changed by hand during it) and turning one effect on leaves the
others' noise unchanged.

### Scenarios

Shared by the live demo and the experiment (`scenarios.py`). Unless stated,
fix σ = 0.1 m, gyro σ = 0.01 rad/s, 1 fix/s, and the filters assume the true
starting noise.

- **Steady noise:** the sensors behave as expected — adapting should not hurt.
- **Position fixes get noisier:** at 30 s fix σ 0.1 → 0.4 m.
- **Gyro gets noisier:** at 30 s gyro σ 0.01 → 0.08 rad/s.
- **Wrong assumptions:** the filters assume fix σ 0.02 m and gyro σ 0.002 rad/s.
- **Occasional bad fixes:** one fix in ten has σ 1 m.
- **Slippery wheels:** wheel slip σ 0.05.

### Estimators

All three start from the known initial pose.

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
  - Noise levels are the assumed ones from the filter settings.
  - *Gate* (optional): a fix whose normalised innovation squared
    `ν ᵀ S⁻¹ ν` exceeds 9.21 (χ², 2 dof, 99%) is rejected as an outlier.
- **Adaptive EKF** (`ekf.AdaptiveEKF`): the same filter, but it estimates the
  fix and gyro noise instead of assuming them (the assumed values are only its
  starting guesses). Both estimates forget old data over the *memory* time τ
  (default 20 s).
  - *Fix σ (R):* residual-based covariance matching (Akhlaghi et al., 2017).
    After each fix, with post-fit residual `ε = z − H x⁺`,
    `R ← αR + (1−α)(ε εᵀ + H P⁺ Hᵀ)`, `α = exp(−Δt/τ)`. If it starts far too
    confident in the fixes it follows them closely, so residuals are small and
    R grows only slowly towards the truth.
  - *Gyro σ (heading part of Q):* from the gyro readings themselves. The true
    yaw rate changes smoothly, so the second difference `g[k+1] − 2g[k] + g[k−1]`
    is almost pure noise with σ√6; `σ = 1.4826·median|Δ²g|/√6` over the last τ
    seconds ignores the moments the robot starts or stops turning.
  - Distance and slip σ (the rest of Q) are taken from the filter settings.
  - *Adaptive EKF (R only)* adapts only the fix σ and keeps the assumed gyro σ,
    to separate the effect of the two estimates.

`localization.Tracker` runs dead reckoning, the EKF and the chosen Adaptive
EKF variants on each sample (all get identical inputs) and keeps the paths,
error statistics and noise history shown in the app. It also checks each
filter's honesty about its uncertainty: the fraction of samples where the
true position lies inside its 95% position ellipse (ideally 95%).

## Experiment

`experiment.py` (or the app's Experiment page) runs every chosen scenario ×
route × run without the viewer, two laps each, in parallel. Each run has its
own seed, shared across scenarios, and every estimator sees the same
readings, so each run gives one *paired* observation.

```sh
.venv/bin/python experiment.py --runs 20 --out output/experiment
.venv/bin/python experiment.py --help    # choose scenarios, routes, gate, memory
```

`analysis.py` then compares each Adaptive EKF variant with the EKF per
scenario using the paired differences `Dᵢ = RMSE(adaptive) − RMSE(EKF)`:

- Primary test: one-sided paired t-test, H₀: E[D] = 0 vs. H₁: E[D] < 0, with
  a 95% confidence interval for E[D] and effect size d_z.
- Checks: Shapiro–Wilk on D (the t-test's normality assumption) and a
  Wilcoxon signed-rank test, which does not need it.
- p-values are Holm-adjusted across all comparisons in the table.
- Verdict: *Better* if adjusted p < 0.05 and E[D] < 0; *Worse* if the whole
  confidence interval is above 0; otherwise *No clear difference*.

Saved output: `runs.csv` (one row per run), `summary.csv` (one row per
scenario and variant) and PNG figures: `difference_*`, `errors_*`,
`calibration` (95% ellipse coverage) and `noise_check` (simulated sensor
errors against the N(0, σ²) they should follow).

## Robot model

`models/differential_drive.xml`: two independently driven 6 cm rear wheels,
a 30 cm wheel track, and a front low-friction ball support approximating a
caster. The orange stripe marks the front (+x); the orange dot between the
wheels is the axle midpoint used as the robot's pose.

`simulate.py` is an older standalone demo that drives the robot with the
arrow keys (`mjpython simulate.py`); the app does not use it.

Run the tests with `.venv/bin/python -m pytest -q` (about 10 s).
