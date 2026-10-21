"""Hover tooltips and the "How it works" page shown in the app."""

TIPS = {
    # Live demo
    'route': 'The path the robot drives. An autopilot steers it using the true position, '
             'so the true path is always the same shape.',
    'scenario': 'What happens to the sensors during the run. Picking one fills in the settings below.',
    'launch': 'Opens the robot in its own MuJoCo window and starts the run. '
              'Stop closes it; the plot and errors stay until the next launch.',
    'speed': 'How fast the simulation runs compared with real time. Can be changed during a run.',
    'settings': 'Show or hide every setting. The scenario fills these in; change any of them to experiment.',
    'fix_noise': 'Standard deviation σ of the Gaussian error added to each position fix (like GPS). '
                 'About 68% of fixes land within σ. Can be changed during a run.',
    'gyro_noise': 'Standard deviation σ of the Gaussian noise added to every gyro yaw-rate reading. '
                  'Can be changed during a run.',
    'fix_rate': 'How often a position fix arrives. Between fixes the filters rely on the wheels and gyro.',
    'outlier_rate': 'Percentage of position fixes that are bad: they get the much larger "bad fix σ" instead.',
    'outlier_noise': 'Standard deviation σ (m) of a bad position fix.',
    'slip': 'How much the wheels slip. Each wheel turns further than the robot moves, by a random '
            'fraction |N(0, σ)| of its travel, so the encoders over-read the distance.',
    'step': 'Ticked: the sensor noise switches to the values below at a set time. It happens at exactly '
            'that simulation time, at any speed, so the run can be repeated.',
    'step_time': 'Simulation time (s) at which the sensor noise changes.',
    'step_fix_noise': 'Position-fix σ (m) from the step on.',
    'step_gyro_noise': 'Gyro σ (rad/s) from the step on.',
    'seed': 'Starting number for the random noise. The same seed and settings give exactly the same run.',
    'match_noise': 'Ticked: the filters assume the sensor noise set above (at launch). Untick to give them '
                   'different, wrong assumptions. During a run the EKF keeps its assumptions even if the '
                   'noise changes; that is the point of the comparison.',
    'tune_fix_noise': 'Measurement noise R: how noisy the filters believe position fixes are. '
                      'Larger: trust fixes less and smooth more. The Adaptive EKF only starts from this.',
    'tune_gyro_noise': 'Part of process noise Q: how noisy the filters believe the gyro is. '
                       'The Adaptive EKF (R and gyro σ) only starts from this.',
    'tune_distance_noise': 'Part of process noise Q: expected encoder distance error, as a percentage '
                           'of the distance travelled.',
    'tune_slip_noise': 'Part of process noise Q: extra position drift allowed per 0.05 s step for '
                       'wheel slip the model does not capture.',
    'tune_initial_sigma': 'Starting covariance P₀: how sure the filters are of the start pose '
                          '(σ in metres and radians). Applies at launch.',
    'gate': 'Ticked: every filter rejects a position fix that is too far from where it expected, using a '
            'χ² test on the normalised innovation (99% level). Protects against bad fixes, but can also '
            'reject good ones when the filter is overconfident.',
    'variant': 'R and gyro σ: the Adaptive EKF estimates both the position-fix noise and the gyro noise. '
               'R only: it estimates just the fix noise and keeps the assumed gyro σ.',
    'memory': 'How many seconds of recent sensor data the Adaptive EKF bases its noise estimates on. '
              'Shorter: reacts faster to a change, but its estimates are noisier.',
    # Experiment
    'scenarios': 'The conditions to test. Each is run on every chosen route.',
    'routes': 'The routes to repeat each scenario on.',
    'runs': 'Independent runs (each with its own random seed) per scenario and route. '
            'Each run is one paired observation for the statistics; 20 or more is a good number.',
    'run_experiment': 'Runs every chosen scenario and route the chosen number of times, without the '
                      'MuJoCo window, then compares the filters statistically.',
    'export': 'Saves the raw runs and summary as CSV files and every figure as a PNG, for the report.',
    'compare': 'Which Adaptive EKF to compare with the EKF in the table and difference plot.',
    'explainer': 'Show or hide an explanation.',
}

ERROR_TIPS = {
    'now': 'Distance between the estimate and the true position at this moment (metres).',
    'rms': 'Root-mean-square position error over the whole run so far (metres). '
           'The main single number for comparing methods.',
    'dr': 'Dead reckoning: position from the wheel encoders only, never corrected.',
    'ekf': 'Extended Kalman filter: wheels and gyro predict, position fixes correct. '
           'Uses fixed noise assumptions.',
    'aekf': 'Adaptive EKF: the same filter, but it estimates the fix and gyro noise from the sensor '
            'data as it goes, instead of assuming them.',
    'aekf_r': 'Adaptive EKF (R only): estimates only the position-fix noise R; the gyro noise stays '
              'as assumed.',
}

RESULT_TIPS = {
    'scenario': 'The test condition.',
    'ekf': 'Mean position RMSE of the EKF over all runs (m).',
    'adaptive': 'Mean position RMSE of the Adaptive EKF over all runs (m).',
    'difference': 'Mean of D = adaptive RMSE − EKF RMSE over the paired runs, with its 95% confidence '
                  'interval (mm). Negative: the adaptive filter did better.',
    'p': 'One-sided paired t-test of "the adaptive filter has lower average error", Holm-adjusted for '
         'testing several scenarios. Below 0.05 counts as significant.',
    'verdict': 'Better: significantly lower error. Worse: the whole confidence interval is above zero. '
               'Otherwise there is no clear difference.',
}

HELP = '''
<h2>What this app does</h2>
<p>A two-wheeled robot drives a route in a MuJoCo physics simulation. Its sensors are noisy:
the <b>wheel encoders</b> measure how far each wheel turned (fooled by wheel slip), the
<b>gyro</b> measures how fast it turns, and <b>position fixes</b> (like GPS) give x and y with
noise, once a second. Because the simulation knows the true position, we can measure exactly how
wrong each method is. The question: <i>can a filter improve when it learns how sensor noise
changes?</i></p>

<h2>The three methods</h2>
<p><b>Dead reckoning</b> adds up wheel motion from the start. Small errors are never corrected,
so it drifts.</p>
<p><b>EKF</b> (extended Kalman filter) keeps an estimate of [x, y, heading] and its uncertainty,
the covariance P. Each 0.05 s it <i>predicts</i> with the wheels and gyro: μ⁻ = f(μ, u),
P⁻ = F P Fᵀ + Q. When a fix z arrives it <i>corrects</i>: innovation ν = z − Hμ⁻, its covariance
S = H P⁻ Hᵀ + R, gain K = P⁻ Hᵀ S⁻¹, and μ⁺ = μ⁻ + Kν. It assumes fixed noise levels: R for the
fixes and Q for the motion. If those assumptions are wrong, or the noise changes, it weighs the
sensors wrongly.</p>
<p><b>Adaptive EKF</b> is the same filter, but it estimates the noise as it goes.
<i>Fix σ (R)</i>: after each fix it looks at the residual ε = z − Hμ⁺ and updates
R ← αR + (1−α)(εεᵀ + H P⁺ Hᵀ), with α = exp(−Δt / memory), so older fixes count less
(residual-based covariance matching, Akhlaghi et al. 2017). <i>Gyro σ (part of Q)</i>: the true
turn rate changes smoothly, so the second difference of consecutive gyro readings is almost pure
noise, with standard deviation σ√6; the median over the memory time gives σ while ignoring the
moments the robot starts or stops turning. The <i>R only</i> variant adapts just R, keeping Q
fixed, which separates the effect of sensor-noise adaptation.</p>

<h2>Live demo</h2>
<p>Pick a route and a scenario and press <b>Launch</b>. The plot shows the true path, each
estimate and the position fixes (bad fixes as red crosses). The thin rings around the filters
are their <b>95% ellipses</b>: if a filter is honest about its uncertainty, the true position
should be inside its ring 95% of the time. Below, the noise plots show the true fix and gyro σ,
what the EKF assumes and what the Adaptive EKF has estimated. Open <b>Settings</b> to change
anything; the sensor noise sliders also work during a run.</p>

<h2>Experiment</h2>
<p>One run is one observation; thousands of correlated time steps are not thousands of
independent trials. So the Experiment page repeats each scenario on each route many times, each
run with its own random seed, without the 3-D window. Within a run every filter gets exactly the
same readings, so the runs are <b>paired</b>. For each run i,
D<sub>i</sub> = RMSE(adaptive, i) − RMSE(EKF, i); negative means the adaptive filter did better.</p>
<p>The test, fixed in advance, is a one-sided <b>paired t-test</b> of H₀: E[D] = 0 against
H₁: E[D] &lt; 0. The table reports the mean difference with a <b>95% confidence interval</b> and
the p-value, <b>Holm-adjusted</b> because several scenarios are tested at once (without it the
chance of at least one false "better" grows with every test). The saved summary also has the
effect size d<sub>z</sub> = mean(D)/sd(D), a <b>Shapiro–Wilk</b> check of the t-test's normality
assumption and a <b>Wilcoxon signed-rank</b> test, which does not need that assumption.</p>
<p><b>Uncertainty check</b>: how often the truth was inside each filter's 95% ellipse (ideal
95%). Below 95% means overconfident. <b>Noise check</b>: the simulated sensor errors against the
N(0, σ²) distribution they are drawn from.</p>

<h2>Scenarios</h2>
<p><b>Steady noise</b>: the baseline; adapting should not hurt. <b>Position fixes / gyro get
noisier</b>: the noise steps up at 30 s; the EKF keeps its old assumptions. <b>Wrong
assumptions</b>: the filters start far too confident in their sensors. <b>Occasional bad
fixes</b>: one fix in ten is wildly wrong. <b>Slippery wheels</b>: the encoders over-read the
distance travelled, a bias that changing noise levels cannot fix.</p>

<h2>Notes</h2>
<p>The autopilot uses the true position to follow the route: this tests estimation, not
navigation. Units are metres, radians and seconds unless marked. Noise comes from a fixed random
seed, so the same settings repeat exactly.</p>
'''
