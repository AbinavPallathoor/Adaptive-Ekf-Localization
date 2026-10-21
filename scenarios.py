"""Ready-made test conditions, shared by the live demo and the experiments."""

from dataclasses import dataclass, replace

from ekf import EKFSettings


@dataclass(frozen=True)
class Scenario:
    key: str
    name: str
    short: str  # label for plots
    description: str
    fix_noise: float = .1  # m, position-fix σ at the start
    gyro_noise: float = .01  # rad/s, gyro σ at the start
    fix_rate: float = 1  # Hz
    outlier_rate: float = 0  # fraction of fixes that are bad
    outlier_noise: float = 1  # m, σ of a bad fix
    slip: float = 0  # wheel slip σ, as a fraction of wheel travel
    step_time: float | None = None  # s; the noise changes to the step values at this time
    step_fix_noise: float | None = None
    step_gyro_noise: float | None = None
    # What the filters assume about the noise; None means the true starting values.
    assumed_fix_noise: float | None = None
    assumed_gyro_noise: float | None = None

    def filter_settings(self, base=EKFSettings()):
        """The filters' assumptions for this scenario, on top of base."""
        fix = self.fix_noise if self.assumed_fix_noise is None else self.assumed_fix_noise
        gyro = self.gyro_noise if self.assumed_gyro_noise is None else self.assumed_gyro_noise
        return replace(base, fix_noise=fix, gyro_noise=gyro)


SCENARIOS = [
    Scenario('steady', 'Steady noise', 'Steady noise',
             'The sensors behave exactly as the filters expect. The baseline: adapting should not hurt.'),
    Scenario('fix_jump', 'Position fixes get noisier', 'Noisier fixes',
             'At 30 s the position fixes become four times noisier (σ 0.1 → 0.4 m).',
             step_time=30, step_fix_noise=.4),
    Scenario('gyro_jump', 'Gyro gets noisier', 'Noisier gyro',
             'At 30 s the gyro becomes eight times noisier (σ 0.01 → 0.08 rad/s).',
             step_time=30, step_gyro_noise=.08),
    Scenario('wrong', 'Wrong assumptions', 'Wrong assumptions',
             'The filters start out trusting the fixes and gyro far too much (assumed σ 0.02 m, 0.002 rad/s).',
             assumed_fix_noise=.02, assumed_gyro_noise=.002),
    Scenario('outliers', 'Occasional bad fixes', 'Bad fixes',
             'One fix in ten is wildly wrong (σ 1 m instead of 0.1 m).', outlier_rate=.1),
    Scenario('slip', 'Slippery wheels', 'Slippery wheels',
             'The wheels slip, so the encoders over-read how far the robot moved (by about 4%).', slip=.05),
]
BY_KEY = {scenario.key: scenario for scenario in SCENARIOS}
