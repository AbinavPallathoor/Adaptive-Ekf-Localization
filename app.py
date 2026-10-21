"""Desktop app: watch dead reckoning, the EKF and the Adaptive EKF live, and test them statistically.

Launch with: .venv/bin/python app.py
"""

from pathlib import Path
import sys

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QRadioButton, QSpinBox, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from ekf import MEMORY, EKFSettings
from experiment_page import ExperimentPage
from help_text import ERROR_TIPS, HELP, TIPS
from localization import NAMES, Tracker
from plots import NoisePlot, PathPlot
import robot
from scenarios import BY_KEY, SCENARIOS
from ui import ROUTE_NAMES, SIDEBAR_WIDTH, THEMES, SliderField, add_row, note, scrolling, section, stylesheet, value_label

ROOT = Path(__file__).resolve().parent
# Filter tuning fields: (setting, label, maximum, step, decimals, display units per setting unit).
TUNING = [('fix_noise', 'Fix σ (m)', 2, .01, 3, 1),
          ('gyro_noise', 'Gyro σ (rad/s)', 1, .001, 3, 1),
          ('distance_noise', 'Distance σ (%)', 50, .5, 1, 100),
          ('slip_noise', 'Slip σ (mm/step)', 20, .1, 2, 1000),
          ('initial_sigma', 'Initial σ', 1, .001, 3, 1)]
VERDICT_AFTER = 10  # s of simulation before comparing the filters' errors


def default_viewer_command():
    # MuJoCo's viewer on macOS must run under mjpython, installed beside python.
    # Run it through this interpreter so a moved venv's stale shebang is ignored.
    mjpython = Path(sys.executable).with_name('mjpython')
    launcher = [str(mjpython)] if sys.platform == 'darwin' and mjpython.exists() else []
    return [sys.executable, *launcher, str(ROOT / 'viewer.py')]


class LivePage(QWidget):
    """Pick a route and a scenario, launch the MuJoCo window and watch the estimators."""

    def __init__(self, theme, status, viewer_command=None):
        super().__init__()
        self.status = status
        self.viewer_command = viewer_command or default_viewer_command()
        self.process = None
        self.tracker = Tracker()
        self.latest = None
        self.step_pending = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        layout.addWidget(self.build_sidebar())
        plots = QVBoxLayout()
        plots.setSpacing(6)
        self.plot = PathPlot(theme)
        plots.addWidget(self.plot, 1)
        self.noise_plot = NoisePlot(theme)
        plots.addWidget(self.noise_plot)
        layout.addLayout(plots, 1)
        self.choose_scenario(SCENARIOS[0].key)

    # Layout

    def build_sidebar(self):
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(2, 4, 8, 4)
        column.setSpacing(8)
        column.addWidget(section('1  Route'))
        routes = QHBoxLayout()
        routes.setSpacing(6)
        self.routes = QButtonGroup(self)
        for key, name in ROUTE_NAMES.items():
            button = QPushButton(name)
            button.setObjectName('segment')
            button.setCheckable(True)
            button.setToolTip(TIPS['route'])
            button.setProperty('key', key)
            self.routes.addButton(button)
            routes.addWidget(button)
        self.routes.buttons()[0].setChecked(True)
        column.addLayout(routes)
        column.addSpacing(6)
        column.addWidget(section('2  What happens to the sensors'))
        self.scenarios = QButtonGroup(self)
        self.scenario_buttons = {}
        for scenario in SCENARIOS:
            button = QRadioButton(scenario.name)
            button.setToolTip(f'{scenario.description}\n\n{TIPS["scenario"]}')
            button.toggled.connect(lambda on, key=scenario.key: on and self.choose_scenario(key))
            self.scenarios.addButton(button)
            self.scenario_buttons[scenario.key] = button
            column.addWidget(button)
        self.description = note('')
        self.description.setObjectName('description')
        column.addWidget(self.description)
        column.addSpacing(6)
        column.addWidget(section('3  Run'))
        run_row = QHBoxLayout()
        self.launch_button = QPushButton('Launch')
        self.launch_button.setObjectName('primary')
        self.launch_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.launch_button.setToolTip(TIPS['launch'])
        self.launch_button.clicked.connect(self.launch_or_stop)
        run_row.addWidget(self.launch_button, 1)
        self.speed = QComboBox()
        self.speed.setToolTip(TIPS['speed'])
        self.speed.addItems([f'{speed}×' for speed in robot.SPEEDS])
        self.speed.currentIndexChanged.connect(self.send_speed)
        self.speed.setEnabled(False)
        run_row.addWidget(self.speed)
        column.addLayout(run_row)
        column.addWidget(self.build_errors())
        self.settings_toggle = QToolButton()
        self.settings_toggle.setObjectName('explainer')
        self.settings_toggle.setText('Settings')
        self.settings_toggle.setToolTip(TIPS['settings'])
        self.settings_toggle.setCheckable(True)
        self.settings_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.settings_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.settings_toggle.toggled.connect(self.show_settings)
        column.addWidget(self.settings_toggle)
        self.settings = self.build_settings()
        self.settings.setVisible(False)
        column.addWidget(self.settings)
        column.addStretch()
        scroll = scrolling(page)
        scroll.setFixedWidth(SIDEBAR_WIDTH)
        return scroll

    def build_errors(self):
        group = QGroupBox('Position error (m)')
        column = QVBoxLayout(group)
        grid = QGridLayout()
        column.addLayout(grid)
        for col, (kind, heading) in enumerate((('now', 'Now'), ('rms', 'RMS')), start=1):
            header = value_label(heading, ERROR_TIPS[kind])
            header.setObjectName('heading')
            grid.addWidget(header, 0, col)
        self.errors, self.error_names = {}, {}
        for row, name in enumerate(('dr', 'ekf', 'adaptive'), start=1):
            self.error_names[name] = QLabel()
            grid.addWidget(self.error_names[name], row, 0)
            for col, kind in enumerate(('now', 'rms'), start=1):
                self.errors[f'{name}_{kind}'] = value_label('–', ERROR_TIPS[kind])
                grid.addWidget(self.errors[f'{name}_{kind}'], row, col)
        self.verdict = QLabel('Launch a run to compare the methods.')
        self.verdict.setObjectName('verdict')
        self.verdict.setWordWrap(True)
        column.addWidget(self.verdict)
        return group

    def build_settings(self):
        container = QWidget()
        column = QVBoxLayout(container)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)

        sensors = QGroupBox('Sensor noise (also during a run)')
        form = QFormLayout(sensors)
        self.fix_noise = SliderField(robot.MAX_NOISE['fix_noise'], .01, .1, 3)
        add_row(form, 'Fix σ (m)', self.fix_noise, TIPS['fix_noise'])
        self.gyro_noise = SliderField(robot.MAX_NOISE['gyro_noise'], .005, .01, 3)
        add_row(form, 'Gyro σ (rad/s)', self.gyro_noise, TIPS['gyro_noise'])
        self.fix_noise.valueChanged.connect(lambda value: self.send_noise('fix_noise', value))
        self.gyro_noise.valueChanged.connect(lambda value: self.send_noise('gyro_noise', value))
        column.addWidget(sensors)

        self.setup = QGroupBox('Simulation (set at launch)')
        form = QFormLayout(self.setup)
        self.fix_rate = QComboBox()
        for rate in robot.FIX_RATES:
            self.fix_rate.addItem(f'{rate:g} Hz', rate)
        add_row(form, 'Fix rate', self.fix_rate, TIPS['fix_rate'])
        self.outlier_rate = SliderField(50, 1, 0, 0)
        add_row(form, 'Bad fixes (%)', self.outlier_rate, TIPS['outlier_rate'])
        self.outlier_noise = SliderField(3, .1, 1, 1)
        add_row(form, 'Bad fix σ (m)', self.outlier_noise, TIPS['outlier_noise'])
        self.slip = SliderField(.3, .01, 0, 2)
        add_row(form, 'Wheel slip σ', self.slip, TIPS['slip'])
        self.step = QCheckBox('Change the noise at a set time')
        self.step.setToolTip(TIPS['step'])
        form.addRow(self.step)
        self.step_fields = {'time': SliderField(300, 5, 30, 0),
                            'fix_noise': SliderField(robot.MAX_NOISE['fix_noise'], .01, .1, 3),
                            'gyro_noise': SliderField(robot.MAX_NOISE['gyro_noise'], .005, .01, 3)}
        for field, label in (('time', 'At time (s)'), ('fix_noise', 'Fix σ becomes'),
                             ('gyro_noise', 'Gyro σ becomes')):
            add_row(form, label, self.step_fields[field], TIPS[f'step_{field}'])
        self.step.toggled.connect(self.enable_step_fields)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(42)
        add_row(form, 'Random seed', self.seed, TIPS['seed'])
        column.addWidget(self.setup)

        filters = QGroupBox('Filter assumptions')
        form = QFormLayout(filters)
        self.match_noise = QCheckBox('Assume the sensor noise above')
        self.match_noise.setToolTip(TIPS['match_noise'])
        form.addRow(self.match_noise)
        self.tuning = {}
        defaults = EKFSettings()
        for field, label, maximum, step, decimals, scale in TUNING:
            box = SliderField(maximum, step, getattr(defaults, field)*scale, decimals)
            box.valueChanged.connect(self.apply_tuning)
            self.tuning[field] = box
            add_row(form, label, box, TIPS[f'tune_{field}'])
        self.gate = QCheckBox('Reject bad fixes (χ² test)')
        self.gate.setToolTip(TIPS['gate'])
        self.gate.toggled.connect(self.apply_tuning)
        form.addRow(self.gate)
        column.addWidget(filters)

        adaptive = QGroupBox('Adaptive EKF')
        form = QFormLayout(adaptive)
        self.variant = QComboBox()
        self.variant.addItem('R and gyro σ', 'aekf')
        self.variant.addItem('R only', 'aekf_r')
        add_row(form, 'Adapts', self.variant, TIPS['variant'])
        self.memory = SliderField(120, 1, MEMORY, 0)
        self.memory.valueChanged.connect(self.apply_tuning)
        add_row(form, 'Memory (s)', self.memory, TIPS['memory'])
        column.addWidget(adaptive)

        self.match_noise.toggled.connect(self.match_sensor_noise)
        self.fix_noise.valueChanged.connect(self.match_sensor_noise)
        self.gyro_noise.valueChanged.connect(self.match_sensor_noise)
        self.variant.currentIndexChanged.connect(self.show_names)
        self.enable_step_fields()
        self.show_names()
        return container

    def show_settings(self, shown):
        self.settings.setVisible(shown)
        self.settings_toggle.setArrowType(Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow)

    def show_names(self):
        variant = self.variant.currentData()
        for name, key in (('dr', 'dr'), ('ekf', 'ekf'), ('adaptive', variant)):
            self.error_names[name].setText(NAMES[key])
            self.error_names[name].setToolTip(ERROR_TIPS[key])

    def enable_step_fields(self):
        for box in self.step_fields.values():
            box.setEnabled(self.step.isChecked())

    # Scenario and settings

    def route(self):
        return self.routes.checkedButton().property('key')

    def choose_scenario(self, key):
        """Fill every setting from the scenario."""
        scenario = BY_KEY[key]
        if not self.scenario_buttons[key].isChecked():
            self.scenario_buttons[key].setChecked(True)
        self.description.setText(scenario.description)
        self.fix_noise.setValue(scenario.fix_noise)
        self.gyro_noise.setValue(scenario.gyro_noise)
        self.fix_rate.setCurrentIndex(robot.FIX_RATES.index(scenario.fix_rate))
        self.outlier_rate.setValue(scenario.outlier_rate*100)
        self.outlier_noise.setValue(scenario.outlier_noise)
        self.slip.setValue(scenario.slip)
        self.step.setChecked(scenario.step_time is not None)
        if scenario.step_time is not None:
            self.step_fields['time'].setValue(scenario.step_time)
            self.step_fields['fix_noise'].setValue(scenario.step_fix_noise or scenario.fix_noise)
            self.step_fields['gyro_noise'].setValue(scenario.step_gyro_noise or scenario.gyro_noise)
        settings = scenario.filter_settings()
        self.match_noise.setChecked(scenario.assumed_fix_noise is None and scenario.assumed_gyro_noise is None)
        for field, *_, scale in TUNING:
            self.tuning[field].setValue(getattr(settings, field)*scale)
        self.match_sensor_noise()

    def match_sensor_noise(self):
        """With matching on, the filters assume the sensor noise; only before a run, so the
        EKF keeps its assumptions when the noise changes during one."""
        matched = self.match_noise.isChecked()
        for field, sensor in (('fix_noise', self.fix_noise), ('gyro_noise', self.gyro_noise)):
            self.tuning[field].setEnabled(not matched)
            if matched and self.process is None:
                self.tuning[field].setValue(sensor.value())

    def filter_settings(self):
        values = {field: round(self.tuning[field].value()/scale, 6) for field, *_, scale in TUNING}
        return EKFSettings(**values, gate=self.gate.isChecked())

    def apply_tuning(self):
        self.tracker.tune(self.filter_settings(), self.memory.value())

    # Running the simulation

    def launch_or_stop(self):
        if self.process is not None:
            self.status.setText('Stopping…')
            self.process.closeWriteChannel()
            return
        variant = self.variant.currentData()
        self.tracker = Tracker(self.filter_settings(), self.memory.value(), adaptive=(variant,))
        self.plot.configure(('dr', 'ekf', variant))
        self.noise_plot.configure(variant)
        self.verdict.setText('Collecting data…')
        for label in self.errors.values():
            label.setText('–')
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.readyReadStandardOutput.connect(self.read_link)
        self.process.finished.connect(self.simulation_closed)
        self.process.errorOccurred.connect(self.simulation_error)
        self.set_running(True)
        self.status.setText('Starting…')
        number = lambda value: f'{round(value, 3):g}'
        arguments = [*self.viewer_command[1:], '--route', self.route(),
                     '--speed', str(robot.SPEEDS[self.speed.currentIndex()]), '--seed', str(self.seed.value()),
                     '--gyro-noise', number(self.gyro_noise.value()), '--fix-noise', number(self.fix_noise.value()),
                     '--fix-rate', number(self.fix_rate.currentData()),
                     '--outlier-rate', number(self.outlier_rate.value()/100),
                     '--outlier-noise', number(self.outlier_noise.value()), '--slip', number(self.slip.value())]
        self.step_pending = self.step.isChecked()
        if self.step_pending:
            arguments += ['--step-time', number(self.step_fields['time'].value()),
                          '--step-fix-noise', number(self.step_fields['fix_noise'].value()),
                          '--step-gyro-noise', number(self.step_fields['gyro_noise'].value())]
        self.process.start(self.viewer_command[0], arguments)

    def set_running(self, running):
        for widget in (self.setup, self.variant, self.tuning['initial_sigma'], *self.routes.buttons(),
                       *self.scenario_buttons.values()):
            widget.setEnabled(not running)
        self.speed.setEnabled(running)
        self.launch_button.setText('Stop' if running else 'Launch')
        # The stylesheet turns the button red while running.
        self.launch_button.setProperty('running', running)
        self.launch_button.style().unpolish(self.launch_button)
        self.launch_button.style().polish(self.launch_button)

    def send(self, command):
        if self.process is not None and self.process.state() == QProcess.ProcessState.Running:
            self.process.write(f'{command}\n'.encode())

    def send_speed(self):
        self.send(f'speed {robot.SPEEDS[self.speed.currentIndex()]}')

    def send_noise(self, field, value):
        self.send(f'{field} {round(value, 3):g}')

    def read_link(self):
        while self.process is not None and self.process.canReadLine():
            line = bytes(self.process.readLine()).decode().strip()
            if line == 'ready':
                self.status.setText('Running')
            elif line:
                try:
                    sample = robot.decode_sample(line)
                except (ValueError, KeyError, TypeError):
                    continue
                self.show_sample(sample)

    def show_sample(self, sample):
        if self.step_pending and sample['time'] >= self.step_fields['time'].value():
            # The simulation has made the scripted step: show the new noise in the sliders.
            self.step_pending = False
            for field in ('fix_noise', 'gyro_noise'):
                getattr(self, field).setValue(self.step_fields[field].value())
        self.tracker.add(sample)
        self.latest = sample
        self.status.setText(f"Running · t = {sample['time']:.1f} s")
        errors = self.tracker.errors()
        variant = self.tracker.adaptive[0]
        for name, key in (('dr', 'dr'), ('ekf', 'ekf'), ('adaptive', variant)):
            for kind in ('now', 'rms'):
                self.errors[f'{name}_{kind}'].setText(f'{errors[f"{key}_{kind}"]:.3f}')
        if sample['time'] >= VERDICT_AFTER:
            change = errors[f'{variant}_rms'] / max(errors['ekf_rms'], 1e-12) - 1
            word = 'lower' if change < 0 else 'higher'
            self.verdict.setText(f"So far the {NAMES[variant]}'s RMS error is "
                                 f"{abs(change)*100:.0f}% {word} than the EKF's.")
        self.plot.refresh(self.tracker, force=self.tracker.count <= 2)
        self.noise_plot.refresh(self.tracker)

    def simulation_closed(self):
        self.process = None
        self.set_running(False)
        self.status.setText('Simulation closed')
        if self.tracker.count:
            self.plot.refresh(self.tracker, force=True)
            self.noise_plot.refresh(self.tracker, force=True)

    def simulation_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.process = None
            self.set_running(False)
            self.status.setText('Could not start the simulation')

    def stop(self):
        """Close the simulation (when the app quits)."""
        if self.process is not None:
            process = self.process
            process.finished.disconnect(self.simulation_closed)
            process.closeWriteChannel()
            if not process.waitForFinished(3000):
                process.kill()
                process.waitForFinished(1000)
            self.process = None


class MainWindow(QMainWindow):
    def __init__(self, viewer_command=None):
        super().__init__()
        self.setWindowTitle('Adaptive EKF Localization')
        self.resize(1320, 840)
        dark = QApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
        theme = THEMES['dark' if dark else 'light']
        self.setStyleSheet(stylesheet(theme))
        self.status = QLabel('Not running')
        self.statusBar().addWidget(self.status)
        self.pages = QTabWidget()
        self.pages.setObjectName('pages')
        self.pages.setDocumentMode(True)
        self.live = LivePage(theme, self.status, viewer_command)
        self.experiment = ExperimentPage(theme)
        help_label = QLabel(HELP)
        help_label.setObjectName('help')
        help_label.setWordWrap(True)
        help_label.setTextFormat(Qt.TextFormat.RichText)
        help_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        help_label.setMaximumWidth(820)
        help_page = QWidget()
        QHBoxLayout(help_page).addWidget(help_label)
        for page, title in ((self.live, 'Live demo'), (self.experiment, 'Experiment'),
                            (scrolling(help_page), 'How it works')):
            holder = QWidget()
            holder.setObjectName('root')
            layout = QVBoxLayout(holder)
            layout.setContentsMargins(14, 10, 14, 8)
            layout.addWidget(page)
            self.pages.addTab(holder, title)
        self.setCentralWidget(self.pages)

    def closeEvent(self, event):
        self.live.stop()
        self.experiment.stop()
        event.accept()


def main():
    application = QApplication(sys.argv[:1])
    application.setApplicationName('Adaptive EKF Localization')
    application.setStyle('Fusion')  # renders the stylesheet the same on every platform
    window = MainWindow()
    window.show()
    sys.exit(application.exec())


if __name__ == '__main__':
    main()
