"""Desktop app: launch a route in MuJoCo and compare dead reckoning with the EKF.

Launch with: .venv/bin/python app.py
"""

import math
from pathlib import Path
import sys
import time

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QPushButton, QToolButton, QVBoxLayout, QWidget,
)

from localization import Tracker
import robot

ROOT = Path(__file__).resolve().parent
SEED = 42
ROUTE_NAMES = {'square': 'Square', 'hexagon': 'Hexagon', 'figure8': 'Figure 8'}
# Sensor readouts: (field, label, scale, decimals); hidden until expanded.
SENSORS = [('time', 'Time (s)', 1, 2), ('x', 'True x (m)', 1, 3), ('y', 'True y (m)', 1, 3),
           ('heading', 'True heading (°)', 180/math.pi, 1),
           ('left_angle', 'Left wheel (rad)', 1, 2), ('right_angle', 'Right wheel (rad)', 1, 2),
           ('gyro_z', 'Gyro z (rad/s)', 1, 3), ('fix_x', 'Last fix x (m)', 1, 3),
           ('fix_y', 'Last fix y (m)', 1, 3)]
# Validated categorical slots 1-3 and text/surface steps, light and dark.
THEMES = {
    'light': {'surface': '#fcfcfb', 'text': '#0b0b0b', 'muted': '#52514e', 'grid': '#e4e3df',
              'fix': '#9a9893', 'true': '#2a78d6', 'dr': '#eb6834', 'ekf': '#1baf7a'},
    'dark': {'surface': '#1a1a19', 'text': '#ffffff', 'muted': '#c3c2b7', 'grid': '#33332f',
             'fix': '#77756e', 'true': '#3987e5', 'dr': '#d95926', 'ekf': '#199e70'},
}


def default_viewer_command():
    # MuJoCo's viewer on macOS must run under mjpython, installed beside python.
    # Run it through this interpreter so a moved venv's stale shebang is ignored.
    mjpython = Path(sys.executable).with_name('mjpython')
    launcher = [str(mjpython)] if sys.platform == 'darwin' and mjpython.exists() else []
    return [sys.executable, *launcher, str(ROOT / 'viewer.py')]


class PathPlot(FigureCanvasQTAgg):
    """Top-down true path, both estimates and the position fixes, in metres."""

    def __init__(self, theme):
        figure = Figure(facecolor=theme['surface'])
        # Fixed margins: automatic layout would shift the axes as tick labels change.
        figure.subplots_adjust(left=.15, right=.97, bottom=.1, top=.86)
        super().__init__(figure)
        self.setMinimumSize(420, 380)
        self.axes = figure.add_subplot(facecolor=theme['surface'])
        self.axes.set_aspect('equal', adjustable='box')
        self.axes.set_xlabel('x (m)', color=theme['muted'])
        self.axes.set_ylabel('y (m)', color=theme['muted'])
        self.axes.tick_params(colors=theme['muted'], length=0)
        self.axes.grid(color=theme['grid'], linewidth=.8)
        for spine in self.axes.spines.values():
            spine.set_visible(False)
        self.fix_dots, = self.axes.plot([], [], linestyle='none', marker='o', markersize=3,
                                        color=theme['fix'], label='Position fixes')
        # The EKF usually sits on the true path: draw it wider underneath so both show.
        self.ekf_line, = self.axes.plot([], [], color=theme['ekf'], linewidth=4, label='EKF')
        self.dr_line, = self.axes.plot([], [], color=theme['dr'], linewidth=2,
                                       linestyle=(0, (5, 3)), label='Dead reckoning')
        self.true_line, = self.axes.plot([], [], color=theme['true'], linewidth=1.5, label='True path')
        marker = dict(marker='o', markersize=9, markeredgewidth=2, markeredgecolor=theme['surface'])
        self.markers = [self.axes.plot([], [], color=theme[name], **marker)[0] for name in ('true', 'dr', 'ekf')]
        # Legend in two rows above the plot, so it never covers the path.
        handles = [self.true_line, self.dr_line, self.ekf_line, self.fix_dots]
        self.axes.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, 1.01), ncol=2,
                         frameon=False, labelcolor=theme['text'], fontsize=9,
                         handlelength=2.2, columnspacing=1.2, borderaxespad=0)
        self.last_draw = 0.0
        self.clear()

    def clear(self):
        for line in (self.fix_dots, self.true_line, self.dr_line, self.ekf_line, *self.markers):
            line.set_data([], [])
        self.axes.set_xlim(-.7, .7)
        self.axes.set_ylim(-.7, .7)
        self.draw_idle()

    def refresh(self, tracker, force=False):
        """Redraw from the tracker's paths, at most 10 times a second."""
        if not force and time.monotonic() - self.last_draw < .1:
            return
        self.last_draw = time.monotonic()
        self.fix_dots.set_data(*tracker.fixes)
        paths = (tracker.true_path, tracker.dr_path, tracker.ekf_path)
        for line, marker, (xs, ys) in zip((self.true_line, self.dr_line, self.ekf_line), self.markers, paths):
            line.set_data(xs, ys)
            marker.set_data(xs[-1:], ys[-1:])
        xs = [x for path in paths for x in path[0]]
        ys = [y for path in paths for y in path[1]]
        # At least a 1.4 m view so the start doesn't look zoomed in; position fixes may fall outside.
        span = max(1.4, max(xs) - min(xs) + .4, max(ys) - min(ys) + .4)
        cx, cy = (max(xs) + min(xs))/2, (max(ys) + min(ys))/2
        self.axes.set_xlim(cx - span/2, cx + span/2)
        self.axes.set_ylim(cy - span/2, cy + span/2)
        self.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self, viewer_command=None):
        super().__init__()
        self.viewer_command = viewer_command or default_viewer_command()
        self.process = None
        self.tracker = Tracker(fix_noise=.1, gyro_noise=.01)
        self.last_fix = {'fix_x': None, 'fix_y': None}
        self.latest = None
        self.setWindowTitle('Adaptive EKF Localization')
        self.resize(1280, 760)
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.addWidget(self.build_controls())
        dark = QApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
        self.plot = PathPlot(THEMES['dark' if dark else 'light'])
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.build_results())
        self.setCentralWidget(root)
        self.status = QLabel('Not running')
        self.statusBar().addWidget(self.status)

    def build_controls(self):
        panel = QWidget()
        panel.setFixedWidth(250)
        column = QVBoxLayout(panel)
        column.setContentsMargins(0, 0, 0, 0)
        self.setup = QGroupBox('Setup')
        form = QFormLayout(self.setup)
        self.route = QComboBox()
        for key, name in ROUTE_NAMES.items():
            self.route.addItem(name, key)
        form.addRow('Route', self.route)
        self.fix_noise = self.spin_box(0, 1, .01, .10)
        form.addRow('Position fix σ (m)', self.fix_noise)
        self.fix_rate = QComboBox()
        for rate in robot.FIX_RATES:
            self.fix_rate.addItem(f'{rate:g} Hz', rate)
        self.fix_rate.setCurrentText('1 Hz')
        form.addRow('Fix rate', self.fix_rate)
        self.gyro_noise = self.spin_box(0, .5, .005, .01)
        form.addRow('Gyro σ (rad/s)', self.gyro_noise)
        column.addWidget(self.setup)
        self.launch_button = QPushButton('Launch')
        self.launch_button.setDefault(True)
        self.launch_button.clicked.connect(self.launch_or_stop)
        column.addWidget(self.launch_button)
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel('Speed'))
        self.speed = QComboBox()
        self.speed.addItems([f'{speed}×' for speed in robot.SPEEDS])
        self.speed.currentIndexChanged.connect(self.send_speed)
        self.speed.setEnabled(False)
        speed_row.addWidget(self.speed, 1)
        column.addLayout(speed_row)
        column.addStretch()
        return panel

    @staticmethod
    def spin_box(minimum, maximum, step, value):
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(3)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    def build_results(self):
        # Fixed-width column so changing values never resize the plot.
        panel = QWidget()
        panel.setFixedWidth(350)
        column = QVBoxLayout(panel)
        column.setContentsMargins(0, 0, 0, 0)
        group = QGroupBox('Error vs. true position')
        grid = QGridLayout(group)
        for col, heading in enumerate(('Now (m)', 'RMS (m)', 'Heading (°)'), start=1):
            grid.addWidget(self.value_label(heading), 0, col)
        self.errors = {}
        for row, (name, label) in enumerate((('dr', 'Dead reckoning'), ('ekf', 'EKF')), start=1):
            grid.addWidget(QLabel(label), row, 0)
            for col, kind in enumerate(('now', 'rms', 'heading'), start=1):
                self.errors[f'{name}_{kind}'] = self.value_label('–')
                grid.addWidget(self.errors[f'{name}_{kind}'], row, col)
        column.addWidget(group)
        self.sensor_toggle = QToolButton()
        self.sensor_toggle.setText('Sensor readings')
        self.sensor_toggle.setCheckable(True)
        self.sensor_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.sensor_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.sensor_toggle.setAutoRaise(True)
        self.sensor_toggle.toggled.connect(self.show_sensors)
        column.addWidget(self.sensor_toggle)
        self.sensor_panel = QGroupBox()
        form = QFormLayout(self.sensor_panel)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.sensors = {}
        for field, label, _, _ in SENSORS:
            self.sensors[field] = self.value_label('–')
            form.addRow(label, self.sensors[field])
        self.sensor_panel.setVisible(False)
        column.addWidget(self.sensor_panel)
        column.addStretch()
        return panel

    @staticmethod
    def value_label(text):
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return label

    def show_sensors(self, shown):
        self.sensor_panel.setVisible(shown)
        self.sensor_toggle.setArrowType(Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow)
        self.update_sensors()

    def update_sensors(self):
        if self.latest is None or not self.sensor_toggle.isChecked():
            return
        for field, _, scale, decimals in SENSORS:
            if self.latest[field] is not None:
                self.sensors[field].setText(f'{self.latest[field]*scale:.{decimals}f}')

    def launch_or_stop(self):
        if self.process is not None:
            self.status.setText('Stopping…')
            self.process.closeWriteChannel()
            return
        fix_noise, gyro_noise = round(self.fix_noise.value(), 3), round(self.gyro_noise.value(), 3)
        self.tracker = Tracker(fix_noise=fix_noise, gyro_noise=gyro_noise)
        self.plot.clear()
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.readyReadStandardOutput.connect(self.read_link)
        self.process.finished.connect(self.simulation_closed)
        self.process.errorOccurred.connect(self.simulation_error)
        self.set_running(True)
        self.status.setText('Starting…')
        arguments = [*self.viewer_command[1:], '--route', self.route.currentData(),
                     '--speed', str(robot.SPEEDS[self.speed.currentIndex()]), '--seed', str(SEED),
                     '--gyro-noise', f'{gyro_noise:g}', '--fix-noise', f'{fix_noise:g}',
                     '--fix-rate', f'{self.fix_rate.currentData():g}']
        self.process.start(self.viewer_command[0], arguments)

    def set_running(self, running):
        self.setup.setEnabled(not running)
        self.speed.setEnabled(running)
        self.launch_button.setText('Stop' if running else 'Launch')

    def send_speed(self):
        if self.process is not None and self.process.state() == QProcess.ProcessState.Running:
            self.process.write(f'speed {robot.SPEEDS[self.speed.currentIndex()]}\n'.encode())

    def read_link(self):
        while self.process is not None and self.process.canReadLine():
            line = bytes(self.process.readLine()).decode().strip()
            if line == 'ready':
                self.status.setText('Linked')
            elif line:
                try:
                    sample = robot.decode_sample(line)
                except (ValueError, KeyError, TypeError):
                    continue
                self.show_sample(sample)

    def show_sample(self, sample):
        self.tracker.add(sample)
        self.status.setText(f"Linked · t = {sample['time']:.1f} s")
        for key, value in self.tracker.errors().items():
            self.errors[key].setText(f'{math.degrees(value):.1f}' if key.endswith('heading') else f'{value:.3f}')
        if sample['fix_x'] is not None:
            self.last_fix = {'fix_x': sample['fix_x'], 'fix_y': sample['fix_y']}
        self.latest = {**sample, **self.last_fix}
        self.update_sensors()
        self.plot.refresh(self.tracker, force=self.tracker.count <= 2)

    def simulation_closed(self):
        self.process = None
        self.set_running(False)
        self.status.setText('Simulation closed')
        if self.tracker.count:
            self.plot.refresh(self.tracker, force=True)

    def simulation_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.process = None
            self.set_running(False)
            self.status.setText('Could not start simulation')

    def closeEvent(self, event):
        if self.process is not None:
            process = self.process
            process.finished.disconnect(self.simulation_closed)
            process.closeWriteChannel()
            if not process.waitForFinished(3000):
                process.kill()
                process.waitForFinished(1000)
            self.process = None
        event.accept()


def main():
    application = QApplication(sys.argv[:1])
    application.setApplicationName('Adaptive EKF Localization')
    window = MainWindow()
    window.show()
    sys.exit(application.exec())


if __name__ == '__main__':
    main()
