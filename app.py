"""Qt launcher for the MuJoCo simulation window. Launch with: .venv/bin/python app.py."""

import math
from pathlib import Path
import sys

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QApplication, QFormLayout, QGridLayout, QGroupBox, QLabel, QMainWindow,
    QPushButton, QVBoxLayout, QWidget,
)

from viewer import decode_state

ROOT = Path(__file__).resolve().parent
DEGREES = 180/math.pi
# Group title, then (field, label, scale, decimals) for each displayed reading.
READINGS = [
    ('Pose', [('x', 'x (m)', 1, 3), ('y', 'y (m)', 1, 3), ('heading', 'Heading (°)', DEGREES, 1)]),
    ('Motion', [('speed', 'Forward speed (m/s)', 1, 3), ('turn_rate', 'Turn rate (°/s)', DEGREES, 1)]),
    ('Wheel encoders', [('left_angle', 'Left angle (rad)', 1, 2), ('right_angle', 'Right angle (rad)', 1, 2),
                        ('left_speed', 'Left speed (rad/s)', 1, 3), ('right_speed', 'Right speed (rad/s)', 1, 3)]),
    ('IMU', [('gyro_z', 'Gyro z (rad/s)', 1, 3), ('accel_x', 'Accel x (m/s²)', 1, 3),
             ('accel_y', 'Accel y (m/s²)', 1, 3)]),
]


def default_viewer_command():
    # MuJoCo's viewer on macOS must run under mjpython, installed beside python.
    # Run it through this interpreter so a moved venv's stale shebang is ignored.
    mjpython = Path(sys.executable).with_name('mjpython')
    launcher = [str(mjpython)] if sys.platform == 'darwin' and mjpython.exists() else []
    return [sys.executable, *launcher, str(ROOT / 'viewer.py')]


class MainWindow(QMainWindow):
    def __init__(self, viewer_command=None):
        super().__init__()
        self.viewer_command = viewer_command or default_viewer_command()
        self.process = None
        self.setWindowTitle('Adaptive EKF Localization')
        root = QWidget()
        layout = QVBoxLayout(root)
        self.launch_button = QPushButton('Launch simulation')
        self.launch_button.clicked.connect(self.launch)
        layout.addWidget(self.launch_button)
        self.status = QLabel('Not running')
        layout.addWidget(self.status)
        grid = QGridLayout()
        self.readings = {}
        for i, (title, fields) in enumerate(READINGS):
            group = QGroupBox(title)
            form = QFormLayout(group)
            for field, label, _, _ in fields:
                self.readings[field] = QLabel('–')
                form.addRow(label, self.readings[field])
            grid.addWidget(group, i//2, i%2)
        layout.addLayout(grid)
        self.setCentralWidget(root)

    def launch(self):
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.readyReadStandardOutput.connect(self.read_link)
        self.process.finished.connect(self.simulation_closed)
        self.process.errorOccurred.connect(self.simulation_error)
        self.launch_button.setEnabled(False)
        self.status.setText('Starting…')
        self.process.start(self.viewer_command[0], self.viewer_command[1:])

    def read_link(self):
        while self.process is not None and self.process.canReadLine():
            line = bytes(self.process.readLine()).decode().strip()
            if line == 'ready':
                self.status.setText('Linked')
            elif line:
                try:
                    state = decode_state(line)
                except (ValueError, KeyError, TypeError):
                    continue
                self.status.setText(f"Linked · t = {state['time']:.1f} s")
                for _, fields in READINGS:
                    for field, _, scale, decimals in fields:
                        self.readings[field].setText(f'{state[field]*scale:.{decimals}f}')

    def simulation_closed(self):
        self.process = None
        self.launch_button.setEnabled(True)
        self.status.setText('Simulation closed')

    def simulation_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.process = None
            self.launch_button.setEnabled(True)
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
