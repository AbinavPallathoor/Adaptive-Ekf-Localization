"""Experiment page: repeated paired runs, the statistical comparison and figures for the report."""

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import math
import os
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QProgressBar,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

import analysis
import experiment
from help_text import RESULT_TIPS, TIPS
from localization import ADAPTIVE, NAMES
from plots import FigureCanvas
from scenarios import SCENARIOS
from ui import ROUTE_NAMES, SIDEBAR_WIDTH, note, scrolling, section

ROOT = Path(__file__).resolve().parent
SECONDS_PER_RUN = .3  # rough wall time of one run on one core, for the estimate


class ExperimentPage(QWidget):
    def __init__(self, theme):
        super().__init__()
        self.theme = theme
        self.pool = None
        self.futures = []
        self.workers = None  # processes; None: one per core but one
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)
        self.results = self.table = self.noise = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        layout.addWidget(self.build_sidebar())
        layout.addLayout(self.build_results(), 1)
        self.update_estimate()

    def build_sidebar(self):
        page = QWidget()
        column = QVBoxLayout(page)
        column.setContentsMargins(2, 4, 8, 4)
        column.setSpacing(8)
        column.addWidget(section('1  Scenarios to test'))
        self.scenario_boxes = {}
        for scenario in SCENARIOS:
            box = QCheckBox(scenario.name)
            box.setChecked(True)
            box.setToolTip(f'{scenario.description}\n\n{TIPS["scenarios"]}')
            box.toggled.connect(self.update_estimate)
            self.scenario_boxes[scenario.key] = box
            column.addWidget(box)
        column.addSpacing(6)
        column.addWidget(section('2  Routes'))
        routes = QHBoxLayout()
        self.route_boxes = {}
        for key, name in ROUTE_NAMES.items():
            box = QCheckBox(name)
            box.setChecked(True)
            box.setToolTip(TIPS['routes'])
            box.toggled.connect(self.update_estimate)
            self.route_boxes[key] = box
            routes.addWidget(box)
        column.addLayout(routes)
        column.addSpacing(6)
        column.addWidget(section('3  Runs'))
        runs = QHBoxLayout()
        label = QLabel('Runs per scenario and route')
        label.setToolTip(TIPS['runs'])
        runs.addWidget(label, 1)
        self.runs = QSpinBox()
        self.runs.setRange(2, 500)
        self.runs.setValue(20)
        self.runs.setToolTip(TIPS['runs'])
        self.runs.valueChanged.connect(self.update_estimate)
        runs.addWidget(self.runs)
        column.addLayout(runs)
        self.gate = QCheckBox('Filters reject bad fixes (χ² test)')
        self.gate.setToolTip(TIPS['gate'])
        column.addWidget(self.gate)
        self.estimate = note('')
        column.addWidget(self.estimate)
        self.run_button = QPushButton('Run experiment')
        self.run_button.setObjectName('primary')
        self.run_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_button.setToolTip(TIPS['run_experiment'])
        self.run_button.clicked.connect(self.run_or_stop)
        column.addWidget(self.run_button)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setValue(0)
        column.addWidget(self.progress)
        self.export_button = QPushButton('Save results for the report…')
        self.export_button.setToolTip(TIPS['export'])
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.choose_export)
        column.addWidget(self.export_button)
        self.saved = note('')
        column.addWidget(self.saved)
        column.addWidget(note('Each run drives two laps without the 3-D window, with its own random seed. '
                              'All filters get identical readings, so every run is a fair paired comparison.'))
        column.addStretch()
        scroll = scrolling(page)
        scroll.setFixedWidth(SIDEBAR_WIDTH)
        return scroll

    def build_results(self):
        column = QVBoxLayout()
        column.setSpacing(8)
        self.headline = QLabel('Choose what to test on the left and press Run experiment.')
        self.headline.setObjectName('headline')
        self.headline.setWordWrap(True)
        column.addWidget(self.headline)
        compare = QHBoxLayout()
        label = QLabel('Compare the EKF with')
        label.setToolTip(TIPS['compare'])
        compare.addWidget(label)
        self.variants = QButtonGroup(self)
        for key in ADAPTIVE:
            button = QPushButton(NAMES[key])
            button.setObjectName('segment')
            button.setCheckable(True)
            button.setToolTip(TIPS['compare'])
            button.setProperty('key', key)
            self.variants.addButton(button)
            compare.addWidget(button)
        self.variants.buttons()[0].setChecked(True)
        self.variants.buttonClicked.connect(self.show_results)
        compare.addStretch()
        column.addLayout(compare)
        self.result_table = QTableWidget(0, 6)
        headers = [('Scenario', 'scenario'), ('EKF RMSE (m)', 'ekf'), ('Adaptive RMSE (m)', 'adaptive'),
                   ('Difference (mm, 95% CI)', 'difference'), ('p (Holm)', 'p'), ('Result', 'verdict')]
        for col, (text, tip) in enumerate(headers):
            item = QTableWidgetItem(text)
            item.setToolTip(RESULT_TIPS[tip])
            self.result_table.setHorizontalHeaderItem(col, item)
        self.result_table.setToolTip(RESULT_TIPS['difference'])
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.result_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.result_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.result_table.setMinimumHeight(120)
        self.result_table.setMaximumHeight(240)
        column.addWidget(self.result_table)
        self.figures = QTabWidget()
        self.figures.setDocumentMode(True)
        self.canvases = {}
        for key, title in (('difference', 'Differences'), ('errors', 'Error of each run'),
                           ('calibration', 'Uncertainty check'), ('noise', 'Noise check')):
            self.canvases[key] = FigureCanvas(self.theme)
            self.figures.addTab(self.canvases[key], title)
        column.addWidget(self.figures, 1)
        return column

    def chosen(self):
        scenarios = [key for key, box in self.scenario_boxes.items() if box.isChecked()]
        routes = [key for key, box in self.route_boxes.items() if box.isChecked()]
        return scenarios, routes

    def update_estimate(self):
        scenarios, routes = self.chosen()
        total = len(scenarios) * len(routes) * self.runs.value()
        cores = max(1, (os.cpu_count() or 2) - 1)
        seconds = math.ceil(total * SECONDS_PER_RUN / cores) + 3
        self.estimate.setText(f'{total} runs in total, about {seconds} s.' if total else
                              'Choose at least one scenario and one route.')
        if self.pool is None:
            self.run_button.setEnabled(total > 0)

    def variant(self):
        return self.variants.checkedButton().property('key')

    # Running

    def run_or_stop(self):
        if self.pool is not None:
            self.close_pool()
            self.headline.setText('Stopped. Press Run experiment to start again.')
            return
        scenarios, routes = self.chosen()
        job_list = experiment.jobs(scenarios, routes, self.runs.value(), self.gate.isChecked())
        self.progress.setRange(0, len(job_list))
        self.progress.setValue(0)
        self.headline.setText(f'Running {len(job_list)} runs…')
        self.set_running(True)
        # Runs go to other processes; the timer collects them, so the window stays responsive.
        workers = self.workers or max(1, min(len(job_list), (os.cpu_count() or 2) - 1))
        self.pool = ProcessPoolExecutor(workers)
        self.futures = [self.pool.submit(experiment.run, *job) for job in job_list]
        self.noise_future = self.pool.submit(experiment.noise_sample)
        self.timer.start()

    def poll(self):
        done = sum(future.done() for future in self.futures)
        self.progress.setValue(done)
        if done < len(self.futures) or not self.noise_future.done():
            return
        try:
            results = pd.DataFrame([future.result() for future in self.futures])
            noise = self.noise_future.result()
        except Exception as error:  # a run failed: show why instead of losing the window
            self.close_pool()
            self.headline.setText(f'The experiment failed: {error}')
            return
        self.close_pool()
        self.finished(results, noise)

    def close_pool(self):
        self.timer.stop()
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.pool, self.futures = None, []
        self.set_running(False)
        self.update_estimate()

    def set_running(self, running):
        self.run_button.setText('Stop' if running else 'Run experiment')
        self.run_button.setProperty('running', running)
        self.run_button.style().unpolish(self.run_button)
        self.run_button.style().polish(self.run_button)
        for box in (*self.scenario_boxes.values(), *self.route_boxes.values(), self.runs, self.gate):
            box.setEnabled(not running)

    def finished(self, results, noise):
        self.results, self.noise = results, noise
        self.table = analysis.summary(results)
        self.export_button.setEnabled(True)
        self.saved.setText('')
        self.show_results()

    def show_results(self):
        if self.table is None:
            return
        variant = self.variant()
        self.headline.setText(analysis.headline(self.table, variant))
        rows = self.table[self.table['variant'] == variant]
        self.result_table.setRowCount(len(rows))
        for row, (_, values) in enumerate(rows.iterrows()):
            ci = '' if math.isnan(values['ci_low']) else \
                f"  ({values['ci_low']*1000:+.2f} to {values['ci_high']*1000:+.2f})"
            p = '–' if math.isnan(values['p_holm']) else \
                '< 0.001' if values['p_holm'] < .001 else f"{values['p_holm']:.3f}"
            cells = [values['scenario_name'], f"{values['ekf_mean']:.4f}", f"{values['adaptive_mean']:.4f}",
                     f"{values['mean_d']*1000:+.2f}{ci}", p, values['verdict']]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                                          if col < 5 else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                if col == 5:
                    if text.startswith(('Better', 'Worse')):
                        item.setForeground(QColor(self.theme[variant if text.startswith('Better') else 'danger']))
                    item.setToolTip(RESULT_TIPS['verdict'])
                self.result_table.setItem(row, col, item)
        theme = self.theme
        self.canvases['difference'].show_figure(lambda f: analysis.difference_figure(f, self.table, variant, theme))
        self.canvases['errors'].show_figure(lambda f: analysis.errors_figure(f, self.results, variant, theme))
        self.canvases['calibration'].show_figure(lambda f: analysis.calibration_figure(f, self.table, theme))
        if self.noise is not None:
            self.canvases['noise'].show_figure(lambda f: analysis.noise_figure(f, *self.noise, theme))

    # Saving

    def choose_export(self):
        default = ROOT / 'output' / f"experiment-{datetime.now():%Y-%m-%d-%H%M}"
        folder = QFileDialog.getExistingDirectory(self, 'Save results into a folder', str(default.parent))
        if folder:
            self.export(Path(folder) / default.name)

    def export(self, folder):
        paths = analysis.export(self.results, folder, *self.noise)
        self.saved.setText(f'Saved {len(paths)} files to {folder}')
        return paths

    def stop(self):
        """Cancel a running experiment (when the app quits)."""
        if self.pool is not None:
            self.timer.stop()
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
