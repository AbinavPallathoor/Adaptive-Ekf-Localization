"""Plot widgets: the live path and noise plots, and a canvas for the experiment figures."""

import math
import time

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

from localization import COVERAGE_LIMIT, NAMES

ELLIPSE = np.array([(math.cos(a), math.sin(a)) for a in np.linspace(0, math.tau, 60)]).T


def style_axes(axes, theme):
    axes.set_facecolor(theme['surface'])
    axes.tick_params(colors=theme['muted'], length=0)
    axes.grid(color=theme['grid'], linewidth=.8)
    for spine in axes.spines.values():
        spine.set_visible(False)


def ellipse(center, covariance):
    """Points of the 95% ellipse of a 2-D Gaussian."""
    values, vectors = np.linalg.eigh(covariance)
    radii = np.sqrt(np.maximum(values, 0) * COVERAGE_LIMIT)
    return (vectors @ (radii[:, None] * ELLIPSE)) + np.asarray(center)[:, None]


class PathPlot(FigureCanvasQTAgg):
    """Top-down true path, the estimates, the filters' 95% ellipses and the position fixes, in metres."""

    def __init__(self, theme):
        figure = Figure(facecolor=theme['surface'])
        super().__init__(figure)
        self.theme = theme
        self.setMinimumSize(420, 340)
        self.axes = figure.add_subplot()
        self.axes.set_aspect('equal', adjustable='box')
        self.axes.set_xlabel('x (m)', color=theme['muted'])
        self.axes.set_ylabel('y (m)', color=theme['muted'])
        style_axes(self.axes, theme)
        self.last_draw = 0.0
        self.configure(('dr', 'ekf', 'aekf'))

    def configure(self, names):
        """Set which estimators are drawn (dead reckoning, EKF, one adaptive variant)."""
        theme = self.theme
        for artist in (*self.axes.lines, *self.axes.texts):
            artist.remove()
        self.names = names
        self.fix_dots, = self.axes.plot([], [], linestyle='none', marker='o', markersize=3,
                                        color=theme['fix'], label='Position fixes')
        self.outlier_dots, = self.axes.plot([], [], linestyle='none', marker='x', markersize=5,
                                            color=theme['danger'], label='Bad fixes')
        styles = {'dr': dict(linewidth=2, linestyle=(0, (5, 3))), 'ekf': dict(linewidth=5)}
        # The filters usually sit on the true path: draw them wider underneath so all show.
        self.lines = {}
        for name in sorted(names, key=lambda n: ('ekf', 'aekf', 'aekf_r', 'dr').index(n)):
            self.lines[name], = self.axes.plot([], [], color=theme[name], label=NAMES[name],
                                               **styles.get(name, dict(linewidth=2.5)))
        self.lines['true'], = self.axes.plot([], [], color=theme['true'], linewidth=1.5, label='True path')
        self.ellipses = {name: self.axes.plot([], [], color=theme[name], linewidth=1.2, alpha=.9)[0]
                         for name in names if name != 'dr'}
        marker = dict(marker='o', markersize=9, markeredgewidth=2, markeredgecolor=theme['surface'])
        self.markers = {name: self.axes.plot([], [], color=theme[name], **marker)[0] for name in ('true', *names)}
        # Legend in one row above the plot, so it never covers the path.
        handles = [self.lines[name] for name in ('true', *names)] + [self.fix_dots, self.outlier_dots]
        self.axes.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, 1.01), ncol=len(handles),
                         frameon=False, labelcolor=theme['text'], fontsize=9,
                         handlelength=2, columnspacing=1, borderaxespad=0)
        self.clear()

    def clear(self):
        for line in (self.fix_dots, self.outlier_dots, *self.lines.values(), *self.markers.values(),
                     *self.ellipses.values()):
            line.set_data([], [])
        self.view = (0.0, 0.0, 1.4)
        self.fit_view()
        self.draw_idle()

    def refresh(self, tracker, force=False):
        """Redraw from the tracker, at most 10 times a second."""
        if not force and time.monotonic() - self.last_draw < .1:
            return
        self.last_draw = time.monotonic()
        self.fix_dots.set_data(*tracker.fixes)
        self.outlier_dots.set_data(*tracker.outliers)
        for name, line in self.lines.items():
            xs, ys = tracker.paths[name]
            line.set_data(xs, ys)
            self.markers[name].set_data(xs[-1:], ys[-1:])
        for name, line in self.ellipses.items():
            estimate = tracker.estimators[name]
            line.set_data(*ellipse((estimate.x, estimate.y), estimate.P[:2, :2]))
        xs = [x for name in self.lines for x in tracker.paths[name][0]]
        ys = [y for name in self.lines for y in tracker.paths[name][1]]
        # At least a 1.4 m view so the start doesn't look zoomed in; position fixes may fall outside.
        span = max(1.4, max(xs) - min(xs) + .4, max(ys) - min(ys) + .4)
        self.view = ((max(xs) + min(xs))/2, (max(ys) + min(ys))/2, span)
        self.fit_view()
        self.draw_idle()

    def fit_view(self):
        """Fill the canvas: widen the shorter range to match its shape, keeping equal scale."""
        width, height = self.figure.bbox.width, self.figure.bbox.height
        # Margins for tick labels, axis labels and the legend row, in screen points
        # (figure pixels are larger on high-DPI screens).
        scale = self.figure.dpi / 100
        left, right, bottom, top = (margin*scale for margin in (64, 16, 48, 34))
        self.figure.subplots_adjust(left=left/width, right=1 - right/width,
                                    bottom=bottom/height, top=1 - top/height)
        ratio = max(width - left - right, 1) / max(height - bottom - top, 1)
        cx, cy, span = self.view
        x_span, y_span = (span*ratio, span) if ratio >= 1 else (span, span/ratio)
        self.axes.set_xlim(cx - x_span/2, cx + x_span/2)
        self.axes.set_ylim(cy - y_span/2, cy + y_span/2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_view()
        self.draw_idle()


class NoisePlot(FigureCanvasQTAgg):
    """Fix σ and gyro σ over time: true, assumed by the EKF and estimated by the Adaptive EKF."""

    def __init__(self, theme):
        figure = Figure(facecolor=theme['surface'])
        super().__init__(figure)
        self.theme = theme
        self.setFixedHeight(190)
        self.axes = figure.subplots(1, 2)
        for axes, title in zip(self.axes, ('Position fix σ (m)', 'Gyro σ (rad/s)')):
            axes.set_title(title, color=theme['text'], fontsize=9, loc='left')
            axes.set_xlabel('time (s)', color=theme['muted'], fontsize=8)
            axes.tick_params(labelsize=8)
            style_axes(axes, theme)
        figure.subplots_adjust(left=.08, right=.98, bottom=.24, top=.86, wspace=.18)
        self.last_draw = 0.0
        self.configure('aekf')

    def configure(self, variant):
        """Show the estimates of this adaptive variant."""
        for axes in self.axes:
            for line in list(axes.lines):
                line.remove()
            legend = axes.get_legend()
            if legend:
                legend.remove()
        self.variant, self.lines = variant, {}
        for axes, kind in zip(self.axes, ('fix', 'gyro')):
            for key, color, style, width, label in (
                    (f'{kind}_sigma', 'true', '-', 1.5, 'True'),
                    (f'ekf_{kind}_sigma', 'ekf', (0, (4, 2)), 2, 'EKF assumes'),
                    (f'{variant}_{kind}_sigma', variant, '-', 2, f'{NAMES[variant]} estimate')):
                self.lines[key], = axes.plot([], [], color=self.theme[color], linestyle=style,
                                             linewidth=width, label=label, drawstyle='steps-post')
        self.axes[1].legend(loc='upper left', frameon=False, labelcolor=self.theme['text'], fontsize=8)
        self.clear()

    def clear(self):
        for line in self.lines.values():
            line.set_data([], [])
        for axes in self.axes:
            axes.set_xlim(0, 10)
            axes.set_ylim(0, .1)
        self.draw_idle()

    def refresh(self, tracker, force=False):
        """Redraw from the tracker's noise history, at most twice a second."""
        if not force and time.monotonic() - self.last_draw < .5:
            return
        self.last_draw = time.monotonic()
        history = tracker.noise_history
        if not history['time']:
            return
        for key, line in self.lines.items():
            line.set_data(history['time'], history[key])
        for axes, kind in zip(self.axes, ('fix', 'gyro')):
            axes.set_xlim(0, max(10, history['time'][-1]))
            top = max(max(history[f'{prefix}{kind}_sigma']) for prefix in ('', 'ekf_', f'{self.variant}_'))
            axes.set_ylim(0, max(1.15*top, .01))
        self.draw_idle()


class FigureCanvas(FigureCanvasQTAgg):
    """A canvas for one experiment figure; draw(figure) fills it."""

    def __init__(self, theme):
        super().__init__(Figure(facecolor=theme['surface']))
        self.setMinimumSize(380, 260)
        self.drawing = None

    def show_figure(self, draw):
        self.drawing = draw
        draw(self.figure)
        self.draw_idle()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.drawing:
            # Lay the figure out again for the new size.
            self.drawing(self.figure)
            self.draw_idle()
