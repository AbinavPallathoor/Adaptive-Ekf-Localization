"""Colours, the app stylesheet, a slider-with-number input and small layout helpers."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QScrollArea, QSlider, QWidget,
)

SIDEBAR_WIDTH = 340
ROUTE_NAMES = {'square': 'Square', 'hexagon': 'Hexagon', 'figure8': 'Figure 8'}

# Plot series use validated categorical slots 1-3 plus violet and pink for the Adaptive EKFs;
# the rest are surface and text steps.
THEMES = {
    'light': {'surface': '#fcfcfb', 'panel': '#f3f2ef', 'input': '#ffffff', 'border': '#dcdad5',
              'text': '#0b0b0b', 'muted': '#52514e', 'grid': '#e4e3df', 'fix': '#9a9893',
              'true': '#2a78d6', 'dr': '#eb6834', 'ekf': '#1baf7a', 'aekf': '#8a5cd6', 'aekf_r': '#d4508f',
              'accent': '#2a78d6', 'accent_hover': '#1f66bd', 'danger': '#e34948', 'danger_hover': '#c93b3a'},
    'dark': {'surface': '#1a1a19', 'panel': '#242423', 'input': '#2e2e2c', 'border': '#3a3a37',
             'text': '#ffffff', 'muted': '#c3c2b7', 'grid': '#33332f', 'fix': '#77756e',
             'true': '#3987e5', 'dr': '#d95926', 'ekf': '#199e70', 'aekf': '#9b73e0', 'aekf_r': '#e0699f',
             'accent': '#3987e5', 'accent_hover': '#5a9cec', 'danger': '#e66767', 'danger_hover': '#ec8585'},
}


def stylesheet(t):
    return f'''
QMainWindow, QWidget#root, QScrollArea, QScrollArea > QWidget > QWidget {{
    background: {t['surface']}; }}
QWidget {{ color: {t['text']}; font-size: 13px; }}
QToolTip {{ background: {t['panel']}; color: {t['text']}; border: 1px solid {t['border']};
    border-radius: 6px; padding: 6px 8px; }}
QLabel#note, QLabel#heading {{ color: {t['muted']}; }}
QLabel#note {{ font-size: 12px; }}
QLabel#heading {{ font-size: 12px; font-weight: 600; }}

QGroupBox {{ background: {t['panel']}; border: 1px solid {t['border']}; border-radius: 10px;
    margin-top: 18px; padding: 10px 12px 8px 12px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 4px; top: 2px; color: {t['muted']}; }}
QGroupBox QLabel {{ font-weight: 400; }}

QPushButton {{ background: {t['input']}; border: 1px solid {t['border']}; border-radius: 8px;
    padding: 9px 14px; font-weight: 600; }}
QPushButton:hover {{ border-color: {t['accent']}; }}
QPushButton#primary {{ background: {t['accent']}; border: none; color: white; }}
QPushButton#primary:hover {{ background: {t['accent_hover']}; }}
QPushButton#primary[running="true"] {{ background: {t['danger']}; }}
QPushButton#primary[running="true"]:hover {{ background: {t['danger_hover']}; }}

QComboBox, QAbstractSpinBox {{ background: {t['input']}; border: 1px solid {t['border']};
    border-radius: 7px; padding: 5px 8px; min-height: 18px; }}
QComboBox:hover, QAbstractSpinBox:hover {{ border-color: {t['accent']}; }}
QComboBox:focus, QAbstractSpinBox:focus {{ border-color: {t['accent']}; }}
QComboBox:disabled, QAbstractSpinBox:disabled, QPushButton:disabled {{ color: {t['muted']}; }}
QDoubleSpinBox#number {{ padding: 5px 6px; }}
QDoubleSpinBox#number::up-button, QDoubleSpinBox#number::down-button {{
    subcontrol-origin: border; width: 0; height: 0; border: none; }}
QComboBox QAbstractItemView {{ background: {t['input']}; border: 1px solid {t['border']};
    border-radius: 7px; padding: 4px; selection-background-color: {t['accent']}; }}

QSlider::groove:horizontal {{ height: 4px; border-radius: 2px; background: {t['border']}; }}
QSlider::sub-page:horizontal {{ border-radius: 2px; background: {t['accent']}; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -6px 0; border-radius: 7px;
    background: {t['text']}; border: 2px solid {t['accent']}; }}
QSlider::sub-page:horizontal:disabled {{ background: {t['muted']}; }}
QSlider::handle:horizontal:disabled {{ border-color: {t['muted']}; background: {t['muted']}; }}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid {t['border']}; background: {t['input']}; }}
QCheckBox::indicator:checked {{ background: {t['accent']}; border-color: {t['accent']}; }}

QTabWidget::pane {{ border: none; }}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{ background: transparent; color: {t['muted']}; padding: 7px 14px; margin-right: 4px;
    border-radius: 8px; font-weight: 600; }}
QTabBar::tab:selected {{ background: {t['panel']}; color: {t['text']}; border: 1px solid {t['border']}; }}
QTabBar::tab:hover:!selected {{ color: {t['text']}; }}

QToolButton#explainer {{ border: none; color: {t['accent']}; font-weight: 600; padding: 4px 2px; }}
QToolButton#explainer:hover {{ color: {t['accent_hover']}; }}
QGroupBox#help {{ margin-top: 0; font-weight: 400; }}
QGroupBox#help QLabel {{ color: {t['muted']}; line-height: 140%; }}

QScrollBar:vertical {{ width: 8px; background: transparent; }}
QScrollBar::handle:vertical {{ background: {t['border']}; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QStatusBar {{ background: {t['panel']}; color: {t['muted']}; border-top: 1px solid {t['border']}; }}
QStatusBar QLabel {{ color: {t['muted']}; padding: 2px 6px; }}

QLabel#section {{ font-size: 13px; font-weight: 700; color: {t['text']}; padding-top: 2px; }}
QLabel#description {{ color: {t['muted']}; font-size: 12px; padding: 0 2px 0 26px; }}
QLabel#verdict {{ color: {t['text']}; font-size: 12px; padding-top: 4px; }}
QLabel#headline {{ font-size: 15px; font-weight: 600; padding: 2px 0 4px 0; }}
QLabel#help {{ font-size: 13px; line-height: 150%; padding: 4px 8px 16px 4px; }}
QRadioButton {{ spacing: 8px; padding: 2px 0; }}
QRadioButton::indicator {{ width: 14px; height: 14px; border-radius: 8px;
    border: 1px solid {t['border']}; background: {t['input']}; }}
QRadioButton::indicator:checked {{ border: 4px solid {t['accent']}; background: {t['input']}; }}

QPushButton#segment {{ padding: 6px 10px; font-weight: 600; color: {t['muted']}; }}
QPushButton#segment:checked {{ background: {t['accent']}; border-color: {t['accent']}; color: white; }}
QPushButton#segment:disabled:checked {{ background: {t['muted']}; border-color: {t['muted']}; }}

QTabWidget#pages > QTabBar::tab {{ padding: 8px 18px; font-size: 14px; }}

QTableWidget {{ background: {t['panel']}; border: 1px solid {t['border']}; border-radius: 8px;
    gridline-color: {t['border']}; }}
QHeaderView::section {{ background: {t['panel']}; color: {t['muted']}; border: none;
    border-bottom: 1px solid {t['border']}; padding: 6px 8px; font-weight: 600; font-size: 12px; }}
QTableWidget::item {{ padding: 4px 8px; }}
QSpinBox {{ min-width: 64px; }}
QProgressBar {{ background: {t['input']}; border: 1px solid {t['border']}; border-radius: 6px;
    text-align: center; color: {t['text']}; min-height: 16px; }}
QProgressBar::chunk {{ background: {t['accent']}; border-radius: 5px; }}
'''


class SliderField(QWidget):
    """A slider with an editable number beside it; behaves like a spin box."""

    valueChanged = Signal(float)

    def __init__(self, maximum, step, value, decimals):
        super().__init__()
        self.step = step
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, round(maximum/step))
        self.number = QDoubleSpinBox()
        self.number.setObjectName('number')
        self.number.setRange(0, maximum)
        self.number.setDecimals(decimals)
        self.number.setSingleStep(step)
        self.number.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.number.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.number.setFixedWidth(64)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.number)
        self.slider.valueChanged.connect(lambda steps: self.number.setValue(steps*self.step))
        self.number.valueChanged.connect(self.number_changed)
        self.setValue(value)

    def number_changed(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(round(value/self.step))
        self.slider.blockSignals(False)
        self.valueChanged.emit(value)

    def value(self):
        return self.number.value()

    def setValue(self, value):
        self.number.setValue(value)

    def setToolTip(self, tip):
        for widget in (self, self.slider, self.number):
            QWidget.setToolTip(widget, tip)


def section(text):
    label = QLabel(text)
    label.setObjectName('section')
    return label


def note(text):
    label = QLabel(text)
    label.setObjectName('note')
    label.setWordWrap(True)
    return label


def add_row(form, text, widget, tip):
    label = QLabel(text)
    label.setToolTip(tip)
    widget.setToolTip(tip)
    form.addRow(label, widget)


def value_label(text='–', tip=''):
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    label.setToolTip(tip)
    return label


def scrolling(widget):
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(widget)
    return scroll
