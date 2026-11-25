"""Fixed-size plot and criterion widgets for WP9 Detection Mode (§ 13.5, § 13.6)."""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QLocale, QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QDoubleValidator, QFontMetrics, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from optocurator.models import MISSING_METRIC_SENTINEL
from optocurator.ui.figure_export import install_figure_export_menu
from optocurator.ui.plots import apply_plot_style
from optocurator.ui.theme import (
    COLOR_ACCENT,
    COLOR_ACCENT_BG,
    COLOR_AXIS,
    COLOR_BORDER_STRONG,
    COLOR_DIVIDER,
    COLOR_KDE_POP_FILL,
    COLOR_PANEL,
    COLOR_PASS_FILL,
    COLOR_SUCCESS,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_SECONDARY,
    DETECTION_CRITERION_ICON_FAIL_STYLE,
    DETECTION_CRITERION_ICON_PASS_STYLE,
    DETECTION_CRITERION_INPUT_STYLE,
    DETECTION_CRITERION_META_DIM_STYLE,
    DETECTION_CRITERION_META_LABEL_STYLE,
    DETECTION_CRITERION_PASS_COUNT_DIM_STYLE,
    DETECTION_CRITERION_PASS_COUNT_STYLE,
    DETECTION_CRITERION_PLOT_SIZE,
    DETECTION_CRITERION_ROW_STYLE,
    DETECTION_CRITERION_VALUE_FAIL_STYLE,
    DETECTION_CRITERION_VALUE_NA_STYLE,
    DETECTION_CRITERION_VALUE_PASS_STYLE,
    DETECTION_PAIRPLOT_AXIS_LABEL_STYLE,
    DETECTION_PAIRPLOT_CELL_SIZE,
    DETECTION_PAIRPLOT_LEGEND_HTML,
    DETECTION_PAIRPLOT_LEGEND_STYLE,
    DETECTION_PLOT_LEFT_MARGIN,
    DETECTION_SUBPANEL_TITLE_STYLE,
    rgba_from_hex,
)

_CRITERION_AXIS_HEIGHT: int = 20
_CRITERION_ROW_HEIGHT: int = 66
_PAIRPLOT_LEFT_MARGIN: int = DETECTION_PLOT_LEFT_MARGIN
_PAIRPLOT_BOTTOM_HEIGHT: int = 34


def salt_p_resolution_floor(n_permutations: int) -> float:
    """Return the exact minimum non-zero SALT p-value for `n_permutations`."""
    return 1.0 / (max(1, int(n_permutations)) + 1)


def format_salt_p_value(val: float, n_permutations: int = 100) -> str:
    """Format a SALT p-value relative to the permutation resolution floor."""
    n_perm = max(1, int(n_permutations))
    p_floor = 1.0 / (n_perm + 1)
    v = float(val)
    if not math.isfinite(v):
        return "—"
    if v <= p_floor + 1e-12:
        bound = 1.0 / n_perm
        if bound >= 1e-4:
            bound_str = f"{bound:.6f}".rstrip("0").rstrip(".")
        else:
            bound_str = f"{bound:g}"
        return f"≤ {bound_str}"
    dec = min(6, max(3, int(math.ceil(math.log10(n_perm)))))
    return f"{v:.{dec}f}"


def salt_log_axis_spec(
    n_permutations: int = 100,
) -> Tuple[float, float, List[Tuple[float, str]]]:
    """Compute `(p_min, p_max, log10_ticks)` for a SALT p log-axis given `n_permutations`."""
    n_perm = max(1, int(n_permutations))
    min_exp = min(-3, int(math.floor(math.log10(1.0 / n_perm) + 1e-12)))
    ticks: List[Tuple[float, str]] = []
    for exp in range(min_exp, 1):
        if exp == 0:
            label = "1"
        elif exp == -1:
            label = "0.1"
        elif exp == -2:
            label = "0.01"
        elif exp == -3:
            label = "0.001"
        else:
            label = f"1e{exp}"
        ticks.append((float(exp), label))
    return (10.0 ** min_exp, 1.0, ticks)


def _compute_kde(samples: np.ndarray, grid: np.ndarray, span: float) -> np.ndarray:
    """Compute a normalized Gaussian KDE over `grid` for finite 1D `samples`."""
    if len(samples) == 0 or len(grid) == 0:
        return np.zeros_like(grid, dtype=np.float64)
    std = float(np.std(samples)) if len(samples) > 1 else 0.0
    if len(samples) > 1 and std > 0.0:
        q75, q25 = np.percentile(samples, [75, 25])
        iqr = float(q75 - q25)
        sigma = min(std, iqr / 1.34) if iqr > 0.0 else std
        bw = 0.5 * sigma * (len(samples) ** (-0.2))
    else:
        bw = span * 0.04
    bw = max(bw, span * 0.025, 1e-6)

    diff = (grid[:, None] - samples[None, :]) / bw
    density = np.mean(np.exp(-0.5 * diff * diff), axis=1) / (bw * np.sqrt(2.0 * np.pi))
    max_d = float(np.max(density))
    if max_d > 0.0:
        density = density / max_d
    return density


def _default_ticks_for_range(
    axis_range: Tuple[float, float],
    is_log: bool,
) -> List[Tuple[float, str]]:
    """Build clean default tick positions and labels for a criterion distribution plot."""
    lo, hi = float(axis_range[0]), float(axis_range[1])
    if is_log:
        return [
            (float(np.log10(0.001)), "0.001"),
            (float(np.log10(0.01)), "0.01"),
            (float(np.log10(0.1)), "0.1"),
            (float(np.log10(1.0)), "1"),
        ]
    if lo == 0.0 and hi == 20.0:
        return [(float(v), str(v)) for v in (0, 5, 10, 15, 20)]
    if lo == 0.0 and hi == 6.0:
        return [(float(v), str(v)) for v in (0, 2, 4, 6)]
    if lo == 0.0 and hi == 1.0:
        return [(0.0, "0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"), (1.0, "1")]
    if abs(lo - 0.4) < 1e-6 and abs(hi - 1.0) < 1e-6:
        return [(0.4, "0.4"), (0.6, "0.6"), (0.8, "0.8"), (1.0, "1")]
    mid = 0.5 * (lo + hi)
    return [(lo, f"{lo:g}"), (mid, f"{mid:g}"), (hi, f"{hi:g}")]


def install_combo_chevron(combo: Any) -> None:
    """Install a crisp downward chevron overlay on a Detection QComboBox."""
    from PyQt6.QtCore import QEvent, QObject, QPointF
    from PyQt6.QtGui import QPolygonF

    class _ChevronFilter(QObject):
        def __init__(self, parent: Any = None) -> None:
            super().__init__(parent)
            self._painting = False

        def eventFilter(self, obj: Any, event: Any) -> bool:
            if (
                obj is combo
                and not self._painting
                and event is not None
                and event.type() == QEvent.Type.Paint
            ):
                self._painting = True
                try:
                    obj.event(event)
                finally:
                    self._painting = False
                painter = QPainter(obj)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                cx = float(obj.width()) - 11.0
                cy = float(obj.height()) / 2.0 + 0.5
                tri = QPolygonF(
                    [
                        QPointF(cx - 3.5, cy - 2.0),
                        QPointF(cx + 3.5, cy - 2.0),
                        QPointF(cx, cy + 2.5),
                    ]
                )
                color = QColor(COLOR_TEXT_SECONDARY if obj.isEnabled() else COLOR_TEXT_MUTED)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                painter.drawPolygon(tri)
                painter.end()
                return True
            return False

    filt = _ChevronFilter(combo)
    combo.installEventFilter(filt)
    combo._chevron_filter = filt


class TickCheckBox(QCheckBox):
    """Lightweight QCheckBox painting a crisp 14x14 rounded indicator with a white checkmark."""

    _BOX_SIZE: int = 14
    _TEXT_GAP: int = 6

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def sizeHint(self) -> QSize:
        fm = QFontMetrics(self.font())
        text_w = fm.horizontalAdvance(self.text()) if self.text() else 0
        w = self._BOX_SIZE + (self._TEXT_GAP + text_w if text_w > 0 else 0) + 4
        h = max(self._BOX_SIZE + 4, fm.height() + 2)
        return QSize(w, h)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def hitButton(self, pos: Any) -> bool:
        return self.rect().contains(pos)

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        box = float(self._BOX_SIZE)
        y_top = 0.5 * (self.height() - box)
        box_rect = QRectF(0.5, y_top + 0.5, box - 1.0, box - 1.0)

        enabled = self.isEnabled()
        checked = self.isChecked()

        if checked:
            fill_color = QColor(COLOR_ACCENT if enabled else COLOR_TEXT_MUTED)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill_color)
            painter.drawRoundedRect(box_rect, 3.0, 3.0)

            tick_path = QPainterPath()
            tick_path.moveTo(box_rect.left() + 2.8, box_rect.top() + 6.7)
            tick_path.lineTo(box_rect.left() + 5.4, box_rect.top() + 9.4)
            tick_path.lineTo(box_rect.left() + 10.4, box_rect.top() + 3.8)
            tick_pen = QPen(
                QColor(COLOR_PANEL),
                1.8,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            painter.setPen(tick_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(tick_path)
        else:
            painter.setBrush(QColor(COLOR_PANEL))
            border_color = QColor(COLOR_BORDER_STRONG if enabled else COLOR_DIVIDER)
            painter.setPen(QPen(border_color, 1.2))
            painter.drawRoundedRect(box_rect, 3.0, 3.0)

        if self.text():
            text_color = QColor(COLOR_TEXT if (enabled and checked) else COLOR_TEXT_MUTED)
            painter.setPen(text_color)
            text_rect = QRectF(
                self._BOX_SIZE + self._TEXT_GAP,
                0.0,
                max(0.0, float(self.width() - self._BOX_SIZE - self._TEXT_GAP)),
                float(self.height()),
            )
            painter.drawText(
                text_rect,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                self.text(),
            )


class DetectionCriterionRow(QWidget):
    """Single tagging criterion row with threshold input and 240x24 KDE plot (§ 13.6)."""

    threshold_changed = pyqtSignal(float)
    enabled_changed = pyqtSignal(bool)

    def __init__(
        self,
        label: str,
        operator: str = "≤",
        unit: str = "",
        axis_range: Tuple[float, float] = (0.0, 20.0),
        ticks: Optional[Sequence[Any]] = None,
        default_threshold: float = 8.0,
        min_val: float = 0.0,
        max_val: float = 100.0,
        decimals: int = 2,
        is_log: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("detectionCriterionRow")
        self.setStyleSheet(DETECTION_CRITERION_ROW_STYLE)
        self.setFixedHeight(_CRITERION_ROW_HEIGHT)

        self._operator = operator
        self._unit = unit
        self._axis_range = (float(axis_range[0]), float(axis_range[1]))
        self._min_val = float(min_val)
        self._max_val = float(max_val)
        self._decimals = int(decimals)
        self._is_log = bool(is_log)
        self._salt_n_permutations: int = 100
        self._threshold = float(default_threshold)
        self._enabled = True

        if self._is_log:
            self._plot_min = float(np.log10(max(self._axis_range[0], 1e-6)))
            self._plot_max = float(np.log10(max(self._axis_range[1], 1e-5)))
        else:
            self._plot_min = self._axis_range[0]
            self._plot_max = self._axis_range[1]

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 1, 8, 1)
        root.setSpacing(0)

        # Top line: checkbox · operator · input · unit · [stretch] · focused value · pass/fail icon
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(5)

        self.chk_enabled = TickCheckBox(label, self)
        self.chk_enabled.setChecked(True)
        self.chk_enabled.toggled.connect(self._on_enabled_toggled)
        top_row.addWidget(self.chk_enabled)

        self.lbl_operator = QLabel(operator)
        self.lbl_operator.setStyleSheet(DETECTION_CRITERION_META_LABEL_STYLE)
        top_row.addWidget(self.lbl_operator)

        self.edit_threshold = QLineEdit(self._format_threshold(self._threshold))
        self.edit_threshold.setFixedWidth(62)
        self.edit_threshold.setFixedHeight(20)
        self.edit_threshold.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.edit_threshold.setStyleSheet(DETECTION_CRITERION_INPUT_STYLE)
        validator = QDoubleValidator(self._min_val, self._max_val, max(self._decimals, 4), self.edit_threshold)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        c_locale = QLocale.c()
        c_locale.setNumberOptions(QLocale.NumberOption.RejectGroupSeparator)
        validator.setLocale(c_locale)
        self.edit_threshold.setValidator(validator)
        self.edit_threshold.textChanged.connect(self._on_threshold_text_changed)
        self.edit_threshold.editingFinished.connect(self._on_threshold_edited)
        top_row.addWidget(self.edit_threshold)

        self.lbl_unit = QLabel(unit)
        self.lbl_unit.setStyleSheet(DETECTION_CRITERION_META_LABEL_STYLE)
        if not unit:
            self.lbl_unit.hide()
        top_row.addWidget(self.lbl_unit)

        top_row.addStretch()

        self.lbl_focused_value = QLabel("n/a")
        self.lbl_focused_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_focused_value.setStyleSheet(DETECTION_CRITERION_VALUE_NA_STYLE)
        top_row.addWidget(self.lbl_focused_value)

        self.lbl_focused_icon = QLabel("")
        self.lbl_focused_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_focused_icon.hide()
        top_row.addWidget(self.lbl_focused_icon)

        root.addLayout(top_row)

        # Bottom line: fixed 240x24 distribution plot + vertically centered pass count
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(10)

        plot_w = DETECTION_CRITERION_PLOT_SIZE[0]
        plot_h = DETECTION_CRITERION_PLOT_SIZE[1] + _CRITERION_AXIS_HEIGHT
        self.plot_dist = pg.PlotWidget(self)
        self.plot_dist.setFixedSize(plot_w, plot_h)
        self.plot_dist.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.plot_dist.setMouseEnabled(x=False, y=False)
        apply_plot_style(
            self.plot_dist,
            compact=True,
            show_bottom_values=True,
        )
        p_item = self.plot_dist.getPlotItem()
        p_item.hideAxis("left")
        p_item.setContentsMargins(12, 2, 12, 2)
        p_item.setXRange(self._plot_min, self._plot_max, padding=0.05)
        p_item.setYRange(0.0, 1.08, padding=0.0)

        bottom_ax = p_item.getAxis("bottom")
        if bottom_ax is not None:
            bottom_ax.setHeight(_CRITERION_AXIS_HEIGHT)
            bottom_ax.setStyle(tickTextOffset=2, tickLength=-3, showValues=True)
        self._ticks: List[Tuple[float, str]] = []
        self.set_ticks(ticks)

        # Plot items: light blue passing band, grey population fill + outline, passing fill, threshold line, focused line
        self._pass_region = pg.LinearRegionItem(
            orientation="vertical",
            brush=pg.mkBrush(COLOR_ACCENT_BG),
            pen=pg.mkPen(None),
            movable=False,
        )
        self._pass_region.setZValue(1)

        self._kde_curve = pg.PlotDataItem(
            pen=pg.mkPen(COLOR_AXIS, width=1.2),
        )
        self._kde_curve.setZValue(10)

        self._pop_zero_curve = pg.PlotDataItem(pen=None)
        self._pop_fill = pg.FillBetweenItem(
            self._pop_zero_curve,
            self._kde_curve,
            brush=pg.mkBrush(COLOR_KDE_POP_FILL),
        )
        self._pop_fill.setZValue(2)

        self._zero_curve = pg.PlotDataItem(pen=None)
        self._pass_curve = pg.PlotDataItem(pen=None)
        self._pass_fill = pg.FillBetweenItem(
            self._zero_curve,
            self._pass_curve,
            brush=pg.mkBrush(COLOR_PASS_FILL),
        )
        self._pass_fill.setZValue(5)

        self._threshold_line = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(COLOR_ACCENT, width=1.6),
        )
        self._threshold_line.setZValue(15)

        self._focused_line = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(COLOR_TEXT, width=1.8),
        )
        self._focused_line.setZValue(20)
        self._focused_line.hide()

        self.plot_dist.addItem(self._pass_region)
        self.plot_dist.addItem(self._pop_fill)
        self.plot_dist.addItem(self._pass_fill)
        self.plot_dist.addItem(self._kde_curve)
        self.plot_dist.addItem(self._threshold_line)
        self.plot_dist.addItem(self._focused_line)

        bottom_row.addWidget(self.plot_dist)

        self.lbl_pass_count = QLabel("<b>0</b> of 0 pass")
        self.lbl_pass_count.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_pass_count.setStyleSheet(DETECTION_CRITERION_PASS_COUNT_STYLE)
        bottom_row.addWidget(self.lbl_pass_count)
        bottom_row.addStretch()

        root.addLayout(bottom_row)

    def set_axis_range(
        self,
        axis_range: Tuple[float, float],
        ticks: Optional[Sequence[Any]] = None,
        min_val: Optional[float] = None,
        decimals: Optional[int] = None,
    ) -> None:
        """Update criterion axis bounds, ticks, and optional validator precision."""
        self._axis_range = (float(axis_range[0]), float(axis_range[1]))
        if self._is_log:
            self._plot_min = float(np.log10(max(self._axis_range[0], 1e-12)))
            self._plot_max = float(np.log10(max(self._axis_range[1], 1e-11)))
        else:
            self._plot_min = self._axis_range[0]
            self._plot_max = self._axis_range[1]

        self.plot_dist.getPlotItem().setXRange(self._plot_min, self._plot_max, padding=0.05)
        self.set_ticks(ticks)

        if min_val is not None or decimals is not None:
            if min_val is not None:
                self._min_val = float(min_val)
            if decimals is not None:
                self._decimals = int(decimals)
            validator = QDoubleValidator(
                self._min_val,
                self._max_val,
                max(self._decimals, 4),
                self.edit_threshold,
            )
            validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            c_locale = QLocale.c()
            c_locale.setNumberOptions(QLocale.NumberOption.RejectGroupSeparator)
            validator.setLocale(c_locale)
            self.edit_threshold.setValidator(validator)

    def set_ticks(self, ticks: Optional[Sequence[Any]] = None) -> None:
        """Set bottom axis tick positions (in data units) and render both end ticks clearly."""
        if ticks is None:
            resolved = _default_ticks_for_range(self._axis_range, self._is_log)
        else:
            resolved = []
            for item in ticks:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    resolved.append((float(item[0]), str(item[1])))
                else:
                    val = float(item)
                    pos = float(np.log10(max(val, 1e-12))) if self._is_log else val
                    resolved.append((pos, f"{val:g}"))
        self._ticks = resolved
        bottom_ax = self.plot_dist.getPlotItem().getAxis("bottom")
        if bottom_ax is not None:
            bottom_ax.setTicks([self._ticks])

    def _format_threshold(self, val: float) -> str:
        if self._is_log:
            return f"{val:.4g}"
        return f"{val:.{self._decimals}f}"

    def _format_value(self, val: float) -> str:
        if self._is_log:
            return format_salt_p_value(val, self._salt_n_permutations)
        return f"{val:.{self._decimals}f}"

    def _to_plot_x(self, val: float) -> float:
        if self._is_log:
            floor_clip = min(1e-6, max(self._axis_range[0], 1e-12))
            return float(np.log10(max(val, floor_clip)))
        return float(val)

    def _passes(self, val: float, threshold: float) -> bool:
        if self._operator in ("≤", "<="):
            return val <= threshold
        return val >= threshold

    def is_enabled(self) -> bool:
        """Return whether this criterion is currently enabled."""
        return self.chk_enabled.isChecked()

    def threshold(self) -> float:
        """Return the current numeric threshold."""
        return self._threshold

    def _on_enabled_toggled(self, checked: bool) -> None:
        self._enabled = bool(checked)
        self._apply_enabled_visuals(self._enabled)
        self.enabled_changed.emit(self._enabled)

    def _on_threshold_text_changed(self, text: str) -> None:
        stripped = text.strip()
        if not stripped or stripped in ("-", ".", "-."):
            return
        try:
            val = float(stripped)
        except ValueError:
            return
        val = max(self._min_val, min(self._max_val, val))
        if abs(val - self._threshold) > 1e-12:
            self._threshold = val
            self.threshold_changed.emit(val)

    def _on_threshold_edited(self) -> None:
        text = self.edit_threshold.text().strip()
        try:
            val = float(text)
        except ValueError:
            self.edit_threshold.blockSignals(True)
            self.edit_threshold.setText(self._format_threshold(self._threshold))
            self.edit_threshold.blockSignals(False)
            return
        val = max(self._min_val, min(self._max_val, val))
        changed = abs(val - self._threshold) > 1e-12
        self._threshold = val
        self.edit_threshold.blockSignals(True)
        self.edit_threshold.setText(self._format_threshold(val))
        self.edit_threshold.blockSignals(False)
        if changed:
            self.threshold_changed.emit(val)

    def _apply_enabled_visuals(self, enabled: bool) -> None:
        self.edit_threshold.setEnabled(enabled)
        meta_style = DETECTION_CRITERION_META_LABEL_STYLE if enabled else DETECTION_CRITERION_META_DIM_STYLE
        self.lbl_operator.setStyleSheet(meta_style)
        self.lbl_unit.setStyleSheet(meta_style)
        self.lbl_pass_count.setStyleSheet(
            DETECTION_CRITERION_PASS_COUNT_STYLE if enabled else DETECTION_CRITERION_PASS_COUNT_DIM_STYLE
        )
        self._pass_region.setVisible(enabled)
        self._pass_fill.setVisible(enabled)
        self._threshold_line.setVisible(enabled)
        dim_opacity = 1.0 if enabled else 0.45
        self._pop_fill.setOpacity(dim_opacity)
        self._kde_curve.setOpacity(dim_opacity)
        self._focused_line.setOpacity(dim_opacity)
        bottom_ax = self.plot_dist.getPlotItem().getAxis("bottom")
        if bottom_ax is not None:
            bottom_ax.setTextPen(pg.mkPen(COLOR_AXIS if enabled else COLOR_TEXT_MUTED))
        self.setProperty("criterionDisabled", not enabled)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_population_data(
        self,
        values: Optional[Sequence[float]],
        threshold: float,
        enabled: bool,
        focused_value: Optional[float],
        n_pass: int,
        n_total: int,
    ) -> None:
        """Update population KDE, passing fill, threshold/focused markers, and readouts."""
        self._threshold = float(threshold)
        self._enabled = bool(enabled)

        self.chk_enabled.blockSignals(True)
        self.chk_enabled.setChecked(self._enabled)
        self.chk_enabled.blockSignals(False)

        if not self.edit_threshold.hasFocus():
            self.edit_threshold.setText(self._format_threshold(self._threshold))

        self._apply_enabled_visuals(self._enabled)

        # Focused unit readout
        if focused_value is None or not np.isfinite(focused_value):
            self.lbl_focused_value.setText("n/a")
            self.lbl_focused_value.setStyleSheet(DETECTION_CRITERION_VALUE_NA_STYLE)
            self.lbl_focused_icon.setText("")
            self.lbl_focused_icon.hide()
            self._focused_line.hide()
        else:
            f_val = float(focused_value)
            val_str = self._format_value(f_val)
            unit_suffix = f" {self._unit}" if self._unit else ""
            self.lbl_focused_value.setText(f"{val_str}{unit_suffix}")
            if self._enabled:
                passed = self._passes(f_val, self._threshold)
                self.lbl_focused_value.setStyleSheet(
                    DETECTION_CRITERION_VALUE_PASS_STYLE if passed else DETECTION_CRITERION_VALUE_FAIL_STYLE
                )
                self.lbl_focused_icon.setText("✓" if passed else "✗")
                self.lbl_focused_icon.setStyleSheet(
                    DETECTION_CRITERION_ICON_PASS_STYLE if passed else DETECTION_CRITERION_ICON_FAIL_STYLE
                )
                self.lbl_focused_icon.show()
            else:
                self.lbl_focused_value.setStyleSheet(DETECTION_CRITERION_VALUE_NA_STYLE)
                self.lbl_focused_icon.setText("")
                self.lbl_focused_icon.hide()

            fx = max(self._plot_min, min(self._plot_max, self._to_plot_x(f_val)))
            self._focused_line.setValue(fx)
            self._focused_line.show()

        # Pass count readout
        self.lbl_pass_count.setText(f"<b>{int(n_pass)}</b> of {int(n_total)} pass")

        # KDE curve + grey population fill + blue passing fill
        arr = np.asarray(values if values is not None else [], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if self._is_log:
            arr = arr[arr > 0.0]
            floor_clip = min(1e-6, max(self._axis_range[0], 1e-12))
            samples = np.log10(np.clip(arr, floor_clip, None))
        else:
            samples = arr

        grid = np.linspace(self._plot_min, self._plot_max, 120)
        density = _compute_kde(samples, grid, self._plot_max - self._plot_min)
        self._kde_curve.setData(grid, density)
        self._pop_zero_curve.setData(grid, np.zeros_like(grid))

        tx = self._to_plot_x(self._threshold)
        tx_clamped = max(self._plot_min, min(self._plot_max, tx))
        self._threshold_line.setValue(tx_clamped)

        if self._operator in ("≤", "<="):
            self._pass_region.setRegion((self._plot_min, tx_clamped))
            mask = grid <= tx_clamped
        else:
            self._pass_region.setRegion((tx_clamped, self._plot_max))
            mask = grid >= tx_clamped

        if np.any(mask):
            pass_x = grid[mask]
            pass_y = density[mask]
            self._zero_curve.setData(pass_x, np.zeros_like(pass_x))
            self._pass_curve.setData(pass_x, pass_y)
        else:
            self._zero_curve.setData([], [])
            self._pass_curve.setData([], [])


class DetectionPairplotWidget(QWidget):
    """3x3 lower-triangle pairplot for (latency, jitter, reliability, log10 SALT p) (§ 13.5.4)."""

    unit_clicked = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._unit_meta: Dict[int, str] = {}
        self._salt_n_permutations: int = 100
        self._salt_min_p: float = 1e-3
        self._salt_y_bounds: Tuple[float, float] = (-3.0, 0.0)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self.lbl_title = QLabel("All units: tagging metrics")
        self.lbl_title.setStyleSheet(DETECTION_SUBPANEL_TITLE_STYLE)
        root.addWidget(self.lbl_title)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)

        cell_w, cell_h = DETECTION_PAIRPLOT_CELL_SIZE

        # (0, 0): Jitter (0..6) vs Latency (0..20)
        self.plot_jit_lat = self._make_scatter_plot(
            width=cell_w + _PAIRPLOT_LEFT_MARGIN,
            height=cell_h,
            left_label="Jitter (ms)",
            bottom_label=None,
            show_left=True,
            show_bottom=False,
            x_range=(0.0, 20.0),
            y_range=(0.0, 6.0),
        )
        grid.addWidget(self.plot_jit_lat, 0, 0)

        # (0, 1): Legend cell
        legend_box = QWidget(self)
        legend_box.setFixedSize(cell_w, cell_h)
        legend_box.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        leg_layout = QVBoxLayout(legend_box)
        leg_layout.setContentsMargins(8, 8, 4, 4)
        leg_layout.setSpacing(4)
        leg_layout.addStretch()

        self._lbl_legend = QLabel(DETECTION_PAIRPLOT_LEGEND_HTML)
        self._lbl_legend.setStyleSheet(DETECTION_PAIRPLOT_LEGEND_STYLE)
        leg_layout.addWidget(self._lbl_legend)
        leg_layout.addStretch()
        grid.addWidget(legend_box, 0, 1)

        # (1, 0): Reliability (0..1) vs Latency (0..20)
        self.plot_rel_lat = self._make_scatter_plot(
            width=cell_w + _PAIRPLOT_LEFT_MARGIN,
            height=cell_h,
            left_label="Reliability",
            bottom_label=None,
            show_left=True,
            show_bottom=False,
            x_range=(0.0, 20.0),
            y_range=(0.0, 1.0),
        )
        grid.addWidget(self.plot_rel_lat, 1, 0)

        # (1, 1): Reliability (0..1) vs Jitter (0..6)
        self.plot_rel_jit = self._make_scatter_plot(
            width=cell_w,
            height=cell_h,
            left_label=None,
            bottom_label=None,
            show_left=False,
            show_bottom=False,
            x_range=(0.0, 6.0),
            y_range=(0.0, 1.0),
        )
        grid.addWidget(self.plot_rel_jit, 1, 1)

        # (2, 0): SALT p (log10: -3..0) vs Latency (0..20)
        self.plot_salt_lat = self._make_scatter_plot(
            width=cell_w + _PAIRPLOT_LEFT_MARGIN,
            height=cell_h + _PAIRPLOT_BOTTOM_HEIGHT,
            left_label="SALT p",
            bottom_label="Latency (ms)",
            show_left=True,
            show_bottom=True,
            x_range=(0.0, 20.0),
            y_range=(-3.0, 0.0),
        )
        salt_left_ax = self.plot_salt_lat.getPlotItem().getAxis("left")
        if salt_left_ax is not None:
            salt_left_ax.setTicks([[(-3.0, "0.001"), (-2.0, "0.01"), (-1.0, "0.1"), (0.0, "1")]])
        grid.addWidget(self.plot_salt_lat, 2, 0)

        # (2, 1): SALT p (log10: -3..0) vs Jitter (0..6)
        self.plot_salt_jit = self._make_scatter_plot(
            width=cell_w,
            height=cell_h + _PAIRPLOT_BOTTOM_HEIGHT,
            left_label=None,
            bottom_label="Jitter (ms)",
            show_left=False,
            show_bottom=True,
            x_range=(0.0, 6.0),
            y_range=(-3.0, 0.0),
        )
        grid.addWidget(self.plot_salt_jit, 2, 1)

        # (2, 2): SALT p (log10: -3..0) vs Reliability (0..1)
        self.plot_salt_rel = self._make_scatter_plot(
            width=cell_w,
            height=cell_h + _PAIRPLOT_BOTTOM_HEIGHT,
            left_label=None,
            bottom_label="Reliability",
            show_left=False,
            show_bottom=True,
            x_range=(0.0, 1.0),
            y_range=(-3.0, 0.0),
        )
        grid.addWidget(self.plot_salt_rel, 2, 2)

        # Shared axis linking across columns and rows
        self.plot_jit_lat.setXLink(self.plot_salt_lat)
        self.plot_rel_lat.setXLink(self.plot_salt_lat)
        self.plot_rel_jit.setXLink(self.plot_salt_jit)
        self.plot_rel_jit.setYLink(self.plot_rel_lat)
        self.plot_salt_jit.setYLink(self.plot_salt_lat)
        self.plot_salt_rel.setYLink(self.plot_salt_lat)

        root.addLayout(grid)

        # Build per-plot overlay items (passing rect, threshold lines, scatters)
        self.export_name_provider: Optional[Callable[[], str]] = None
        self._cells: Dict[str, Dict[str, Any]] = {}
        for key, plot_w in (
            ("jit_lat", self.plot_jit_lat),
            ("rel_lat", self.plot_rel_lat),
            ("rel_jit", self.plot_rel_jit),
            ("salt_lat", self.plot_salt_lat),
            ("salt_jit", self.plot_salt_jit),
            ("salt_rel", self.plot_salt_rel),
        ):
            self._cells[key] = self._init_cell_items(plot_w)
            install_figure_export_menu(
                plot_w,
                export_target=self,
                default_name=self._export_base_name,
                label="Save pairplot as SVG…",
                subplot_target=plot_w,
                subplot_default_name=lambda k=key: f"{self._export_base_name()}_{k}",
                subplot_label="Save this subplot as SVG…",
            )

        self.set_salt_permutations(self._salt_n_permutations)

    def set_salt_permutations(self, n_permutations: int) -> None:
        """Update log10 SALT p axis bounds and ticks to match `n_permutations`."""
        self._salt_n_permutations = max(1, int(n_permutations))
        p_min, _p_max, ticks = salt_log_axis_spec(self._salt_n_permutations)
        min_exp = float(ticks[0][0]) if ticks else -3.0
        self._salt_min_p = float(p_min)
        self._salt_y_bounds = (min_exp, 0.0)

        for pw in (self.plot_salt_lat, self.plot_salt_jit, self.plot_salt_rel):
            pw.getPlotItem().setYRange(min_exp, 0.0, padding=0.0)
        salt_left_ax = self.plot_salt_lat.getPlotItem().getAxis("left")
        if salt_left_ax is not None:
            salt_left_ax.setTicks([ticks])

    def _export_base_name(self) -> str:
        """Return base filename stem for pairplot SVG export."""
        if callable(self.export_name_provider):
            try:
                name = str(self.export_name_provider() or "").strip()
                if name:
                    return name
            except Exception:
                pass
        return "tagging_pairplot"

    def _make_scatter_plot(
        self,
        width: int,
        height: int,
        left_label: Optional[str],
        bottom_label: Optional[str],
        show_left: bool,
        show_bottom: bool,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
    ) -> pg.PlotWidget:
        pw = pg.PlotWidget(self)
        pw.setFixedSize(width, height)
        pw.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        pw.setMouseEnabled(x=False, y=False)
        apply_plot_style(
            pw,
            compact=False,
            left_label=left_label if show_left else None,
            bottom_label=bottom_label if show_bottom else None,
            show_bottom_values=show_bottom,
            left_margin=_PAIRPLOT_LEFT_MARGIN if show_left else None,
        )
        p_item = pw.getPlotItem()
        p_item.setContentsMargins(0, 8, 6, 0)
        if not show_left:
            p_item.hideAxis("left")
        if not show_bottom:
            p_item.hideAxis("bottom")
        p_item.getViewBox().disableAutoRange()
        p_item.setXRange(x_range[0], x_range[1], padding=0.0)
        p_item.setYRange(y_range[0], y_range[1], padding=0.0)
        return pw

    def _init_cell_items(self, plot_w: pg.PlotWidget) -> Dict[str, Any]:
        pass_rect = pg.QtWidgets.QGraphicsRectItem()
        pass_rect.setBrush(pg.mkBrush(COLOR_ACCENT_BG))
        pass_rect.setPen(pg.mkPen(None))
        pass_rect.setZValue(1)
        pass_rect.hide()
        plot_w.addItem(pass_rect)

        thresh_pen = pg.mkPen(COLOR_ACCENT, width=1.1, style=Qt.PenStyle.DashLine)
        vline = pg.InfiniteLine(angle=90, pen=thresh_pen)
        vline.setZValue(3)
        vline.hide()
        plot_w.addItem(vline)

        hline = pg.InfiniteLine(angle=0, pen=thresh_pen)
        hline.setZValue(3)
        hline.hide()
        plot_w.addItem(hline)

        untagged = pg.ScatterPlotItem(
            size=5,
            symbol="o",
            brush=pg.mkBrush(rgba_from_hex(COLOR_TEXT_MUTED, alpha=130)),
            pen=None,
            hoverable=True,
        )
        untagged.setZValue(10)
        untagged.sigClicked.connect(self._on_points_clicked)
        untagged.sigHovered.connect(self._on_points_hovered)
        plot_w.addItem(untagged)

        tagged = pg.ScatterPlotItem(
            size=6,
            symbol="o",
            brush=pg.mkBrush(rgba_from_hex(COLOR_SUCCESS, alpha=210)),
            pen=None,
            hoverable=True,
        )
        tagged.setZValue(12)
        tagged.sigClicked.connect(self._on_points_clicked)
        tagged.sigHovered.connect(self._on_points_hovered)
        plot_w.addItem(tagged)

        focused = pg.ScatterPlotItem(
            size=10,
            symbol="o",
            brush=pg.mkBrush(0, 0, 0, 0),
            pen=pg.mkPen(COLOR_ACCENT, width=2.0),
        )
        focused.setZValue(20)
        plot_w.addItem(focused)

        return {
            "pass_rect": pass_rect,
            "vline": vline,
            "hline": hline,
            "untagged": untagged,
            "tagged": tagged,
            "focused": focused,
        }

    def _on_points_clicked(self, _scatter: Any, points: Any) -> None:
        if not points:
            return
        uid = points[0].data()
        if uid is not None:
            self.unit_clicked.emit(int(uid))

    def _on_points_hovered(self, scatter: Any, points: Any, _ev: Any = None) -> None:
        if points is not None and len(points) > 0:
            uid = points[0].data()
            if uid is not None and int(uid) in self._unit_meta:
                scatter.getViewWidget().setToolTip(self._unit_meta[int(uid)])
                return
        if scatter.getViewWidget() is not None:
            scatter.getViewWidget().setToolTip("")

    def set_data(
        self,
        unit_ids: Sequence[int],
        latencies: Sequence[float],
        jitters: Sequence[float],
        reliabilities: Sequence[float],
        is_tagged: Sequence[bool],
        focused_unit_id: Optional[int] = None,
        lat_max: float = 8.0,
        lat_enabled: bool = True,
        jit_max: float = 2.0,
        jit_enabled: bool = True,
        rel_min: float = 0.60,
        rel_enabled: bool = True,
        salt_p_values: Optional[Sequence[float]] = None,
        salt_max: float = 0.05,
        salt_enabled: bool = True,
        salt_n_permutations: Optional[int] = None,
    ) -> None:
        """Update pairplot scatters, threshold lines, passing regions, and focused ring."""
        if salt_n_permutations is not None:
            self.set_salt_permutations(salt_n_permutations)

        uids = np.asarray(unit_ids, dtype=np.int64)
        lat = np.asarray(latencies, dtype=np.float64)
        jit = np.asarray(jitters, dtype=np.float64)
        rel = np.asarray(reliabilities, dtype=np.float64)
        tag = np.asarray(is_tagged, dtype=bool)
        has_salt = salt_p_values is not None
        salt = np.asarray(salt_p_values, dtype=np.float64) if has_salt else None

        valid = (
            np.isfinite(lat)
            & (lat < (MISSING_METRIC_SENTINEL - 1.0))
            & np.isfinite(jit)
            & (jit < (MISSING_METRIC_SENTINEL - 1.0))
            & np.isfinite(rel)
        )
        if salt is not None:
            valid = valid & np.isfinite(salt)

        n_omitted = int(np.sum(~valid))
        tooltip = (
            f"{n_omitted} units without a light response omitted"
            if n_omitted > 0
            else "All session units shown"
        )
        self.setToolTip(tooltip)
        self._lbl_legend.setToolTip(tooltip)
        self.lbl_title.setToolTip(tooltip)

        v_uids = uids[valid]
        v_lat = lat[valid]
        v_jit = jit[valid]
        v_rel = rel[valid]
        v_tag = tag[valid]
        v_salt = salt[valid] if salt is not None else None
        v_salt_log = np.log10(np.clip(v_salt, self._salt_min_p, 1.0)) if v_salt is not None else None

        self._unit_meta.clear()
        if v_salt is not None:
            for uid, la, ji, re_val, sa_val, tg in zip(v_uids, v_lat, v_jit, v_rel, v_salt, v_tag):
                status = "Tagged" if tg else "Untagged"
                salt_txt = format_salt_p_value(float(sa_val), self._salt_n_permutations)
                self._unit_meta[int(uid)] = (
                    f"Unit u{int(uid):04d} ({status}) · "
                    f"Lat {la:.2f} ms · Jit {ji:.2f} ms · Rel {re_val:.2f} · SALT p {salt_txt}"
                )
        else:
            for uid, la, ji, re_val, tg in zip(v_uids, v_lat, v_jit, v_rel, v_tag):
                status = "Tagged" if tg else "Untagged"
                self._unit_meta[int(uid)] = (
                    f"Unit u{int(uid):04d} ({status}) · "
                    f"Lat {la:.2f} ms · Jit {ji:.2f} ms · Rel {re_val:.2f}"
                )

        salt_thresh_log = (
            float(np.log10(np.clip(salt_max, self._salt_min_p, 1.0))) if salt_enabled else None
        )

        self._update_cell(
            "jit_lat",
            v_uids,
            v_lat,
            v_jit,
            v_tag,
            focused_unit_id,
            x_bounds=(0.0, 20.0),
            y_bounds=(0.0, 6.0),
            x_thresh=lat_max if lat_enabled else None,
            x_le=True,
            y_thresh=jit_max if jit_enabled else None,
            y_le=True,
        )
        self._update_cell(
            "rel_lat",
            v_uids,
            v_lat,
            v_rel,
            v_tag,
            focused_unit_id,
            x_bounds=(0.0, 20.0),
            y_bounds=(0.0, 1.0),
            x_thresh=lat_max if lat_enabled else None,
            x_le=True,
            y_thresh=rel_min if rel_enabled else None,
            y_le=False,
        )
        self._update_cell(
            "rel_jit",
            v_uids,
            v_jit,
            v_rel,
            v_tag,
            focused_unit_id,
            x_bounds=(0.0, 6.0),
            y_bounds=(0.0, 1.0),
            x_thresh=jit_max if jit_enabled else None,
            x_le=True,
            y_thresh=rel_min if rel_enabled else None,
            y_le=False,
        )

        if v_salt_log is not None:
            s_uids, s_lat, s_jit, s_rel, s_log, s_tag = v_uids, v_lat, v_jit, v_rel, v_salt_log, v_tag
        else:
            s_uids = np.empty(0, dtype=np.int64)
            s_lat = np.empty(0, dtype=np.float64)
            s_jit = np.empty(0, dtype=np.float64)
            s_rel = np.empty(0, dtype=np.float64)
            s_log = np.empty(0, dtype=np.float64)
            s_tag = np.empty(0, dtype=bool)

        self._update_cell(
            "salt_lat",
            s_uids,
            s_lat,
            s_log,
            s_tag,
            focused_unit_id,
            x_bounds=(0.0, 20.0),
            y_bounds=self._salt_y_bounds,
            x_thresh=lat_max if lat_enabled else None,
            x_le=True,
            y_thresh=salt_thresh_log,
            y_le=True,
        )
        self._update_cell(
            "salt_jit",
            s_uids,
            s_jit,
            s_log,
            s_tag,
            focused_unit_id,
            x_bounds=(0.0, 6.0),
            y_bounds=self._salt_y_bounds,
            x_thresh=jit_max if jit_enabled else None,
            x_le=True,
            y_thresh=salt_thresh_log,
            y_le=True,
        )
        self._update_cell(
            "salt_rel",
            s_uids,
            s_rel,
            s_log,
            s_tag,
            focused_unit_id,
            x_bounds=(0.0, 1.0),
            y_bounds=self._salt_y_bounds,
            x_thresh=rel_min if rel_enabled else None,
            x_le=False,
            y_thresh=salt_thresh_log,
            y_le=True,
        )

    def _update_cell(
        self,
        key: str,
        uids: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        tags: np.ndarray,
        focused_unit_id: Optional[int],
        x_bounds: Tuple[float, float],
        y_bounds: Tuple[float, float],
        x_thresh: Optional[float],
        x_le: bool,
        y_thresh: Optional[float],
        y_le: bool,
    ) -> None:
        cell = self._cells[key]
        untag_mask = ~tags
        cell["untagged"].setData(x=xs[untag_mask], y=ys[untag_mask], data=uids[untag_mask].tolist())
        cell["tagged"].setData(x=xs[tags], y=ys[tags], data=uids[tags].tolist())

        if focused_unit_id is not None:
            f_mask = uids == int(focused_unit_id)
            if np.any(f_mask):
                cell["focused"].setData(x=xs[f_mask], y=ys[f_mask])
            else:
                cell["focused"].setData(x=[], y=[])
        else:
            cell["focused"].setData(x=[], y=[])

        if x_thresh is not None:
            cell["vline"].setValue(x_thresh)
            cell["vline"].show()
        else:
            cell["vline"].hide()

        if y_thresh is not None:
            cell["hline"].setValue(y_thresh)
            cell["hline"].show()
        else:
            cell["hline"].hide()

        if x_thresh is not None or y_thresh is not None:
            rx0 = x_bounds[0] if (x_thresh is None or x_le) else max(x_bounds[0], x_thresh)
            rx1 = x_bounds[1] if (x_thresh is None or not x_le) else min(x_bounds[1], x_thresh)
            ry0 = y_bounds[0] if (y_thresh is None or y_le) else max(y_bounds[0], y_thresh)
            ry1 = y_bounds[1] if (y_thresh is None or not y_le) else min(y_bounds[1], y_thresh)
            if rx1 > rx0 and ry1 > ry0:
                cell["pass_rect"].setRect(QRectF(rx0, ry0, rx1 - rx0, ry1 - ry0))
                cell["pass_rect"].show()
            else:
                cell["pass_rect"].hide()
        else:
            cell["pass_rect"].hide()
