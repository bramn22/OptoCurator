"""Live pyqtgraph plot helpers for Phase 1 Detection View and Phase 2 Inspection (§ 8.2)."""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QTransform
from PyQt6.QtWidgets import QSizePolicy, QTableWidgetItem, QVBoxLayout, QWidget

from optocurator.models import DEFAULT_PULSE_DURATION_MS
from optocurator.ui.theme import (
    COLOR_ACCENT,
    COLOR_ACCENT_HOVER,
    COLOR_AXIS,
    COLOR_CANDIDATE_MARKER,
    COLOR_LIGHT_PULSE_BAND,
    COLOR_PANEL,
    COLOR_PROBE_SITE,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    FONT_SIZE_CAPTION,
    FONT_SIZE_TICK,
    PLOT_AXIS_LABEL_COMPACT_STYLE,
    PLOT_AXIS_LABEL_STYLE,
    PLOT_HIST_BRUSH,
    PLOT_PROBE_SIZE,
    PLOT_UNIT_RESPONSE_GAP,
    PLOT_UNIT_RESPONSE_LEFT_MARGIN,
    PLOT_UNIT_RESPONSE_PSTH_SIZE,
    PLOT_UNIT_RESPONSE_RASTER_SIZE,
    PSTH_LINE_WIDTH,
    make_font,
    rgba_from_hex,
)

DEFAULT_Y_AXIS_WIDTH: int = 58


class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem supporting true numeric and custom sort keys."""

    def __init__(self, text: str, sort_key: Any = None):
        super().__init__(text)
        self.sort_key = sort_key if sort_key is not None else text

    def __lt__(self, other: "QTableWidgetItem") -> bool:
        if isinstance(other, NumericTableWidgetItem):
            try:
                return self.sort_key < other.sort_key
            except TypeError:
                return str(self.sort_key) < str(other.sort_key)
        return super().__lt__(other)


def infer_pulse_duration_ms(
    onsets: Optional[np.ndarray],
    offsets: Optional[np.ndarray] = None,
    default_ms: float = DEFAULT_PULSE_DURATION_MS,
) -> float:
    """Infer median stimulation pulse duration in ms from onset/offset arrays (seconds)."""
    if onsets is None or offsets is None:
        return float(default_ms)
    try:
        on_arr = np.asarray(onsets, dtype=np.float64).ravel()
        off_arr = np.asarray(offsets, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return float(default_ms)
    n = min(len(on_arr), len(off_arr))
    if n == 0:
        return float(default_ms)
    durs_ms = (off_arr[:n] - on_arr[:n]) * 1000.0
    valid = durs_ms[np.isfinite(durs_ms) & (durs_ms > 0.0)]
    if len(valid) == 0:
        return float(default_ms)
    return round(float(np.median(valid)), 3)


class _SafeLinearRegionItem(pg.LinearRegionItem):
    """LinearRegionItem that safely handles viewRect() returning None in headless mode."""

    def boundingRect(self) -> QRectF:
        vr = self.viewRect()
        if vr is None:
            return QRectF()
        return super().boundingRect()


def _get_plot_item(plot_widget: Union[pg.PlotWidget, pg.PlotItem]) -> pg.PlotItem:
    """Extract pg.PlotItem from either a PlotWidget or PlotItem instance."""
    if hasattr(plot_widget, "getPlotItem"):
        return plot_widget.getPlotItem()
    return plot_widget


def compute_standard_plot_ticks(
    w_start: float = -30.0,
    w_end: float = 50.0,
    compact: bool = False,
) -> List[List[Tuple[float, str]]]:
    """Compute standard x-axis tick marks for raster and PSTH plots (§ 8.2).

    For the default window (-30 to +50 ms), tick marks are placed at
    [-20, -10, 0, 5, 10, 20, 30, 40] ms. Full-size plots label all of them;
    small plots (compact=True, e.g. cluster cards) label [-20, 0, 5, 10, 20, 40].
    """
    base_candidates = [-20, -10, 0, 5, 10, 20, 30, 40]
    if w_start < -30.0 or w_end > 50.0:
        start_step = int(np.ceil(w_start / 10.0)) * 10
        end_step = int(np.floor(w_end / 10.0)) * 10
        extra = list(range(start_step, end_step + 1, 10))
        candidates = sorted(set(base_candidates + extra))
    else:
        candidates = base_candidates

    in_range = [t for t in candidates if w_start <= t <= w_end]
    compact_unlabeled = {-10, 5, 30}

    major_ticks: List[Tuple[float, str]] = []
    for t in in_range:
        if compact and t in compact_unlabeled:
            major_ticks.append((float(t), ""))
        else:
            major_ticks.append((float(t), str(int(t))))
    return [major_ticks]


def compute_clean_y_ticks(y_top: float, max_ticks: int = 3) -> List[List[Tuple[float, str]]]:
    """Compute at most `max_ticks` clean, non-overlapping y-axis ticks in [0, y_top]."""
    top = float(y_top)
    if not np.isfinite(top) or top <= 0.0:
        return [[(0.0, "0")]]
    raw_step = top / max(1, max_ticks - 1)
    mag = 10.0 ** float(np.floor(np.log10(raw_step)))
    norm = raw_step / mag
    if norm <= 1.0:
        step = 1.0 * mag
    elif norm <= 2.0:
        step = 2.0 * mag
    elif norm <= 5.0:
        step = 5.0 * mag
    else:
        step = 10.0 * mag

    ticks: List[Tuple[float, str]] = []
    val = 0.0
    while val <= top * 1.001 and len(ticks) < max_ticks:
        label = "0" if abs(val) < 1e-12 else f"{val:g}"
        ticks.append((float(val), label))
        val += step
    return [ticks]


def _reposition_axis_label(axis: pg.AxisItem, gap_px: float = 2.0) -> None:
    """Position an AxisItem label close to its tick numbers instead of pinned to outer edges."""
    if getattr(axis, "label", None) is None or not axis.label.isVisible():
        return
    br = axis.label.boundingRect()
    w = float(axis.size().width())
    h = float(axis.size().height())
    offset = axis.style.get("tickTextOffset", [3, 2])
    off_x = float(offset[0]) if isinstance(offset, (list, tuple)) else float(offset)
    off_y = float(offset[1]) if isinstance(offset, (list, tuple)) else float(offset)

    if axis.orientation == "left":
        tick_w = max(float(getattr(axis, "textWidth", 0.0) or 0.0), 10.0)
        tick_left = w - off_x - tick_w
        # QGraphicsTextItem adds ~4 px documentMargin; align visible right glyph edge beside ticks.
        label_visual_h = max(0.0, br.height() - 4.0)
        x = tick_left - gap_px - label_visual_h
        x = max(-6.0, min(x, w - br.height() + 5.0))
        y = int(h / 2.0 + br.width() / 2.0)
        axis.label.setPos(float(x), float(y))
        axis.picture = None
    elif axis.orientation == "bottom":
        tick_font = axis.style.get("tickFont")
        px = tick_font.pixelSize() if (tick_font is not None and tick_font.pixelSize() > 0) else FONT_SIZE_TICK
        tick_bottom = off_y + float(px) + 2.0
        y = tick_bottom + gap_px - 4.0
        y = min(y, h - br.height() + 5.0)
        x = int(w / 2.0 - br.width() / 2.0)
        axis.label.setPos(float(x), float(y))
        axis.picture = None


def _install_tight_axis_label(axis: pg.AxisItem) -> None:
    """Hook AxisItem resize and tick-width updates so axis labels stay close to tick numbers."""
    if getattr(axis, "_optocurator_tight_label", False):
        _reposition_axis_label(axis)
        return
    axis._optocurator_tight_label = True
    orig_resize = axis.resizeEvent
    orig_update_max = axis._updateMaxTextSize
    orig_update_label = axis._updateLabel

    def _wrapped_resize(ev=None, _orig=orig_resize, _ax=axis):
        res = _orig(ev)
        _reposition_axis_label(_ax)
        return res

    def _wrapped_update_max(x: int, _orig=orig_update_max, _ax=axis):
        res = _orig(x)
        _reposition_axis_label(_ax)
        return res

    def _wrapped_update_label(_orig=orig_update_label, _ax=axis):
        res = _orig()
        _reposition_axis_label(_ax)
        return res

    axis.resizeEvent = _wrapped_resize
    axis._updateMaxTextSize = _wrapped_update_max
    axis._updateLabel = _wrapped_update_label
    _reposition_axis_label(axis)


def apply_plot_style(
    plot_item: Union[pg.PlotWidget, pg.PlotItem],
    compact: bool = False,
    left_label: Optional[str] = None,
    bottom_label: Optional[str] = None,
    show_bottom_values: Optional[bool] = None,
    left_margin: Optional[int] = None,
) -> None:
    """Apply consistent scientific figure styling to a PlotWidget or PlotItem (§ 8.2).

    - White background, no top/right spines, no gridlines, no redundant title
    - 10–11 px ticks in COLOR_TEXT_MUTED, 11–12 px axis labels in COLOR_TEXT_SECONDARY
    - Axis pen in COLOR_AXIS (width 1)
    - Disables pyqtgraph auto SI prefix notation on left and bottom axes
    """
    if hasattr(plot_item, "setBackground"):
        plot_item.setBackground(COLOR_PANEL)
    if hasattr(plot_item, "setMenuEnabled"):
        plot_item.setMenuEnabled(False)
    if hasattr(plot_item, "hideButtons"):
        plot_item.hideButtons()

    p_item = _get_plot_item(plot_item)
    p_item.showAxis("top", False)
    p_item.showAxis("right", False)
    p_item.showGrid(x=False, y=False)
    p_item.setTitle(None)
    if hasattr(p_item, "hideButtons"):
        p_item.hideButtons()

    for axis_name in ("left", "bottom", "right", "top"):
        axis = p_item.getAxis(axis_name)
        if axis is not None and axis.getViewWidget() is None:
            axis.deviceTransform_ = lambda a=axis: a.deviceTransform(QTransform())

    axis_pen = pg.mkPen(COLOR_AXIS, width=1)
    text_pen = pg.mkPen(COLOR_TEXT_MUTED)
    tick_font_size = FONT_SIZE_TICK if compact else FONT_SIZE_CAPTION
    tick_font = make_font(tick_font_size)
    label_style = PLOT_AXIS_LABEL_COMPACT_STYLE if compact else PLOT_AXIS_LABEL_STYLE

    for axis_name in ("left", "bottom"):
        axis = p_item.getAxis(axis_name)
        if axis is None:
            continue
        axis.enableAutoSIPrefix(False)
        axis.setPen(axis_pen)
        axis.setTextPen(text_pen)
        axis.setStyle(
            tickFont=tick_font,
            tickTextOffset=3 if compact else 4,
            tickLength=-3 if compact else -4,
            autoExpandTextSpace=True,
        )

    left_ax = p_item.getAxis("left")
    if left_ax is not None:
        if left_margin is not None:
            left_ax.setWidth(int(left_margin))
        if left_label is not None:
            p_item.setLabel("left", left_label, **label_style)
        if left_margin is not None and p_item.layout is not None:
            p_item.layout.activate()
        _install_tight_axis_label(left_ax)

    bottom_ax = p_item.getAxis("bottom")
    if bottom_ax is not None:
        if show_bottom_values is not None:
            bottom_ax.setStyle(showValues=bool(show_bottom_values))
        if bottom_label is not None:
            p_item.setLabel("bottom", bottom_label, **label_style)
        _install_tight_axis_label(bottom_ax)


def configure_plot_theme(
    plot_item: Union[pg.PlotWidget, pg.PlotItem],
    compact: bool = False,
) -> None:
    """Apply clean, publication-ready light styling to a pyqtgraph plot widget or PlotItem."""
    apply_plot_style(plot_item, compact=compact)


def render_raster(
    plot_widget: Union[pg.PlotWidget, pg.PlotItem],
    raster_x: np.ndarray,
    raster_y: np.ndarray,
    n_trials: int,
    title: Optional[str] = None,
    x_range: Tuple[float, float] = (-30.0, 50.0),
    order_by: str = "presentation",
    show_x_axis: bool = False,
    marker_size: float = 3.0,
    y_axis_width: Optional[int] = DEFAULT_Y_AXIS_WIDTH,
    pulse_duration_ms: float = DEFAULT_PULSE_DURATION_MS,
    compact: bool = False,
    y_label: Optional[str] = None,
    x_label: Optional[str] = None,
) -> None:
    """Render trial-aligned spike raster on the specified pyqtgraph PlotWidget or PlotItem in-place.

    Follows § 8.2:
    - Light-pulse band (0–10 ms in COLOR_LIGHT_PULSE_BAND) and thin dashed t = 0 line
    - Raster ticks in dark ink (COLOR_TEXT)
    - Standard x-axis tick marks
    - No redundant plot title
    """
    p_item = _get_plot_item(plot_widget)
    cache = getattr(p_item, "_raster_cache", None)
    valid_cache = (
        cache is not None
        and cache.get("scatter") in p_item.items
        and cache.get("onset_line") in p_item.items
        and cache.get("pulse_band") in p_item.items
    )

    if not valid_cache:
        plot_widget.clear()
        apply_plot_style(plot_widget, compact=compact)

        pulse_band = _SafeLinearRegionItem(
            values=(0.0, max(0.0, float(pulse_duration_ms))),
            orientation="vertical",
            brush=pg.mkBrush(*rgba_from_hex(COLOR_LIGHT_PULSE_BAND, 255)),
            pen=pg.mkPen(None),
            movable=False,
        )
        pulse_band.setZValue(-20)
        plot_widget.addItem(pulse_band)

        onset_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            pen=pg.mkPen(COLOR_ACCENT, width=1.0, style=Qt.PenStyle.DashLine),
        )
        onset_line.setZValue(10)
        plot_widget.addItem(onset_line)

        scatter = pg.ScatterPlotItem(
            size=marker_size,
            symbol="|",
            pen=pg.mkPen(COLOR_TEXT, width=1.0),
            brush=pg.mkBrush(COLOR_TEXT),
            pxMode=True,
        )
        scatter.setZValue(5)
        plot_widget.addItem(scatter)

        cache = {
            "pulse_band": pulse_band,
            "onset_line": onset_line,
            "scatter": scatter,
        }
        p_item._raster_cache = cache
    else:
        pulse_band = cache["pulse_band"]
        onset_line = cache["onset_line"]
        scatter = cache["scatter"]

    pulse_band.setRegion((0.0, max(0.0, float(pulse_duration_ms))))

    rx = np.asarray(raster_x, dtype=np.float64)
    ry = np.asarray(raster_y, dtype=np.float64)

    if len(rx) > 0:
        scatter.setData(x=rx, y=ry, size=marker_size)
        scatter.setVisible(True)
    else:
        scatter.setData(x=np.array([]), y=np.array([]))
        scatter.setVisible(False)

    onset_line.setValue(0.0)
    onset_line.setVisible(True)

    label_style = PLOT_AXIS_LABEL_COMPACT_STYLE if compact else PLOT_AXIS_LABEL_STYLE
    if y_label is not None:
        resolved_y_label = y_label
    elif compact:
        resolved_y_label = "Trial #"
    else:
        resolved_y_label = (
            "Pulse Trial # (Latency Order)"
            if order_by == "latency"
            else "Pulse Trial # (Presentation Order)"
        )
    plot_widget.setLabel("left", resolved_y_label, **label_style)
    if y_axis_width is not None:
        plot_widget.getAxis("left").setWidth(y_axis_width)

    if show_x_axis:
        resolved_x_label = x_label if x_label is not None else "Time from Pulse Onset (ms)"
        plot_widget.setLabel("bottom", resolved_x_label, **label_style)
        plot_widget.showAxis("bottom", True)
        plot_widget.getAxis("bottom").setTicks(
            compute_standard_plot_ticks(x_range[0], x_range[1], compact=compact)
        )
    else:
        plot_widget.hideAxis("bottom")

    if n_trials > 0:
        plot_widget.setYRange(-0.5, max(1.0, float(n_trials - 0.5)), padding=0.02)

    plot_widget.setXRange(x_range[0], x_range[1], padding=0.0)
    plot_widget.getViewBox().disableAutoRange(pg.ViewBox.XAxis)
    plot_widget.setTitle(None)


def render_psth(
    plot_widget: Union[pg.PlotWidget, pg.PlotItem],
    psth_bins: np.ndarray,
    psth_rates: np.ndarray,
    baseline_mean: float = 0.0,
    baseline_3sigma: float = 0.0,
    title: Optional[str] = None,
    x_range: Tuple[float, float] = (-30.0, 50.0),
    show_x_axis: bool = True,
    y_axis_width: Optional[int] = DEFAULT_Y_AXIS_WIDTH,
    is_smooth: bool = True,
    pulse_duration_ms: float = DEFAULT_PULSE_DURATION_MS,
    compact: bool = False,
    color_hex: Optional[str] = None,
    y_label: Optional[str] = None,
    x_label: Optional[str] = None,
    y_min_top: float = 10.0,
) -> None:
    """Render peristimulus time histogram (PSTH) in-place.

    Follows § 8.2:
    - Light-pulse band (0–10 ms in COLOR_LIGHT_PULSE_BAND) and thin dashed t = 0 line
    - PSTH line 1.5 px with light fill
    - No baseline mean or 3σ lines (baseline_line is kept hidden in _psth_cache for API compatibility)
    - Standard x-axis tick marks
    - No redundant plot title
    """
    stroke_hex = color_hex if (color_hex and str(color_hex).strip()) else COLOR_ACCENT
    fill_rgba = (
        rgba_from_hex(stroke_hex, 45)
        if (color_hex and str(color_hex).strip())
        else PLOT_HIST_BRUSH
    )

    p_item = _get_plot_item(plot_widget)
    cache = getattr(p_item, "_psth_cache", None)
    valid_cache = (
        cache is not None
        and cache.get("pulse_band") in p_item.items
        and cache.get("onset_line") in p_item.items
        and cache.get("baseline_line") in p_item.items
        and cache.get("smooth_curve") in p_item.items
        and cache.get("bar_item") in p_item.items
    )

    if not valid_cache:
        plot_widget.clear()
        apply_plot_style(plot_widget, compact=compact)

        pulse_band = _SafeLinearRegionItem(
            values=(0.0, max(0.0, float(pulse_duration_ms))),
            orientation="vertical",
            brush=pg.mkBrush(*rgba_from_hex(COLOR_LIGHT_PULSE_BAND, 255)),
            pen=pg.mkPen(None),
            movable=False,
        )
        pulse_band.setZValue(-20)
        plot_widget.addItem(pulse_band)

        onset_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            pen=pg.mkPen(COLOR_ACCENT, width=1.0, style=Qt.PenStyle.DashLine),
        )
        onset_line.setZValue(10)
        plot_widget.addItem(onset_line)

        smooth_curve = pg.PlotCurveItem(
            pen=pg.mkPen(stroke_hex, width=PSTH_LINE_WIDTH),
            fillLevel=0.0,
            brush=pg.mkBrush(*fill_rgba),
        )
        smooth_curve.setZValue(5)
        plot_widget.addItem(smooth_curve)

        bar_item = pg.BarGraphItem(
            x0=np.array([]),
            x1=np.array([]),
            height=np.array([]),
            brush=pg.mkBrush(*fill_rgba),
            pen=pg.mkPen(stroke_hex, width=0.8),
        )
        bar_item.setZValue(3)
        plot_widget.addItem(bar_item)

        # Retained hidden in _psth_cache for backward compatibility while strictly
        # enforcing § 8.2 (no baseline mean or 3σ lines in any PSTH).
        baseline_line = pg.InfiniteLine(
            pos=0.0,
            angle=0,
            pen=pg.mkPen(None),
        )
        baseline_line.setVisible(False)
        plot_widget.addItem(baseline_line)

        cache = {
            "pulse_band": pulse_band,
            "onset_line": onset_line,
            "baseline_line": baseline_line,
            "smooth_curve": smooth_curve,
            "bar_item": bar_item,
        }
        p_item._psth_cache = cache
    else:
        pulse_band = cache["pulse_band"]
        onset_line = cache["onset_line"]
        baseline_line = cache["baseline_line"]
        smooth_curve = cache["smooth_curve"]
        bar_item = cache["bar_item"]

    pulse_band.setRegion((0.0, max(0.0, float(pulse_duration_ms))))
    smooth_curve.setPen(pg.mkPen(stroke_hex, width=PSTH_LINE_WIDTH))
    smooth_curve.setBrush(pg.mkBrush(*fill_rgba))

    rates = np.asarray(psth_rates, dtype=np.float64)
    bins = np.asarray(psth_bins, dtype=np.float64)

    label_style = PLOT_AXIS_LABEL_COMPACT_STYLE if compact else PLOT_AXIS_LABEL_STYLE
    resolved_y_label = y_label if y_label is not None else (None if compact else "Firing Rate (Hz)")
    if resolved_y_label is not None:
        plot_widget.setLabel("left", resolved_y_label, **label_style)
    else:
        plot_widget.setLabel("left", None)
    if y_axis_width is not None:
        plot_widget.getAxis("left").setWidth(y_axis_width)
    if show_x_axis:
        resolved_x_label = x_label if x_label is not None else (None if compact else "Time from Pulse Onset (ms)")
        if resolved_x_label is not None:
            plot_widget.setLabel("bottom", resolved_x_label, **label_style)
        else:
            plot_widget.setLabel("bottom", None)
        plot_widget.showAxis("bottom", True)
        plot_widget.getAxis("bottom").setTicks(
            compute_standard_plot_ticks(x_range[0], x_range[1], compact=compact)
        )
    else:
        plot_widget.hideAxis("bottom")

    max_rate = float(np.max(rates)) if len(rates) > 0 else float(y_min_top)
    y_top = max(float(y_min_top), max_rate * 1.15)
    plot_widget.setYRange(0.0, y_top, padding=0.02)
    left_ax = plot_widget.getAxis("left")
    if left_ax is not None:
        left_ax.setTicks(compute_clean_y_ticks(y_top, max_ticks=3))

    onset_line.setValue(0.0)
    onset_line.setVisible(True)

    if is_smooth:
        bar_item.setVisible(False)
        if len(rates) > 0:
            if len(bins) == len(rates) + 1:
                x_pts = (bins[:-1] + bins[1:]) / 2.0
            else:
                x_pts = bins
            smooth_curve.setData(x=x_pts, y=rates)
            smooth_curve.setVisible(True)
        else:
            smooth_curve.setData(x=np.array([]), y=np.array([]))
            smooth_curve.setVisible(False)
    else:
        smooth_curve.setVisible(False)
        if len(rates) > 0 and len(bins) == len(rates) + 1:
            bar_item.setOpts(
                x0=bins[:-1],
                x1=bins[1:],
                height=rates,
                brush=pg.mkBrush(*fill_rgba),
                pen=pg.mkPen(COLOR_ACCENT_HOVER if stroke_hex == COLOR_ACCENT else stroke_hex, width=0.5),
            )
            bar_item.setVisible(True)
        else:
            bar_item.setOpts(x0=np.array([]), x1=np.array([]), height=np.array([]))
            bar_item.setVisible(False)

    if baseline_3sigma > 0:
        baseline_line.setValue(baseline_3sigma)
    baseline_line.setVisible(False)

    plot_widget.setXRange(x_range[0], x_range[1], padding=0.0)
    plot_widget.getViewBox().disableAutoRange(pg.ViewBox.XAxis)
    plot_widget.setTitle(None)


def build_raster_plot(
    plot: Union[pg.PlotWidget, pg.PlotItem],
    spike_times_ms: np.ndarray,
    trial_indices: np.ndarray,
    window_ms: Tuple[float, float] = (-30.0, 50.0),
    n_trials: Optional[int] = None,
    compact: bool = False,
    marker_size: float = 5.0,
    show_x_label: bool = True,
    pulse_duration_ms: float = DEFAULT_PULSE_DURATION_MS,
) -> None:
    """Convenience wrapper around render_raster()."""
    inferred_trials = (
        int(n_trials)
        if (n_trials is not None and n_trials > 0)
        else (int(np.max(trial_indices)) + 1 if (trial_indices is not None and len(trial_indices) > 0) else 20)
    )
    render_raster(
        plot_widget=plot,
        raster_x=spike_times_ms if spike_times_ms is not None else np.array([]),
        raster_y=trial_indices if trial_indices is not None else np.array([]),
        n_trials=inferred_trials,
        x_range=window_ms,
        show_x_axis=show_x_label,
        marker_size=marker_size,
        y_axis_width=None,
        pulse_duration_ms=pulse_duration_ms,
        compact=compact,
    )


def build_psth_plot(
    plot: Union[pg.PlotWidget, pg.PlotItem],
    bin_centers_ms: np.ndarray,
    firing_rate_hz: np.ndarray,
    bin_size_ms: float = 1.0,
    window_ms: Tuple[float, float] = (-30.0, 50.0),
    baseline_rate_hz: Optional[float] = None,
    compact: bool = False,
    smooth: bool = True,
    gaussian_sigma_ms: float = 2.0,
    color_hex: Optional[str] = None,
    pulse_duration_ms: float = DEFAULT_PULSE_DURATION_MS,
) -> None:
    """Convenience wrapper around render_psth()."""
    if bin_centers_ms is not None and len(bin_centers_ms) > 0:
        centers = np.asarray(bin_centers_ms, dtype=np.float64)
        half = float(bin_size_ms) / 2.0
        bins = np.concatenate([centers - half, [centers[-1] + half]])
        rates = np.asarray(firing_rate_hz, dtype=np.float64)
        if smooth and len(rates) > 1:
            try:
                from scipy.ndimage import gaussian_filter1d
                sigma_bins = max(0.1, float(gaussian_sigma_ms) / max(0.1, float(bin_size_ms)))
                rates = gaussian_filter1d(rates, sigma=sigma_bins, mode="nearest")
            except Exception:
                pass
    else:
        bins = np.array([])
        rates = np.array([])
    render_psth(
        plot_widget=plot,
        psth_bins=bins,
        psth_rates=rates,
        baseline_mean=float(baseline_rate_hz or 0.0),
        baseline_3sigma=0.0,
        x_range=window_ms,
        show_x_axis=True,
        y_axis_width=None,
        is_smooth=smooth,
        pulse_duration_ms=pulse_duration_ms,
        compact=compact,
        color_hex=color_hex,
    )


def format_unit_display(
    session_id: str = "",
    unit_id: Any = None,
    global_unit_id: str = "",
) -> Tuple[str, str]:
    """Parse and format session_id and unit_id cleanly without underscore concatenation."""
    sess_str = str(session_id).strip() if session_id is not None and str(session_id).strip() != "None" else ""
    uid_str = ""
    if unit_id is not None and str(unit_id).strip() and str(unit_id).strip() != "None":
        try:
            uid_str = str(int(unit_id))
        except (ValueError, TypeError):
            uid_str = str(unit_id).strip()

    gid_str = str(global_unit_id).strip() if global_unit_id is not None else ""

    if (not sess_str or not uid_str) and gid_str:
        m_u = re.match(r"^(.*)_u(\d+)$", gid_str)
        if m_u:
            if not sess_str:
                sess_str = m_u.group(1)
            if not uid_str:
                uid_str = str(int(m_u.group(2)))
        else:
            m_d = re.match(r"^(.*)_(\d+)$", gid_str)
            if m_d:
                if not sess_str:
                    sess_str = m_d.group(1)
                if not uid_str:
                    uid_str = str(int(m_d.group(2)))
            elif not sess_str:
                sess_str = gid_str

    if not sess_str:
        sess_str = "Unknown"
    if not uid_str:
        uid_str = "—"

    return sess_str, uid_str


class UnitResponsePlot(QWidget):
    """Reusable fixed-size stacked Raster + PSTH widget for Clustering mode (§ 8, § 8.1).

    Used identically for both the Focused Unit panel and Inspector member rows so
    they are pixel-identical:
    - Raster: 232 x 110 px data area + 36 px left margin = 268 x 110 px
    - Vertical gap: 6 px
    - PSTH: 232 x 56 px data area + 36 px left margin (+ bottom axis)
    - Light-pulse band (0–10 ms) and thin dashed t = 0 line; no baseline/3-sigma lines
    - Live rendering from spike arrays / psth_data (never loads static PNGs)
    """

    PSTH_AXIS_HEIGHT: int = 34

    @classmethod
    def fixed_plot_size(cls) -> QSize:
        """Return exact fixed outer dimensions matching every UnitResponsePlot instance."""
        total_w = PLOT_UNIT_RESPONSE_RASTER_SIZE[0] + PLOT_UNIT_RESPONSE_LEFT_MARGIN
        raster_h = PLOT_UNIT_RESPONSE_RASTER_SIZE[1]
        psth_h = PLOT_UNIT_RESPONSE_PSTH_SIZE[1] + cls.PSTH_AXIS_HEIGHT
        total_h = raster_h + PLOT_UNIT_RESPONSE_GAP + psth_h
        return QSize(total_w, total_h)

    def __init__(self, parent: Optional[QWidget] = None, defer_initial_render: bool = False) -> None:
        super().__init__(parent)
        total_w = PLOT_UNIT_RESPONSE_RASTER_SIZE[0] + PLOT_UNIT_RESPONSE_LEFT_MARGIN
        raster_h = PLOT_UNIT_RESPONSE_RASTER_SIZE[1]
        psth_h = PLOT_UNIT_RESPONSE_PSTH_SIZE[1] + self.PSTH_AXIS_HEIGHT
        total_h = raster_h + PLOT_UNIT_RESPONSE_GAP + psth_h

        self.setFixedSize(total_w, total_h)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(PLOT_UNIT_RESPONSE_GAP)

        self.raster_plot = pg.PlotWidget(self)
        self.raster_plot.setFixedSize(total_w, raster_h)
        self.raster_plot.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.raster_plot.setMouseEnabled(x=False, y=False)
        apply_plot_style(
            self.raster_plot,
            compact=True,
            left_label="Trial",
            show_bottom_values=False,
            left_margin=PLOT_UNIT_RESPONSE_LEFT_MARGIN,
        )
        self.raster_plot.hideAxis("bottom")

        self.psth_plot = pg.PlotWidget(self)
        self.psth_plot.setFixedSize(total_w, psth_h)
        self.psth_plot.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.psth_plot.setMouseEnabled(x=False, y=False)
        self.psth_plot.setXLink(self.raster_plot)
        self.psth_plot.setToolTip("Firing rate (Hz)")
        apply_plot_style(
            self.psth_plot,
            compact=True,
            left_label="Rate (Hz)",
            bottom_label="Time from pulse onset (ms)",
            show_bottom_values=True,
            left_margin=PLOT_UNIT_RESPONSE_LEFT_MARGIN,
        )

        layout.addWidget(self.raster_plot)
        layout.addWidget(self.psth_plot)

        if not defer_initial_render:
            self.clear_preview()

    def set_psth_tooltip(self, text: str) -> None:
        """Update the PSTH plot tooltip (e.g. to reflect bin size or normalisation)."""
        self.psth_plot.setToolTip(str(text) if text else "Firing rate (Hz)")

    def set_data(
        self,
        spike_times_ms: Optional[np.ndarray],
        trial_indices: Optional[np.ndarray],
        psth_bins: Optional[np.ndarray] = None,
        psth_rates: Optional[np.ndarray] = None,
        n_trials: Optional[int] = None,
        window_ms: Tuple[float, float] = (-30.0, 50.0),
        order_by: str = "presentation",
        marker_size: float = 3.0,
        is_smooth: bool = True,
        pulse_duration_ms: float = DEFAULT_PULSE_DURATION_MS,
        as_probability: bool = False,
    ) -> None:
        """Render live spike raster and PSTH from in-memory spike/PSTH arrays."""
        rx = np.asarray(spike_times_ms, dtype=np.float64) if spike_times_ms is not None else np.array([], dtype=np.float64)
        ry = np.asarray(trial_indices, dtype=np.float64) if trial_indices is not None else np.array([], dtype=np.float64)

        inferred_trials = (
            int(n_trials)
            if (n_trials is not None and n_trials > 0)
            else (int(np.max(ry)) + 1 if len(ry) > 0 else 0)
        )

        render_raster(
            self.raster_plot,
            raster_x=rx,
            raster_y=ry,
            n_trials=inferred_trials,
            title=None,
            x_range=window_ms,
            order_by=order_by,
            show_x_axis=False,
            marker_size=marker_size,
            y_axis_width=PLOT_UNIT_RESPONSE_LEFT_MARGIN,
            pulse_duration_ms=pulse_duration_ms,
            compact=True,
            y_label="Trial",
        )

        if psth_bins is not None and psth_rates is not None:
            bins = np.asarray(psth_bins, dtype=np.float64)
            vals = np.asarray(psth_rates, dtype=np.float64)
            if as_probability and len(bins) >= 2 and len(vals) > 0:
                dt_ms = float(np.median(np.diff(bins)))
                if dt_ms > 0:
                    vals = vals * (dt_ms / 1000.0)
        elif len(rx) > 0 and inferred_trials > 0:
            bin_w = 2.0
            bins = np.arange(float(window_ms[0]), float(window_ms[1]) + bin_w * 0.5, bin_w)
            counts, _ = np.histogram(rx, bins=bins)
            if as_probability:
                vals = counts.astype(np.float64) / float(inferred_trials)
            else:
                vals = counts.astype(np.float64) / (float(inferred_trials) * (bin_w / 1000.0))
        else:
            bins = np.array([], dtype=np.float64)
            vals = np.array([], dtype=np.float64)

        render_psth(
            self.psth_plot,
            psth_bins=bins,
            psth_rates=vals,
            baseline_mean=0.0,
            baseline_3sigma=0.0,
            title=None,
            x_range=window_ms,
            show_x_axis=True,
            y_axis_width=PLOT_UNIT_RESPONSE_LEFT_MARGIN,
            is_smooth=is_smooth,
            pulse_duration_ms=pulse_duration_ms,
            compact=True,
            y_label="Prob./bin" if as_probability else "Rate (Hz)",
            x_label="Time from pulse onset (ms)",
            y_min_top=0.1 if as_probability else 10.0,
        )

    def set_psth_data(
        self,
        psth_data: Optional[Dict[str, Any]],
        window_ms: Tuple[float, float] = (-30.0, 50.0),
        order_by: str = "presentation",
        marker_size: float = 3.0,
        pulse_duration_ms: float = DEFAULT_PULSE_DURATION_MS,
        as_probability: bool = False,
    ) -> None:
        """Render from a compute_psth_and_raster() result dict, or clear gracefully if None."""
        if not psth_data:
            self.clear_preview(window_ms=window_ms, pulse_duration_ms=pulse_duration_ms)
            return

        self.set_data(
            spike_times_ms=psth_data.get("raster_x"),
            trial_indices=psth_data.get("raster_y"),
            psth_bins=psth_data.get("psth_bins"),
            psth_rates=psth_data.get("psth_rates"),
            n_trials=psth_data.get("n_trials", 0),
            window_ms=window_ms,
            order_by=order_by,
            marker_size=marker_size,
            is_smooth=bool(psth_data.get("is_smooth", True)),
            pulse_duration_ms=pulse_duration_ms,
            as_probability=as_probability,
        )

    def clear_preview(
        self,
        window_ms: Tuple[float, float] = (-30.0, 50.0),
        pulse_duration_ms: float = DEFAULT_PULSE_DURATION_MS,
    ) -> None:
        """Reset raster and PSTH to empty state while preserving fixed axes and styling."""
        self.set_data(
            spike_times_ms=np.array([], dtype=np.float64),
            trial_indices=np.array([], dtype=np.float64),
            psth_bins=np.array([], dtype=np.float64),
            psth_rates=np.array([], dtype=np.float64),
            n_trials=0,
            window_ms=window_ms,
            pulse_duration_ms=pulse_duration_ms,
        )


class SiliconProbeWidget(QWidget):
    """Reusable fixed-size silicon probe plot widget (§ 7.3, § 8.1)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedSize(*PLOT_PROBE_SIZE)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.plot_widget = pg.PlotWidget(self)
        self.plot_widget.setFixedSize(*PLOT_PROBE_SIZE)
        self.plot_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        apply_plot_style(
            self.plot_widget,
            compact=True,
            left_label="Depth (µm)",
        )
        left_ax = self.plot_widget.getAxis("left")
        if left_ax is not None:
            left_ax.enableAutoSIPrefix(False)
            left_ax.setStyle(
                tickFont=make_font(FONT_SIZE_TICK),
                tickTextOffset=2,
                tickLength=-3,
                autoExpandTextSpace=True,
            )
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.getPlotItem().setContentsMargins(2, 4, 2, 14)
        layout.addWidget(self.plot_widget)

        self.sites_scatter = pg.ScatterPlotItem(
            size=5,
            symbol="s",
            brush=pg.mkBrush(COLOR_PROBE_SITE),
            pen=None,
        )
        self.sites_scatter.setZValue(5)
        self.plot_widget.addItem(self.sites_scatter)

        self.units_scatter = pg.ScatterPlotItem(
            size=9,
            symbol="o",
            brush=pg.mkBrush(COLOR_ACCENT),
            pen=pg.mkPen(COLOR_PANEL, width=0.8),
            hoverable=True,
        )
        self.units_scatter.setZValue(10)
        self.plot_widget.addItem(self.units_scatter)

        self.cand_line = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen(COLOR_CANDIDATE_MARKER, width=1.5, style=Qt.PenStyle.DashLine),
        )
        self.cand_line.setZValue(15)
        self.plot_widget.addItem(self.cand_line)

        self.cand_marker = pg.ScatterPlotItem(
            size=12,
            symbol="d",
            brush=pg.mkBrush(COLOR_CANDIDATE_MARKER),
            pen=pg.mkPen(COLOR_PANEL, width=1.2),
        )
        self.cand_marker.setZValue(20)
        self.plot_widget.addItem(self.cand_marker)

    @property
    def plotItem(self) -> pg.PlotItem:
        """Compatibility alias returning the underlying PlotItem."""
        return self.plot_widget.getPlotItem()


