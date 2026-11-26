"""Response Clustering & Active Learning Workspace."""

import atexit
import html
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import weakref
import numpy as np
import pandas as pd
import pyqtgraph as pg
from scipy.stats import gaussian_kde
from PyQt6.QtCore import (
    QEasingCurve,
    QEvent,
    QItemSelectionModel,
    QMimeData,
    QObject,
    QParallelAnimationGroup,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QRectF,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QBrush,
    QColor,
    QCursor,
    QDrag,
    QFont,
    QFontMetrics,
    QIcon,
    QKeyEvent,
    QKeySequence,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
    QPolygonF,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsPolygonItem,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from optocurator.analysis.clustering import (
    ActiveLearningCandidate,
    AffinityPropagationParameters,
    CLUSTER_PALETTE,
    COLOR_UNASSIGNED,
    ClusteringParameters,
    ClusteringResult,
    ClusterSummary,
    EmbeddingParameters,
    FeatureConfiguration,
    color_for_cluster,
    compute_2d_embedding,
    compute_cluster_summaries,
    compute_pca_embedding,
    extract_feature_matrix,
    generate_psth_matrix,
    normalize_psth_waveform,
    perform_clustering,
    run_active_learning_cycle,
    run_clustering_algorithm,
    run_hierarchical_clustering,
    seed_clusters_affinity_propagation,
)
from optocurator.io import (
    ClusteringProjectData,
    ExperimentRef,
    InvalidPhase1ExportError,
    InvalidProjectConfigError,
    MatNpyStimulusReader,
    MissingExperimentDirectoryError,
    MissingPhase1ExportError,
    Phase1ExportMetadata,
    PhyKilosortReader,
    ProjectError,
    ProjectNotFoundError,
    UnitAssignmentRecord,
    compute_portable_path,
    deserialize_model_parameters,
    load_clustering_project,
    load_phase1_session_arrays,
    resolve_experiment_path,
    save_clustering_project,
    serialize_model_parameters,
    validate_phase1_export,
)
from optocurator.analysis.metrics import compute_psth_and_raster
from optocurator.models import MISSING_METRIC_SENTINEL, is_valid_metric_value
from optocurator.ui.active_learning_worker import ActiveLearningWorker


_ACTIVE_AL_WORKERS: Set[ActiveLearningWorker] = set()


def _wait_active_al_workers_at_exit() -> None:
    """Wait briefly for any detached ActiveLearningWorker threads before interpreter teardown."""
    for worker in list(_ACTIVE_AL_WORKERS):
        try:
            if worker.isRunning():
                worker.wait(5000)
        except Exception:
            pass
        _ACTIVE_AL_WORKERS.discard(worker)


atexit.register(_wait_active_al_workers_at_exit)
from optocurator.ui.detection_widgets import install_combo_chevron
from optocurator.ui.dialog_clustering import (
    EmbeddingSettingsDialog,
    HierarchicalClusteringDialog,
)
from optocurator.ui.dialog_clustering_settings import (
    ActiveLearningSettingsDialog,
    ConfidenceThresholdDialog,
    ResetClusteringDialog,
    RevertAutoAssignmentsDialog,
    VisualizationSettingsDialog,
)
from optocurator.ui.dialog_clusters_customize import CustomizeClustersDialog, normalize_hotkey
from optocurator.ui.dialog_feature_settings import FeatureSettingsDialog
from optocurator.ui.dialog_seed_clusters import SeedClustersDialog
from optocurator.ui.figure_export import (
    install_figure_export_menu,
    populate_figure_export_menu,
)
from optocurator.ui.plots import (
    DEFAULT_PULSE_DURATION_MS,
    NumericTableWidgetItem,
    SiliconProbeWidget,
    UnitResponsePlot,
    _SafeLinearRegionItem,
    apply_plot_style,
    compute_clean_y_ticks,
    compute_standard_plot_ticks,
    format_unit_display as _format_unit_display,
    infer_pulse_duration_ms,
)
from optocurator.ui.splitters import ResettableSplitter
from optocurator.ui.theme import (
    CARD_GRID_SCROLL_STYLE,
    CENTRE_PANEL_WIDTH,
    CLUSTERS_ACCEPT_BTN_STYLE,
    CLUSTERS_AREA_PANEL_STYLE,
    CLUSTER_CARD_COUNT_STYLE,
    CLUSTER_CARD_DRAG_MERGE_STYLE,
    CLUSTER_CARD_DRAG_MOVE_STYLE,
    CLUSTER_CARD_KEYCAP_HOVER_STYLE,
    CLUSTER_CARD_KEYCAP_STYLE,
    CLUSTER_CARD_MENU_DANGER_BTN_STYLE,
    CLUSTER_CARD_PROB_BADGE_DEFAULT_STYLE,
    CLUSTER_CARD_SIZE,
    CLUSTER_CARD_SUBTITLE_STYLE,
    CLUSTER_CARD_SUGGESTED_BADGE_STYLE,
    CLUSTER_GRID_MIN_HEIGHT,
    CLUSTER_GRID_SPACING,
    CLUSTERS_CANDIDATE_LABEL_STYLE,
    CLUSTERS_COMPARISON_CLOSE_BTN_STYLE,
    CLUSTERS_COMPARISON_EMPTY_LABEL_STYLE,
    CLUSTERS_COMPARISON_PANE_DEFAULT_STYLE,
    CLUSTERS_COMPARISON_SCROLL_STYLE,
    CLUSTERS_COUNT_CHIP_STYLE,
    CLUSTERS_CURATION_BTN_SKIP_STYLE,
    CLUSTERS_CURATION_BTN_UNDO_STYLE,
    CLUSTERS_FILTER_COMBO_MAX_POPUP_HEIGHT,
    CLUSTERS_FILTER_COMBO_MAX_VISIBLE,
    CLUSTERS_FILTER_COMBO_STYLE,
    CLUSTERS_GHOST_ICON_BTN_STYLE,
    CLUSTERS_HELP_BTN_STYLE,
    CLUSTERS_INSPECTOR_SPLITTER_STYLE,
    CLUSTERS_MEMBERS_SECTION_STYLE,
    CLUSTERS_MENU_STYLE,
    CLUSTERS_POOLED_BADGE_STYLE,
    CLUSTERS_PROBE_EXP_LABEL_STYLE,
    CLUSTERS_PROBE_INFO_LABEL_STYLE,
    CLUSTERS_PROGRESS_BAR_CONTAINER_STYLE,
    CLUSTERS_PROGRESS_LEGEND_STYLE,
    CLUSTERS_SECTION_SUBHEADER_STYLE,
    CLUSTERS_SHORTCUTS_HELP_TEXT_STYLE,
    CLUSTERS_SUGGESTED_TOOLBAR_BTN_PRIMARY_STYLE,
    CLUSTERS_TOOLBAR_ACTION_BTN_STYLE,
    CLUSTERS_TOOLBAR_COMBO_STYLE,
    CLUSTERS_TOOLBAR_DANGER_BTN_STYLE,
    CLUSTERS_TOOLBAR_DIVIDER_STYLE,
    CLUSTERS_TOOLBAR_ICON_BTN_STYLE,
    CLUSTERS_TOOLBAR_PRIMARY_BTN_STYLE,
    CLUSTERS_TOOLBAR_RESET_BTN_STYLE,
    CLUSTERS_TOOLBAR_SECONDARY_BTN_STYLE,
    CLUSTERS_TOOLBAR_SLIDER_STYLE,
    CLUSTERS_TOOLBAR_SPIN_STYLE,
    CLUSTERS_TOOLBAR_SUBLABEL_STYLE,
    CLUSTERS_TRANSPARENT_BG_STYLE,
    CLUSTERS_UNIT_COUNT_LABEL_STYLE,
    CLUSTERS_VERTICAL_DIVIDER_STYLE,
    CLUSTERS_WORKFLOW_HELP_TEXT_STYLE,
    COLOR_ACCENT,
    COLOR_ACCENT_BG,
    COLOR_AUTO_ASSIGNED,
    COLOR_AXIS,
    COLOR_CANDIDATE_MARKER,
    COLOR_CANVAS,
    COLOR_DANGER,
    COLOR_DIVIDER,
    COLOR_LIGHT_PULSE_BAND,
    COLOR_PANEL,
    COLOR_PASSING_FILL,
    COLOR_POPULATION_CURVE,
    COLOR_PROBE_SHANK_BG,
    COLOR_PROBE_SHANK_BORDER,
    COLOR_PROBE_SITE,
    COLOR_SEEDED,
    COLOR_SUCCESS,
    COLOR_SURFACE_MUTED,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_SECONDARY,
    COLOR_WARNING,
    ClusterColorService,
    DIALOG_GROUP_BOX_STYLE,
    DIALOG_LINE_EDIT_STYLE,
    DIALOG_MODAL_STYLE,
    DIALOG_SECTION_TITLE_STYLE,
    EMBEDDING_DIM_ALPHA,
    EMBEDDING_FOCUSED_RING_SIZE,
    EMBEDDING_FOCUSED_RING_WIDTH,
    EMBEDDING_POINT_EMPHASIS_SIZE,
    EMBEDDING_POINT_SIZE,
    FOCUSED_UNIT_CONTENT_WIDTH,
    FOCUSED_UNIT_HEADER_HEIGHT,
    FOCUSED_UNIT_LEGEND_STYLE,
    FOCUSED_UNIT_PANEL_STYLE,
    FOCUSED_UNIT_SESSION_STYLE,
    FOCUSED_UNIT_SUGGESTED_BTN_STYLE,
    FOCUSED_UNIT_TITLE_STYLE,
    FOCUSED_UNIT_UNASSIGN_BTN_STYLE,
    FOCUSED_UNIT_WIDTH,
    FONT_SIZE_BODY,
    FONT_SIZE_PANEL_TITLE,
    FONT_SIZE_TICK,
    FONT_SIZE_UNIT_TITLE,
    INSPECTOR_CLOSE_BTN_SIZE,
    INSPECTOR_EMPTY_LABEL_STYLE,
    INSPECTOR_EMPTY_STATE_STYLE,
    INSPECTOR_HEADER_CID_STYLE,
    INSPECTOR_HEADER_HEIGHT,
    INSPECTOR_HEADER_NAME_MAX_WIDTH,
    INSPECTOR_HEADER_NAME_STYLE,
    INSPECTOR_HEADER_SUMMARY_STYLE,
    INSPECTOR_SLOT_PANEL_STYLE,
    INSPECTOR_SLOT_WIDTH,
    INSPECTOR_SWATCH_SIZE,
    LIST_PANEL_CHEVRON_BTN_STYLE,
    LIST_PANEL_COUNT_CHIP_STYLE,
    LIST_PANEL_HEADER_STYLE,
    LIST_PANEL_HINT_STYLE,
    LIST_PANEL_MIN_HEIGHT,
    LIST_PANEL_STYLE,
    LIST_PANEL_TABLE_STYLE,
    CLUSTERS_POOLED_TABLE_STYLE,
    CLUSTERS_COMPARISON_PANE_NEUTRAL_STYLE,
    NEW_CLUSTER_CARD_KEYCAP_STYLE,
    NEW_CLUSTER_CARD_LABEL_STYLE,
    NEW_CLUSTER_CARD_STYLE,
    PLOT_AXIS_LABEL_COMPACT_STYLE,
    PLOT_CARD_PSTH_LEFT_MARGIN,
    PLOT_CARD_PSTH_SIZE,
    PLOT_CLUSTER_PROB_SIZE,
    PLOT_EMBEDDING_SIZE,
    PLOT_METRIC_KDE_SIZE,
    PLOT_PROBE_SIZE,
    PLOT_UNIT_RESPONSE_LEFT_MARGIN,
    PROBE_DIM_ALPHA,
    PROBE_UNIT_EMPHASIS_SIZE,
    PROBE_UNIT_FOCUSED_SIZE,
    PROBE_UNIT_SIZE,
    PROGRESS_DIALOG_STYLE,
    PSTH_LINE_WIDTH,
    RIGHT_PANEL_MIN_WIDTH,
    SALT_P_RESOLUTION_FLOOR,
    SECONDARY_BTN_STYLE,
    UNIT_LIST_DEFAULT_WIDTH,
    UNIT_LIST_MAX_WIDTH,
    UNIT_LIST_MIN_WIDTH,
    UNIT_LIST_PANEL_STYLE,
    UNIT_RESPONSE_MEMBER_CARD_ACTIVE_STYLE,
    UNIT_RESPONSE_MEMBER_CARD_STYLE,
    UNIT_RESPONSE_MEMBER_HEADER_STYLE,
    build_marginal_legend_html,
    UNIT_RESPONSE_MEMBER_META_STYLE,
    build_active_learning_button_style,
    build_classification_progress_summary_html,
    build_cluster_card_header_style,
    build_cluster_card_hint_style,
    build_cluster_card_prob_badge_style,
    build_cluster_card_widget_style,
    build_comparison_pane_style,
    build_comparison_title_style,
    build_inspector_swatch_style,
    build_list_panel_candidates_title_html,
    build_marginal_plot_title_html,
    build_progress_segment_style,
    make_font,
    make_mono_font,
    rgba_from_hex,
    text_color_for_cluster,
)


def _make_swatch_icon(color_hex: str, size: int = 10) -> QIcon:
    """Create a crisp rounded-square colour swatch QIcon."""
    pm = QPixmap(size, size)
    pm.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(color_hex)))
    painter.drawRoundedRect(QRectF(0.5, 0.5, size - 1.0, size - 1.0), 2.5, 2.5)
    painter.end()
    return QIcon(pm)


def _make_status_icon(status: str, size: int = 14) -> QIcon:
    """Create a crisp status QIcon (hollow grey circle, green check, violet dot, or blue dot)."""
    pm = QPixmap(size, size)
    pm.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    margin = 2.0
    rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)

    if status == "Curated":
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(COLOR_SUCCESS)))
        painter.drawEllipse(rect)
        pen = QPen(QColor(COLOR_PANEL), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(QPointF(size * 0.28, size * 0.52), QPointF(size * 0.44, size * 0.68))
        painter.drawLine(QPointF(size * 0.44, size * 0.68), QPointF(size * 0.72, size * 0.34))
    elif status == "Seeded":
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(COLOR_SEEDED)))
        painter.drawEllipse(rect.adjusted(1.0, 1.0, -1.0, -1.0))
    elif status == "Auto-assigned":
        painter.setPen(QPen(QColor(COLOR_ACCENT), 1.4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(rect)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(COLOR_ACCENT)))
        painter.drawEllipse(rect.adjusted(2.5, 2.5, -2.5, -2.5))
    else:
        painter.setPen(QPen(QColor(COLOR_UNASSIGNED), 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(rect.adjusted(0.5, 0.5, -0.5, -0.5))

    painter.end()
    return QIcon(pm)


def _short_session_id(session_id: str) -> str:
    """Return short session identifier (first '_'-separated token) for compact table display."""
    s = str(session_id or "").strip()
    if "_" in s:
        return s.split("_", 1)[0]
    return s


def _compute_clean_range_ticks(
    vmin: float,
    vmax: float,
    max_ticks: int = 3,
) -> List[List[Tuple[float, str]]]:
    """Compute 2 to `max_ticks` clean, non-overlapping axis ticks strictly inside [vmin, vmax]."""
    lo = float(vmin)
    hi = float(vmax)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return [[(0.0, "0")]]
    span = hi - lo
    # Keep a small inner margin so PyQtGraph does not clip edge tick labels.
    inner_lo = lo + 0.04 * span
    inner_hi = hi - 0.04 * span
    raw_step = span / max(1, max_ticks - 1)
    mag = 10.0 ** float(np.floor(np.log10(raw_step)))

    candidate_steps = [
        10.0 * mag,
        5.0 * mag,
        2.0 * mag,
        1.0 * mag,
        0.5 * mag,
        0.2 * mag,
        0.1 * mag,
    ]
    ticks: List[Tuple[float, str]] = []
    for step in candidate_steps:
        if step <= 0:
            continue
        start = float(np.ceil((inner_lo - 1e-12) / step) * step)
        cand: List[Tuple[float, str]] = []
        val = start
        while val <= inner_hi + 1e-9 * max(1.0, abs(inner_hi)):
            clean_v = 0.0 if abs(val) < 1e-12 else float(val)
            label = (
                "0"
                if clean_v == 0.0
                else (
                    str(int(round(clean_v)))
                    if abs(clean_v - round(clean_v)) < 1e-6
                    else f"{clean_v:.3g}"
                )
            )
            cand.append((clean_v, label))
            val += step
            if len(cand) > 12:
                break
        if len(cand) >= 2:
            if len(cand) > max_ticks:
                idxs = np.linspace(0, len(cand) - 1, max_ticks, dtype=int)
                cand = [cand[int(idx)] for idx in idxs]
            ticks = cand
            break

    if len(ticks) < 2:
        t0 = lo + 0.08 * span
        t1 = hi - 0.08 * span
        ticks = [
            (t0, f"{t0:.2g}"),
            (t1, f"{t1:.2g}"),
        ]
    return [ticks]


def _make_sliders_icon(size: int = 16, color_hex: str = COLOR_TEXT_SECONDARY) -> QIcon:
    """Paint a crisp 3-line horizontal sliders icon using QPainter."""
    pix = QPixmap(size, size)
    pix.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    col = QColor(color_hex)
    pen = QPen(col, 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)

    ys = [size * 0.24, size * 0.50, size * 0.76]
    thumb_xs = [size * 0.34, size * 0.68, size * 0.44]
    left_x = size * 0.14
    right_x = size * 0.86
    r = 2.0
    for y, tx in zip(ys, thumb_xs):
        painter.setPen(pen)
        painter.drawLine(QPointF(left_x, y), QPointF(right_x, y))
        painter.setBrush(QColor(COLOR_PANEL))
        painter.drawEllipse(QPointF(tx, y), r, r)
    painter.end()
    return QIcon(pix)


class MarginalDistributionsWidget(QWidget):
    """Mini population distribution curves with vertical marker tracking focused unit."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 4, 0, 4)
        outer_layout.setSpacing(4)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)

        self.lbl_title = QLabel("Response metrics")
        self.lbl_title.setStyleSheet(CLUSTERS_SECTION_SUBHEADER_STYLE)
        header_row.addWidget(self.lbl_title)
        header_row.addStretch(1)

        self.lbl_legend = QLabel()
        self.lbl_legend.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_legend.setStyleSheet(FOCUSED_UNIT_LEGEND_STYLE)
        self.lbl_legend.setText(build_marginal_legend_html())
        header_row.addWidget(self.lbl_legend)
        outer_layout.addLayout(header_row)

        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setHorizontalSpacing(14)
        grid_layout.setVerticalSpacing(6)
        outer_layout.addLayout(grid_layout)

        self.plots: Dict[str, pg.PlotWidget] = {}
        self.lines: Dict[str, pg.InfiniteLine] = {}
        self.curves: Dict[str, pg.PlotDataItem] = {}
        self.pop_curves: Dict[str, pg.PlotDataItem] = self.curves
        self.cluster_curves: Dict[str, pg.PlotDataItem] = {}
        self.value_items: Dict[str, pg.TextItem] = {}
        self.value_labels: Dict[str, QLabel] = {}
        self._pop_bounds: Dict[str, Tuple[float, float]] = {}

        metrics = [
            ("latency_ms", "Latency (ms)"),
            ("jitter_ms", "Jitter (ms)"),
            ("reliability", "Reliability"),
            ("salt_p_value", "SALT p"),
        ]

        for i, (key, label) in enumerate(metrics):
            r = i // 2
            c = i % 2

            cell = QWidget(self)
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(1)

            cell_header = QHBoxLayout()
            cell_header.setContentsMargins(0, 0, 0, 0)
            cell_header.setSpacing(4)
            lbl_name = QLabel(label, cell)
            lbl_name.setStyleSheet(FOCUSED_UNIT_LEGEND_STYLE)
            lbl_val = QLabel("—", cell)
            lbl_val.setStyleSheet(FOCUSED_UNIT_LEGEND_STYLE)
            lbl_val.setFont(make_font(FONT_SIZE_TICK, bold=True))
            cell_header.addWidget(lbl_name)
            cell_header.addStretch(1)
            cell_header.addWidget(lbl_val)
            cell_layout.addLayout(cell_header)

            p = pg.PlotWidget()
            p.setFixedSize(*PLOT_METRIC_KDE_SIZE)
            p.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            p.setMouseEnabled(x=False, y=False)
            apply_plot_style(p, compact=True)
            p.hideAxis("left")

            # Population curve (subtle gray)
            curve = p.plot(
                pen=pg.mkPen(COLOR_POPULATION_CURVE, width=1.5),
                fillLevel=0,
                brush=pg.mkBrush(*rgba_from_hex(COLOR_POPULATION_CURVE, 80)),
            )
            curve.setZValue(1)

            # Comparison cluster overlay curve (colored)
            cluster_curve = p.plot(
                pen=pg.mkPen(COLOR_ACCENT, width=2.0),
                fillLevel=0,
                brush=pg.mkBrush(*rgba_from_hex(COLOR_ACCENT, 60)),
            )
            cluster_curve.setZValue(2)

            # Focused unit vertical indicator line (solid dark)
            line = pg.InfiniteLine(
                angle=90,
                pen=pg.mkPen(COLOR_TEXT, width=1.8, style=Qt.PenStyle.SolidLine),
            )
            line.setZValue(10)
            line.setVisible(False)
            p.addItem(line)

            # Hidden TextItem kept in sync for API/test compatibility
            val_item = pg.TextItem(text="—", color=COLOR_TEXT, anchor=(1, 0))
            val_item.setVisible(False)
            p.addItem(val_item)

            self.plots[key] = p
            self.lines[key] = line
            self.curves[key] = curve
            self.cluster_curves[key] = cluster_curve
            self.value_items[key] = val_item
            self.value_labels[key] = lbl_val
            p.setYRange(0.0, 1.2, padding=0.0)
            install_figure_export_menu(
                p,
                export_target=self,
                default_name="marginal_distributions",
                label="Save marginal distributions as SVG…",
                subplot_target=p,
                subplot_default_name=f"marginal_{key}",
                subplot_label="Save this subplot as SVG…",
            )
            cell_layout.addWidget(p)
            grid_layout.addWidget(cell, r, c)

    @staticmethod
    def _format_metric_value(key: str, val: Optional[float]) -> str:
        """Format focused unit metric value for top-right mini-plot readout."""
        if val is None or not np.isfinite(val):
            return "—"
        if key in ("latency_ms", "jitter_ms"):
            return f"{val:.1f}"
        if key == "salt_p_value":
            if val <= SALT_P_RESOLUTION_FLOOR:
                return f"≤ {SALT_P_RESOLUTION_FLOOR:.2f}"
            return f"{val:.2f}"
        return f"{val:.2f}"

    def update_distributions(self, df: pd.DataFrame) -> None:
        """Update density curves based on current pooled population records."""
        if df.empty or len(df) == 0:
            for curve in self.curves.values():
                curve.setData([], [])
            return

        for key, curve in self.curves.items():
            vals = df[key].dropna().to_numpy(dtype=np.float64) if key in df.columns else np.array([], dtype=np.float64)
            if len(vals) == 0:
                curve.setData([], [])
                continue

            min_v, max_v = float(np.min(vals)), float(np.max(vals))
            if min_v == max_v:
                min_v -= 0.5
                max_v += 0.5
            span = max(max_v - min_v, 0.01)
            pad = span * 0.08
            pop_min, pop_max = min_v - pad, max_v + pad
            self._pop_bounds[key] = (pop_min, pop_max)
            grid = np.linspace(pop_min, pop_max, 100)

            if len(vals) >= 3:
                try:
                    kde = gaussian_kde(vals)
                    density = kde(grid)
                except Exception:
                    counts, bin_edges = np.histogram(vals, bins=np.linspace(pop_min, pop_max, 25), density=True)
                    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    density = np.interp(grid, centers, counts, left=0.0, right=0.0)
            else:
                counts, bin_edges = np.histogram(vals, bins=np.linspace(pop_min, pop_max, 20), density=True)
                centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                density = np.interp(grid, centers, counts, left=0.0, right=0.0)

            if len(density) > 0 and np.max(density) > 0:
                density = density / np.max(density)

            x = np.concatenate(([grid[0]], grid, [grid[-1]]))
            y = np.concatenate(([0.0], density, [0.0]))

            p = self.plots[key]
            p.setXRange(pop_min, pop_max, padding=0.0)
            p.setYRange(0.0, 1.2, padding=0.0)
            p.getAxis("bottom").setTicks(_compute_clean_range_ticks(pop_min, pop_max, max_ticks=3))
            curve.setData(x, y)
            if key in self.value_items:
                self.value_items[key].setPos(pop_max, 1.18)

    def update_cluster_distribution(
        self,
        cluster_df: Optional[pd.DataFrame],
        color_hex: str = COLOR_ACCENT,
        cluster_name: str = "",
    ) -> None:
        """Update cluster distribution overlay curves and legend."""
        if cluster_df is None or cluster_df.empty:
            self.clear_cluster_distribution()
            return

        self.lbl_legend.setText(
            build_marginal_legend_html(cluster_name=cluster_name or "Cluster", color_hex=color_hex)
        )

        for key, curve in self.cluster_curves.items():
            if key not in cluster_df.columns:
                curve.setData([], [])
                continue

            vals = cluster_df[key].dropna().to_numpy(dtype=np.float64)
            if len(vals) == 0:
                curve.setData([], [])
                continue

            pop_bounds = self._pop_bounds.get(key)
            if pop_bounds is not None:
                pop_min, pop_max = pop_bounds
            else:
                pop_min, pop_max = float(np.min(vals)) - 0.1, float(np.max(vals)) + 0.1
            grid = np.linspace(pop_min, pop_max, 100)

            if len(vals) >= 3 and np.std(vals) > 1e-6:
                try:
                    kde = gaussian_kde(vals)
                    density = kde(grid)
                except Exception:
                    counts, bin_edges = np.histogram(vals, bins=np.linspace(pop_min, pop_max, 25), density=True)
                    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    density = np.interp(grid, centers, counts, left=0.0, right=0.0)
            else:
                mean_v = float(np.mean(vals))
                span = max(pop_max - pop_min, 0.1)
                sigma = max(float(np.std(vals)), span * 0.05)
                density = np.exp(-((grid - mean_v) ** 2) / (2.0 * (sigma ** 2)))

            if len(density) > 0 and np.max(density) > 0:
                density = density / np.max(density)

            x = np.concatenate(([grid[0]], grid, [grid[-1]]))
            y = np.concatenate(([0.0], density, [0.0]))

            c = QColor(color_hex)
            c.setAlpha(60)
            curve.setPen(pg.mkPen(color_hex, width=2.0))
            curve.setBrush(pg.mkBrush(c))
            curve.setData(x, y)

    def clear_cluster_distribution(self) -> None:
        """Clear all comparison cluster distribution overlay curves and reset legend."""
        for curve in self.cluster_curves.values():
            curve.setData([], [])
        if hasattr(self, "lbl_legend"):
            self.lbl_legend.setText(build_marginal_legend_html())

    def set_candidate_values(self, feature_values: Dict[str, float]) -> None:
        """Update vertical indicator lines and top-right value readouts for focused unit."""
        for key, line in self.lines.items():
            raw_val = feature_values.get(key)
            try:
                val = float(raw_val) if raw_val is not None else float("nan")
            except (TypeError, ValueError):
                val = float("nan")

            if np.isfinite(val):
                line.setValue(val)
                line.setVisible(True)
            else:
                line.setVisible(False)

            formatted_val = self._format_metric_value(key, val if np.isfinite(val) else None)
            if key in self.value_labels:
                self.value_labels[key].setText(formatted_val)
            if key in self.value_items:
                self.value_items[key].setText(formatted_val)
                pop_bounds = self._pop_bounds.get(key)
                if pop_bounds is not None:
                    self.value_items[key].setPos(pop_bounds[1], 1.18)


class _BoldTopClusterAxisItem(pg.AxisItem):
    """Bottom axis item that renders the top-probability cluster tick label in bold."""

    def __init__(self, orientation: str = "bottom") -> None:
        super().__init__(orientation=orientation)
        self._bold_label: Optional[str] = None

    def set_bold_label(self, label: Optional[str]) -> None:
        """Set the tick text that should be rendered in bold (e.g. 'C2')."""
        self._bold_label = label
        self.picture = None
        self.update()

    def drawPicture(self, p: Any, axisSpec: Any, tickSpecs: Any, textSpecs: Any) -> None:
        p.setRenderHint(p.RenderHint.Antialiasing, False)
        p.setRenderHint(p.RenderHint.TextAntialiasing, True)

        pen, p1, p2 = axisSpec
        p.setPen(pen)
        p.drawLine(p1, p2)

        for t_pen, tp1, tp2 in tickSpecs:
            p.setPen(t_pen)
            p.drawLine(tp1, tp2)

        base_font = self.style.get("tickFont") or p.font()
        p.setFont(base_font)
        p.setPen(self.textPen())
        bounding = self.boundingRect().toAlignedRect()
        p.setClipRect(bounding)
        for rect, flags, text in textSpecs:
            if self._bold_label is not None and text == self._bold_label:
                bold_font = QFont(base_font)
                bold_font.setBold(True)
                p.setFont(bold_font)
                p.drawText(rect, int(flags), text)
                p.setFont(base_font)
            else:
                p.drawText(rect, int(flags), text)


class ClusterProbabilityBarWidget(QWidget):
    """Vertical bar plot showing model probabilities across all clusters with threshold."""

    cluster_clicked = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(2)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)

        self.lbl_title = QLabel("Cluster probability")
        self.lbl_title.setStyleSheet(CLUSTERS_SECTION_SUBHEADER_STYLE)
        header_row.addWidget(self.lbl_title)
        header_row.addStretch(1)

        self.lbl_threshold_legend = QLabel("- - Auto-assign 85%")
        self.lbl_threshold_legend.setStyleSheet(FOCUSED_UNIT_LEGEND_STYLE)
        header_row.addWidget(self.lbl_threshold_legend)
        layout.addLayout(header_row)

        self._bottom_axis = _BoldTopClusterAxisItem("bottom")
        self.plot_widget = pg.PlotWidget(axisItems={"bottom": self._bottom_axis})
        self.plot_widget.setFixedSize(
            PLOT_CLUSTER_PROB_SIZE[0] + PLOT_UNIT_RESPONSE_LEFT_MARGIN,
            PLOT_CLUSTER_PROB_SIZE[1],
        )
        self.plot_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setYRange(0.0, 1.0, padding=0.05)
        apply_plot_style(
            self.plot_widget,
            compact=True,
            bottom_label=None,
            left_margin=PLOT_UNIT_RESPONSE_LEFT_MARGIN,
        )
        self.plot_widget.showAxis("left")
        self.plot_widget.showAxis("bottom")
        self.plot_widget.getAxis("left").setTicks([[(0.0, "0%"), (0.5, "50%"), (1.0, "100%")]])

        self.threshold_line = pg.InfiniteLine(
            angle=0,
            pos=0.85,
            pen=pg.mkPen(COLOR_WARNING, width=1.5, style=Qt.PenStyle.DashLine),
        )
        self.plot_widget.addItem(self.threshold_line)

        layout.addWidget(self.plot_widget)
        self.bar_items: List[pg.BarGraphItem] = []
        self._bar_cids: List[int] = []
        self._bar_probs: Dict[int, float] = {}
        self._bar_names: Dict[int, str] = {}
        self._inspected_cluster_id: Optional[int] = None

        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        install_figure_export_menu(
            self.plot_widget,
            export_target=self.plot_widget,
            default_name="cluster_probability",
        )

    def _bar_index_at_scene_pos(self, scene_pos: QPointF) -> Optional[int]:
        """Map scene position to bar index if pointer is over a bar column."""
        if not self._bar_cids:
            return None
        vb = self.plot_widget.getViewBox()
        if not vb.sceneBoundingRect().contains(scene_pos):
            return None
        pt = vb.mapSceneToView(scene_pos)
        if pt.y() < -0.05 or pt.y() > 1.05:
            return None
        idx = int(round(pt.x()))
        if 0 <= idx < len(self._bar_cids) and abs(pt.x() - idx) <= 0.35:
            return idx
        return None

    def _on_mouse_moved(self, scene_pos: QPointF) -> None:
        """Show tooltip with cluster name, ID, and probability percentage on bar hover."""
        idx = self._bar_index_at_scene_pos(scene_pos)
        if idx is not None:
            cid = self._bar_cids[idx]
            cname = self._bar_names.get(cid, f"Cluster {cid}")
            prob = self._bar_probs.get(cid, 0.0)
            tip = f"{cname} (C{cid}): {prob * 100:.0f}%"
            self.plot_widget.setToolTip(tip)
            QToolTip.showText(QCursor.pos(), tip, self.plot_widget)
        else:
            self.plot_widget.setToolTip("")
            QToolTip.hideText()

    def _on_mouse_clicked(self, event: Any) -> None:
        """Emit cluster_clicked(cid) when user clicks on a cluster bar."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        idx = self._bar_index_at_scene_pos(event.scenePos())
        if idx is not None:
            cid = self._bar_cids[idx]
            self.cluster_clicked.emit(cid)

    def set_inspected_cluster(self, cluster_id: Optional[int]) -> None:
        """Highlight the currently inspected cluster bar with a dark outline."""
        self._inspected_cluster_id = cluster_id
        for cid, bar in zip(self._bar_cids, self.bar_items):
            pen = (
                pg.mkPen(COLOR_TEXT, width=2.0)
                if (cluster_id is not None and cid == cluster_id)
                else pg.mkPen(COLOR_DIVIDER, width=0.5)
            )
            bar.setOpts(pen=pen)

    def set_probabilities(
        self,
        probabilities: Dict[int, float],
        cluster_colors: Dict[int, str],
        cluster_names: Dict[int, str],
        threshold: float,
        cluster_order: Optional[List[int]] = None,
    ) -> None:
        """Render cluster probability bars in cluster order with threshold line."""
        self.threshold_line.setValue(float(threshold))
        self.lbl_threshold_legend.setText(f"- - Auto-assign {float(threshold) * 100:.0f}%")

        # Remove old bar items
        for item in self.bar_items:
            self.plot_widget.removeItem(item)
        self.bar_items.clear()
        self._bar_cids.clear()
        self._bar_probs.clear()
        self._bar_names = dict(cluster_names)

        if cluster_order is not None:
            cids = [c for c in cluster_order if c >= 0]
            for c in sorted(probabilities.keys()):
                if c >= 0 and c not in cids:
                    cids.append(c)
        else:
            cids = [c for c in sorted(probabilities.keys()) if c >= 0]

        if not cids:
            self._bottom_axis.set_bold_label(None)
            self.plot_widget.getAxis("bottom").setTicks([])
            return

        top_cid: Optional[int] = None
        top_prob = -1.0
        for cid in cids:
            p_val = max(0.0, min(1.0, float(probabilities.get(cid, 0.0))))
            if p_val > top_prob:
                top_prob = p_val
                top_cid = cid
        self._bottom_axis.set_bold_label(f"C{top_cid}" if (top_cid is not None and top_prob > 0.0) else None)

        ticks = []
        for idx, cid in enumerate(cids):
            prob = max(0.0, min(1.0, float(probabilities.get(cid, 0.0))))
            color = color_for_cluster(cid, cluster_colors)
            pen = (
                pg.mkPen(COLOR_TEXT, width=2.0)
                if (self._inspected_cluster_id is not None and cid == self._inspected_cluster_id)
                else pg.mkPen(COLOR_DIVIDER, width=0.5)
            )
            bar = pg.BarGraphItem(
                x=[idx],
                height=[prob],
                width=0.6,
                brush=pg.mkBrush(color),
                pen=pen,
            )
            self.plot_widget.addItem(bar)
            self.bar_items.append(bar)
            self._bar_cids.append(cid)
            self._bar_probs[cid] = prob
            ticks.append((idx, f"C{cid}"))

        self.plot_widget.getAxis("bottom").setTicks([ticks])
        self.plot_widget.setXRange(-0.6, len(cids) - 0.4, padding=0.02)
        self.plot_widget.setYRange(0.0, 1.0, padding=0.05)

    def clear(self) -> None:
        """Clear all probability bars and reset ticks."""
        for item in self.bar_items:
            self.plot_widget.removeItem(item)
        self.bar_items.clear()
        self._bar_cids.clear()
        self._bar_probs.clear()
        self._bottom_axis.set_bold_label(None)
        self.plot_widget.getAxis("bottom").setTicks([])


class StripedProgressBarSegment(QFrame):
    """Progress bar segment rendering 45-degree diagonal stripes."""

    def __init__(
        self,
        base_color: str = COLOR_ACCENT,
        stripe_color: str = COLOR_PASSING_FILL,
        stripe_width: float = 2.5,
        stripe_spacing: float = 6.0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.base_color = QColor(base_color)
        self.stripe_color = QColor(stripe_color)
        self.stripe_width = stripe_width
        self.stripe_spacing = stripe_spacing
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()
        painter.setClipRect(rect)

        # 1. Fill base solid color
        painter.fillRect(rect, self.base_color)

        # 2. Draw 45-degree diagonal stripes
        pen = QPen(self.stripe_color, self.stripe_width)
        painter.setPen(pen)

        w = rect.width()
        h = rect.height()
        step = max(1.0, float(self.stripe_spacing))
        x = -h
        while x < w + h:
            painter.drawLine(int(round(x)), h, int(round(x + h)), 0)
            x += step

        painter.end()


class ClassificationProgressBar(QWidget):
    """Sleek multi-segment progress bar displaying breakdown of curation status."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.curated_count = 0
        self.auto_count = 0
        self.seeded_count = 0
        self.unassigned_count = 0
        self.suggested_count = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        self.lbl_summary = QLabel("Classified: 0 / 0 (0%)")
        self.lbl_summary.setStyleSheet(CLUSTERS_PROGRESS_LEGEND_STYLE)
        layout.addWidget(self.lbl_summary)

        # Bar container
        self.bar_frame = QFrame()
        self.bar_frame.setFixedHeight(12)
        self.bar_frame.setStyleSheet(CLUSTERS_PROGRESS_BAR_CONTAINER_STYLE)
        bar_layout = QHBoxLayout(self.bar_frame)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(1)

        self.seg_curated = QFrame()
        self.seg_curated.setStyleSheet(build_progress_segment_style(COLOR_SUCCESS, is_left=True))
        self.seg_curated.setToolTip("Curated (Manually Confirmed)")

        self.seg_auto = QFrame()
        self.seg_auto.setStyleSheet(build_progress_segment_style(COLOR_ACCENT))
        self.seg_auto.setToolTip("Auto-assigned (Confidence >= Threshold)")

        self.seg_suggested = StripedProgressBarSegment(COLOR_ACCENT, COLOR_PASSING_FILL)
        self.seg_suggested.setToolTip("Putative Auto-labeled (Confidence >= Threshold)")

        self.seg_seeded = QFrame()
        self.seg_seeded.setStyleSheet(build_progress_segment_style(COLOR_SEEDED))
        self.seg_seeded.setToolTip("Seeded (Affinity Propagation)")

        self.seg_unassigned = QFrame()
        self.seg_unassigned.setStyleSheet(build_progress_segment_style(COLOR_DIVIDER, is_right=True))
        self.seg_unassigned.setToolTip("Unassigned / Remaining")

        bar_layout.addWidget(self.seg_curated)
        bar_layout.addWidget(self.seg_auto)
        bar_layout.addWidget(self.seg_suggested)
        bar_layout.addWidget(self.seg_seeded)
        bar_layout.addWidget(self.seg_unassigned)

        layout.addWidget(self.bar_frame)
        self._bar_layout = bar_layout

        self.update_counts(0, 0, 0, 0)

    def update_counts(
        self,
        curated: int,
        auto_assigned: int,
        seeded: int,
        unassigned: int,
        suggested: int = 0,
        threshold: float = 0.85,
    ) -> None:
        """Update segment stretch factors and summary label."""
        self.curated_count = max(0, int(curated))
        self.auto_count = max(0, int(auto_assigned))
        self.seeded_count = max(0, int(seeded))
        self.unassigned_count = max(0, int(unassigned))
        self.suggested_count = max(0, int(suggested))
        remaining_unassigned = max(0, self.unassigned_count - self.suggested_count)

        total = self.curated_count + self.auto_count + self.seeded_count + self.unassigned_count
        classified = self.curated_count + self.auto_count + self.seeded_count

        if total == 0:
            pct = 0
            self.lbl_summary.setText("Classified: 0 / 0 (0%) • 0 Curated | 0 Auto | 0 Seeded | 0 Remaining")
            self._bar_layout.setStretch(0, 0)
            self._bar_layout.setStretch(1, 0)
            self._bar_layout.setStretch(2, 0)
            self._bar_layout.setStretch(3, 0)
            self._bar_layout.setStretch(4, 1)
            self.seg_curated.setVisible(False)
            self.seg_auto.setVisible(False)
            self.seg_suggested.setVisible(False)
            self.seg_seeded.setVisible(False)
            self.seg_unassigned.setVisible(True)
        else:
            pct = int(round((classified / total) * 100))
            if self.suggested_count > 0:
                auto_str = f"{self.auto_count} Auto ({self.suggested_count} Putative)"
            else:
                auto_str = f"{self.auto_count} Auto"

            self.lbl_summary.setText(
                build_classification_progress_summary_html(
                    classified=classified,
                    total=total,
                    pct=pct,
                    curated_count=self.curated_count,
                    auto_str=auto_str,
                    seeded_count=self.seeded_count,
                    remaining_unassigned=remaining_unassigned,
                )
            )
            self.seg_curated.setVisible(self.curated_count > 0)
            self.seg_auto.setVisible(self.auto_count > 0)
            self.seg_suggested.setVisible(self.suggested_count > 0)
            self.seg_suggested.setToolTip(
                f"Putative Auto-labeled: {self.suggested_count} cells (Confidence >= {threshold:.0%})"
            )
            self.seg_seeded.setVisible(self.seeded_count > 0)
            self.seg_unassigned.setVisible(remaining_unassigned > 0)

            self._bar_layout.setStretch(0, max(0, self.curated_count))
            self._bar_layout.setStretch(1, max(0, self.auto_count))
            self._bar_layout.setStretch(2, max(0, self.suggested_count))
            self._bar_layout.setStretch(3, max(0, self.seeded_count))
            self._bar_layout.setStretch(4, max(0, remaining_unassigned))


class ClusterCardWidget(QFrame):
    """Card widget representing an active cluster with waveform preview and confidence envelope."""

    card_clicked = pyqtSignal(int)          # cluster_id
    assign_requested = pyqtSignal(int)      # cluster_id
    merge_requested = pyqtSignal(int, int)  # (source_cluster_id, target_cluster_id)
    move_requested = pyqtSignal(int, int)   # (source_cluster_id, target_cluster_id)
    rename_requested = pyqtSignal(int)      # cluster_id
    recolor_requested = pyqtSignal(int)     # cluster_id
    dissolve_requested = pyqtSignal(int)    # cluster_id

    _KEYCAP_ANIM_MS: int = 180

    def __init__(
        self,
        cluster_id: Union[int, ClusterSummary],
        count: int = 0,
        color_hex: Optional[str] = None,
        summary: Optional[ClusterSummary] = None,
        pulse_duration_ms: float = DEFAULT_PULSE_DURATION_MS,
        parent: QWidget | None = None,
        custom_name: Optional[str] = None,
        is_smooth: bool = True,
        norm_mode: str = "peak",
        hotkey: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        if isinstance(cluster_id, ClusterSummary):
            summary = cluster_id
            cluster_id = summary.cluster_id
            count = summary.count
            color_hex = getattr(summary, "color_hex", None)
            custom_name = custom_name or getattr(summary, "name", None)
            if hasattr(summary, "norm_mode") and summary.norm_mode and norm_mode == "peak":
                norm_mode = summary.norm_mode

        self.cluster_id = int(cluster_id)
        self.count = int(count)
        self.color_hex = str(color_hex) if color_hex else color_for_cluster(self.cluster_id)
        self.summary = summary
        self.pulse_duration_ms = pulse_duration_ms
        self.custom_name = custom_name
        self.is_smooth = is_smooth
        self.norm_mode = norm_mode
        self.hotkey = normalize_hotkey(hotkey) if hotkey else ""
        self.display_name = custom_name if custom_name else f"Cluster {self.cluster_id}"
        self._drag_start_pos: Optional[QPoint] = None
        self._is_comparing: bool = False
        self._keycap_expanded: bool = False
        self._keycap_anim_group: Optional[QParallelAnimationGroup] = None
        self._other_clusters_provider: Optional[Any] = None

        self.setFixedSize(*CLUSTER_CARD_SIZE)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAcceptDrops(True)
        self.setToolTip("Click to compare · Drag to move · Shift+Drag to merge")
        self._reset_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)

        header = QLabel(self.display_name)
        header.setStyleSheet(build_cluster_card_header_style(self.color_hex))
        header.setMinimumWidth(10)
        header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        header.setToolTip(self.display_name)
        header_layout.addWidget(header, stretch=1)
        self.header = header

        self.lbl_count = QLabel(f"n = {self.count}")
        self.lbl_count.setStyleSheet(CLUSTER_CARD_COUNT_STYLE)
        self.lbl_count.setWordWrap(False)
        self.lbl_count.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        header_layout.addWidget(self.lbl_count)

        self.lbl_suggested_badge = QLabel("")
        self.lbl_suggested_badge.setStyleSheet(CLUSTER_CARD_SUGGESTED_BADGE_STYLE)
        self.lbl_suggested_badge.setVisible(False)
        self.lbl_suggested_badge.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        header_layout.addWidget(self.lbl_suggested_badge)

        self.lbl_prob = QLabel("", self)
        self.lbl_prob.setStyleSheet(CLUSTER_CARD_PROB_BADGE_DEFAULT_STYLE)
        self.lbl_prob.setVisible(False)
        self.lbl_cand_prob = self.lbl_prob

        # Hidden compatibility label
        self.lbl_hotkey = QLabel("", self)
        self.lbl_hotkey.setStyleSheet(CLUSTER_CARD_SUBTITLE_STYLE)
        self.lbl_hotkey.setVisible(False)

        self.btn_keycap = QPushButton("", self)
        self.btn_keycap.setStyleSheet(CLUSTER_CARD_KEYCAP_STYLE)
        self.btn_keycap.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_keycap.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_keycap.setFixedHeight(20)
        self.btn_keycap.clicked.connect(lambda: self.assign_requested.emit(self.cluster_id))
        header_layout.addWidget(self.btn_keycap)

        layout.addLayout(header_layout)

        # Mini PSTH waveform preview (fixed size per § 8.1)
        plot = pg.PlotWidget()
        plot.setFixedSize(
            PLOT_CARD_PSTH_SIZE[0] + PLOT_CARD_PSTH_LEFT_MARGIN,
            PLOT_CARD_PSTH_SIZE[1],
        )
        plot.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        plot.getViewBox().setDefaultPadding(0.06)
        self._y_zero_anchor = plot.plot([0.0], [0.0], pen=None)
        plot.setMouseEnabled(x=False, y=False)

        y_label = (
            "Norm. rate"
            if self.norm_mode == "peak"
            else ("Probability" if self.norm_mode == "probability" else "Firing Rate (Hz)")
        )
        apply_plot_style(
            plot,
            compact=True,
            left_label=y_label,
            bottom_label="Time (ms)",
            left_margin=PLOT_CARD_PSTH_LEFT_MARGIN,
        )
        plot.showAxis("bottom")
        plot.showAxis("left")

        # Optical stimulus shaded region between t = 0 and t = pulse_duration_ms
        stim_region = _SafeLinearRegionItem(
            values=[0.0, pulse_duration_ms],
            orientation="vertical",
            movable=False,
            brush=pg.mkBrush(COLOR_LIGHT_PULSE_BAND),
            pen=pg.mkPen(None),
        )
        stim_region.setZValue(-20)
        plot.addItem(stim_region)

        # Configure PSTH time ticks guaranteeing -20, 0, 5, 10, 20, 40 ms labeled
        w_start, w_end = -30.0, 50.0
        if summary is not None and len(summary.time_bins_ms) > 0:
            w_start = float(summary.time_bins_ms[0])
            w_end = float(summary.time_bins_ms[-1])
        plot.getAxis("bottom").setTicks(compute_standard_plot_ticks(w_start, w_end, compact=True))

        # Onset vertical bar at t = 0.0
        onset_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            pen=pg.mkPen(COLOR_AXIS, width=1.0, style=Qt.PenStyle.DashLine),
        )
        plot.addItem(onset_line)

        # Kept hidden for backward compatibility (§ 8.2: only dashed t = 0 line shown)
        offset_line = pg.InfiniteLine(
            pos=pulse_duration_ms,
            angle=90,
            pen=pg.mkPen(None),
        )
        offset_line.setVisible(False)
        plot.addItem(offset_line)

        self.plot = plot
        self.plot_psth = plot
        self.onset_line = onset_line
        self.offset_line = offset_line
        self.stim_region = stim_region

        # Confidence bound lines and fill
        self.c_upper = plot.plot(pen=pg.mkPen(None))
        self.c_lower = plot.plot(pen=pg.mkPen(None))
        c_color = pg.mkColor(self.color_hex)
        c_color.setAlpha(45)
        self.fill_item = pg.FillBetweenItem(self.c_lower, self.c_upper, brush=c_color)
        plot.addItem(self.fill_item)

        # Main mean waveform trace
        self.mean_curve = plot.plot(pen=pg.mkPen(self.color_hex, width=PSTH_LINE_WIDTH))
        self._update_plot_curves(summary)

        # Subtle thin dark focused-unit PSTH curve overlay (§ 9.2)
        if getattr(self, "is_smooth", False):
            self.candidate_psth_curve = plot.plot(pen=pg.mkPen(COLOR_TEXT, width=1.2))
        else:
            self.candidate_psth_curve = plot.plot(pen=pg.mkPen(COLOR_TEXT, width=1.2), stepMode="center")
        self.candidate_psth_curve.setZValue(5)

        layout.addWidget(plot)

        # Hidden compatibility label (hint moved to card tooltip in § 9.2)
        self.drag_hint = QLabel("", self)
        self.drag_hint.setVisible(False)

        # Set mouse transparency on plot & non-interactive labels so card receives drag/click
        # NOTE: Do NOT set WA_TransparentForMouseEvents on lbl_suggested_badge or btn_keycap
        # so the badge tooltip and keycap click work reliably (§ 9.2).
        plot.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        if plot.viewport() is not None:
            plot.viewport().setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.lbl_count.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.lbl_prob.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        # Forward any direct ContextMenu events on plot/viewport to the card's unified menu
        class _CardPlotContextForwarder(QObject):
            def __init__(self, card: "ClusterCardWidget") -> None:
                super().__init__(card)
                self._card = card

            def eventFilter(self, _obj: QObject, ev: QEvent) -> bool:  # noqa: N802
                if ev.type() == QEvent.Type.ContextMenu:
                    self._card.contextMenuEvent(ev)
                    ev.accept()
                    return True
                return False

        self._plot_ctx_forwarder = _CardPlotContextForwarder(self)
        plot.installEventFilter(self._plot_ctx_forwarder)
        if plot.viewport() is not None:
            plot.viewport().installEventFilter(self._plot_ctx_forwarder)

        self.set_hotkey(hotkey)
        self._update_header_text()

    def _update_plot_curves(self, summary: Optional[ClusterSummary]) -> None:
        """Update mean waveform and confidence envelope in-place."""
        color_hex = getattr(self, "color_hex", color_for_cluster(getattr(self, "cluster_id", 0)))
        if summary is not None and len(summary.time_bins_ms) > 0 and len(summary.mean_psth) > 0:
            x = np.asarray(summary.time_bins_ms, dtype=np.float64)
            y = np.asarray(summary.mean_psth, dtype=np.float64)
            sem = np.asarray(summary.sem_psth, dtype=np.float64)

            upper = y + sem
            lower = np.maximum(0.0, y - sem)

            c_color = pg.mkColor(color_hex)
            c_color.setAlpha(45)
            self.fill_item.setBrush(pg.mkBrush(c_color))
            self.mean_curve.setPen(pg.mkPen(color_hex, width=PSTH_LINE_WIDTH))

            if getattr(self, "is_smooth", False):
                x_pts = (x[:-1] + x[1:]) / 2.0 if len(x) == len(y) + 1 else x
                self.c_upper.setData(x_pts, upper, stepMode=False)
                self.c_lower.setData(x_pts, lower, stepMode=False)
                self.mean_curve.setData(x_pts, y, stepMode=False)
                self.plot.setXRange(x_pts[0], x_pts[-1], padding=0.02)
            else:
                if len(x) > 1:
                    dx = float(x[1] - x[0])
                else:
                    dx = 2.0
                x_edges = np.append(x - dx / 2.0, x[-1] + dx / 2.0)
                self.c_upper.setData(x_edges, upper, stepMode="center")
                self.c_lower.setData(x_edges, lower, stepMode="center")
                self.mean_curve.setData(x_edges, y, stepMode="center")
                self.plot.setXRange(x_edges[0], x_edges[-1], padding=0.02)

            self.c_upper.setVisible(True)
            self.c_lower.setVisible(True)
            self.fill_item.setVisible(True)
            self.mean_curve.setVisible(True)
            y_top = float(np.nanmax(upper)) if len(upper) > 0 else 1.0
            self._auto_y_top = max(1e-3, y_top)
            left_ax = self.plot.getAxis("left")
            if left_ax is not None:
                left_ax.setTicks(compute_clean_y_ticks(self._auto_y_top, max_ticks=3))
        elif getattr(self, "cluster_id", -1) >= 0 and getattr(self, "count", 0) > 0:
            cluster_id = self.cluster_id
            x = np.linspace(-50, 50, 40)
            y = np.exp(-((x - (cluster_id * 3 + 2)) ** 2) / 15.0) * (30 + cluster_id * 10)
            self.mean_curve.setPen(pg.mkPen(color_hex, width=PSTH_LINE_WIDTH))
            if getattr(self, "is_smooth", False):
                self.mean_curve.setData(x, y, stepMode=False)
                self.plot.setXRange(x[0], x[-1], padding=0.02)
            else:
                dx = float(x[1] - x[0])
                x_edges = np.append(x - dx / 2.0, x[-1] + dx / 2.0)
                self.mean_curve.setData(x_edges, y, stepMode="center")
                self.plot.setXRange(x_edges[0], x_edges[-1], padding=0.02)
            self.mean_curve.setVisible(True)
            self.c_upper.setVisible(False)
            self.c_lower.setVisible(False)
            self.fill_item.setVisible(False)
        else:
            self.c_upper.setData([], [])
            self.c_lower.setData([], [])
            self.mean_curve.setData([], [])
            self.c_upper.setVisible(False)
            self.c_lower.setVisible(False)
            self.fill_item.setVisible(False)
            self.mean_curve.setVisible(False)

    def update_card(
        self,
        summary: Optional[ClusterSummary],
        count: int,
        color_hex: str,
        custom_name: Optional[str] = None,
        pulse_duration_ms: float = 10.0,
        is_smooth: bool = True,
        norm_mode: str = "raw",
        hotkey: Optional[str] = None,
    ) -> None:
        """Update existing card elements in-place without widget recreation."""
        self.summary = summary
        self.count = count
        self.color_hex = color_hex if color_hex else color_for_cluster(self.cluster_id)
        self.custom_name = custom_name
        self.pulse_duration_ms = pulse_duration_ms
        self.is_smooth = is_smooth
        self.norm_mode = norm_mode

        # Update header
        self.display_name = custom_name if custom_name else f"Cluster {self.cluster_id}"
        self.lbl_count.setText(f"n = {self.count}")
        self.header.setStyleSheet(build_cluster_card_header_style(self.color_hex))
        self.header.setToolTip(self.display_name)

        # Update hotkey
        self.set_hotkey(hotkey)
        self._update_header_text()

        # Update cluster card y- and x-axis labels
        if norm_mode == "peak":
            self.plot.setLabel("left", "Norm. rate", **PLOT_AXIS_LABEL_COMPACT_STYLE)
        elif norm_mode == "probability":
            self.plot.setLabel("left", "Probability", **PLOT_AXIS_LABEL_COMPACT_STYLE)
        else:
            self.plot.setLabel("left", "Firing Rate (Hz)", **PLOT_AXIS_LABEL_COMPACT_STYLE)
        self.plot.setLabel("bottom", "Time (ms)", **PLOT_AXIS_LABEL_COMPACT_STYLE)

        # Update stimulus region and lines
        self.stim_region.setRegion([0.0, pulse_duration_ms])
        self.onset_line.setValue(0.0)
        self.offset_line.setValue(pulse_duration_ms)

        # Update ticks
        w_start, w_end = -30.0, 50.0
        if summary is not None and len(summary.time_bins_ms) > 0:
            w_start = float(summary.time_bins_ms[0])
            w_end = float(summary.time_bins_ms[-1])
        self.plot.getAxis("bottom").setTicks(compute_standard_plot_ticks(w_start, w_end, compact=True))

        # Update waveform curves
        self._update_plot_curves(summary)
        self._reset_style()

    def set_candidate_psth(self, time_bins_ms: Optional[np.ndarray], psth_hz: Optional[np.ndarray]) -> None:
        """Overlay subtle thin light-gray candidate PSTH trace."""
        if time_bins_ms is None or psth_hz is None or len(time_bins_ms) == 0 or len(psth_hz) == 0:
            self.candidate_psth_curve.setData([], [])
            return
        t_arr = np.asarray(time_bins_ms, dtype=np.float64)
        y_raw = np.asarray(psth_hz, dtype=np.float64)
        y_arr = normalize_psth_waveform(y_raw, getattr(self, "norm_mode", "peak"))
        if getattr(self, "is_smooth", False):
            if len(t_arr) == len(y_arr) + 1:
                t_arr = (t_arr[:-1] + t_arr[1:]) / 2.0
            self.candidate_psth_curve.setData(t_arr, y_arr, stepMode=False)
        else:
            if len(t_arr) == len(y_arr) + 1:
                self.candidate_psth_curve.setData(t_arr, y_arr, stepMode="center")
            else:
                dx = float(t_arr[1] - t_arr[0]) if len(t_arr) > 1 else 2.0
                x_edges = np.append(t_arr - dx / 2.0, t_arr[-1] + dx / 2.0)
                self.candidate_psth_curve.setData(x_edges, y_arr, stepMode="center")

    def set_visualized_window(self, window_ms: Tuple[float, float]) -> None:
        """Update bottom axis ticks for new visualized time window."""
        self.plot.getAxis("bottom").setTicks(
            compute_standard_plot_ticks(float(window_ms[0]), float(window_ms[1]), compact=True)
        )

    def set_comparison_active(self, active: bool) -> None:
        """Toggle active comparison highlight styling."""
        self._is_comparing = bool(active)
        self._reset_style()

    def set_y_range(self, y_min: Optional[float], y_max: Optional[float]) -> None:
        """Apply shared y-axis bounds or restore per-card auto-ranging (§ 9.1)."""
        if y_min is None or y_max is None or y_max <= y_min:
            y_top = max(1e-3, float(getattr(self, "_auto_y_top", 1.0)))
            self.plot.enableAutoRange(axis="y", enable=True)
            left_ax = self.plot.getAxis("left")
            if left_ax is not None:
                left_ax.setTicks(compute_clean_y_ticks(y_top, max_ticks=3))
        else:
            self.plot.enableAutoRange(axis="y", enable=False)
            self.plot.setYRange(float(y_min), float(y_max), padding=0.06)
            left_ax = self.plot.getAxis("left")
            if left_ax is not None:
                left_ax.setTicks(compute_clean_y_ticks(float(y_max), max_ticks=3))

    def _reset_style(self) -> None:
        """Reset default card styling."""
        self.setStyleSheet(
            build_cluster_card_widget_style(
                self.color_hex,
                is_comparing=getattr(self, "_is_comparing", False),
                is_top=False,
            )
        )

    def set_candidate_probability(self, prob: Optional[float], is_top: bool = False) -> None:
        """Update top candidate card highlighting (probability badge kept hidden on card header)."""
        if prob is not None:
            pct = int(round(float(prob) * 100))
            self.lbl_prob.setText(f"{pct}%")
        else:
            self.lbl_prob.setText("")
        self.lbl_prob.setVisible(False)

        self.setStyleSheet(
            build_cluster_card_widget_style(
                self.color_hex,
                is_comparing=getattr(self, "_is_comparing", False),
                is_top=is_top,
            )
        )
        self.lbl_prob.setStyleSheet(build_cluster_card_prob_badge_style(self.color_hex, is_top=is_top))
        self._update_header_text()

    def set_suggested_count(self, count: int, threshold: float = 0.85) -> None:
        """Update or hide putative suggested cells badge (§ 9.2)."""
        c = max(0, int(count))
        if c > 0:
            self.lbl_suggested_badge.setText(f"! {c}")
            self.lbl_suggested_badge.setToolTip("Cells above the confidence threshold")
            self.lbl_suggested_badge.setVisible(True)
        else:
            self.lbl_suggested_badge.setText("")
            self.lbl_suggested_badge.setToolTip("")
            self.lbl_suggested_badge.setVisible(False)
        self._update_header_text()

    def _compact_keycap_width(self) -> int:
        if not self.hotkey:
            return 0
        fm = QFontMetrics(self.btn_keycap.font())
        return max(20, fm.horizontalAdvance(self.hotkey) + 12)

    def _expanded_keycap_width(self) -> int:
        if not self.hotkey:
            return 0
        fm = QFontMetrics(self.btn_keycap.font())
        return max(56, fm.horizontalAdvance(f"Assign {self.hotkey}") + 14)

    def _animate_keycap(self, expanded: bool) -> None:
        """Smoothly animate keycap between compact 'K' and accent-blue 'Assign K' (§ 9.2)."""
        self._keycap_expanded = bool(expanded)
        if not self.hotkey or not self.btn_keycap.isVisible():
            return

        if self._keycap_anim_group is not None:
            self._keycap_anim_group.stop()
            self._keycap_anim_group = None

        if expanded:
            self.btn_keycap.setText(f"Assign {self.hotkey}")
            self.btn_keycap.setStyleSheet(CLUSTER_CARD_KEYCAP_HOVER_STYLE)
            target_w = self._expanded_keycap_width()
        else:
            self.btn_keycap.setText(self.hotkey)
            self.btn_keycap.setStyleSheet(CLUSTER_CARD_KEYCAP_STYLE)
            target_w = self._compact_keycap_width()

        start_w = self.btn_keycap.width() or self._compact_keycap_width()
        group = QParallelAnimationGroup(self)
        for prop in (b"minimumWidth", b"maximumWidth"):
            anim = QPropertyAnimation(self.btn_keycap, prop, group)
            anim.setDuration(self._KEYCAP_ANIM_MS)
            anim.setStartValue(start_w)
            anim.setEndValue(target_w)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            group.addAnimation(anim)
        group.finished.connect(self._update_header_text)
        self._keycap_anim_group = group
        group.start()

    def _update_header_text(self) -> None:
        """Elide cluster name so count, badge, and keycap never wrap or clip (§ 9.2)."""
        self.header.setToolTip(self.display_name)
        reserved = 24  # card horizontal margins + spacing
        if hasattr(self, "lbl_count"):
            reserved += self.lbl_count.sizeHint().width() + 4
        if hasattr(self, "lbl_suggested_badge") and self.lbl_suggested_badge.isVisible():
            reserved += self.lbl_suggested_badge.sizeHint().width() + 4
        if hasattr(self, "lbl_prob") and self.lbl_prob.isVisible():
            reserved += self.lbl_prob.sizeHint().width() + 4
        if hasattr(self, "btn_keycap") and self.btn_keycap.isVisible():
            kw = self._expanded_keycap_width() if self._keycap_expanded else self._compact_keycap_width()
            reserved += kw + 4
        avail_w = max(24, CLUSTER_CARD_SIZE[0] - reserved)
        fm = QFontMetrics(self.header.font())
        elided = fm.elidedText(self.display_name, Qt.TextElideMode.ElideRight, avail_w)
        self.header.setText(elided)

    def set_hotkey(self, hotkey: Optional[str]) -> None:
        """Update keycap hotkey displayed on top-right of cluster card (§ 9.2)."""
        self.hotkey = normalize_hotkey(str(hotkey).strip()) if hotkey else ""
        if self.hotkey:
            self.lbl_hotkey.setText(f"[{self.hotkey}]")
            self.btn_keycap.setToolTip(f"Assign focused unit to {self.display_name} ({self.hotkey})")
            self.btn_keycap.setVisible(True)
            if self._keycap_expanded:
                self.btn_keycap.setText(f"Assign {self.hotkey}")
                self.btn_keycap.setStyleSheet(CLUSTER_CARD_KEYCAP_HOVER_STYLE)
                w = self._expanded_keycap_width()
            else:
                self.btn_keycap.setText(self.hotkey)
                self.btn_keycap.setStyleSheet(CLUSTER_CARD_KEYCAP_STYLE)
                w = self._compact_keycap_width()
            self.btn_keycap.setMinimumWidth(w)
            self.btn_keycap.setMaximumWidth(w)
        else:
            self.lbl_hotkey.setText("")
            self.btn_keycap.setText("")
            self.btn_keycap.setVisible(False)
        self._update_header_text()

    def set_display_name(self, name: str) -> None:
        """Update display name on header label."""
        self.custom_name = name
        self.display_name = name if name else f"Cluster {self.cluster_id}"
        self._update_header_text()

    def set_color(self, color_hex: str) -> None:
        """Update card accent color and header style."""
        self.color_hex = color_hex
        self.header.setStyleSheet(build_cluster_card_header_style(color_hex))
        self._update_plot_curves(self.summary)
        self._reset_style()

    def enterEvent(self, event: Any) -> None:
        super().enterEvent(event)
        self._animate_keycap(True)

    def leaveEvent(self, event: Any) -> None:
        super().leaveEvent(event)
        if not self.hasFocus():
            self._animate_keycap(False)

    def focusInEvent(self, event: Any) -> None:
        super().focusInEvent(event)
        self._animate_keycap(True)

    def focusOutEvent(self, event: Any) -> None:
        super().focusOutEvent(event)
        if not self.underMouse():
            self._animate_keycap(False)

    def contextMenuEvent(self, event: Any) -> None:
        """Show right-click context menu: Rename, Change colour, Merge into, Dissolve (§ 9.2)."""
        menu = QMenu(self)
        menu.setStyleSheet(CLUSTERS_MENU_STYLE)

        act_rename = menu.addAction("Rename…")
        act_rename.triggered.connect(lambda: self.rename_requested.emit(self.cluster_id))

        act_recolor = menu.addAction("Change colour…")
        act_recolor.triggered.connect(lambda: self.recolor_requested.emit(self.cluster_id))

        merge_menu = menu.addMenu("Merge into…")
        merge_menu.setStyleSheet(CLUSTERS_MENU_STYLE)
        others: List[Tuple[int, str]] = []
        if callable(self._other_clusters_provider):
            others = self._other_clusters_provider(self.cluster_id)
        if others:
            for tgt_cid, tgt_name in others:
                act_m = merge_menu.addAction(tgt_name)
                act_m.triggered.connect(
                    lambda _checked=False, s=self.cluster_id, t=tgt_cid: self.merge_requested.emit(s, t)
                )
        else:
            act_none = merge_menu.addAction("No other clusters")
            act_none.setEnabled(False)

        populate_figure_export_menu(
            menu,
            export_target=self.plot,
            default_name=lambda: f"cluster_{self.cluster_id}_{self.display_name}",
            label="Save figure as SVG…",
            add_separator=True,
        )

        menu.addSeparator()

        from PyQt6.QtWidgets import QWidgetAction
        dissolve_wa = QWidgetAction(menu)
        dissolve_wa.setText("Dissolve cluster…")
        dissolve_wa.triggered.connect(lambda: self.dissolve_requested.emit(self.cluster_id))
        dissolve_btn = QPushButton("Dissolve cluster…", menu)
        dissolve_btn.setFlat(True)
        dissolve_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        dissolve_btn.setStyleSheet(CLUSTER_CARD_MENU_DANGER_BTN_STYLE)
        dissolve_btn.clicked.connect(dissolve_wa.trigger)
        dissolve_btn.clicked.connect(menu.close)
        dissolve_wa.setDefaultWidget(dissolve_btn)
        menu.addAction(dissolve_wa)

        menu.exec(event.globalPos())

    def _is_shift_pressed(self, event: Any = None) -> bool:
        """Detect whether Shift modifier key is held during drag/drop."""
        if event is not None:
            if hasattr(event, "modifiers"):
                try:
                    mods = event.modifiers() if callable(event.modifiers) else event.modifiers
                    if bool(mods & Qt.KeyboardModifier.ShiftModifier):
                        return True
                except Exception:
                    pass
        try:
            return bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)
        except Exception:
            return False

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)
        self.card_clicked.emit(self.cluster_id)

    def mouseMoveEvent(self, event: Any) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            if self._drag_start_pos is not None:
                dist = (event.pos() - self._drag_start_pos).manhattanLength()
                if dist >= QApplication.startDragDistance():
                    drag = QDrag(self)
                    mime = QMimeData()
                    mime.setData("application/x-cluster-id", str(self.cluster_id).encode("utf-8"))
                    drag.setMimeData(mime)
                    drag.exec(Qt.DropAction.MoveAction)
                    self._drag_start_pos = None
                    return
        super().mouseMoveEvent(event)

    def dragEnterEvent(self, event: Any) -> None:
        if event.mimeData().hasFormat("application/x-cluster-id"):
            try:
                src_id = int(event.mimeData().data("application/x-cluster-id").data().decode("utf-8"))
                if src_id != self.cluster_id:
                    event.acceptProposedAction()
                    if self._is_shift_pressed(event):
                        self.setStyleSheet(CLUSTER_CARD_DRAG_MERGE_STYLE)
                    else:
                        self.setStyleSheet(CLUSTER_CARD_DRAG_MOVE_STYLE)
                    return
            except Exception:
                pass
        event.ignore()

    def dragMoveEvent(self, event: Any) -> None:
        if event.mimeData().hasFormat("application/x-cluster-id"):
            try:
                src_id = int(event.mimeData().data("application/x-cluster-id").data().decode("utf-8"))
                if src_id != self.cluster_id:
                    event.acceptProposedAction()
                    if self._is_shift_pressed(event):
                        self.setStyleSheet(CLUSTER_CARD_DRAG_MERGE_STYLE)
                    else:
                        self.setStyleSheet(CLUSTER_CARD_DRAG_MOVE_STYLE)
                    return
            except Exception:
                pass
        try:
            super().dragMoveEvent(event)
        except (TypeError, AttributeError):
            pass

    def dragLeaveEvent(self, event: Any) -> None:
        self._reset_style()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: Any) -> None:
        self._reset_style()
        if event.mimeData().hasFormat("application/x-cluster-id"):
            try:
                src_id = int(event.mimeData().data("application/x-cluster-id").data().decode("utf-8"))
                if src_id != self.cluster_id:
                    event.acceptProposedAction()
                    if self._is_shift_pressed(event):
                        self.merge_requested.emit(src_id, self.cluster_id)
                    else:
                        self.move_requested.emit(src_id, self.cluster_id)
                    return
            except Exception:
                pass
        if hasattr(event, "ignore"):
            event.ignore()



class NewClusterCardWidget(QFrame):
    """Dashed '+ New cluster' card placed at the end of the cluster card grid (§ 9.2)."""

    clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(*CLUSTER_CARD_SIZE)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(NEW_CLUSTER_CARD_STYLE)
        self.setToolTip("Create a new cluster and assign the focused unit to it (Hotkey: N)")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)
        layout.addStretch(1)

        self.lbl_title = QLabel("+ New cluster", self)
        self.lbl_title.setStyleSheet(NEW_CLUSTER_CARD_LABEL_STYLE)
        self.lbl_title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.lbl_title)

        self.lbl_keycap = QLabel("N", self)
        self.lbl_keycap.setFixedSize(22, 22)
        self.lbl_keycap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_keycap.setStyleSheet(NEW_CLUSTER_CARD_KEYCAP_STYLE)
        self.lbl_keycap.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.lbl_keycap)

        layout.addStretch(1)

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class _ListPanelBarDelegate(QStyledItemDelegate):
    """Delegate drawing a compact probability or uncertainty bar in List Panel cells (§ 9.3)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: Any) -> None:
        val = index.data(Qt.ItemDataRole.UserRole + 1)
        bar_role = index.data(Qt.ItemDataRole.UserRole + 2)
        if val is None or bar_role not in ("prob", "uncertainty"):
            super().paint(painter, option, index)
            return

        painter.save()
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        if opt.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(opt.rect, QColor(COLOR_ACCENT_BG))

        frac = max(0.0, min(1.0, float(val)))
        rect = opt.rect.adjusted(4, 2, -4, -2)

        # Draw small horizontal bar on the left half, numeric text on the right
        bar_w = max(24, min(42, rect.width() // 2 - 4))
        bar_h = 5
        bar_y = rect.center().y() - bar_h // 2
        bar_x = rect.left()

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLOR_SURFACE_MUTED))
        painter.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 2.0, 2.0)

        fill_w = max(0.0, bar_w * frac)
        if fill_w > 0.0:
            fill_color = QColor(COLOR_WARNING if bar_role == "uncertainty" else COLOR_ACCENT)
            painter.setBrush(fill_color)
            painter.drawRoundedRect(QRectF(bar_x, bar_y, fill_w, bar_h), 2.0, 2.0)

        text_rect = rect.adjusted(bar_w + 6, 0, 0, 0)
        painter.setPen(QColor(COLOR_TEXT))
        painter.setFont(opt.font)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, opt.text)
        painter.restore()


class _BestMatchColorDelegate(QStyledItemDelegate):
    """Preserve cluster foreground colour on selected rows in Best match column."""

    def initStyleOption(self, option: QStyleOptionViewItem, index: Any) -> None:
        super().initStyleOption(option, index)
        fg = index.data(Qt.ItemDataRole.ForegroundRole)
        if fg is not None:
            color = fg.color() if isinstance(fg, QBrush) else QColor(fg)
            if color.isValid():
                option.palette.setColor(QPalette.ColorRole.Text, color)
                option.palette.setColor(QPalette.ColorRole.HighlightedText, color)


class _PrewarmSignals(QObject):
    """Thread-safe signal bridge from background PSTH/raster workers to ClusteringView."""

    batch_ready = pyqtSignal(int, list)


class _PrewarmBatchRunnable(QRunnable):
    """Compute compute_psth_and_raster() batches off the GUI thread using read-only slices."""

    def __init__(self, generation: int, items: List[Tuple[Any, ...]], signals: _PrewarmSignals) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self.generation = int(generation)
        self.items = items
        self.signals = signals

    def run(self) -> None:
        results: List[Tuple[Any, Any, Dict[str, Any]]] = []
        for cache_key, psth_key, spikes, onsets, win_ms, bin_ms, order_by, smooth, sigma_ms in self.items:
            try:
                psth_data = compute_psth_and_raster(
                    spikes,
                    onsets,
                    window_ms=win_ms,
                    bin_size_ms=bin_ms,
                    order_by=order_by,
                    smooth=smooth,
                    gaussian_sigma_ms=sigma_ms,
                )
                if psth_data is not None:
                    results.append((cache_key, psth_key, psth_data))
            except Exception:
                continue
        try:
            self.signals.batch_ready.emit(self.generation, results)
        except RuntimeError:
            # Target ClusteringView was already destroyed before worker completion.
            pass


class _MemberRowFrame(QFrame):
    """Lightweight Inspector member card using cached HiDPI pixmaps and lazy live plot promotion."""

    def __init__(
        self,
        member_gid: str,
        owner: "ClusteringView",
        unit_title: str,
        unit_tooltip: str,
        meta_text: str,
        on_click: Any,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.gid = member_gid
        self._owner = owner
        self._on_click = on_click
        self._rendered_key: Optional[Tuple[Any, ...]] = None
        self._live_plot: Optional[UnitResponsePlot] = None

        self.setObjectName("unitResponseMemberCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(1, 6, 1, 6)
        vbox.setSpacing(2)

        lbl_unit = QLabel(unit_title, self)
        lbl_unit.setStyleSheet(UNIT_RESPONSE_MEMBER_HEADER_STYLE)
        lbl_unit.setToolTip(unit_tooltip)
        lbl_meta = QLabel(meta_text, self)
        lbl_meta.setStyleSheet(UNIT_RESPONSE_MEMBER_META_STYLE)
        vbox.addWidget(lbl_unit)
        vbox.addWidget(lbl_meta)

        slot_size = UnitResponsePlot.fixed_plot_size()
        self.plot_slot = QWidget(self)
        self.plot_slot.setFixedSize(slot_size)
        self.plot_slot.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        slot_layout = QVBoxLayout(self.plot_slot)
        slot_layout.setContentsMargins(0, 0, 0, 0)
        slot_layout.setSpacing(0)

        self._pixmap_label = QLabel(self.plot_slot)
        self._pixmap_label.setFixedSize(slot_size)
        self._pixmap_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._pixmap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slot_layout.addWidget(self._pixmap_label)

        vbox.addWidget(self.plot_slot, alignment=Qt.AlignmentFlag.AlignHCenter)

    def set_active(self, active: bool) -> None:
        self.setStyleSheet(
            UNIT_RESPONSE_MEMBER_CARD_ACTIVE_STYLE if active else UNIT_RESPONSE_MEMBER_CARD_STYLE
        )

    def hydrate_if_cached(self) -> bool:
        """Apply an already-cached HiDPI QPixmap without rendering; return True when up-to-date."""
        key = self._owner._member_render_key(self.gid)
        if self._live_plot is not None:
            if self._rendered_key != key:
                self._owner._populate_unit_response_for_unit(self._live_plot, self.gid)
                self._rendered_key = key
            return True
        if self._rendered_key == key:
            return True
        pix = self._owner._member_pixmap_cache.get(key)
        if pix is not None:
            self._pixmap_label.setPixmap(pix)
            self._rendered_key = key
            return True
        return False

    def hydrate(self, force: bool = False) -> None:
        """Populate this member card for current view parameters (cached pixmap or live plot)."""
        key = self._owner._member_render_key(self.gid)
        if self._live_plot is not None:
            if force or self._rendered_key != key:
                self._owner._populate_unit_response_for_unit(self._live_plot, self.gid)
                self._rendered_key = key
            return
        if not force and self._rendered_key == key:
            return
        pix = self._owner._get_or_render_member_pixmap(self.gid)
        if pix is not None:
            self._pixmap_label.setPixmap(pix)
            self._rendered_key = key

    @property
    def unit_response_plot(self) -> UnitResponsePlot:
        """Lazily promote the static snapshot slot to a live UnitResponsePlot on explicit access."""
        if self._live_plot is None:
            live = UnitResponsePlot(self.plot_slot, defer_initial_render=True)
            live.export_name_provider = lambda gid=self.gid: f"{gid}_raster_psth"
            self._owner._populate_unit_response_for_unit(live, self.gid)
            self._pixmap_label.hide()
            self._pixmap_label.clear()
            slot_layout = self.plot_slot.layout()
            if slot_layout is not None:
                slot_layout.addWidget(live)
                slot_layout.activate()
            live.setGeometry(0, 0, live.width(), live.height())
            live.ensurePolished()
            if live.layout() is not None:
                live.layout().activate()
            self._live_plot = live
            self._rendered_key = self._owner._member_render_key(self.gid)
        return self._live_plot

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_click(self.gid)
            event.accept()
            return
        super().mousePressEvent(event)


class ClusteringView(QWidget):
    """Workspace for multi-experiment response clustering, PCA, and active learning."""

    request_home = pyqtSignal()
    request_detection_view = pyqtSignal()
    experiment_added = pyqtSignal(str)
    experiment_removed = pyqtSignal(str)
    unit_selected = pyqtSignal(str)
    clustering_completed = pyqtSignal(int)  # n_clusters
    cluster_merged = pyqtSignal(int, int)   # (src_cluster_id, tgt_cluster_id)
    cluster_moved = pyqtSignal(int, int)    # (src_cluster_id, tgt_cluster_id)
    candidate_updated = pyqtSignal(object)  # ActiveLearningCandidate
    view_settings_changed = pyqtSignal(tuple)  # (start_ms, end_ms)
    bin_size_changed = pyqtSignal(float)  # bin_size_ms
    psth_norm_mode_changed = pyqtSignal(str)  # norm_mode ("peak", "probability", "raw")
    progress_updated = pyqtSignal(int, int, int, int, int)  # curated, seeded, auto, suggested, total
    experiments_changed = pyqtSignal()
    dirty_changed = pyqtSignal(bool)
    confidence_threshold_changed = pyqtSignal(float)
    auto_assign_changed = pyqtSignal(bool)
    active_learning_toggled = pyqtSignal(bool)

    def __init__(
        self,
        parent: QWidget | None = None,
        start_with_clusters: bool = False,
        load_mock: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("clusteringWorkspace")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.loaded_experiments: Dict[str, Phase1ExportMetadata] = {}
        self._experiment_probes: Dict[str, np.ndarray] = {}
        self._current_probe_session_id: Optional[str] = None
        self._probe_shank_items: List[QGraphicsPolygonItem] = []
        self._probe_shank_labels: List[pg.TextItem] = []
        self._experiment_spikes: Dict[str, Dict[int, np.ndarray]] = {}
        self._experiment_stimulus: Dict[str, np.ndarray] = {}
        self._experiment_pulse_durations: Dict[str, float] = {}
        self._unit_psth_cache: Dict[Tuple[str, Tuple[float, float], float], np.ndarray] = {}
        self._is_mock: bool = load_mock
        self._pooled_rows: List[Dict[str, Any]] = []
        self._pooled_units_map: Dict[str, Dict[str, Any]] = {}
        self._unit_status_map: Dict[str, str] = {}
        self._auto_confidences: Dict[str, float] = {}
        self.auto_assign_threshold: float = 0.85
        self.auto_assign_enabled: bool = False
        self.al_n_estimators: int = 100
        self.al_random_state: int = 42
        self.al_max_distance_sigma: float = 3.5
        self._is_dirty: bool = False
        self.current_window_ms: Tuple[float, float] = (-30.0, 50.0)
        self.current_bin_size_ms: float = 2.0
        self.psth_smooth: bool = True
        self.psth_gaussian_sigma_ms: float = 2.0
        self.cluster_psth_norm: str = "probability"
        self.raster_order_by: str = "presentation"
        self.raster_marker_size: float = 3.0
        self._unit_raster_psth_cache: Dict[Tuple[str, Tuple[float, float], float, str, bool, float], Dict[str, Any]] = {}
        self._member_pixmap_cache: Dict[Tuple[Any, ...], QPixmap] = {}
        self._member_snapshot_plot: Optional[UnitResponsePlot] = None
        self._prewarm_queue: List[str] = []
        self._prewarm_data_queue: List[str] = []
        self._prewarm_generation: int = 0
        self._prewarm_worker_active: bool = False
        self._prewarm_signals = _PrewarmSignals(self)
        self._prewarm_signals.batch_ready.connect(self._on_prewarm_batch_ready)
        self._prewarm_timer: QTimer = QTimer(self)
        self._prewarm_timer.setInterval(10)
        self._prewarm_timer.setSingleShot(True)
        self._prewarm_timer.timeout.connect(self._process_next_prewarm_unit)
        self._scroll_hydrate_timer: QTimer = QTimer(self)
        self._scroll_hydrate_timer.setInterval(16)
        self._scroll_hydrate_timer.setSingleShot(True)
        self._scroll_hydrate_timer.timeout.connect(self._on_scroll_hydrate_timeout)
        self.current_project_dir: Optional[Path] = None
        self._feature_window_customized: bool = False
        self.feature_config: FeatureConfiguration = FeatureConfiguration(
            psth_smooth=self.psth_smooth,
            psth_gaussian_sigma_ms=self.psth_gaussian_sigma_ms,
        )
        self.embedding_params: EmbeddingParameters = EmbeddingParameters()
        self.ap_params: AffinityPropagationParameters = AffinityPropagationParameters()
        self._active_learning_worker: Optional[ActiveLearningWorker] = None
        self._tracked_al_workers: List[ActiveLearningWorker] = []
        self._is_closing: bool = False
        self._start_with_clusters: bool = start_with_clusters

        self.active_clustering_params: ClusteringParameters = ClusteringParameters(
            metric="euclidean",
            linkage="ward",
            cutoff_mode="k",
            n_clusters=4,
            feature_config=self.feature_config,
            presented_window_ms=self.current_window_ms,
            presented_bin_size_ms=self.current_bin_size_ms,
            psth_smooth=self.psth_smooth,
            psth_gaussian_sigma_ms=self.psth_gaussian_sigma_ms,
            cluster_psth_norm=self.cluster_psth_norm,
        )
        self.current_clustering_result: Optional[ClusteringResult] = None
        self._scatter_points_map: Dict[str, Tuple[float, float, int]] = {}
        self._current_selected_gid: Optional[str] = None

        # Active Learning state
        self.active_candidate: Optional[ActiveLearningCandidate] = None
        self.ranked_candidates: List[ActiveLearningCandidate] = []
        self._skipped_gids: Set[str] = set()
        self._all_unit_probabilities: Dict[str, Dict[int, float]] = {}
        self._unit_candidates_map: Dict[str, ActiveLearningCandidate] = {}

        # Cluster card layout order and widget references
        self._cluster_card_order: List[int] = []
        self._cluster_cards: Dict[int, ClusterCardWidget] = {}
        self._new_cluster_card: Optional[NewClusterCardWidget] = None
        self.overlay_focused_unit_enabled: bool = True
        self.shared_y_axis_enabled: bool = False
        self._cluster_sort_mode: str = "id"
        self._list_panel_collapsed: bool = False
        self._list_panel_expanded_height: int = 260
        self._suppress_list_panel_selection: bool = False

        # Cluster customizations: names, colors, and hotkeys
        self._cluster_names: Dict[int, str] = {}
        self._cluster_colors: Dict[int, str] = {}
        self.cluster_color_service: ClusterColorService = ClusterColorService(self._cluster_colors)
        self._cluster_hotkeys: Dict[int, str] = {}
        self._cluster_shortcuts: Dict[int, QShortcut] = {}

        # Active Learning toggle and Cluster Comparison state
        self.active_learning_enabled: bool = True
        self.auto_assign_enabled: bool = False
        self._suggested_auto_assignments: Dict[str, int] = {}
        self._comparison_cluster_id: Optional[int] = None
        self.comparison_viewers: List[QWidget] = []
        self._dismissed_cluster_candidates: Dict[int, Set[str]] = {}
        self._undo_stack: List[Dict[str, Any]] = []
        self._redo_stack: List[Dict[str, Any]] = []
        self._is_applying_undo_redo: bool = False
        self._focus_source: str = "queue"
        self._programmatic_unit_select: bool = False

        # Performance optimization caches
        self._cached_spike_data_map: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
        self._cached_spike_data_map_key: Optional[Tuple[str, ...]] = None
        self._cached_psth_matrix_tuple: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._cached_psth_matrix_key: Optional[Any] = None
        self._probe_shanks_cache: Dict[str, List[Dict[str, float]]] = {}
        self._probe_layout_cache: Dict[Tuple[str, int], Dict[str, Tuple[float, float]]] = {}
        self._comparison_member_viewers_map: Dict[str, QWidget] = {}
        self._comparison_rendered_cid: Optional[int] = None

        self._init_ui()
        if parent is None:
            self.show()
        if load_mock or start_with_clusters:
            self._is_mock = True
            self._load_mock_data()
        else:
            self._is_mock = False
            self.table_exps.setRowCount(0)
            self.table_pooled_units.setRowCount(0)
            self.lbl_pool_count.setText("0 units")
            self.lbl_candidate.setText("No Experiment Selected")
            self.image_viewer.clear_preview()
            self._populate_cluster_filter_options()
            self.scatter_item.clear()
            self.cand_scatter.clear()

    def _get_cluster_color(self, cluster_id: int) -> str:
        """Resolve true swatch/plot hex colour for cluster_id via ClusterColorService."""
        if not hasattr(self, "cluster_color_service") or self.cluster_color_service.custom_colors is not self._cluster_colors:
            self.cluster_color_service = ClusterColorService(self._cluster_colors)
        return self.cluster_color_service.get_color(int(cluster_id))

    def _get_cluster_text_color(self, cluster_id: int) -> str:
        """Resolve readable text hex colour on white for cluster_id via ClusterColorService."""
        if not hasattr(self, "cluster_color_service") or self.cluster_color_service.custom_colors is not self._cluster_colors:
            self.cluster_color_service = ClusterColorService(self._cluster_colors)
        return self.cluster_color_service.get_text_color(int(cluster_id))

    def _init_ui(self) -> None:
        """Build the 4-region Clustering workspace layout (§ 5, § 5.1, § 5.2)."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Hidden compatibility progress bar (mirrored in MainWindow header)
        self.progress_bar = ClassificationProgressBar(self)
        self.progress_bar.setVisible(False)

        # Main horizontal splitter: [Unit List | Centre (600 px fixed) | Right]
        self.main_splitter = ResettableSplitter(
            Qt.Orientation.Horizontal,
            self,
            default_sizes=[UNIT_LIST_DEFAULT_WIDTH, CENTRE_PANEL_WIDTH, 934],
            settings_key="clustering_main",
        )
        main_layout.addWidget(self.main_splitter, stretch=1)

        # =====================================================================
        # REGION 1: Unit List Column (Left: ~380 px, min 300, max 600)
        # =====================================================================
        self.unit_list_panel = QFrame(self)
        self.unit_list_panel.setObjectName("unitListPanel")
        self.unit_list_panel.setMinimumWidth(UNIT_LIST_MIN_WIDTH)
        self.unit_list_panel.setMaximumWidth(UNIT_LIST_MAX_WIDTH)
        self.unit_list_panel.setStyleSheet(UNIT_LIST_PANEL_STYLE)
        unit_list_layout = QVBoxLayout(self.unit_list_panel)
        unit_list_layout.setContentsMargins(10, 10, 10, 10)
        unit_list_layout.setSpacing(8)

        # Hidden compatibility experiments table for ingestion/project tests
        self.table_exps = QTableWidget(self.unit_list_panel)
        self.table_exps.setColumnCount(3)
        self.table_exps.setHorizontalHeaderLabels(["Experiment", "Protocol", "Tagged Cells"])
        self.table_exps.verticalHeader().setVisible(False)
        self.table_exps.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_exps.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table_exps.setVisible(False)

        # Filter row ("Show" dropdown + unit count)
        units_header_layout = QHBoxLayout()
        units_header_layout.setContentsMargins(0, 0, 0, 0)
        units_header_layout.setSpacing(6)
        self.lbl_pooled_units_title = QLabel("Show")
        self.lbl_pooled_units_title.setStyleSheet(CLUSTERS_SECTION_SUBHEADER_STYLE)
        units_header_layout.addWidget(self.lbl_pooled_units_title)

        self.combo_filter_cluster = QComboBox()
        self.combo_filter_cluster.setToolTip("Filter pooled units by status or cluster")
        self.combo_filter_cluster.setStyleSheet(CLUSTERS_FILTER_COMBO_STYLE)
        self.combo_filter_cluster.setMaxVisibleItems(CLUSTERS_FILTER_COMBO_MAX_VISIBLE)
        self.combo_filter_cluster.setIconSize(QSize(10, 10))
        self.combo_filter_cluster.view().setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.combo_filter_cluster.view().setMaximumHeight(CLUSTERS_FILTER_COMBO_MAX_POPUP_HEIGHT)
        self.combo_filter_cluster.addItem("All units", -2)
        self.combo_filter_cluster.currentIndexChanged.connect(self._apply_cluster_filter)
        units_header_layout.addWidget(self.combo_filter_cluster, stretch=1)

        self.lbl_pooled_units_badge = QLabel("0 units")
        self.lbl_pool_count = self.lbl_pooled_units_badge
        self.lbl_unit_count = self.lbl_pooled_units_badge
        self.lbl_pooled_units_badge.setStyleSheet(CLUSTERS_UNIT_COUNT_LABEL_STYLE)
        units_header_layout.addWidget(self.lbl_pooled_units_badge)
        unit_list_layout.addLayout(units_header_layout)

        # Pooled units table
        self.table_pooled_units = QTableWidget()
        self.table_pooled_units.setStyleSheet(CLUSTERS_POOLED_TABLE_STYLE)
        self.table_pooled_units.setColumnCount(7)
        self.table_pooled_units.setHorizontalHeaderLabels([
            "Unit", "Session", "Cluster", "Lat", "Rel.", "SALT p", "Status"
        ])
        pooled_header_tooltips = [
            "Unit ID",
            "Recording session",
            "Assigned response cluster",
            "First-spike latency (ms)",
            "Response reliability (fraction of pulses with spike)",
            "SALT test p-value",
            "Curation status",
        ]
        for col_idx, tip in enumerate(pooled_header_tooltips):
            hdr_item = self.table_pooled_units.horizontalHeaderItem(col_idx)
            if hdr_item is not None:
                hdr_item.setToolTip(tip)
        self.table_pooled_units.verticalHeader().setVisible(False)
        self.table_pooled_units.verticalHeader().setDefaultSectionSize(22)
        for c in range(7):
            self.table_pooled_units.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeMode.Interactive)
        self.table_pooled_units.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table_pooled_units.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.table_pooled_units.setWordWrap(False)
        self.table_pooled_units.setIconSize(QSize(12, 12))
        self.table_pooled_units.horizontalHeader().setStretchLastSection(False)
        col_widths = {0: 36, 1: 50, 2: 76, 3: 42, 4: 42, 5: 54, 6: 46}
        for col_idx, width in col_widths.items():
            self.table_pooled_units.setColumnWidth(col_idx, width)
        self.table_pooled_units.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_pooled_units.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table_pooled_units.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table_pooled_units.customContextMenuRequested.connect(self._show_pooled_units_context_menu)
        self.table_pooled_units.horizontalHeader().setSortIndicatorShown(True)
        self.table_pooled_units.horizontalHeader().setSectionsClickable(True)
        self.table_pooled_units.horizontalHeader().sectionClicked.connect(self._on_units_header_clicked)
        self.table_pooled_units.itemSelectionChanged.connect(self._on_pooled_unit_selected)
        self.table_pooled_units.installEventFilter(self)
        unit_list_layout.addWidget(self.table_pooled_units, stretch=1)

        # Bottom of Region 1: Embedding + Probe side-by-side (inside horizontal scroll guard so plots never shrink)
        embed_probe_scroll = QScrollArea()
        embed_probe_scroll.setWidgetResizable(True)
        embed_probe_scroll.setFrameShape(QFrame.Shape.NoFrame)
        embed_probe_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        embed_probe_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        embed_probe_scroll.setFixedHeight(318)
        embed_probe_container = QWidget()
        embed_probe_layout = QHBoxLayout(embed_probe_container)
        embed_probe_layout.setContentsMargins(0, 0, 0, 0)
        embed_probe_layout.setSpacing(8)

        # Embedding column
        pca_container = QWidget()
        pca_vbox = QVBoxLayout(pca_container)
        pca_vbox.setContentsMargins(0, 0, 0, 0)
        pca_vbox.setSpacing(4)

        pca_header = QHBoxLayout()
        pca_header.setContentsMargins(0, 0, 0, 0)
        pca_header.setSpacing(4)
        lbl_method = QLabel("Embedding")
        lbl_method.setStyleSheet(CLUSTERS_SECTION_SUBHEADER_STYLE)
        lbl_method.setFixedHeight(24)
        pca_header.addWidget(lbl_method)
        pca_header.addStretch()

        self.combo_embedding_method = QComboBox()
        self.combo_embedding_method.addItem("PCA", "PCA")
        self.combo_embedding_method.addItem("UMAP", "UMAP")
        self.combo_embedding_method.setStyleSheet(CLUSTERS_TOOLBAR_COMBO_STYLE)
        install_combo_chevron(self.combo_embedding_method)
        self.combo_embedding_method.currentIndexChanged.connect(self._on_embedding_method_changed)
        pca_header.addWidget(self.combo_embedding_method)

        self.btn_embedding_settings = QPushButton("⚙")
        self.btn_configure_embedding = self.btn_embedding_settings
        self.btn_embed_settings = self.btn_embedding_settings
        self.btn_embedding_settings.setToolTip("Configure 2D embedding hyperparameters (PCA components / UMAP neighbors)")
        self.btn_embedding_settings.setStyleSheet(CLUSTERS_GHOST_ICON_BTN_STYLE)
        self.btn_embedding_settings.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_embedding_settings.clicked.connect(self.prompt_embedding_settings)
        pca_header.addWidget(self.btn_embedding_settings)
        pca_vbox.addLayout(pca_header)

        self.pca_plot = pg.PlotWidget()
        self.pca_plot.setFixedSize(*PLOT_EMBEDDING_SIZE)
        self.pca_plot.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.pca_plot.setAspectLocked(True)
        apply_plot_style(
            self.pca_plot,
            compact=True,
            left_label="PC2 (0.0% var.)",
            bottom_label="PC1 (0.0% var.)",
        )
        self.scatter_item = pg.ScatterPlotItem(
            size=EMBEDDING_POINT_SIZE,
            pen=pg.mkPen(COLOR_PANEL, width=0.5),
            hoverable=True,
        )
        self.scatter_item.sigClicked.connect(self._on_scatter_point_clicked)
        self.scatter_item.sigHovered.connect(self._on_embedding_point_hovered)
        self.pca_plot.addItem(self.scatter_item)

        self.cand_scatter = pg.ScatterPlotItem(
            size=EMBEDDING_FOCUSED_RING_SIZE,
            pen=pg.mkPen(COLOR_TEXT, width=EMBEDDING_FOCUSED_RING_WIDTH),
            brush=None,
            symbol="o",
        )
        self.cand_scatter.setZValue(20)
        self.pca_plot.addItem(self.cand_scatter)
        install_figure_export_menu(
            self.pca_plot,
            export_target=self.pca_plot,
            default_name=lambda: (
                f"embedding_{str(self.combo_embedding_method.currentData() or self.combo_embedding_method.currentText() or 'pca').lower()}"
            ),
        )
        pca_vbox.addWidget(self.pca_plot, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        embed_probe_layout.addWidget(pca_container)

        # Probe column
        probe_container = QWidget()
        probe_vbox = QVBoxLayout(probe_container)
        probe_vbox.setContentsMargins(0, 0, 0, 0)
        probe_vbox.setSpacing(4)

        probe_header = QHBoxLayout()
        probe_header.setContentsMargins(0, 0, 0, 0)
        probe_header.setSpacing(4)
        lbl_probe_hdr = QLabel("Probe")
        lbl_probe_hdr.setStyleSheet(CLUSTERS_SECTION_SUBHEADER_STYLE)
        lbl_probe_hdr.setFixedHeight(24)
        probe_header.addWidget(lbl_probe_hdr)
        probe_header.addStretch(1)
        probe_vbox.addLayout(probe_header)

        self.lbl_probe_experiment = QLabel("Experiment: None")
        self.lbl_probe_experiment.setStyleSheet(CLUSTERS_PROBE_EXP_LABEL_STYLE)
        self.lbl_probe_experiment.setVisible(False)

        self.lbl_probe_unit_info = QLabel("Depth: -  |  Channel: -")
        self.lbl_probe_unit_info.setStyleSheet(CLUSTERS_PROBE_INFO_LABEL_STYLE)
        self.lbl_probe_unit_info.setVisible(False)

        self.probe_widget = SiliconProbeWidget(self)
        self.probe_widget.export_name_provider = lambda: (
            f"{self._current_probe_session_id}_silicon_probe"
            if self._current_probe_session_id
            else "silicon_probe"
        )
        self.probe_plot = self.probe_widget.plot_widget
        self._init_probe_view()
        probe_vbox.addWidget(self.probe_widget, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        probe_vbox.addStretch(1)
        embed_probe_layout.addWidget(probe_container)
        embed_probe_layout.addStretch(1)

        embed_probe_scroll.setWidget(embed_probe_container)
        unit_list_layout.addWidget(embed_probe_scroll)

        self.main_splitter.addWidget(self.unit_list_panel)

        # =====================================================================
        # REGION 2 + 3: Centre (600 px fixed = 300 px Focused Unit + 300 px Inspector Slot)
        # =====================================================================
        self.centre_panel = QFrame(self)
        self.centre_panel.setObjectName("centrePanel")
        self.centre_panel.setFixedWidth(CENTRE_PANEL_WIDTH)
        centre_layout = QHBoxLayout(self.centre_panel)
        centre_layout.setContentsMargins(0, 0, 0, 0)
        centre_layout.setSpacing(0)

        # --- Region 2: Focused Unit (300 px fixed, non-resizable against inspector) ---
        self.focused_unit_panel = QFrame(self.centre_panel)
        self.focused_unit_panel.setObjectName("focusedUnitPanel")
        self.focused_unit_panel.setFixedWidth(FOCUSED_UNIT_WIDTH)
        self.focused_unit_panel.setStyleSheet(FOCUSED_UNIT_PANEL_STYLE)
        self.viewer_pane = self.focused_unit_panel
        self.curation_pane = self.focused_unit_panel
        focused_layout = QVBoxLayout(self.focused_unit_panel)
        focused_layout.setContentsMargins(16, 10, 16, 10)
        focused_layout.setSpacing(6)

        # Focused Unit Header (64 px height, 268 px content width)
        self.focused_unit_header = QWidget(self.focused_unit_panel)
        self.focused_unit_header.setFixedHeight(FOCUSED_UNIT_HEADER_HEIGHT)
        focused_header_vbox = QVBoxLayout(self.focused_unit_header)
        focused_header_vbox.setContentsMargins(0, 2, 0, 2)
        focused_header_vbox.setSpacing(2)

        cand_header_layout = QHBoxLayout()
        cand_header_layout.setContentsMargins(0, 0, 0, 0)
        cand_header_layout.setSpacing(6)

        self.lbl_focused_unit_title = QLabel("No unit selected")
        self.lbl_focused_unit_title.setStyleSheet(FOCUSED_UNIT_TITLE_STYLE)
        cand_header_layout.addWidget(self.lbl_focused_unit_title, stretch=1)

        self.btn_return_to_candidate = QPushButton("↩ Suggested unit")
        self.btn_return_to_candidate.setToolTip("Return to suggested unit (G)")
        self.btn_return_to_candidate.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_return_to_candidate.setStyleSheet(FOCUSED_UNIT_SUGGESTED_BTN_STYLE)
        self.btn_return_to_candidate.clicked.connect(self.return_to_active_candidate)
        self.btn_return_to_candidate.setVisible(False)
        cand_header_layout.addWidget(self.btn_return_to_candidate)
        focused_header_vbox.addLayout(cand_header_layout)

        self.lbl_focused_unit_session = QLabel("")
        self.lbl_focused_unit_session.setStyleSheet(FOCUSED_UNIT_SESSION_STYLE)
        focused_header_vbox.addWidget(self.lbl_focused_unit_session)

        # Hidden compatibility label so existing tests inspecting lbl_candidate.text() keep working
        self.lbl_candidate = QLabel("Candidate Unit: Calculating Uncertainty...")
        self.lbl_candidate.setStyleSheet(CLUSTERS_CANDIDATE_LABEL_STYLE)
        self.lbl_candidate.setVisible(False)

        # Compatibility buttons (controlled from status bar / menu in WP8)
        self.btn_toggle_active_learning = QPushButton("Active Learning: ON")
        self.btn_toggle_active_learning.setCheckable(True)
        self.btn_toggle_active_learning.setChecked(True)
        self._update_active_learning_button_style()
        self.btn_toggle_active_learning.toggled.connect(self.set_active_learning_mode)
        self.btn_toggle_active_learning.setVisible(False)

        self.btn_shortcuts_help = QPushButton("?")
        self.btn_shortcuts_help.setFixedSize(26, 26)
        self.btn_shortcuts_help.setToolTip("Curation Keyboard Shortcuts & Help")
        self.btn_shortcuts_help.setStyleSheet(CLUSTERS_HELP_BTN_STYLE)
        self.btn_shortcuts_help.clicked.connect(self.show_curation_shortcuts_help)
        self.btn_shortcuts_help.setVisible(False)

        focused_layout.addWidget(self.focused_unit_header)

        self._focused_align_spacer = QWidget(self.focused_unit_panel)
        self._focused_align_spacer.setFixedHeight(0)
        focused_layout.addWidget(self._focused_align_spacer)

        self.unit_response_plot = UnitResponsePlot(self)
        self.unit_response_plot.export_name_provider = lambda: (
            f"{self._current_selected_gid}_raster_psth"
            if self._current_selected_gid
            else (
                f"{self.active_candidate.global_unit_id}_raster_psth"
                if self.active_candidate is not None
                else "unit_response"
            )
        )
        self.image_viewer = self.unit_response_plot
        focused_layout.addWidget(self.unit_response_plot, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        self.cluster_prob_bar = ClusterProbabilityBarWidget(self)
        self.cluster_prob_bar.cluster_clicked.connect(self.set_cluster_comparison)
        focused_layout.addWidget(self.cluster_prob_bar)

        self.marginal_widget = MarginalDistributionsWidget(self)
        focused_layout.addWidget(self.marginal_widget)

        # Action row at bottom of Focused Unit column: Undo · Skip · Unassign
        focused_actions_layout = QHBoxLayout()
        focused_actions_layout.setContentsMargins(0, 4, 0, 0)
        focused_actions_layout.setSpacing(6)

        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setToolTip("Undo last curation action (Ctrl+Z)")
        self.btn_undo.setStyleSheet(SECONDARY_BTN_STYLE)
        self.btn_undo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_undo.clicked.connect(self.undo)
        self.btn_undo.setEnabled(False)
        focused_actions_layout.addWidget(self.btn_undo)

        self.btn_skip_candidate = QPushButton("Skip")
        self.btn_skip_candidate.setToolTip("Skip current unit (S / Space)")
        self.btn_skip_candidate.setStyleSheet(SECONDARY_BTN_STYLE)
        self.btn_skip_candidate.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_skip_candidate.clicked.connect(self.skip_current_candidate)
        focused_actions_layout.addWidget(self.btn_skip_candidate)

        self.btn_unassign = QPushButton("Unassign")
        self.btn_unassign.setToolTip("Unassign focused unit (Backspace)")
        self.btn_unassign.setStyleSheet(FOCUSED_UNIT_UNASSIGN_BTN_STYLE)
        self.btn_unassign.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_unassign.clicked.connect(self.unassign_focused_unit)
        self.btn_unassign.setVisible(False)
        focused_actions_layout.addWidget(self.btn_unassign)

        focused_actions_layout.addStretch(1)

        # Hidden compatibility Save button
        self.btn_save_pool = QPushButton("Save")
        self.btn_save_pool.setStyleSheet(CLUSTERS_SUGGESTED_TOOLBAR_BTN_PRIMARY_STYLE)
        self.btn_save_pool.clicked.connect(self._on_save_pooled_curation)
        self.btn_save_pool.setVisible(False)

        focused_layout.addLayout(focused_actions_layout)
        focused_layout.addStretch(1)
        centre_layout.addWidget(self.focused_unit_panel)

        # --- Region 3: Reserved Inspector Slot (300 px fixed, always visible) ---
        self.inspector_slot = QFrame(self.centre_panel)
        self.inspector_slot.setObjectName("inspectorSlotPanel")
        self.inspector_slot.setFixedWidth(INSPECTOR_SLOT_WIDTH)
        self.inspector_slot.setStyleSheet(INSPECTOR_SLOT_PANEL_STYLE)
        inspector_slot_layout = QVBoxLayout(self.inspector_slot)
        inspector_slot_layout.setContentsMargins(2, 8, 2, 8)
        inspector_slot_layout.setSpacing(0)

        # Empty state shown when no cluster is open
        self.inspector_empty_state = QFrame(self.inspector_slot)
        self.inspector_empty_state.setObjectName("inspectorEmptyState")
        self.inspector_empty_state.setStyleSheet(INSPECTOR_EMPTY_STATE_STYLE)
        empty_layout = QVBoxLayout(self.inspector_empty_state)
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_inspector_empty = QLabel("Click a cluster to compare its members")
        self.lbl_inspector_empty.setObjectName("inspectorEmptyLabel")
        self.lbl_inspector_empty.setStyleSheet(INSPECTOR_EMPTY_LABEL_STYLE)
        self.lbl_inspector_empty.setWordWrap(True)
        self.lbl_inspector_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(self.lbl_inspector_empty)
        inspector_slot_layout.addWidget(self.inspector_empty_state, stretch=1)

        # Cluster comparison pane (fills inspector_slot when a cluster is open)
        self.comparison_pane = QWidget(self.inspector_slot)
        self.comparison_pane.setObjectName("comparisonPane")
        self.comparison_pane.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.comparison_pane.setMinimumWidth(0)
        self.comparison_pane.setStyleSheet(CLUSTERS_COMPARISON_PANE_NEUTRAL_STYLE)
        comp_layout = QVBoxLayout(self.comparison_pane)
        comp_layout.setContentsMargins(2, 6, 2, 6)
        comp_layout.setSpacing(6)

        # Inspector Header (exact 46 px height, § 10.1)
        self.inspector_header_widget = QWidget(self.comparison_pane)
        self.inspector_header_widget.setFixedHeight(INSPECTOR_HEADER_HEIGHT)
        inspector_hdr_vbox = QVBoxLayout(self.inspector_header_widget)
        inspector_hdr_vbox.setContentsMargins(2, 2, 2, 2)
        inspector_hdr_vbox.setSpacing(2)

        inspector_row1 = QHBoxLayout()
        inspector_row1.setContentsMargins(0, 0, 0, 0)
        inspector_row1.setSpacing(6)

        self.lbl_inspector_swatch = QLabel(self.inspector_header_widget)
        self.lbl_inspector_swatch.setFixedSize(INSPECTOR_SWATCH_SIZE, INSPECTOR_SWATCH_SIZE)
        self.lbl_inspector_swatch.setStyleSheet(build_inspector_swatch_style(COLOR_ACCENT))
        inspector_row1.addWidget(self.lbl_inspector_swatch)

        self.lbl_inspector_name = QLabel("Cluster", self.inspector_header_widget)
        self.lbl_inspector_name.setStyleSheet(INSPECTOR_HEADER_NAME_STYLE)
        self.lbl_inspector_name.setMinimumWidth(0)
        inspector_row1.addWidget(self.lbl_inspector_name)

        self.lbl_inspector_cid = QLabel("C0", self.inspector_header_widget)
        self.lbl_inspector_cid.setStyleSheet(INSPECTOR_HEADER_CID_STYLE)
        inspector_row1.addWidget(self.lbl_inspector_cid)

        inspector_row1.addStretch(1)

        self.btn_comp_close = QPushButton("✕", self.inspector_header_widget)
        self.btn_comp_close.setToolTip("Close cluster inspector (Esc)")
        self.btn_comp_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_comp_close.setFixedSize(INSPECTOR_CLOSE_BTN_SIZE, INSPECTOR_CLOSE_BTN_SIZE)
        self.btn_comp_close.setStyleSheet(CLUSTERS_COMPARISON_CLOSE_BTN_STYLE)
        self.btn_comp_close.clicked.connect(lambda _checked=False: self.clear_cluster_comparison(return_to_suggestion=True))
        inspector_row1.addWidget(self.btn_comp_close)
        inspector_hdr_vbox.addLayout(inspector_row1)

        self.lbl_inspector_summary = QLabel("0 members · mean latency —", self.inspector_header_widget)
        self.lbl_inspector_summary.setStyleSheet(INSPECTOR_HEADER_SUMMARY_STYLE)
        inspector_hdr_vbox.addWidget(self.lbl_inspector_summary)

        comp_layout.addWidget(self.inspector_header_widget)

        # Hidden compatibility widgets kept for legacy test attribute access
        self.lbl_comparison_title = QLabel("Cluster Comparison", self.comparison_pane)
        self.lbl_comparison_title.setStyleSheet(build_comparison_title_style(COLOR_TEXT))
        self.lbl_comparison_title.setVisible(False)

        self.btn_comp_assign = QPushButton("✓", self.comparison_pane)
        self.btn_comp_assign.setToolTip("Assign current active unit to this cluster")
        self.btn_comp_assign.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_comp_assign.setStyleSheet(CLUSTERS_SUGGESTED_TOOLBAR_BTN_PRIMARY_STYLE)
        self.btn_comp_assign.clicked.connect(self._on_assign_to_comparison_cluster)
        self.btn_comp_assign.setVisible(False)
        self.btn_assign_candidate = self.btn_comp_assign

        self._inspector_align_spacer = QWidget(self.comparison_pane)
        self._inspector_align_spacer.setFixedHeight(0)
        comp_layout.addWidget(self._inspector_align_spacer)

        self.curated_members_panel = QWidget()
        panel1_layout = QVBoxLayout(self.curated_members_panel)
        panel1_layout.setContentsMargins(0, 0, 0, 0)
        panel1_layout.setSpacing(0)

        self.lbl_members_section = QLabel("▼ CURATED MEMBERS (N=0)")
        self.lbl_members_section.setStyleSheet(CLUSTERS_MEMBERS_SECTION_STYLE)
        self.lbl_members_section.setVisible(False)

        self.comparison_scroll_area = QScrollArea()
        self.comparison_scroll_area.setWidgetResizable(True)
        self.comparison_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.comparison_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.comparison_scroll_area.setStyleSheet(CLUSTERS_COMPARISON_SCROLL_STYLE)
        self.comparison_scroll_area.installEventFilter(self)
        self.comparison_scroll_area.viewport().installEventFilter(self)
        self.comparison_scroll_area.verticalScrollBar().valueChanged.connect(self._on_inspector_scrolled)
        self.comparison_list_widget = QWidget()
        self.comparison_list_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.comparison_list_widget.setStyleSheet(CLUSTERS_TRANSPARENT_BG_STYLE)
        self.comparison_list_layout = QVBoxLayout(self.comparison_list_widget)
        self.comparison_list_layout.setContentsMargins(0, 0, 0, 2)
        self.comparison_list_layout.setSpacing(6)
        self.comparison_scroll_area.setWidget(self.comparison_list_widget)
        panel1_layout.addWidget(self.comparison_scroll_area)
        comp_layout.addWidget(self.curated_members_panel, stretch=1)

        self.comparison_pane.setVisible(False)
        if hasattr(self, "inspector_empty_state"):
            self.inspector_empty_state.setVisible(True)
        self._refresh_embedding_and_probe_emphasis()
        self._refresh_list_panel()
        inspector_slot_layout.addWidget(self.comparison_pane, stretch=1)

        centre_layout.addWidget(self.inspector_slot)
        self.main_splitter.addWidget(self.centre_panel)

        # =====================================================================
        # REGION 4: Right (Response Clusters Card Grid + List Panel with Vertical Splitter)
        # =====================================================================
        self.right_panel = QWidget(self)
        self.right_panel.setMinimumWidth(RIGHT_PANEL_MIN_WIDTH)
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.right_splitter = ResettableSplitter(
            Qt.Orientation.Vertical,
            self.right_panel,
            default_sizes=[620, 260],
            settings_key="clustering_right",
        )
        right_layout.addWidget(self.right_splitter, stretch=1)

        # Top pane: Clusters area (toolbar + scrollable card grid)
        self.clusters_area_panel = QFrame(self.right_splitter)
        self.clusters_area_panel.setObjectName("clustersAreaPanel")
        self.clusters_area_panel.setMinimumHeight(CLUSTER_GRID_MIN_HEIGHT)
        self.clusters_area_panel.setStyleSheet(CLUSTERS_AREA_PANEL_STYLE)
        self.clusters_container = self.clusters_area_panel
        clusters_layout = QVBoxLayout(self.clusters_area_panel)
        clusters_layout.setContentsMargins(12, 10, 12, 8)
        clusters_layout.setSpacing(8)

        # Row 1 (Single-line Cluster Actions Toolbar, § 9.1)
        row1_action_bar = QHBoxLayout()
        row1_action_bar.setSpacing(6)

        lbl_clusters_hdr = QLabel("Response clusters")
        lbl_clusters_hdr.setStyleSheet(CLUSTERS_SECTION_SUBHEADER_STYLE)
        row1_action_bar.addWidget(lbl_clusters_hdr)

        self.lbl_clusters_count_chip = QLabel("0")
        self.lbl_clusters_count_chip.setStyleSheet(CLUSTERS_COUNT_CHIP_STYLE)
        self.lbl_clusters_count_chip.setToolTip("Number of active response clusters")
        row1_action_bar.addWidget(self.lbl_clusters_count_chip)

        row1_action_bar.addStretch(1)

        self.btn_seed_clusters = QPushButton("Seed clusters")
        self.btn_seed_clusters.setToolTip("Discover initial response clusters using Affinity Propagation with exemplar filtering")
        self.btn_seed_clusters.setStyleSheet(CLUSTERS_TOOLBAR_SECONDARY_BTN_STYLE)
        self.btn_seed_clusters.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_seed_clusters.clicked.connect(self.prompt_seed_clusters)
        row1_action_bar.addWidget(self.btn_seed_clusters)

        self.btn_cluster = QPushButton("Run clustering")
        self.btn_cluster.setToolTip("Configure and run response clustering (Hierarchical, K-Means, GMM)")
        self.btn_cluster.setStyleSheet(CLUSTERS_TOOLBAR_SECONDARY_BTN_STYLE)
        self.btn_cluster.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cluster.clicked.connect(self.prompt_hierarchical_clustering)
        row1_action_bar.addWidget(self.btn_cluster)

        div1 = QFrame(self.clusters_area_panel)
        div1.setFrameShape(QFrame.Shape.VLine)
        div1.setFixedHeight(18)
        div1.setStyleSheet(CLUSTERS_VERTICAL_DIVIDER_STYLE)
        row1_action_bar.addWidget(div1)

        self.btn_reset_clusters = QPushButton("↻ Reset clustering")
        self.btn_reset_clusters.setObjectName("resetClustersButton")
        self.btn_reset_clusters.setToolTip("Reset all clusters and unit assignments to unassigned")
        self.btn_reset_clusters.setStyleSheet(CLUSTERS_TOOLBAR_RESET_BTN_STYLE)
        self.btn_reset_clusters.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset_clusters.clicked.connect(lambda: self.reset_clustering(confirm=True))
        row1_action_bar.addWidget(self.btn_reset_clusters)

        div2 = QFrame(self.clusters_area_panel)
        div2.setFrameShape(QFrame.Shape.VLine)
        div2.setFixedHeight(18)
        div2.setStyleSheet(CLUSTERS_VERTICAL_DIVIDER_STYLE)
        row1_action_bar.addWidget(div2)

        # View options popover (§ 9.1)
        self.btn_view_options = QPushButton("")
        self.btn_view_options.setIcon(_make_sliders_icon(16, COLOR_TEXT_SECONDARY))
        self.btn_view_options.setIconSize(QSize(16, 16))
        self.btn_view_options.setObjectName("clustersViewOptionsButton")
        self.btn_view_options.setToolTip("View options")
        self.btn_view_options.setStyleSheet(CLUSTERS_GHOST_ICON_BTN_STYLE)
        self.btn_view_options.setCursor(Qt.CursorShape.PointingHandCursor)

        self._view_options_menu = QMenu(self.btn_view_options)
        self._view_options_menu.setStyleSheet(CLUSTERS_MENU_STYLE)

        self.act_overlay_focused_unit = QAction("Overlay focused unit", self._view_options_menu)
        self.act_overlay_focused_unit.setCheckable(True)
        self.act_overlay_focused_unit.setChecked(self.overlay_focused_unit_enabled)
        self.act_overlay_focused_unit.toggled.connect(self._on_overlay_focused_unit_toggled)
        self._view_options_menu.addAction(self.act_overlay_focused_unit)

        self.act_shared_y_axis = QAction("Shared y-axis", self._view_options_menu)
        self.act_shared_y_axis.setCheckable(True)
        self.act_shared_y_axis.setChecked(self.shared_y_axis_enabled)
        self.act_shared_y_axis.toggled.connect(self._on_shared_y_axis_toggled)
        self._view_options_menu.addAction(self.act_shared_y_axis)

        self._view_options_menu.addSeparator()
        sort_menu = self._view_options_menu.addMenu("Sort by")
        sort_menu.setStyleSheet(CLUSTERS_MENU_STYLE)
        self._sort_action_group = QActionGroup(sort_menu)
        self._sort_action_group.setExclusive(True)
        self._sort_actions: Dict[str, QAction] = {}
        for mode_key, mode_label in (
            ("id", "Cluster ID"),
            ("size", "Size"),
            ("latency", "Latency"),
            ("name", "Name"),
        ):
            act_s = QAction(mode_label, sort_menu)
            act_s.setCheckable(True)
            act_s.setChecked(mode_key == self._cluster_sort_mode)
            act_s.setData(mode_key)
            act_s.triggered.connect(lambda _checked=False, k=mode_key: self._sort_cluster_cards(k))
            self._sort_action_group.addAction(act_s)
            sort_menu.addAction(act_s)
            self._sort_actions[mode_key] = act_s

        self.btn_view_options.setMenu(self._view_options_menu)
        row1_action_bar.addWidget(self.btn_view_options)

        # More options ('⋯') menu (§ 9.1)
        self.btn_clusters_more = QPushButton("⋯")
        self.btn_clusters_more.setObjectName("clustersMoreButton")
        self.btn_clusters_more.setToolTip("More clustering options")
        self.btn_clusters_more.setStyleSheet(CLUSTERS_GHOST_ICON_BTN_STYLE)
        self.btn_clusters_more.setCursor(Qt.CursorShape.PointingHandCursor)

        self._clusters_more_menu = QMenu(self.btn_clusters_more)
        self._clusters_more_menu.setStyleSheet(CLUSTERS_MENU_STYLE)

        self.act_more_feature_settings = QAction("Feature settings…", self._clusters_more_menu)
        self.act_more_feature_settings.triggered.connect(self.prompt_feature_settings)
        self._clusters_more_menu.addAction(self.act_more_feature_settings)

        self.act_more_customize_clusters = QAction("Customize clusters…", self._clusters_more_menu)
        self.act_more_customize_clusters.triggered.connect(self.prompt_customize_clusters)
        self._clusters_more_menu.addAction(self.act_more_customize_clusters)

        self.act_more_confidence_threshold = QAction("Confidence threshold…", self._clusters_more_menu)
        self.act_more_confidence_threshold.triggered.connect(self.prompt_confidence_threshold)
        self._clusters_more_menu.addAction(self.act_more_confidence_threshold)

        self._clusters_more_menu.addSeparator()

        self.act_more_revert_auto = QAction("Revert auto-assignments…", self._clusters_more_menu)
        self.act_more_revert_auto.triggered.connect(self.revert_auto_assignments)
        self._clusters_more_menu.addAction(self.act_more_revert_auto)

        self.btn_clusters_more.setMenu(self._clusters_more_menu)
        row1_action_bar.addWidget(self.btn_clusters_more)

        self.btn_customize_clusters = QPushButton("Customize clusters")
        self.btn_customize_clusters.setStyleSheet(CLUSTERS_TOOLBAR_ACTION_BTN_STYLE)
        self.btn_customize_clusters.clicked.connect(self.prompt_customize_clusters)
        self.btn_customize_clusters.setVisible(False)

        clusters_layout.addLayout(row1_action_bar)

        # Compatibility hidden/compact Row 2 controls (moved to menus/dialogs in WP8)
        self.edit_win_start = QLineEdit("-30", self.clusters_area_panel)
        self.edit_win_start.setVisible(False)
        self.edit_win_end = QLineEdit("+50", self.clusters_area_panel)
        self.edit_win_end.setVisible(False)

        self.spin_bin_size = QDoubleSpinBox(self.clusters_area_panel)
        self.spin_bin_size.setRange(0.2, 50.0)
        self.spin_bin_size.setSingleStep(0.5)
        self.spin_bin_size.setDecimals(1)
        self.spin_bin_size.setValue(2.0)
        self.spin_bin_size.valueChanged.connect(self._on_bin_size_changed)
        self.spin_bin_size.setVisible(False)

        self.combo_order = QComboBox(self.clusters_area_panel)
        self.combo_order.addItem("Presentation Order", "presentation")
        self.combo_order.addItem("First-Spike Latency", "latency")
        self.combo_order.currentIndexChanged.connect(self._on_combo_order_changed)
        self.combo_order.setVisible(False)

        self.chk_smooth = QCheckBox("Smooth", self.clusters_area_panel)
        self.chk_smooth.setChecked(True)
        self.chk_smooth.toggled.connect(self.set_psth_smooth)
        self.chk_smooth.setVisible(False)

        self.spin_sigma = QDoubleSpinBox(self.clusters_area_panel)
        self.spin_sigma.setRange(0.2, 20.0)
        self.spin_sigma.setSingleStep(0.5)
        self.spin_sigma.setDecimals(1)
        self.spin_sigma.setValue(2.0)
        self.spin_sigma.valueChanged.connect(self.set_psth_gaussian_sigma)
        self.spin_sigma.setVisible(False)

        self.combo_psth_norm = QComboBox(self.clusters_area_panel)
        self.combo_psth_norm.addItem("Peak [0–1]", "peak")
        self.combo_psth_norm.addItem("Probability", "probability")
        self.combo_psth_norm.addItem("Raw (Hz)", "raw")
        self.combo_psth_norm.setCurrentIndex(1)
        self.combo_psth_norm.currentIndexChanged.connect(self._on_psth_norm_changed)
        self.combo_psth_norm.setVisible(False)

        self.chk_auto_assign = QCheckBox("Auto-assign", self.clusters_area_panel)
        self.chk_auto_assign.setChecked(False)
        self.chk_auto_assign.toggled.connect(self._on_auto_assign_toggled)
        self.chk_auto_assign.setVisible(False)

        self.lbl_threshold = QLabel("Threshold: 85%", self.clusters_area_panel)
        self.lbl_threshold.setVisible(False)

        self.slider_threshold = QSlider(Qt.Orientation.Horizontal, self.clusters_area_panel)
        self.slider_threshold.setRange(50, 99)
        self.slider_threshold.setValue(85)
        self.slider_threshold.valueChanged.connect(self._on_threshold_slider_changed)
        self.slider_threshold.sliderReleased.connect(self.update_auto_assignment_threshold)
        self.slider_threshold.setVisible(False)

        self.btn_update_threshold = QPushButton("Update Threshold", self.clusters_area_panel)
        self.btn_update_threshold.clicked.connect(self.update_auto_assignment_threshold)
        self.btn_update_threshold.setVisible(False)

        self.btn_revert_auto = QPushButton("Revert Auto-assigned", self.clusters_area_panel)
        self.btn_revert_auto.clicked.connect(self.revert_auto_assignments)
        self.btn_revert_auto.setVisible(False)

        # Card Grid inside QScrollArea (vertical scroll only, horizontal reflow)
        self.clusters_scroll_area = QScrollArea(self.clusters_area_panel)
        self.clusters_scroll_area.setObjectName("clustersScrollArea")
        self.clusters_scroll_area.setStyleSheet(CARD_GRID_SCROLL_STYLE)
        self.clusters_scroll_area.setWidgetResizable(True)
        self.clusters_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.clusters_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.clusters_scroll_widget = QWidget()
        self.clusters_scroll_widget.setObjectName("clustersScrollContent")
        self.clusters_grid = QGridLayout(self.clusters_scroll_widget)
        self.clusters_grid.setContentsMargins(0, 0, 0, 0)
        self.clusters_grid.setSpacing(CLUSTER_GRID_SPACING)
        self.clusters_grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.clusters_scroll_area.setWidget(self.clusters_scroll_widget)
        self.clusters_scroll_area.viewport().installEventFilter(self)
        clusters_layout.addWidget(self.clusters_scroll_area, stretch=1)

        self.right_splitter.addWidget(self.clusters_area_panel)

        # Bottom pane: List Panel (min height 120 px, ~3 rows, § 9.3)
        self.list_panel = QFrame(self.right_splitter)
        self.list_panel.setObjectName("listPanel")
        self.list_panel.setMinimumHeight(LIST_PANEL_MIN_HEIGHT)
        self.list_panel.setStyleSheet(LIST_PANEL_STYLE)
        list_panel_layout = QVBoxLayout(self.list_panel)
        list_panel_layout.setContentsMargins(12, 6, 12, 6)
        list_panel_layout.setSpacing(6)

        self.list_panel_header = QWidget(self.list_panel)
        self.list_panel_header.setFixedHeight(30)
        self.list_panel_header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        list_hdr_row = QHBoxLayout(self.list_panel_header)
        list_hdr_row.setContentsMargins(0, 0, 0, 0)
        list_hdr_row.setSpacing(6)

        self.btn_list_collapse = QPushButton("▾", self.list_panel_header)
        self.btn_list_collapse.setStyleSheet(LIST_PANEL_CHEVRON_BTN_STYLE)
        self.btn_list_collapse.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_list_collapse.setToolTip("Collapse or expand list panel")
        self.btn_list_collapse.clicked.connect(self._on_list_panel_toggle_collapse)
        list_hdr_row.addWidget(self.btn_list_collapse)

        self.lbl_list_panel_title = QLabel("Suggested next", self.list_panel_header)
        self.lbl_list_panel_title.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_list_panel_title.setStyleSheet(LIST_PANEL_HEADER_STYLE)
        list_hdr_row.addWidget(self.lbl_list_panel_title)

        self.lbl_list_panel_subtitle = QLabel("by classifier uncertainty", self.list_panel_header)
        self.lbl_list_panel_subtitle.setStyleSheet(LIST_PANEL_HINT_STYLE)
        list_hdr_row.addWidget(self.lbl_list_panel_subtitle)

        self.lbl_list_panel_count = QLabel("0", self.list_panel_header)
        self.lbl_list_panel_count.setStyleSheet(LIST_PANEL_COUNT_CHIP_STYLE)
        self.lbl_list_panel_count.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.lbl_list_panel_count.setFixedHeight(20)
        list_hdr_row.addWidget(self.lbl_list_panel_count)

        list_hdr_row.addStretch(1)

        self.btn_list_accept_selected = QPushButton("Accept 0 selected", self.list_panel_header)
        self.btn_list_accept_selected.setStyleSheet(CLUSTERS_ACCEPT_BTN_STYLE)
        self.btn_list_accept_selected.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_list_accept_selected.setVisible(False)
        self.btn_list_accept_selected.setEnabled(False)
        self.btn_list_accept_selected.clicked.connect(self._on_list_panel_accept_selected)
        list_hdr_row.addWidget(self.btn_list_accept_selected)

        list_panel_layout.addWidget(self.list_panel_header)

        self.list_panel_body = QWidget(self.list_panel)
        list_body_layout = QVBoxLayout(self.list_panel_body)
        list_body_layout.setContentsMargins(0, 0, 0, 0)
        list_body_layout.setSpacing(0)

        self.table_list_panel = QTableWidget(self.list_panel_body)
        self.table_list_panel.setObjectName("listPanelTable")
        self.table_list_panel.setColumnCount(9)
        self.table_list_panel.setHorizontalHeaderLabels([
            "Unit", "Session", "Top probability", "Uncertainty",
            "Best match", "Lat (ms)", "Jitter", "Rel.", "SALT p",
        ])
        unc_hdr = self.table_list_panel.horizontalHeaderItem(3)
        if unc_hdr is not None:
            unc_hdr.setToolTip("Classifier entropy (0 = certain, 1 = uncertain)")
        self.table_list_panel.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_list_panel.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table_list_panel.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_list_panel.verticalHeader().setVisible(False)
        self.table_list_panel.verticalHeader().setDefaultSectionSize(22)
        self.table_list_panel.horizontalHeader().setStretchLastSection(True)
        self.table_list_panel.setStyleSheet(LIST_PANEL_TABLE_STYLE)
        self._list_panel_bar_delegate = _ListPanelBarDelegate(self.table_list_panel)
        self._best_match_color_delegate = _BestMatchColorDelegate(self.table_list_panel)
        self.table_list_panel.setItemDelegateForColumn(2, self._list_panel_bar_delegate)
        self.table_list_panel.setItemDelegateForColumn(3, self._list_panel_bar_delegate)
        self.table_list_panel.setItemDelegateForColumn(4, self._best_match_color_delegate)
        self.table_list_panel.cellClicked.connect(self._on_list_panel_cell_clicked)
        self.table_list_panel.currentCellChanged.connect(self._on_list_panel_current_cell_changed)
        self.table_list_panel.itemSelectionChanged.connect(self._on_list_panel_selection_changed)
        self.table_list_panel.installEventFilter(self)
        list_body_layout.addWidget(self.table_list_panel, stretch=1)

        list_panel_layout.addWidget(self.list_panel_body, stretch=1)

        self.right_splitter.addWidget(self.list_panel)
        self.right_splitter.setStretchFactor(0, 3)
        self.right_splitter.setStretchFactor(1, 1)

        self.main_splitter.addWidget(self.right_panel)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 0)
        self.main_splitter.setStretchFactor(2, 1)

        # Restore persisted splitter sizes if present
        self.main_splitter.restore_from_settings()
        self.right_splitter.restore_from_settings()

        self._init_shortcuts()

    def reset_default_layout(self) -> None:
        """Restore default 4-region splitter sizes (§ 5.1)."""
        if hasattr(self, "main_splitter"):
            self.main_splitter.reset_to_default()
        if hasattr(self, "right_splitter"):
            self.right_splitter.reset_to_default()

    _EXTRA_DEFAULT_HOTKEYS: Dict[int, str] = {10: "Q", 11: "W", 12: "E", 13: "R", 14: "T"}

    def _get_cluster_hotkey(self, cluster_id: int) -> str:
        """Get assigned hotkey for a cluster, defaulting to 0–9 then Q W E R T (§ 9.2)."""
        cid = int(cluster_id)
        if hasattr(self, "_cluster_hotkeys") and cid in self._cluster_hotkeys:
            return str(self._cluster_hotkeys[cid])
        if 0 <= cid <= 9:
            return str(cid)
        if cid in self._EXTRA_DEFAULT_HOTKEYS:
            return self._EXTRA_DEFAULT_HOTKEYS[cid]
        return ""

    @staticmethod
    def _is_text_input_widget(widget: Any) -> bool:
        """Return True if widget is an editable text or numeric input control."""
        return isinstance(
            widget, (QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox)
        ) or (isinstance(widget, QComboBox) and widget.isEditable())

    def _can_trigger_curation_shortcut(self) -> bool:
        """Return True when curation QShortcuts may fire (visible, no popup, not in text input)."""
        if not self.isVisible():
            return False
        if QApplication.activePopupWidget() is not None:
            return False
        return not self._is_text_input_widget(QApplication.focusWidget())

    def _push_undo_snapshot(self) -> None:
        """Capture current curation state onto undo stack if not applying undo/redo."""
        if not self._is_applying_undo_redo:
            self._undo_stack.append(self._capture_curation_snapshot())
            self._redo_stack.clear()
            if hasattr(self, "btn_undo"):
                self.btn_undo.setEnabled(True)
            if hasattr(self, "btn_redo"):
                self.btn_redo.setEnabled(False)
            self._set_dirty(True)

    def _refresh_cluster_shortcuts(self) -> None:
        """Create or update QShortcut instances for configured cluster hotkeys."""
        if not hasattr(self, "_cluster_shortcuts"):
            self._cluster_shortcuts = {}

        for sc in list(self._cluster_shortcuts.values()):
            try:
                sc.setEnabled(False)
                sc.setParent(None)
            except Exception:
                pass
        self._cluster_shortcuts.clear()

        cids: List[int] = []
        if hasattr(self, "_cluster_card_order") and self._cluster_card_order:
            cids.extend(self._cluster_card_order)
        if hasattr(self, "current_clustering_result") and self.current_clustering_result is not None:
            for cid in self.current_clustering_result.cluster_summaries.keys():
                if cid not in cids:
                    cids.append(cid)
        if hasattr(self, "_cluster_hotkeys"):
            for cid in self._cluster_hotkeys.keys():
                if cid not in cids:
                    cids.append(cid)

        for cid in cids:
            hk_str = self._get_cluster_hotkey(cid)
            if not hk_str:
                continue
            seq = QKeySequence(hk_str)
            if seq.isEmpty():
                continue
            sc = QShortcut(seq, self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(
                lambda c=cid: self._on_cluster_hotkey_assign(c)
                if self._can_trigger_curation_shortcut()
                else None
            )
            self._cluster_shortcuts[cid] = sc

    def _init_shortcuts(self) -> None:
        """Register application-level keyboard shortcuts for active curation loop."""
        self._shortcuts: List[QShortcut] = []
        self._refresh_cluster_shortcuts()

        sc_space = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        sc_space.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc_space.activated.connect(
            lambda: self.skip_current_candidate() if self._can_trigger_curation_shortcut() else None
        )
        self._shortcuts.append(sc_space)

        sc_s = QShortcut(QKeySequence("S"), self)
        sc_s.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc_s.activated.connect(
            lambda: self.skip_current_candidate() if self._can_trigger_curation_shortcut() else None
        )
        self._shortcuts.append(sc_s)

        sc_n = QShortcut(QKeySequence("N"), self)
        sc_n.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc_n.activated.connect(
            lambda: self.create_new_cluster_for_candidate()
            if self._can_trigger_curation_shortcut()
            else None
        )
        self._shortcuts.append(sc_n)

    def _update_active_learning_button_style(self) -> None:
        """Update toggle button text and styling based on active_learning_enabled state."""
        if not hasattr(self, "btn_toggle_active_learning"):
            return
        if self.active_learning_enabled:
            self.btn_toggle_active_learning.setText("🤖 Active Learning: ON")
        else:
            self.btn_toggle_active_learning.setText("🤖 Active Learning: OFF")
        self.btn_toggle_active_learning.setStyleSheet(
            build_active_learning_button_style(self.active_learning_enabled)
        )

    def _set_dirty(self, dirty: bool) -> None:
        """Update unsaved-changes dirty flag and emit dirty_changed if changed."""
        dirty_val = bool(dirty)
        if getattr(self, "_is_dirty", False) != dirty_val:
            self._is_dirty = dirty_val
            self.dirty_changed.emit(dirty_val)
        else:
            self._is_dirty = dirty_val

    def set_active_learning_mode(self, enabled: bool) -> None:
        """Toggle active learning mode on or off.

        When enabled, entropy-based candidate selection automatically suggests the next unit.
        When disabled, manual curation advances sequentially through the pooled units table.
        """
        self.active_learning_enabled = bool(enabled)
        if hasattr(self, "btn_toggle_active_learning"):
            self.btn_toggle_active_learning.blockSignals(True)
            self.btn_toggle_active_learning.setChecked(self.active_learning_enabled)
            self.btn_toggle_active_learning.blockSignals(False)
        self._update_active_learning_button_style()
        self.active_learning_toggled.emit(self.active_learning_enabled)

        if self.active_learning_enabled:
            if hasattr(self, "slider_threshold"):
                self.slider_threshold.setEnabled(True)
            if hasattr(self, "btn_update_threshold"):
                self.btn_update_threshold.setEnabled(True)
            if hasattr(self, "chk_auto_assign"):
                self.chk_auto_assign.setEnabled(True)
            if self.active_candidate is None:
                self.refresh_active_learning()
            else:
                self._update_return_to_candidate_button()
        else:
            if hasattr(self, "chk_auto_assign"):
                self.chk_auto_assign.setEnabled(False)
            self._update_return_to_candidate_button()
            for card in self._cluster_cards.values():
                card.set_candidate_probability(None)
            if hasattr(self, "cluster_prob_bar"):
                self.cluster_prob_bar.clear()
        self._refresh_list_panel()

    def _update_focused_unit_header(self, gid: Optional[str]) -> None:
        """Update Focused Unit header title, session subtitle, and Unassign button visibility."""
        if not hasattr(self, "lbl_focused_unit_title"):
            return
        if not gid or gid not in self._pooled_units_map:
            self.lbl_focused_unit_title.setText("No unit selected")
            self.lbl_focused_unit_session.setText("")
            if hasattr(self, "btn_unassign"):
                self.btn_unassign.setVisible(False)
                self.btn_unassign.setEnabled(False)
            return

        rec = self._pooled_units_map[gid]
        sess_id = rec.get("session_id", "")
        unit_id = rec.get("unit_id")
        sess_str, uid_str = _format_unit_display(sess_id, unit_id, gid)
        self.lbl_focused_unit_title.setText(f"Unit {uid_str}")
        self.lbl_focused_unit_session.setText(sess_str)

        cid = int(rec.get("cluster_id", -1)) if rec.get("cluster_id") is not None else -1
        has_cluster = cid >= 0
        if hasattr(self, "btn_unassign"):
            self.btn_unassign.setVisible(has_cluster)
            self.btn_unassign.setEnabled(has_cluster)

    def _update_return_to_candidate_button(self) -> None:
        """Show '↩ Suggested unit' only when AL is on, a candidate exists, and focus is elsewhere."""
        if not hasattr(self, "btn_return_to_candidate"):
            return
        has_candidate = self.active_learning_enabled and (self.active_candidate is not None)
        is_different_unit = bool(
            has_candidate and self._current_selected_gid != self.active_candidate.global_unit_id
        )
        self.btn_return_to_candidate.setVisible(is_different_unit)
        self.btn_return_to_candidate.setEnabled(has_candidate)

    def return_to_active_candidate(self) -> bool:
        """Return selection to the highest-entropy active learning candidate cell."""
        cand = self.active_candidate
        if cand is None and self.ranked_candidates:
            cand = self.ranked_candidates[0]
            self.active_candidate = cand

        if cand is not None:
            return self.select_pooled_unit(cand.global_unit_id, source="queue")
        return False

    def _advance_to_next_table_row(self, assigned_gid: Optional[str] = None) -> None:
        """Advance table selection sequentially to the next non-hidden row."""
        total_rows = self.table_pooled_units.rowCount()
        if total_rows == 0:
            return

        current_row = -1
        if assigned_gid is not None:
            for r in range(total_rows):
                if self._get_gid_for_row(r) == assigned_gid:
                    current_row = r
                    break
        if current_row < 0:
            selected = self.table_pooled_units.selectionModel().selectedRows()
            if selected:
                current_row = selected[0].row()

        # Find next visible (non-hidden) row
        next_row = -1
        for r in range(current_row + 1, total_rows):
            if not self.table_pooled_units.isRowHidden(r):
                next_row = r
                break

        if next_row >= 0:
            self.table_pooled_units.blockSignals(True)
            self.table_pooled_units.clearSelection()
            self.table_pooled_units.selectRow(next_row)
            self.table_pooled_units.blockSignals(False)
            self._on_pooled_unit_selected()

    def _on_cluster_hotkey_assign(self, target_cid: int) -> None:
        """Assign current candidate or selected pooled units to target_cid."""
        if self.current_clustering_result is None:
            return

        # Check if multiple rows selected in the active list (table_list_panel vs table_pooled_units)
        target_gids: List[str] = []
        in_list_panel = (
            getattr(self, "_focus_source", None) == "list_candidates"
            or (hasattr(self, "table_list_panel") and self.table_list_panel.hasFocus())
        )
        if in_list_panel:
            target_gids = self._get_list_panel_selected_gids()
        elif hasattr(self, "table_pooled_units") and self.table_pooled_units.selectionModel() is not None:
            selected_model_rows = self.table_pooled_units.selectionModel().selectedRows()
            for model_idx in selected_model_rows:
                r = model_idx.row()
                gid_val = self._get_gid_for_row(r)
                if gid_val and gid_val in self._pooled_units_map:
                    target_gids.append(gid_val)

        if not target_gids:
            gid = self._current_selected_gid
            if gid is None and self.active_candidate is not None:
                gid = self.active_candidate.global_unit_id
            if gid:
                target_gids.append(gid)

        if not target_gids:
            return

        self.assign_candidate_to_cluster(target_cid, unit_id=target_gids)

    def _on_cluster_hotkey(self, key_num: int) -> None:
        """Handle numeric hotkey (0-9) cluster assignment."""
        if self.current_clustering_result is None:
            return

        target_cid = None
        key_str = str(key_num)
        for cid in self._cluster_card_order:
            if self._get_cluster_hotkey(cid) == key_str:
                target_cid = cid
                break

        if target_cid is None:
            summaries = self.current_clustering_result.cluster_summaries
            if key_num in summaries:
                target_cid = key_num
            elif key_num > 0 and 0 <= (key_num - 1) < len(summaries):
                active_cids = sorted(summaries.keys())
                target_cid = active_cids[key_num - 1]
            elif 0 <= key_num < len(self._cluster_card_order):
                target_cid = self._cluster_card_order[key_num]

        if target_cid is not None:
            self._on_cluster_hotkey_assign(target_cid)

    def _on_units_header_clicked(self, logical_index: int) -> None:
        """Handle column header click on pooled units table to sort and toggle order."""
        header = self.table_pooled_units.horizontalHeader()
        order = header.sortIndicatorOrder()
        self.table_pooled_units.sortItems(logical_index, order)
        self._apply_cluster_filter()

    def _on_threshold_slider_changed(self, value: int) -> None:
        """Update label when threshold slider is moved."""
        if hasattr(self, "lbl_threshold"):
            self.lbl_threshold.setText(f"Threshold: {value}%")

    def _on_auto_assign_toggled(self, checked: bool) -> None:
        """Handle toggling of background auto-assignment on/off."""
        self.auto_assign_enabled = bool(checked)
        self.auto_assign_changed.emit(self.auto_assign_enabled)
        if not self.auto_assign_enabled:
            self.revert_auto_assignments()
            self.refresh_active_learning()
        else:
            self.refresh_active_learning()

    def prompt_revert_auto_assignments(self) -> int:
        """Prompt user with RevertAutoAssignmentsDialog and revert auto-assigned units if confirmed."""
        return self.revert_auto_assignments(confirm=True)

    def revert_auto_assignments(self, confirm: bool = False) -> int:
        """Revert all Auto-assigned units back to Unassigned (-1), preserving Curated and Seeded units.

        Args:
            confirm: If True, prompts user with RevertAutoAssignmentsDialog before reverting.

        Returns:
            Number of units reverted.
        """
        auto_gids = [
            gid for gid, status in self._unit_status_map.items()
            if status == "Auto-assigned"
        ]
        if not auto_gids:
            return 0

        if confirm:
            dlg = RevertAutoAssignmentsDialog(len(auto_gids), parent=self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return 0

        self._push_undo_snapshot()

        reverted_count = 0
        for gid in auto_gids:
            if gid in self._pooled_units_map:
                self._pooled_units_map[gid]["cluster_id"] = -1
            self._unit_status_map[gid] = "Unassigned"
            reverted_count += 1

        if reverted_count > 0:
            for r in self._pooled_rows:
                gid = r.get("global_unit_id")
                if gid in self._pooled_units_map:
                    r["cluster_id"] = self._pooled_units_map[gid]["cluster_id"]
            self._update_all_pooled_cluster_items()
            self._populate_cluster_filter_options()
            self._sync_clustering_result_and_cards()
            self._update_classification_progress()
            if self._comparison_cluster_id is not None:
                self.set_cluster_comparison(self._comparison_cluster_id)
            if self._current_selected_gid:
                self._update_probe_view_for_unit(self._current_selected_gid)
            self._set_dirty(True)

        return reverted_count

    def update_auto_assignment_threshold(self) -> None:
        """Apply auto-assignment threshold from slider and trigger re-evaluation."""
        if hasattr(self, "slider_threshold"):
            self.auto_assign_threshold = self.slider_threshold.value() / 100.0
        self.confidence_threshold_changed.emit(self.auto_assign_threshold)
        self.refresh_active_learning()

    def _on_window_text_edited(self) -> None:
        """Parse window start and end from text boxes and apply."""
        text_start = self.edit_win_start.text().strip().replace("+", "")
        text_end = self.edit_win_end.text().strip().replace("+", "")
        try:
            val_start = float(text_start)
            val_end = float(text_end)
        except ValueError:
            self._sync_window_text_boxes()
            return

        if val_start < val_end:
            self.set_visualized_window((val_start, val_end))
        else:
            self._sync_window_text_boxes()

    def _sync_window_text_boxes(self) -> None:
        """Update window text boxes to match self.current_window_ms."""
        if hasattr(self, "edit_win_start") and hasattr(self, "edit_win_end"):
            self.edit_win_start.blockSignals(True)
            self.edit_win_end.blockSignals(True)
            w_start, w_end = self.current_window_ms
            start_str = f"{w_start:.1f}" if w_start % 1 != 0 else f"{int(w_start)}"
            end_str = f"{w_end:.1f}" if w_end % 1 != 0 else f"{int(w_end)}"
            self.edit_win_start.setText(start_str)
            self.edit_win_end.setText(f"+{end_str}" if w_end > 0 else end_str)
            self.edit_win_start.blockSignals(False)
            self.edit_win_end.blockSignals(False)

    def set_visualized_window(self, window_ms: Tuple[float, float]) -> None:
        """Set the presented time window for cluster PSTHs and re-render cluster cards."""
        self.current_window_ms = (float(window_ms[0]), float(window_ms[1]))
        self.active_clustering_params.presented_window_ms = self.current_window_ms
        self._invalidate_clustering_caches()
        self._sync_window_text_boxes()
        df_active = self._get_active_pooled_dataframe()
        if not df_active.empty and self.current_clustering_result is not None:
            spike_map = self._get_spike_data_map(df_active)
            psth_matrix, t_bins = generate_psth_matrix(
                df_active,
                time_window_ms=self.current_window_ms,
                bin_size_ms=self.current_bin_size_ms,
                spike_data_map=spike_map,
                psth_cache=self._unit_psth_cache,
                smooth=self.psth_smooth,
                gaussian_sigma_ms=self.psth_gaussian_sigma_ms,
            )
            active_cids = [c for c in self._cluster_card_order if c >= 0]
            new_summaries = compute_cluster_summaries(
                df_active,
                self.current_clustering_result.cluster_labels,
                psth_matrix,
                t_bins,
                active_cluster_ids=active_cids,
                norm_mode=self.cluster_psth_norm,
            )
            for c_id, summ in new_summaries.items():
                summ.color_hex = self._get_cluster_color(c_id)
            self.current_clustering_result.cluster_summaries = new_summaries
            self._render_cluster_cards(self.current_clustering_result)
        self._refresh_inspected_unit_plots()
        self._schedule_background_plot_prewarm()
        self.view_settings_changed.emit(self.current_window_ms)

    def _on_bin_size_changed(self, value: float) -> None:
        """Handle change in cluster PSTH bin size from spinbox."""
        self.set_cluster_bin_size(value)

    def set_cluster_bin_size(self, bin_size_ms: float) -> None:
        """Set the histogram bin width in ms for cluster PSTHs and re-render cluster cards."""
        self.current_bin_size_ms = float(bin_size_ms)
        self.active_clustering_params.presented_bin_size_ms = self.current_bin_size_ms
        self._invalidate_clustering_caches()
        if hasattr(self, "spin_bin_size"):
            self.spin_bin_size.blockSignals(True)
            self.spin_bin_size.setValue(self.current_bin_size_ms)
            self.spin_bin_size.blockSignals(False)
        self._recompute_and_render_clusters()
        self._refresh_inspected_unit_plots()
        self._schedule_background_plot_prewarm()
        self.bin_size_changed.emit(self.current_bin_size_ms)

    def set_psth_smooth(self, smooth: bool) -> None:
        """Toggle Gaussian smoothed firing rate PSTH estimation."""
        self.psth_smooth = bool(smooth)
        self.feature_config.psth_smooth = self.psth_smooth
        self.active_clustering_params.psth_smooth = self.psth_smooth
        self._invalidate_clustering_caches()
        if hasattr(self, "chk_smooth"):
            self.chk_smooth.blockSignals(True)
            self.chk_smooth.setChecked(self.psth_smooth)
            self.chk_smooth.blockSignals(False)
        self._recompute_and_render_clusters()
        self._refresh_inspected_unit_plots()
        self._schedule_background_plot_prewarm()

    def set_psth_gaussian_sigma(self, sigma_ms: float) -> None:
        """Set standard deviation sigma in ms for Gaussian PSTH smoothing."""
        self.psth_gaussian_sigma_ms = max(0.1, float(sigma_ms))
        self.feature_config.psth_gaussian_sigma_ms = self.psth_gaussian_sigma_ms
        self.active_clustering_params.psth_gaussian_sigma_ms = self.psth_gaussian_sigma_ms
        self._invalidate_clustering_caches()
        if hasattr(self, "spin_sigma"):
            self.spin_sigma.blockSignals(True)
            self.spin_sigma.setValue(self.psth_gaussian_sigma_ms)
            self.spin_sigma.blockSignals(False)
        self._recompute_and_render_clusters()
        self._refresh_inspected_unit_plots()
        self._schedule_background_plot_prewarm()

    def _on_psth_norm_changed(self, index: int) -> None:
        """Handle selection in PSTH Normalization combo box."""
        if hasattr(self, "combo_psth_norm"):
            mode = self.combo_psth_norm.itemData(index) or "probability"
            self.set_psth_norm_mode(str(mode))

    def set_psth_norm_mode(self, mode: str) -> None:
        """Set cluster PSTH waveform normalization mode ("peak", "probability", or "raw")."""
        mode = str(mode or "probability").lower()
        if mode not in ("peak", "probability", "raw"):
            mode = "probability"
        self.cluster_psth_norm = mode
        if hasattr(self, "active_clustering_params") and self.active_clustering_params is not None:
            self.active_clustering_params.cluster_psth_norm = mode

        if hasattr(self, "combo_psth_norm"):
            self.combo_psth_norm.blockSignals(True)
            for idx in range(self.combo_psth_norm.count()):
                if self.combo_psth_norm.itemData(idx) == mode:
                    self.combo_psth_norm.setCurrentIndex(idx)
                    break
            self.combo_psth_norm.blockSignals(False)

        self._recompute_and_render_clusters()
        current_gid = self._current_selected_gid or (self.active_candidate.global_unit_id if self.active_candidate else None)
        self._update_cluster_cards_candidate_psth(current_gid)
        self.psth_norm_mode_changed.emit(mode)

    def _on_combo_order_changed(self, index: int) -> None:
        """Handle selection in Pulse Order combo box."""
        if hasattr(self, "combo_order"):
            data = self.combo_order.itemData(index)
            if isinstance(data, str):
                self.set_pulse_ordering(data)

    def set_pulse_ordering(self, order_by: str) -> None:
        """Set optogenetic pulse trial ordering mode ("presentation" or "latency")."""
        if order_by not in ("presentation", "latency"):
            return
        self.raster_order_by = order_by

        if hasattr(self, "combo_order"):
            self.combo_order.blockSignals(True)
            for idx in range(self.combo_order.count()):
                if self.combo_order.itemData(idx) == order_by:
                    self.combo_order.setCurrentIndex(idx)
                    break
            self.combo_order.blockSignals(False)

        self._refresh_inspected_unit_plots()
        self._schedule_background_plot_prewarm()
        self.view_settings_changed.emit(self.current_window_ms)

    def set_raster_marker_size(self, size: float) -> None:
        """Set raster plot scatter marker size in pixels and re-render."""
        self.raster_marker_size = max(0.1, float(size))
        self._refresh_inspected_unit_plots()
        self._schedule_background_plot_prewarm()
        self.view_settings_changed.emit(self.current_window_ms)

    def prompt_custom_marker_size(self) -> None:
        """Prompt dialog for user to enter custom raster marker size."""
        val, ok = QInputDialog.getDouble(
            self,
            "Custom Marker Size",
            "Marker size in pixels (0.5 – 30.0):",
            self.raster_marker_size,
            0.5,
            30.0,
            1,
        )
        if ok:
            self.set_raster_marker_size(val)

    def _refresh_inspected_unit_plots(self) -> None:
        """Re-render the active inspection image_viewer and all open comparison viewers."""
        current_gid = self._current_selected_gid or (self.active_candidate.global_unit_id if self.active_candidate else None)
        if current_gid and hasattr(self, "image_viewer"):
            self._populate_image_viewer_for_unit(self.image_viewer, current_gid)
            self._update_cluster_cards_candidate_psth(current_gid)

        if self._comparison_cluster_id is not None and hasattr(self, "comparison_viewers"):
            self._member_pixmap_cache.clear()
            for viewer in self.comparison_viewers:
                if getattr(viewer, "_live_plot", None) is not None and hasattr(viewer, "hydrate"):
                    viewer.hydrate(force=True)
            self._hydrate_visible_comparison_members(hydrate_uncached=True)

    def prompt_custom_window(self) -> None:
        """Prompt dialog for user to enter custom analysis/visualization window."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Custom Visualized Window")
        dialog.setFixedWidth(320)
        vbox = QVBoxLayout(dialog)

        form = QFormLayout()
        spin_start = QDoubleSpinBox()
        spin_start.setRange(-500.0, -1.0)
        spin_start.setValue(self.current_window_ms[0])
        spin_start.setSingleStep(5.0)
        spin_start.setSuffix(" ms")
        form.addRow("Pre-pulse Start:", spin_start)

        spin_end = QDoubleSpinBox()
        spin_end.setRange(1.0, 1000.0)
        spin_end.setValue(self.current_window_ms[1])
        spin_end.setSingleStep(5.0)
        spin_end.setSuffix(" ms")
        form.addRow("Post-pulse End:", spin_end)

        vbox.addLayout(form)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        vbox.addWidget(btn_box)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            w_start = float(spin_start.value())
            w_end = float(spin_end.value())
            if w_start < w_end:
                self.set_visualized_window((w_start, w_end))
            else:
                QMessageBox.warning(self, "Invalid Window", "Window start must be less than window end.")

    def _default_feature_time_window_ms(self) -> Tuple[float, float]:
        """Return default PSTH feature window (0.0, pulse_duration_ms + 5.0)."""
        return (0.0, self._get_pulse_duration_ms() + 5.0)

    def prompt_feature_settings(self) -> None:
        """Open the Feature Settings modal dialog to configure clustering/classification features."""
        dlg = FeatureSettingsDialog(
            self.feature_config,
            parent=self,
            default_time_window_ms=self._default_feature_time_window_ms(),
        )
        if dlg.exec():
            new_config = dlg.get_feature_configuration()
            self._feature_window_customized = (
                tuple(new_config.time_window_ms) != self._default_feature_time_window_ms()
            )
            self.feature_config = new_config
            self.active_clustering_params.feature_config = self.feature_config
            self._recompute_2d_embedding()
            self.refresh_active_learning()

    def prompt_seed_clusters(self) -> None:
        """Open modal dialog allowing researchers to configure Affinity Propagation hyperparameters and seed clusters."""
        dlg = SeedClustersDialog(
            current_params=self.ap_params,
            feature_config=self.feature_config,
            parent=self,
        )
        if dlg.exec():
            self.ap_params = dlg.get_parameters()
            new_config = dlg.get_feature_configuration()
            if tuple(new_config.time_window_ms) != tuple(self.feature_config.time_window_ms):
                self._feature_window_customized = (
                    tuple(new_config.time_window_ms) != self._default_feature_time_window_ms()
                )
            self.feature_config = new_config
            self.active_clustering_params.feature_config = self.feature_config
            self.seed_clusters_with_affinity_propagation(ap_params=self.ap_params)

    def prompt_embedding_settings(self) -> None:
        """Open modal dialog allowing researchers to configure 2D projection (PCA / UMAP)."""
        dlg = EmbeddingSettingsDialog(
            current_params=self.embedding_params,
            feature_config=self.feature_config,
            parent=self,
        )
        if dlg.exec():
            self.embedding_params = dlg.get_parameters()
            new_config = dlg.get_feature_configuration()
            if tuple(new_config.time_window_ms) != tuple(self.feature_config.time_window_ms):
                self._feature_window_customized = (
                    tuple(new_config.time_window_ms) != self._default_feature_time_window_ms()
                )
            self.feature_config = new_config
            self.active_clustering_params.feature_config = self.feature_config

            idx = self.combo_embedding_method.findData(self.embedding_params.method.upper())
            if idx >= 0:
                self.combo_embedding_method.blockSignals(True)
                self.combo_embedding_method.setCurrentIndex(idx)
                self.combo_embedding_method.blockSignals(False)

            self._recompute_2d_embedding()

    def _on_embedding_method_changed(self, index: int) -> None:
        """Handle embedding method switch from combo box."""
        method = self.combo_embedding_method.currentData() or "PCA"
        self.embedding_params.method = str(method)
        self._recompute_2d_embedding()

    def _recompute_2d_embedding(self) -> None:
        """Recompute 2D scatter coordinates using active embedding parameters and refresh canvas."""
        df_active = self._get_active_pooled_dataframe()
        if df_active.empty:
            return

        spike_map = self._get_spike_data_map(df_active)
        feat_matrix, _, _ = extract_feature_matrix(
            df_active,
            feature_config=self.feature_config,
            spike_data_map=spike_map,
            psth_cache=self._unit_psth_cache,
            pulse_duration_ms=self._get_pulse_duration_ms(),
        )
        coords_2d, exp_var = compute_2d_embedding(
            feat_matrix,
            embedding_params=self.embedding_params,
            random_state=getattr(self, "al_random_state", 42),
        )

        if self.current_clustering_result is not None:
            self.current_clustering_result.pca_coordinates = coords_2d
            self.current_clustering_result.explained_variance_ratio = exp_var
            self.current_clustering_result.feature_matrix = feat_matrix
            self._render_pca_scatter(self.current_clustering_result)
        else:
            labels = np.array([self._pooled_units_map.get(gid, {}).get("cluster_id", -1) for gid in df_active["global_unit_id"].astype(str)], dtype=np.int32)
            unit_ids = df_active["global_unit_id"].astype(str).tolist()
            res = ClusteringResult(
                cluster_labels=labels,
                pca_coordinates=coords_2d,
                explained_variance_ratio=exp_var,
                cluster_summaries={},
                feature_matrix=feat_matrix,
                unit_ids=unit_ids,
                parameters=self.active_clustering_params,
            )
            self._render_pca_scatter(res)

    def seed_clusters_with_affinity_propagation(
        self, ap_params: Optional[AffinityPropagationParameters] = None
    ) -> bool:
        """Discover clear initial response clusters using Affinity Propagation with exemplar filtering."""
        df = self._get_active_pooled_dataframe()
        if df.empty or len(df) == 0:
            QMessageBox.warning(
                self,
                "No Units to Cluster",
                "No optotagged units are currently loaded in the Phase 2 pool.\n\n"
                "Please add at least one curated experiment containing optotagged cells before clustering.",
            )
            return False

        if ap_params is not None:
            self.ap_params = ap_params
        active_ap = self.ap_params

        self._push_undo_snapshot()
        spike_map = self._get_spike_data_map(df)
        result = seed_clusters_affinity_propagation(
            df=df,
            feature_config=self.feature_config,
            damping=active_ap.damping,
            similarity_percentile=active_ap.similarity_percentile,
            presented_window_ms=self.current_window_ms,
            presented_bin_size_ms=self.current_bin_size_ms,
            spike_data_map=spike_map,
            psth_cache=self._unit_psth_cache,
            ap_params=active_ap,
            embedding_params=self.embedding_params,
            pulse_duration_ms=self._get_pulse_duration_ms(),
            psth_smooth=self.psth_smooth,
            psth_gaussian_sigma_ms=self.psth_gaussian_sigma_ms,
            cluster_psth_norm=self.cluster_psth_norm,
        )
        self.current_clustering_result = result
        preserve_curated = getattr(active_ap, "preserve_curated", True)
        if not preserve_curated:
            self._cluster_card_order = sorted(list(result.cluster_summaries.keys()))

        # Update unit status map and internal records
        for i, gid in enumerate(result.unit_ids):
            cid = int(result.cluster_labels[i])
            if gid in self._pooled_units_map:
                if preserve_curated and self._unit_status_map.get(gid) == "Curated":
                    continue
                self._pooled_units_map[gid]["cluster_id"] = cid
                self._unit_status_map[gid] = "Seeded" if cid >= 0 else "Unassigned"

        for r in self._pooled_rows:
            gid = r.get("global_unit_id")
            if gid in self._pooled_units_map:
                r["cluster_id"] = self._pooled_units_map[gid]["cluster_id"]

        if preserve_curated:
            self._sync_clustering_result_and_cards()
        else:
            for cid, summ in result.cluster_summaries.items():
                if cid not in self._cluster_colors:
                    self._cluster_colors[cid] = self._get_cluster_color(cid)
                summ.color_hex = self._get_cluster_color(cid)
            self._render_cluster_cards(result)

        for cid in self.current_clustering_result.cluster_summaries:
            if cid not in self._cluster_colors:
                self._cluster_colors[cid] = self._get_cluster_color(cid)

        self._render_pca_scatter(self.current_clustering_result)
        self._update_all_pooled_cluster_items()
        self._populate_cluster_filter_options()

        if hasattr(self, "marginal_widget"):
            self.marginal_widget.update_distributions(df)

        self.refresh_active_learning()
        self._schedule_background_plot_prewarm()
        self._set_dirty(True)
        self.clustering_completed.emit(len(result.cluster_summaries))
        return True

    def _resolve_selected_target_gids(self) -> List[str]:
        """Collect selected unit IDs prioritizing list_candidates focus over pooled table/active candidate."""
        target_gids: List[str] = []
        in_list_panel = (
            getattr(self, "_focus_source", None) == "list_candidates"
            or (hasattr(self, "table_list_panel") and self.table_list_panel.hasFocus())
        )
        if in_list_panel:
            target_gids = self._get_list_panel_selected_gids()
        elif hasattr(self, "table_pooled_units") and self.table_pooled_units.selectionModel() is not None:
            selected_model_rows = self.table_pooled_units.selectionModel().selectedRows()
            for model_idx in selected_model_rows:
                r = model_idx.row()
                gid_val = self._get_gid_for_row(r)
                if gid_val and gid_val in self._pooled_units_map:
                    target_gids.append(gid_val)
        if not target_gids:
            if self._current_selected_gid is not None and self._current_selected_gid in self._pooled_units_map:
                target_gids.append(self._current_selected_gid)
            elif self.active_candidate is not None:
                target_gids.append(self.active_candidate.global_unit_id)
        return target_gids

    def create_new_cluster_for_candidate(self) -> int:
        """Create the next sequential cluster and assign the active or selected candidate(s) to it."""
        target_gids = self._resolve_selected_target_gids()

        if not target_gids and self._pooled_rows:
            # First unassigned unit
            for r in self._pooled_rows:
                if r.get("cluster_id", -1) < 0:
                    target_gids.append(r.get("global_unit_id"))
                    break
            if not target_gids and self._pooled_rows:
                target_gids.append(self._pooled_rows[0].get("global_unit_id"))

        if not target_gids:
            return -1

        # Determine next available cluster ID
        used_cids = set(self._cluster_card_order)
        if self.current_clustering_result:
            used_cids.update(self.current_clustering_result.cluster_summaries.keys())
        for rec in self._pooled_units_map.values():
            c = rec.get("cluster_id", -1)
            if c >= 0:
                used_cids.add(c)
        next_cid = 0
        while next_cid in used_cids:
            next_cid += 1

        color = self._get_cluster_color(next_cid)
        self._cluster_colors[next_cid] = color
        if next_cid not in self._cluster_card_order:
            self._cluster_card_order.append(next_cid)

        self.assign_candidate_to_cluster(next_cid, unit_id=target_gids)

        self.clustering_completed.emit(0)
        return next_cid

    def reset_clustering(self, confirm: bool = True) -> bool:
        """Reset clusters and unit assignments to unassigned state.

        Args:
            confirm: If True, prompts user with ResetClusteringDialog.

        Returns:
            True if reset was executed, False if cancelled by the user.
        """
        mode = "everything"
        if confirm:
            dlg = ResetClusteringDialog(parent=self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return False
            mode = dlg.reset_mode()

        self._invalidate_active_learning_worker()
        if mode == "auto_only":
            self._push_undo_snapshot()
            self.revert_auto_assignments(confirm=False)
            self._set_dirty(True)
            return True

        self._push_undo_snapshot()
        self._skipped_gids.clear()
        self.reset_to_unassigned()
        self._set_dirty(True)
        return True

    def reset_to_unassigned(self) -> None:
        """Reset all pooled units to unassigned status (-1) and clear cluster cards."""
        self.clear_cluster_comparison(return_to_suggestion=False)
        self._cluster_card_order.clear()
        self._cluster_cards.clear()
        self._cluster_names.clear()
        self._auto_confidences.clear()
        self._all_unit_probabilities.clear()
        self._unit_candidates_map.clear()
        while self.clusters_grid.count():
            item = self.clusters_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._new_cluster_card = None

        for gid, rec in self._pooled_units_map.items():
            rec["cluster_id"] = -1
            self._unit_status_map[gid] = "Unassigned"
        for r in self._pooled_rows:
            r["cluster_id"] = -1

        df_active = self._get_active_pooled_dataframe()
        if not df_active.empty:
            spike_map = self._get_spike_data_map(df_active)
            feat_matrix, psth_matrix, t_bins = extract_feature_matrix(
                df_active,
                feature_config=self.feature_config,
                spike_data_map=spike_map,
                psth_cache=self._unit_psth_cache,
                pulse_duration_ms=self._get_pulse_duration_ms(),
            )
            pca_coords, exp_var = compute_2d_embedding(
                feat_matrix,
                embedding_params=self.embedding_params,
                random_state=getattr(self, "al_random_state", 42),
            )
            unit_ids = df_active["global_unit_id"].astype(str).tolist()
            labels = np.full(len(df_active), -1, dtype=np.int32)
            self.current_clustering_result = ClusteringResult(
                cluster_labels=labels,
                pca_coordinates=pca_coords,
                explained_variance_ratio=exp_var,
                cluster_summaries={},
                feature_matrix=feat_matrix,
                unit_ids=unit_ids,
                parameters=self.active_clustering_params,
            )
            self._render_pca_scatter(self.current_clustering_result)

        self._update_all_pooled_cluster_items()
        self._populate_cluster_filter_options()
        if hasattr(self, "marginal_widget"):
            self.marginal_widget.update_distributions(df_active)
        self.refresh_active_learning()
        self.clustering_completed.emit(0)

    def _recompute_and_render_clusters(self) -> None:
        """Recompute cluster summaries from current assignments and re-render cluster cards and scatter."""
        self._sync_clustering_result_and_cards(
            recompute_embedding=True,
            prune_and_refresh_list=False,
        )

    def show_curation_shortcuts_help(self) -> QDialog:
        """Display dialog explaining curation keyboard shortcuts and workflow."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Curation Shortcuts & Workflow Help")
        dialog.setFixedWidth(400)
        dialog.setStyleSheet(DIALOG_MODAL_STYLE)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        header = QLabel("Keyboard Shortcuts & Active Learning Help")
        header.setStyleSheet(DIALOG_SECTION_TITLE_STYLE)
        layout.addWidget(header)

        shortcuts_box = QGroupBox("Curation Hotkeys")
        shortcuts_box.setStyleSheet(DIALOG_GROUP_BOX_STYLE)
        s_layout = QVBoxLayout(shortcuts_box)
        s_layout.setSpacing(6)
        shortcuts_text = QLabel(
            "• <b>[0] – [9]</b>: Assign cell to Cluster 0–9<br>"
            "• <b>[N]</b>: Create New Cluster from cell<br>"
            "• <b>[Space]</b> or <b>[S]</b>: Skip Ambiguous Unit (cycles to end)<br>"
            "• <b>[Ctrl + S]</b>: Save Multi-Session Curation"
        )
        shortcuts_text.setStyleSheet(CLUSTERS_SHORTCUTS_HELP_TEXT_STYLE)
        s_layout.addWidget(shortcuts_text)
        layout.addWidget(shortcuts_box)

        workflow_box = QGroupBox("Workflow & Comparison Tips")
        workflow_box.setStyleSheet(shortcuts_box.styleSheet())
        w_layout = QVBoxLayout(workflow_box)
        w_layout.setSpacing(6)
        workflow_text = QLabel(
            "• <b>🤖 Active Learning ON/OFF</b>: In ON mode, the model automatically queries "
            "highest-entropy boundary units. In OFF mode, curation advances sequentially down the table.<br>"
            "• <b>🎯 Suggested Cell</b>: Click anytime to instantly jump back to the model's top uncertain candidate.<br>"
            "• <b>🔍 Click Cluster Card</b>: Toggles side-by-side member comparison in the center pane."
        )
        workflow_text.setStyleSheet(CLUSTERS_WORKFLOW_HELP_TEXT_STYLE)
        workflow_text.setWordWrap(True)
        w_layout.addWidget(workflow_text)
        layout.addWidget(workflow_box)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        dialog.show()
        self._shortcuts_help_dialog = dialog
        return dialog

    def prompt_hierarchical_clustering(self) -> bool:
        """Open the Response Clustering modal configuration dialog and run clustering if accepted."""
        dlg = HierarchicalClusteringDialog(
            current_params=self.active_clustering_params,
            feature_config=self.feature_config,
            parent=self,
        )
        if dlg.exec():
            new_params = dlg.get_parameters()
            self.active_clustering_params = new_params
            self.feature_config = dlg.get_feature_configuration()
            self.active_clustering_params.feature_config = self.feature_config
            preserve_curated = dlg.keep_curated()
            return self.run_clustering(new_params, preserve_curated=preserve_curated)
        return False

    def prompt_customize_clusters(self) -> None:
        """Open modal dialog to customize cluster names, colors, and order."""
        if not self._cluster_card_order:
            QMessageBox.information(
                self,
                "No Clusters Available",
                "Please run clustering first before customizing clusters.",
            )
            return

        counts: Dict[int, int] = {}
        if self.current_clustering_result is not None:
            for cid, summary in self.current_clustering_result.cluster_summaries.items():
                counts[cid] = summary.count

        hotkeys = {cid: self._get_cluster_hotkey(cid) for cid in self._cluster_card_order}

        dlg = CustomizeClustersDialog(
            cluster_order=list(self._cluster_card_order),
            cluster_names=dict(self._cluster_names),
            cluster_colors=dict(self._cluster_colors),
            cluster_counts=counts,
            cluster_hotkeys=hotkeys,
            parent=self,
        )
        if dlg.exec():
            new_order, new_names, new_colors, new_hotkeys = dlg.get_customizations()
            self.apply_cluster_customizations(new_order, new_names, new_colors, new_hotkeys)

    def apply_cluster_customizations(
        self,
        new_order: List[int],
        new_names: Dict[int, str],
        new_colors: Dict[int, str],
        new_hotkeys: Optional[Dict[int, str]] = None,
    ) -> None:
        """Apply cluster customizations including order, names, colors, and hotkeys.

        Args:
            new_order: New list of cluster IDs in customized order.
            new_names: Mapping of cluster ID to custom display name.
            new_colors: Mapping of cluster ID to color hex code.
            new_hotkeys: Optional mapping of cluster ID to hotkey shortcut string.
        """
        self._push_undo_snapshot()
        self._cluster_card_order = list(new_order)
        self._cluster_names = dict(new_names)
        self.cluster_color_service.set_colors(new_colors)
        if new_hotkeys is not None:
            self._cluster_hotkeys = {int(k): normalize_hotkey(v) for k, v in new_hotkeys.items()}
            self._refresh_cluster_shortcuts()

        # Update cluster_summaries color_hex if available
        if self.current_clustering_result is not None:
            for cid in self.current_clustering_result.cluster_summaries:
                self.current_clustering_result.cluster_summaries[cid].color_hex = self._get_cluster_color(cid)

            # Re-render cluster cards with custom names and colors
            self._render_cluster_cards(self.current_clustering_result)

            # Re-render 2D PCA scatter with custom colors
            self._render_pca_scatter(self.current_clustering_result)
        else:
            self._relayout_cluster_cards()

        # Update table column 2 "Cluster" and column 6 "Status" with custom name and color
        self._update_all_pooled_cluster_items()

        # Update combo_filter_cluster items with custom names
        self._populate_cluster_filter_options()

    def run_clustering(
        self,
        params: Optional[ClusteringParameters] = None,
        preserve_curated: bool = True,
    ) -> bool:
        """Execute unsupervised hierarchical clustering and update 2D scatter and cluster cards.

        Args:
            params: Optional ClusteringParameters. If None, uses self.active_clustering_params.

        Returns:
            True if clustering succeeded, False otherwise.
        """
        if params is None:
            params = self.active_clustering_params

        params.presented_window_ms = self.current_window_ms
        params.presented_bin_size_ms = self.current_bin_size_ms

        # Build pooled dataframe from active sessions or mock records
        df = self._get_active_pooled_dataframe()
        if df.empty or len(df) == 0:
            QMessageBox.warning(
                self,
                "No Units to Cluster",
                "No optotagged units are currently loaded in the Phase 2 pool.\n\n"
                "Please add at least one curated experiment containing optotagged cells before clustering.",
            )
            return False

        # Run clustering analysis
        self._push_undo_snapshot()
        spike_map = self._get_spike_data_map(df)
        result = perform_clustering(
            df,
            params=params,
            embedding_params=self.embedding_params,
            spike_data_map=spike_map,
            psth_cache=self._unit_psth_cache,
            pulse_duration_ms=self._get_pulse_duration_ms(),
        )
        self.current_clustering_result = result

        # Update cluster_id assignments in internal units map and table
        for i, gid in enumerate(result.unit_ids):
            cid = int(result.cluster_labels[i])
            if gid in self._pooled_units_map:
                if not preserve_curated or self._unit_status_map.get(gid) != "Curated":
                    self._pooled_units_map[gid]["cluster_id"] = cid
                    self._unit_status_map[gid] = "Seeded" if cid >= 0 else "Unassigned"
        for r in self._pooled_rows:
            gid = r.get("global_unit_id")
            if gid in self._pooled_units_map:
                r["cluster_id"] = self._pooled_units_map[gid]["cluster_id"]

        if preserve_curated:
            self._sync_clustering_result_and_cards()
        else:
            for cid, summ in result.cluster_summaries.items():
                if cid not in self._cluster_colors:
                    self._cluster_colors[cid] = self._get_cluster_color(cid)
                summ.color_hex = self._get_cluster_color(cid)
            self._render_cluster_cards(result)

        for cid in self.current_clustering_result.cluster_summaries:
            if cid not in self._cluster_colors:
                self._cluster_colors[cid] = self._get_cluster_color(cid)

        # Update 2D PCA Scatter plot
        self._render_pca_scatter(self.current_clustering_result)

        # Update cluster column items across table
        self._update_all_pooled_cluster_items()

        # Update marginal distributions with current population
        if hasattr(self, "marginal_widget"):
            self.marginal_widget.update_distributions(df)

        # Initialize or refresh Active Learning Queue
        self.refresh_active_learning()

        # Update cluster filter options and filter rows
        self._populate_cluster_filter_options()

        self._schedule_background_plot_prewarm()
        self._set_dirty(True)
        self.clustering_completed.emit(len(result.cluster_summaries))
        return True

    def _populate_cluster_filter_options(self) -> None:
        """Populate combo_filter_cluster with All units, statuses, separator, and active clusters."""
        prev_data = self.combo_filter_cluster.currentData()
        if prev_data is None:
            prev_data = -2

        self.combo_filter_cluster.blockSignals(True)
        self.combo_filter_cluster.clear()
        total_units = len(self._pooled_rows)

        # Count units per status and per cluster from _pooled_units_map
        status_counts: Dict[str, int] = {
            "Unassigned": 0,
            "Curated": 0,
            "Seeded": 0,
            "Auto-assigned": 0,
        }
        cluster_counts: Dict[int, int] = {}
        if self.current_clustering_result is not None:
            for cid in self.current_clustering_result.cluster_summaries:
                cluster_counts[cid] = 0
        for cid in self._cluster_card_order:
            cluster_counts[cid] = 0

        for gid, rec in self._pooled_units_map.items():
            cid = rec.get("cluster_id", -1)
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
            st = self._unit_status_map.get(gid, rec.get("status", "Unassigned"))
            if cid < 0 or st not in status_counts:
                st = "Unassigned"
            status_counts[st] += 1

        # 1. All units
        self.combo_filter_cluster.addItem("All units", -2)
        # 2-5. Statuses in exact spec order with colour swatches
        status_specs = [
            ("Unassigned", "status:Unassigned", COLOR_UNASSIGNED),
            ("Curated", "status:Curated", COLOR_SUCCESS),
            ("Seeded", "status:Seeded", COLOR_SEEDED),
            ("Auto-assigned", "status:Auto-assigned", COLOR_ACCENT),
        ]
        for label, key, swatch_hex in status_specs:
            self.combo_filter_cluster.addItem(
                _make_swatch_icon(swatch_hex, 10),
                label,
                key,
            )

        # 6. Separator + 7. Active clusters by name, in their colour, with swatch
        active_cids = [c for c in cluster_counts if c >= 0 and cluster_counts[c] > 0]
        ordered_cids = [c for c in self._cluster_card_order if c in active_cids]
        for c in sorted(active_cids):
            if c not in ordered_cids:
                ordered_cids.append(c)

        if ordered_cids:
            self.combo_filter_cluster.insertSeparator(self.combo_filter_cluster.count())
            for cid in ordered_cids:
                count = cluster_counts[cid]
                name = self._cluster_names.get(cid, f"Cluster {cid}")
                swatch_hex = self._get_cluster_color(cid)
                text_hex = text_color_for_cluster(swatch_hex)
                self.combo_filter_cluster.addItem(
                    _make_swatch_icon(swatch_hex, 10),
                    f"{name} ({count})",
                    cid,
                )
                idx = self.combo_filter_cluster.count() - 1
                self.combo_filter_cluster.setItemData(idx, QColor(text_hex), Qt.ItemDataRole.ForegroundRole)

        # Preserve previous selection if still available, else default to -2 (All units)
        target_idx = 0
        found = False
        for idx in range(self.combo_filter_cluster.count()):
            if self.combo_filter_cluster.itemData(idx) == prev_data:
                target_idx = idx
                found = True
                break
        self.combo_filter_cluster.setCurrentIndex(target_idx)
        self.combo_filter_cluster.blockSignals(False)

        cleared_missing_filter = not found and prev_data not in (-2, "all")
        if cleared_missing_filter:
            self._preserve_cleared_selection = True
            self.table_pooled_units.clearSelection()
            self._current_selected_gid = None

        self._apply_cluster_filter(preserve_cleared_selection=cleared_missing_filter)

    def _apply_cluster_filter(self, *_args: Any, preserve_cleared_selection: bool = False) -> None:
        """Filter pooled units table based on selected status or cluster in combo_filter_cluster."""
        selected_filter = self.combo_filter_cluster.currentData()
        if selected_filter is None:
            selected_filter = -2

        total = self.table_pooled_units.rowCount()
        if total == 0:
            self.lbl_pooled_units_badge.setText("0 units")
            self._current_selected_gid = None
            return

        visible = 0
        for row in range(total):
            gid = self._get_gid_for_row(row)
            rec = self._pooled_units_map.get(gid, {})
            cid = rec.get("cluster_id", -1)
            status = self._unit_status_map.get(gid, rec.get("status", "Unassigned"))
            if cid < 0:
                status = "Unassigned"

            if selected_filter in (-2, "all"):
                hide = False
            elif isinstance(selected_filter, str) and selected_filter.startswith("status:"):
                target_status = selected_filter.split(":", 1)[1]
                hide = (status != target_status)
            elif selected_filter == -1:
                hide = (cid >= 0)
            else:
                hide = (cid != selected_filter)

            self.table_pooled_units.setRowHidden(row, hide)
            if not hide:
                visible += 1

        if selected_filter in (-2, "all"):
            self.lbl_pooled_units_badge.setText(f"{total} units")
        else:
            self.lbl_pooled_units_badge.setText(f"{visible} / {total} units")

        # If currently selected row (or currentRow) is hidden or no selection, shift selection to the next visible row
        selected_rows = self.table_pooled_units.selectionModel().selectedRows()
        curr_row = selected_rows[0].row() if selected_rows else self.table_pooled_units.currentRow()
        cur_idx_row = self.table_pooled_units.currentRow()
        if (
            not preserve_cleared_selection
            and not getattr(self, "_preserve_cleared_selection", False)
            and (
                curr_row < 0
                or self.table_pooled_units.isRowHidden(curr_row)
                or (cur_idx_row >= 0 and self.table_pooled_units.isRowHidden(cur_idx_row))
            )
        ):
            next_visible = -1
            if curr_row >= 0:
                # Search downwards from curr_row
                for r in range(curr_row + 1, self.table_pooled_units.rowCount()):
                    if not self.table_pooled_units.isRowHidden(r):
                        next_visible = r
                        break
                # Search upwards if not found downwards
                if next_visible == -1:
                    for r in range(curr_row - 1, -1, -1):
                        if not self.table_pooled_units.isRowHidden(r):
                            next_visible = r
                            break
            else:
                # Find first visible row in table
                for r in range(self.table_pooled_units.rowCount()):
                    if not self.table_pooled_units.isRowHidden(r):
                        next_visible = r
                        break

            self.table_pooled_units.blockSignals(True)
            self.table_pooled_units.clearSelection()
            if next_visible >= 0:
                self.table_pooled_units.setCurrentCell(next_visible, 0)
                self.table_pooled_units.selectRow(next_visible)
                self.table_pooled_units.blockSignals(False)
                self._on_pooled_unit_selected()
            else:
                self.table_pooled_units.blockSignals(False)
                self._current_selected_gid = None

    def _get_active_pooled_dataframe(self) -> pd.DataFrame:
        """Construct pandas DataFrame of currently pooled optotagged units."""
        if self._is_mock:
            if not self._pooled_rows:
                return pd.DataFrame()
            return pd.DataFrame(self._pooled_rows)

        dfs = []
        for meta in self.loaded_experiments.values():
            tdf = meta.tagged_df
            if tdf is not None and not tdf.empty:
                dfs.append(tdf)

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def _get_spike_data_map(self, df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Construct mapping from global_unit_id and session_unit keys to (spike_times, pulse_onsets)."""
        data_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if df.empty or not hasattr(self, "_experiment_spikes"):
            return data_map

        for _, row in df.iterrows():
            gid = str(row.get("global_unit_id", ""))
            sess = str(row.get("session_id", ""))
            uid_raw = row.get("unit_id")
            if pd.isna(uid_raw):
                continue
            try:
                uid = int(uid_raw)
            except (ValueError, TypeError):
                continue

            if sess in self._experiment_spikes and sess in self._experiment_stimulus:
                spikes_for_sess = self._experiment_spikes[sess]
                if uid in spikes_for_sess:
                    spk_times = spikes_for_sess[uid]
                    p_onsets = self._experiment_stimulus[sess]
                    if spk_times is not None and p_onsets is not None:
                        data_map[gid] = (spk_times, p_onsets)
                        data_map[f"{sess}_{uid}"] = (spk_times, p_onsets)
                        data_map[f"{sess}_u{uid:04d}"] = (spk_times, p_onsets)
        return data_map

    def _invalidate_clustering_caches(self) -> None:
        """Invalidate cached spike data map and pooled PSTH matrix when pool or window settings change."""
        self._cached_spike_data_map = None
        self._cached_spike_data_map_key = None
        self._cached_psth_matrix_tuple = None
        self._cached_psth_matrix_key = None

    def _invalidate_probe_caches(self) -> None:
        """Invalidate cached probe shank geometry and jitter layouts."""
        self._probe_shanks_cache.clear()
        self._probe_layout_cache.clear()

    def _get_cached_spike_data_map(self, df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return cached spike data map for df if unit list hasn't changed."""
        if df.empty:
            return {}
        gids_tuple = tuple(str(g) for g in df.get("global_unit_id", []))
        if self._cached_spike_data_map is not None and self._cached_spike_data_map_key == gids_tuple:
            return self._cached_spike_data_map
        spike_map = self._get_spike_data_map(df)
        self._cached_spike_data_map = spike_map
        self._cached_spike_data_map_key = gids_tuple
        return spike_map

    def _get_cached_psth_matrix(
        self, df: pd.DataFrame, spike_map: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return cached (psth_matrix, t_bins) for df if units and PSTH settings haven't changed."""
        if df.empty:
            return np.zeros((0, 0)), np.zeros(0)
        gids_tuple = tuple(str(g) for g in df.get("global_unit_id", []))
        key = (
            gids_tuple,
            (float(self.current_window_ms[0]), float(self.current_window_ms[1])),
            float(self.current_bin_size_ms),
            bool(self.psth_smooth),
            float(self.psth_gaussian_sigma_ms),
        )
        if self._cached_psth_matrix_tuple is not None and self._cached_psth_matrix_key == key:
            return self._cached_psth_matrix_tuple
        psth_matrix, t_bins = generate_psth_matrix(
            df,
            time_window_ms=self.current_window_ms,
            bin_size_ms=self.current_bin_size_ms,
            spike_data_map=spike_map,
            psth_cache=self._unit_psth_cache,
            smooth=self.psth_smooth,
            gaussian_sigma_ms=self.psth_gaussian_sigma_ms,
        )
        self._cached_psth_matrix_tuple = (psth_matrix, t_bins)
        self._cached_psth_matrix_key = key
        return psth_matrix, t_bins

    def _on_embedding_point_hovered(self, plot: Any, points: Any, ev: Any = None) -> None:
        """Show unit + metrics tooltip when hovering over an embedding scatter point."""
        if points is None or len(points) == 0:
            QToolTip.hideText()
            return
        gid = str(points[0].data() or "")
        rec = self._pooled_units_map.get(gid)
        if not rec:
            return
        uid = rec.get("unit_id", "?")
        sess = _short_session_id(str(rec.get("session_id", "")))
        cid = int(rec.get("cluster_id", -1))
        cname = self._cluster_names.get(cid, f"Cluster {cid}") if cid >= 0 else "Unassigned"
        lat = rec.get("latency_ms")
        rel = rec.get("reliability")
        salt_p = rec.get("salt_p_value")
        lat_str = f"{float(lat):.1f} ms" if pd.notna(lat) and float(lat) < 500 else "N/A"
        rel_str = f"{float(rel):.2f}" if pd.notna(rel) else "0.00"
        if pd.notna(salt_p):
            sv = float(salt_p)
            salt_str = f"≤ 0.01 ({sv:.4f})" if sv <= SALT_P_RESOLUTION_FLOOR + 1e-9 else f"{sv:.4f}"
        else:
            salt_str = "1.0000"
        tip = f"Unit {uid} · {sess} ({cname})\nLat: {lat_str} · Rel: {rel_str} · SALT p: {salt_str}"
        global_pos = ev.screenPos().toPoint() if (ev is not None and hasattr(ev, "screenPos")) else self.pca_plot.mapToGlobal(self.pca_plot.rect().center())
        QToolTip.showText(global_pos, tip, self.pca_plot)

    def _refresh_embedding_and_probe_emphasis(self) -> None:
        """Refresh embedding and probe point emphasis/dimming when cluster comparison opens/closes."""
        if self.current_clustering_result is not None:
            self._render_pca_scatter(self.current_clustering_result, auto_range=False)
        gid = self._current_selected_gid or (self.active_candidate.global_unit_id if self.active_candidate else None)
        if gid:
            self._update_probe_view_for_unit(gid)

    def _render_pca_scatter(self, result: ClusteringResult, auto_range: bool = True) -> None:
        """Render 2D embedding scatter canvas colored by cluster assignment."""
        exp_x = result.explained_variance_ratio[0] * 100.0
        exp_y = result.explained_variance_ratio[1] * 100.0

        if getattr(self, "embedding_params", None) and self.embedding_params.method.upper() == "UMAP":
            self.pca_plot.setLabel("bottom", "UMAP 1")
            self.pca_plot.setLabel("left", "UMAP 2")
            if hasattr(self, "dock_pca"):
                p = self.embedding_params
                self.dock_pca.setTitle(
                    f"2D UMAP Embedding Scatter Plot (k={p.umap_n_neighbors}, d={p.umap_min_dist:.2f}, {p.umap_metric})"
                )
        else:
            pc_x = self.embedding_params.pca_component_x if hasattr(self, "embedding_params") else 1
            pc_y = self.embedding_params.pca_component_y if hasattr(self, "embedding_params") else 2
            self.pca_plot.setLabel("bottom", f"PC{pc_x} ({exp_x:.1f}% var.)")
            self.pca_plot.setLabel("left", f"PC{pc_y} ({exp_y:.1f}% var.)")
            if hasattr(self, "dock_pca"):
                self.dock_pca.setTitle(f"2D PCA Scatter Plot (PC{pc_x} vs. PC{pc_y})")

        open_cid = self._comparison_cluster_id
        spots = []
        self._scatter_points_map.clear()
        for i, gid in enumerate(result.unit_ids):
            x, y = float(result.pca_coordinates[i, 0]), float(result.pca_coordinates[i, 1])
            rec = self._pooled_units_map.get(gid)
            cid = int(rec.get("cluster_id", result.cluster_labels[i])) if rec is not None else int(result.cluster_labels[i])
            color = self._get_cluster_color(cid)

            if open_cid is not None:
                if cid == open_cid:
                    brush = pg.mkBrush(color)
                    size = EMBEDDING_POINT_EMPHASIS_SIZE
                    pen = pg.mkPen(COLOR_PANEL, width=0.8)
                else:
                    brush = pg.mkBrush(rgba_from_hex(color, EMBEDDING_DIM_ALPHA))
                    size = max(5, EMBEDDING_POINT_SIZE - 1)
                    pen = None
            else:
                brush = pg.mkBrush(color)
                size = EMBEDDING_POINT_SIZE
                pen = pg.mkPen(COLOR_PANEL, width=0.6)

            spots.append({
                "pos": (x, y),
                "data": gid,
                "brush": brush,
                "size": size,
                "pen": pen,
            })
            self._scatter_points_map[gid] = (x, y, cid)

        self.scatter_item.setData(spots=spots)

        # Ensure active/selected candidate ring is positioned at the updated 2D coordinates
        ring_gid = self._current_selected_gid or (self.active_candidate.global_unit_id if self.active_candidate else None)
        if ring_gid and ring_gid in self._scatter_points_map:
            px, py, _ = self._scatter_points_map[ring_gid]
            self.cand_scatter.setData(x=[px], y=[py])

        if auto_range:
            self.pca_plot.autoRange()
        coords = np.asarray(result.pca_coordinates, dtype=np.float64)
        if coords.ndim == 2 and len(coords) > 0:
            x_min, x_max = float(np.min(coords[:, 0])), float(np.max(coords[:, 0]))
            y_min, y_max = float(np.min(coords[:, 1])), float(np.max(coords[:, 1]))
            if x_max > x_min:
                self.pca_plot.getAxis("bottom").setTicks(_compute_clean_range_ticks(x_min, x_max, max_ticks=3))
            if y_max > y_min:
                self.pca_plot.getAxis("left").setTicks(_compute_clean_range_ticks(y_min, y_max, max_ticks=3))

    def _render_cluster_cards(self, result: ClusteringResult) -> None:
        """Populate the top cluster card widgets grid with grand-average waveforms."""
        # Synchronize _cluster_card_order with result.cluster_summaries.keys()
        active_cids = set(result.cluster_summaries.keys())
        self._cluster_card_order = [cid for cid in self._cluster_card_order if cid in active_cids]
        for cid in sorted(active_cids):
            if cid not in self._cluster_card_order:
                self._cluster_card_order.append(cid)

        # Remove cards whose cid is no longer active
        for cid in list(self._cluster_cards.keys()):
            if cid not in active_cids:
                card = self._cluster_cards.pop(cid)
                self.clusters_grid.removeWidget(card)
                card.deleteLater()

        # Derive pulse_duration_ms from loaded experiments (or default 10.0)
        pulse_duration_ms = self._get_pulse_duration_ms()

        # Configure grid spacing for fixed-size cluster cards
        self.clusters_grid.setSpacing(12)

        # Add responsive card or update existing card for each active cluster
        for cid in self._cluster_card_order:
            summary = result.cluster_summaries[cid]
            color = self._get_cluster_color(cid)
            summary.color_hex = color
            custom_name = self._cluster_names.get(cid)
            hotkey = self._get_cluster_hotkey(cid)
            if cid in self._cluster_cards:
                self._cluster_cards[cid].update_card(
                    summary=summary,
                    count=summary.count,
                    color_hex=color,
                    custom_name=custom_name,
                    pulse_duration_ms=pulse_duration_ms,
                    is_smooth=self.psth_smooth,
                    norm_mode=self.cluster_psth_norm,
                    hotkey=hotkey,
                )
            else:
                card = ClusterCardWidget(
                    cluster_id=cid,
                    count=summary.count,
                    color_hex=color,
                    summary=summary,
                    pulse_duration_ms=pulse_duration_ms,
                    parent=self,
                    custom_name=custom_name,
                    is_smooth=self.psth_smooth,
                    norm_mode=self.cluster_psth_norm,
                    hotkey=hotkey,
                )
                card.card_clicked.connect(self._on_cluster_card_clicked)
                card.assign_requested.connect(lambda c: self.assign_candidate_to_cluster(c))
                card.merge_requested.connect(self._on_cluster_merge_requested)
                card.move_requested.connect(self._on_cluster_move_requested)
                card.rename_requested.connect(self._on_card_rename_requested)
                card.recolor_requested.connect(self._on_card_recolor_requested)
                card.dissolve_requested.connect(lambda c: self.dissolve_cluster(c, confirm=True))
                card._other_clusters_provider = self._other_clusters_for_merge
                self._cluster_cards[cid] = card

        if hasattr(self, "lbl_clusters_count_chip"):
            self.lbl_clusters_count_chip.setText(str(len(self._cluster_card_order)))

        self._relayout_cluster_cards()
        self._refresh_cluster_shortcuts()
        self._apply_shared_y_axis()

        current_gid = self._current_selected_gid or (self.active_candidate.global_unit_id if self.active_candidate else None)
        self._update_cluster_cards_candidate_psth(current_gid)
        if self._comparison_cluster_id is not None and self._comparison_cluster_id in self._cluster_cards:
            self._cluster_cards[self._comparison_cluster_id].set_comparison_active(True)
        self._update_all_cluster_card_suggested_badges()

    def _compute_card_grid_columns(self) -> int:
        """Compute number of cluster card columns that fit in the scroll viewport without clipping (§ 5.2)."""
        if not hasattr(self, "clusters_scroll_area"):
            return 4
        avail_w = self.clusters_scroll_area.viewport().width()
        if avail_w <= 0 or not self.isVisible():
            return 4
        card_w = CLUSTER_CARD_SIZE[0]
        spacing = CLUSTER_GRID_SPACING
        return max(1, (avail_w + spacing) // (card_w + spacing))

    def _relayout_cluster_cards(self) -> None:
        """Reposition existing card widgets in clusters_grid dynamically according to viewport width (§ 5.2)."""
        for card in self._cluster_cards.values():
            self.clusters_grid.removeWidget(card)
        if self._new_cluster_card is not None:
            self.clusters_grid.removeWidget(self._new_cluster_card)

        cols = self._compute_card_grid_columns()
        self._current_grid_cols = cols

        idx = 0
        for cid in self._cluster_card_order:
            card = self._cluster_cards.get(cid)
            if card is not None:
                card.set_hotkey(self._get_cluster_hotkey(cid))
                row = idx // cols
                col = idx % cols
                self.clusters_grid.addWidget(card, row, col)
                card.show()
                idx += 1

        if self._new_cluster_card is None:
            self._new_cluster_card = NewClusterCardWidget(self.clusters_scroll_widget)
            self._new_cluster_card.clicked.connect(self.create_new_cluster_for_candidate)
        self.clusters_grid.addWidget(self._new_cluster_card, idx // cols, idx % cols)
        self._new_cluster_card.show()

        if hasattr(self, "lbl_clusters_count_chip"):
            self.lbl_clusters_count_chip.setText(str(len(self._cluster_card_order)))

    def _on_overlay_focused_unit_toggled(self, enabled: bool) -> None:
        """Toggle focused unit PSTH overlay on all cluster cards (§ 9.1)."""
        self.overlay_focused_unit_enabled = bool(enabled)
        current_gid = self._current_selected_gid or (self.active_candidate.global_unit_id if self.active_candidate else None)
        self._update_cluster_cards_candidate_psth(current_gid)

    def _on_shared_y_axis_toggled(self, enabled: bool) -> None:
        """Toggle shared y-axis scaling across all cluster card plots (§ 9.1)."""
        self.shared_y_axis_enabled = bool(enabled)
        self._apply_shared_y_axis()

    def _apply_shared_y_axis(self) -> None:
        """Synchronize y-axis range across all cluster cards when shared y-axis is enabled."""
        if not getattr(self, "shared_y_axis_enabled", False):
            for card in self._cluster_cards.values():
                card.set_y_range(None, None)
            return

        y_min = 0.0
        y_max = 0.0
        for card in self._cluster_cards.values():
            summ = getattr(card, "summary", None)
            if summ is not None and len(summ.mean_psth) > 0:
                y = np.asarray(summ.mean_psth, dtype=np.float64)
                sem = np.asarray(summ.sem_psth, dtype=np.float64) if len(summ.sem_psth) == len(y) else np.zeros_like(y)
                y_max = max(y_max, float(np.nanmax(y + sem)))
                y_min = min(y_min, float(np.nanmin(np.maximum(0.0, y - sem))))
        if y_max <= y_min:
            y_max = 1.0
        for card in self._cluster_cards.values():
            card.set_y_range(y_min, y_max)

    def _sort_cluster_cards(self, sort_mode: str) -> None:
        """Sort cluster card order by 'id', 'size', 'latency', or 'name' while keeping colours & keys fixed (§ 9.1)."""
        self._push_undo_snapshot()
        self._cluster_sort_mode = sort_mode
        if hasattr(self, "_sort_actions") and sort_mode in self._sort_actions:
            self._sort_actions[sort_mode].setChecked(True)

        if sort_mode == "size":
            def _key(cid: int) -> Tuple[int, int]:
                card = self._cluster_cards.get(cid)
                cnt = card.count if card is not None else 0
                return (-cnt, cid)
        elif sort_mode == "latency":
            lat_by_cid: Dict[int, List[float]] = {}
            for rec in self._pooled_units_map.values():
                c = rec.get("cluster_id", -1)
                if c >= 0:
                    lat = rec.get("first_spike_latency_ms", rec.get("latency_ms"))
                    if lat is not None and np.isfinite(float(lat)):
                        lat_by_cid.setdefault(c, []).append(float(lat))

            def _key(cid: int) -> Tuple[float, int]:
                vals = lat_by_cid.get(cid)
                mean_lat = float(np.mean(vals)) if vals else float("inf")
                return (mean_lat, cid)
        elif sort_mode == "name":
            def _key(cid: int) -> Tuple[str, int]:
                name = self._cluster_names.get(cid, f"Cluster {cid}")
                return (name.lower(), cid)
        else:
            def _key(cid: int) -> int:
                return cid

        self._cluster_card_order = sorted(self._cluster_card_order, key=_key)
        self._relayout_cluster_cards()

    def _apply_threshold_preview(self, pct: int, auto_assign: bool) -> None:
        """Live-update confidence threshold line, card '!' badges, and signals."""
        self.auto_assign_threshold = max(0.50, min(0.99, float(pct) / 100.0))
        if hasattr(self, "slider_threshold"):
            self.slider_threshold.blockSignals(True)
            self.slider_threshold.setValue(int(pct))
            self.slider_threshold.blockSignals(False)
        if hasattr(self, "cluster_prob_bar") and self._current_selected_gid:
            probs = self._all_unit_probabilities.get(self._current_selected_gid, {})
            if probs:
                self.cluster_prob_bar.set_probabilities(
                    probabilities=probs,
                    cluster_colors=self._cluster_colors,
                    cluster_names=self._cluster_names,
                    threshold=self.auto_assign_threshold,
                    cluster_order=sorted(self._cluster_card_order),
                )
        self._update_all_cluster_card_suggested_badges()
        self.confidence_threshold_changed.emit(self.auto_assign_threshold)

    def prompt_confidence_threshold(self) -> None:
        """Open ConfidenceThresholdDialog (50–99%) from status bar, menu, or '⋯' menu (§ 12)."""
        orig_threshold = float(getattr(self, "auto_assign_threshold", 0.85))
        orig_auto_assign = bool(getattr(self, "auto_assign_enabled", False))

        unassigned_max_probs: Dict[str, float] = {}
        for gid, status in self._unit_status_map.items():
            if status == "Unassigned":
                probs = self._all_unit_probabilities.get(gid, {})
                if probs:
                    unassigned_max_probs[gid] = max(probs.values())

        dlg = ConfidenceThresholdDialog(
            current_threshold=orig_threshold,
            auto_assign_enabled=orig_auto_assign,
            unassigned_max_probs=unassigned_max_probs,
            on_live_update=self._apply_threshold_preview,
            parent=self,
        )

        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.auto_assign_threshold = dlg.get_threshold()
            new_auto_assign = dlg.is_auto_assign_enabled()
            if hasattr(self, "slider_threshold"):
                self.slider_threshold.blockSignals(True)
                self.slider_threshold.setValue(dlg.get_threshold_percent())
                self.slider_threshold.blockSignals(False)
            self.confidence_threshold_changed.emit(self.auto_assign_threshold)
            if new_auto_assign != orig_auto_assign:
                if hasattr(self, "chk_auto_assign"):
                    self.chk_auto_assign.blockSignals(True)
                    self.chk_auto_assign.setChecked(new_auto_assign)
                    self.chk_auto_assign.blockSignals(False)
                self._on_auto_assign_toggled(new_auto_assign)
            else:
                self.update_auto_assignment_threshold()
        else:
            self._apply_threshold_preview(int(round(orig_threshold * 100)), orig_auto_assign)

    def prompt_active_learning_settings(self) -> None:
        """Open ActiveLearningSettingsDialog and retrain classifier on accept (§ 12)."""
        dlg = ActiveLearningSettingsDialog(
            feature_config=self.feature_config,
            threshold=self.auto_assign_threshold,
            n_estimators=getattr(self, "al_n_estimators", 100),
            random_state=getattr(self, "al_random_state", 42),
            max_distance_sigma=getattr(self, "al_max_distance_sigma", 3.5),
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_config = dlg.get_feature_configuration()
            if tuple(new_config.time_window_ms) != tuple(self.feature_config.time_window_ms):
                self._feature_window_customized = (
                    tuple(new_config.time_window_ms) != self._default_feature_time_window_ms()
                )
            self.feature_config = new_config
            self.active_clustering_params.feature_config = self.feature_config
            self.auto_assign_threshold = dlg.get_threshold()
            self.al_n_estimators = dlg.get_n_estimators()
            self.al_random_state = dlg.get_random_state()
            self.al_max_distance_sigma = dlg.get_max_distance_sigma()
            if hasattr(self, "slider_threshold"):
                self.slider_threshold.blockSignals(True)
                self.slider_threshold.setValue(int(round(self.auto_assign_threshold * 100)))
                self.slider_threshold.blockSignals(False)
            self.confidence_threshold_changed.emit(self.auto_assign_threshold)
            self._invalidate_clustering_caches()
            self.refresh_active_learning()

    def prompt_visualization_settings(self) -> None:
        """Open VisualizationSettingsDialog and apply display-only settings (§ 12)."""
        dlg = VisualizationSettingsDialog(
            window_ms=self.current_window_ms,
            bin_size_ms=self.current_bin_size_ms,
            pulse_order=self.raster_order_by,
            psth_smooth=self.psth_smooth,
            psth_sigma_ms=self.psth_gaussian_sigma_ms,
            norm_mode=self.cluster_psth_norm,
            marker_size=self.raster_marker_size,
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            s = dlg.get_settings()
            self.set_visualized_window(s["window_ms"])
            self.set_cluster_bin_size(s["bin_size_ms"])
            self.set_pulse_ordering(s["pulse_order"])
            self.set_psth_smooth(s["psth_smooth"])
            self.set_psth_gaussian_sigma(s["psth_sigma_ms"])
            self.set_psth_norm_mode(s["norm_mode"])
            self.set_raster_marker_size(s["marker_size"])

    def _other_clusters_for_merge(self, exclude_cid: int) -> List[Tuple[int, str]]:
        """Return (cluster_id, display_name) pairs for other active clusters."""
        return [
            (cid, self._cluster_names.get(cid, f"Cluster {cid}"))
            for cid in self._cluster_card_order
            if cid != exclude_cid
        ]

    def _on_card_rename_requested(self, cluster_id: int) -> None:
        """Rename a cluster from its right-click context menu (§ 9.2)."""
        cur_name = self._cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        new_name, ok = QInputDialog.getText(
            self,
            "Rename cluster",
            "Cluster name:",
            QLineEdit.EchoMode.Normal,
            cur_name,
        )
        if not ok:
            return
        self._push_undo_snapshot()
        cleaned = new_name.strip()
        if cleaned:
            self._cluster_names[cluster_id] = cleaned
        else:
            self._cluster_names.pop(cluster_id, None)
        display_name = self._cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        card = self._cluster_cards.get(cluster_id)
        if card is not None:
            card.set_display_name(display_name)
        if self.current_clustering_result is not None and cluster_id in self.current_clustering_result.cluster_summaries:
            self.current_clustering_result.cluster_summaries[cluster_id].name = display_name
        self._update_all_pooled_cluster_items()
        self._populate_cluster_filter_options()
        if self._comparison_cluster_id == cluster_id:
            self.set_cluster_comparison(cluster_id)
        self._refresh_list_panel()

    def _on_card_recolor_requested(self, cluster_id: int) -> None:
        """Change a cluster's colour from its right-click context menu (§ 9.2)."""
        cur_hex = self._get_cluster_color(cluster_id)
        color = QColorDialog.getColor(QColor(cur_hex), self, "Change cluster colour")
        if not color.isValid():
            return
        self._push_undo_snapshot()
        new_hex = color.name()
        self.cluster_color_service.set_color(cluster_id, new_hex)
        card = self._cluster_cards.get(cluster_id)
        if card is not None:
            card.set_color(new_hex)
        if self.current_clustering_result is not None:
            if cluster_id in self.current_clustering_result.cluster_summaries:
                self.current_clustering_result.cluster_summaries[cluster_id].color_hex = new_hex
            self._render_pca_scatter(self.current_clustering_result)
        self._update_all_pooled_cluster_items()
        if self._comparison_cluster_id == cluster_id:
            self.set_cluster_comparison(cluster_id)
        if self.active_candidate is not None:
            self._display_candidate(self.active_candidate)
        self._refresh_list_panel()

    def dissolve_cluster(self, cluster_id: int, confirm: bool = True) -> bool:
        """Dissolve a cluster, unassigning all its members with undo support (§ 9.2)."""
        members = [
            gid for gid, rec in self._pooled_units_map.items()
            if rec.get("cluster_id") == cluster_id
        ]
        if not members and cluster_id not in self._cluster_cards:
            return False

        cname = self._cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        if confirm:
            reply = QMessageBox.question(
                self,
                "Dissolve cluster",
                f"Dissolve {cname}?\nIts {len(members)} units become unassigned. You can undo this.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False

        if not self._is_applying_undo_redo:
            self._undo_stack.append(self._capture_curation_snapshot())
            self._redo_stack.clear()
            if hasattr(self, "btn_undo"):
                self.btn_undo.setEnabled(True)
            if hasattr(self, "btn_redo"):
                self.btn_redo.setEnabled(False)

        # Capture pre-action position in current focus source for § 10.2 Advance rule
        for gid in members:
            self._pooled_units_map[gid]["cluster_id"] = -1
            self._pooled_units_map[gid]["status"] = "Unassigned"
            self._pooled_units_map[gid]["curated_status"] = "Unassigned"
            self._unit_status_map[gid] = "Unassigned"

        for rec in self._pooled_rows:
            if rec.get("global_unit_id") in members or rec.get("cluster_id") == cluster_id:
                rec["cluster_id"] = -1
                rec["status"] = "Unassigned"

        if self._comparison_cluster_id == cluster_id:
            self.clear_cluster_comparison(return_to_suggestion=False)

        self._update_all_pooled_cluster_items(target_gids=set(members))
        self._populate_cluster_filter_options()
        self._sync_clustering_result_and_cards()
        self.refresh_active_learning()
        return True

    def _on_list_panel_toggle_collapse(self) -> None:
        """Collapse or expand the List Panel inside right_splitter down to its header bar (§ 9.3)."""
        self._list_panel_collapsed = not getattr(self, "_list_panel_collapsed", False)
        if hasattr(self, "btn_list_collapse"):
            self.btn_list_collapse.setText("▸" if self._list_panel_collapsed else "▾")
        if not hasattr(self, "list_panel"):
            return

        if self._list_panel_collapsed:
            if hasattr(self, "right_splitter"):
                sizes = self.right_splitter.sizes()
                if len(sizes) == 2 and sizes[1] >= LIST_PANEL_MIN_HEIGHT:
                    self._list_panel_expanded_height = int(sizes[1])
            elif self.list_panel.height() >= LIST_PANEL_MIN_HEIGHT:
                self._list_panel_expanded_height = int(self.list_panel.height())

            if hasattr(self, "list_panel_body"):
                self.list_panel_body.setVisible(False)

            hdr_h = self.list_panel_header.height() if hasattr(self, "list_panel_header") else 30
            collapsed_h = max(36, hdr_h + 14)
            self.list_panel.setMinimumHeight(collapsed_h)
            self.list_panel.setMaximumHeight(collapsed_h)
            if hasattr(self, "right_splitter"):
                total_h = sum(self.right_splitter.sizes())
                if total_h > collapsed_h:
                    self.right_splitter.setSizes([total_h - collapsed_h, collapsed_h])
        else:
            self.list_panel.setMaximumHeight(16777215)
            self.list_panel.setMinimumHeight(LIST_PANEL_MIN_HEIGHT)
            if hasattr(self, "list_panel_body"):
                self.list_panel_body.setVisible(True)
            if hasattr(self, "right_splitter"):
                sizes = self.right_splitter.sizes()
                total_h = sum(sizes)
                target_h = max(LIST_PANEL_MIN_HEIGHT, int(getattr(self, "_list_panel_expanded_height", 260)))
                if total_h > LIST_PANEL_MIN_HEIGHT:
                    target_h = min(target_h, max(LIST_PANEL_MIN_HEIGHT, total_h - 120))
                    self.right_splitter.setSizes([total_h - target_h, target_h])

    def _on_list_panel_cell_clicked(self, row: int, _col: int) -> None:
        """Focus the clicked unit in the List Panel (§ 9.3, § 10.2)."""
        if not hasattr(self, "table_list_panel"):
            return
        item = self.table_list_panel.item(row, 0)
        if item is None:
            return
        gid = item.data(Qt.ItemDataRole.UserRole)
        if gid:
            source = "list_candidates" if self._comparison_cluster_id is not None else "queue"
            self.select_pooled_unit(str(gid), source=source)

    def _on_list_panel_current_cell_changed(
        self, row: int, _col: int, _prev_row: int, _prev_col: int
    ) -> None:
        """Focus the unit at `row` when navigating table_list_panel (e.g., via arrow keys)."""
        if row < 0 or getattr(self, "_suppress_list_panel_selection", False):
            return
        if not hasattr(self, "table_list_panel"):
            return
        item = self.table_list_panel.item(row, 0)
        if item is None:
            return
        gid = item.data(Qt.ItemDataRole.UserRole)
        if gid and str(gid) != self._current_selected_gid:
            source = "list_candidates" if self._comparison_cluster_id is not None else "queue"
            self.select_pooled_unit(str(gid), source=source)

    def _get_list_panel_selected_gids(self) -> List[str]:
        """Return ordered global_unit_ids for selected rows in table_list_panel."""
        if not hasattr(self, "table_list_panel"):
            return []
        sel_model = self.table_list_panel.selectionModel()
        if sel_model is None:
            return []
        gids: List[str] = []
        for idx in sorted(sel_model.selectedRows(), key=lambda m: m.row()):
            item = self.table_list_panel.item(idx.row(), 0)
            if item is not None:
                gid = item.data(Qt.ItemDataRole.UserRole)
                if gid and str(gid) in self._pooled_units_map:
                    gids.append(str(gid))
        return gids

    def _on_list_panel_selection_changed(self) -> None:
        """Update 'Accept N selected' button state when List Panel selection changes."""
        if getattr(self, "_suppress_list_panel_selection", False):
            return
        if not hasattr(self, "table_list_panel") or not hasattr(self, "btn_list_accept_selected"):
            return
        if self._comparison_cluster_id is None:
            self.btn_list_accept_selected.setVisible(False)
            return
        selected_rows = self.table_list_panel.selectionModel().selectedRows() if self.table_list_panel.selectionModel() else []
        n = len(selected_rows)
        self.btn_list_accept_selected.setText(f"Accept {n} selected")
        self.btn_list_accept_selected.setEnabled(n > 0)

    def _on_list_panel_accept_selected(self) -> None:
        """Assign all selected candidate rows in List Panel to the currently open cluster (§ 9.3)."""
        cid = self._comparison_cluster_id
        if cid is None or not hasattr(self, "table_list_panel"):
            return
        sel_model = self.table_list_panel.selectionModel()
        if sel_model is None:
            return
        gids: List[str] = []
        for idx in sel_model.selectedRows():
            item = self.table_list_panel.item(idx.row(), 0)
            if item is not None:
                gid = item.data(Qt.ItemDataRole.UserRole)
                if gid:
                    gids.append(str(gid))
        if gids:
            self.assign_candidate_to_cluster(cid, unit_id=gids)

    def _refresh_list_panel(self) -> None:
        """Populate the List Panel in one of its 3 states (§ 9.3)."""
        if not hasattr(self, "table_list_panel"):
            return

        open_cid = getattr(self, "_comparison_cluster_id", None)
        al_on = bool(getattr(self, "active_learning_enabled", True))

        # Collect all unassigned units
        unassigned_gids = [
            gid for gid, rec in self._pooled_units_map.items()
            if rec.get("cluster_id", -1) < 0
        ]

        # Build entropy lookup from ranked_candidates
        entropy_map: Dict[str, float] = {
            c.global_unit_id: float(c.entropy)
            for c in getattr(self, "ranked_candidates", [])
        }

        if open_cid is not None:
            cname = self._cluster_names.get(open_cid, f"Cluster {open_cid}")
            text_col = text_color_for_cluster(self._get_cluster_color(open_cid))
            self.lbl_list_panel_title.setText(build_list_panel_candidates_title_html(cname, text_col))
            self.lbl_list_panel_subtitle.setText("by confidence")
            self.table_list_panel.setHorizontalHeaderItem(2, QTableWidgetItem("Confidence"))
            self.btn_list_accept_selected.setVisible(True)

            def _sort_key(gid: str) -> Tuple[float, str]:
                probs = self._all_unit_probabilities.get(gid, {})
                p = float(probs.get(open_cid, 0.0))
                return (-p, gid)

            ordered_gids = sorted(unassigned_gids, key=_sort_key)
        elif al_on:
            self.lbl_list_panel_title.setText("Suggested next")
            self.lbl_list_panel_subtitle.setText("by classifier uncertainty")
            self.table_list_panel.setHorizontalHeaderItem(2, QTableWidgetItem("Top probability"))
            self.btn_list_accept_selected.setVisible(False)

            def _sort_key(gid: str) -> Tuple[float, str]:
                ent = entropy_map.get(gid, 0.0)
                return (-ent, gid)

            ordered_gids = sorted(unassigned_gids, key=_sort_key)
        else:
            self.lbl_list_panel_title.setText("Unassigned units")
            self.lbl_list_panel_subtitle.setText("by unit ID")
            self.table_list_panel.setHorizontalHeaderItem(2, QTableWidgetItem("Top probability"))
            self.btn_list_accept_selected.setVisible(False)

            def _sort_key(gid: str) -> Tuple[int, str]:
                rec = self._pooled_units_map.get(gid, {})
                uid_raw = str(rec.get("unit_id", gid))
                try:
                    return (0, f"{int(uid_raw):09d}")
                except ValueError:
                    return (1, uid_raw)

            ordered_gids = sorted(unassigned_gids, key=_sort_key)

        self.lbl_list_panel_count.setText(str(len(ordered_gids)))

        self._suppress_list_panel_selection = True
        self.table_list_panel.blockSignals(True)
        self.table_list_panel.setRowCount(len(ordered_gids))

        mono_font = make_mono_font(FONT_SIZE_TICK)
        for r, gid in enumerate(ordered_gids):
            rec = self._pooled_units_map.get(gid, {})
            sess_str, uid_str = _format_unit_display(rec.get("session_id", ""), rec.get("unit_id", gid), gid)
            probs = self._all_unit_probabilities.get(gid, {})

            best_cid: Optional[int] = None
            top_prob = 0.0
            if probs:
                best_cid, top_prob = max(probs.items(), key=lambda kv: float(kv[1]))
                top_prob = float(top_prob)

            shown_prob = float(probs.get(open_cid, 0.0)) if open_cid is not None else top_prob
            ent = entropy_map.get(gid, 0.0)

            u_item = QTableWidgetItem(uid_str)
            u_item.setData(Qt.ItemDataRole.UserRole, gid)
            u_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_list_panel.setItem(r, 0, u_item)

            s_item = QTableWidgetItem(_short_session_id(sess_str))
            s_item.setToolTip(sess_str)
            s_item.setData(Qt.ItemDataRole.UserRole, gid)
            s_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_list_panel.setItem(r, 1, s_item)

            prob_item = QTableWidgetItem(f"{int(round(shown_prob * 100))}%")
            prob_item.setData(Qt.ItemDataRole.UserRole, gid)
            prob_item.setData(Qt.ItemDataRole.UserRole + 1, shown_prob)
            prob_item.setData(Qt.ItemDataRole.UserRole + 2, "prob")
            prob_item.setFont(mono_font)
            self.table_list_panel.setItem(r, 2, prob_item)

            unc_item = QTableWidgetItem(f"{ent:.2f}")
            unc_item.setData(Qt.ItemDataRole.UserRole, gid)
            unc_item.setData(Qt.ItemDataRole.UserRole + 1, ent)
            unc_item.setData(Qt.ItemDataRole.UserRole + 2, "uncertainty")
            unc_item.setFont(mono_font)
            self.table_list_panel.setItem(r, 3, unc_item)

            if best_cid is not None and best_cid >= 0:
                bname = self._cluster_names.get(best_cid, f"Cluster {best_cid}")
                b_item = QTableWidgetItem(bname)
                b_item.setForeground(QColor(text_color_for_cluster(self._get_cluster_color(best_cid))))
            else:
                b_item = QTableWidgetItem("—")
                b_item.setForeground(QColor(COLOR_TEXT_MUTED))
            b_item.setData(Qt.ItemDataRole.UserRole, gid)
            self.table_list_panel.setItem(r, 4, b_item)

            lat_val = rec.get("first_spike_latency_ms", rec.get("latency_ms"))
            lat_txt = f"{float(lat_val):.1f}" if lat_val is not None and np.isfinite(float(lat_val)) else "—"
            lat_item = QTableWidgetItem(lat_txt)
            lat_item.setFont(mono_font)
            lat_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table_list_panel.setItem(r, 5, lat_item)

            jit_val = rec.get("spike_jitter_ms", rec.get("jitter_ms"))
            jit_txt = f"{float(jit_val):.1f}" if jit_val is not None and np.isfinite(float(jit_val)) else "—"
            jit_item = QTableWidgetItem(jit_txt)
            jit_item.setFont(mono_font)
            jit_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table_list_panel.setItem(r, 6, jit_item)

            rel_val = rec.get("reliability")
            rel_txt = f"{float(rel_val):.2f}" if rel_val is not None and np.isfinite(float(rel_val)) else "—"
            rel_item = QTableWidgetItem(rel_txt)
            rel_item.setFont(mono_font)
            rel_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table_list_panel.setItem(r, 7, rel_item)

            salt_val = rec.get("salt_p_value")
            if salt_val is not None and np.isfinite(float(salt_val)):
                sp = float(salt_val)
                salt_txt = f"≤ {SALT_P_RESOLUTION_FLOOR:.2f}" if sp <= SALT_P_RESOLUTION_FLOOR else f"{sp:.3f}"
                salt_tip = f"Exact SALT p = {sp:.4g}"
            else:
                salt_txt = "—"
                salt_tip = ""
            salt_item = QTableWidgetItem(salt_txt)
            salt_item.setFont(mono_font)
            salt_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if salt_tip:
                salt_item.setToolTip(salt_tip)
            self.table_list_panel.setItem(r, 8, salt_item)

        self.table_list_panel.blockSignals(False)
        self._suppress_list_panel_selection = False
        self._on_list_panel_selection_changed()

    def _sync_clustering_result_and_cards(
        self,
        *,
        recompute_embedding: bool = False,
        prune_and_refresh_list: bool = True,
    ) -> None:
        """Synchronize current_clustering_result and cluster card summaries with _pooled_units_map."""
        df_active = self._get_active_pooled_dataframe()
        if df_active.empty:
            return

        need_embedding = (
            recompute_embedding
            or self.current_clustering_result is None
            or len(self.current_clustering_result.unit_ids) != len(df_active)
        )
        if not need_embedding and self.current_clustering_result is not None:
            unit_ids = self.current_clustering_result.unit_ids
        elif "global_unit_id" in df_active.columns:
            unit_ids = df_active["global_unit_id"].astype(str).tolist()
        else:
            unit_ids = [f"u_{i}" for i in range(len(df_active))]

        labels = np.array([
            self._pooled_units_map.get(gid, {}).get("cluster_id", -1)
            for gid in unit_ids
        ], dtype=np.int32)

        spike_map = self._get_cached_spike_data_map(df_active)

        if need_embedding:
            feat_matrix, _, _ = extract_feature_matrix(
                df_active,
                feature_config=self.feature_config,
                spike_data_map=spike_map,
                psth_cache=self._unit_psth_cache,
                pulse_duration_ms=self._get_pulse_duration_ms(),
            )
            pca_coords, exp_var = compute_2d_embedding(
                feat_matrix,
                embedding_params=self.embedding_params,
                random_state=getattr(self, "al_random_state", 42),
            )
            self.current_clustering_result = ClusteringResult(
                cluster_labels=labels,
                pca_coordinates=pca_coords,
                explained_variance_ratio=exp_var,
                cluster_summaries={},
                feature_matrix=feat_matrix,
                unit_ids=unit_ids,
                parameters=self.active_clustering_params,
            )
        else:
            self.current_clustering_result.cluster_labels = labels

        psth_matrix, t_bins = self._get_cached_psth_matrix(df_active, spike_map)

        # Determine non-empty cluster IDs with at least 1 member in labels
        non_empty_cids = set(int(c) for c in np.unique(labels) if c >= 0)
        self._cluster_card_order = [cid for cid in self._cluster_card_order if cid in non_empty_cids]
        for c in sorted(non_empty_cids):
            if c not in self._cluster_card_order:
                self._cluster_card_order.append(c)

        new_summaries = compute_cluster_summaries(
            df_active,
            self.current_clustering_result.cluster_labels,
            psth_matrix,
            t_bins,
            active_cluster_ids=self._cluster_card_order,
            norm_mode=self.cluster_psth_norm,
        )
        for c_id, summ in new_summaries.items():
            summ.color_hex = self._get_cluster_color(c_id)
        self.current_clustering_result.cluster_summaries = new_summaries
        self._render_cluster_cards(self.current_clustering_result)
        self._render_pca_scatter(self.current_clustering_result)
        if prune_and_refresh_list:
            self.prune_empty_clusters()
            if self._comparison_cluster_id is not None and self._comparison_cluster_id not in self._cluster_cards:
                self.clear_cluster_comparison()
            self._refresh_list_panel()

    def _on_scatter_point_clicked(self, scatter: Any, points: Any) -> None:
        """Interactive point selection callback: clicking a PCA point selects the corresponding unit."""
        if points is None or len(points) == 0:
            return

        # If multiple overlapping points were clicked, cycle through them if current is selected
        curr_gid = getattr(self, "_current_selected_gid", None)
        target_gid = None
        if curr_gid and len(points) > 1:
            for idx, pt in enumerate(points):
                if pt.data() and str(pt.data()) == curr_gid:
                    target_gid = points[(idx + 1) % len(points)].data()
                    break
        if not target_gid:
            target_gid = points[0].data()

        if target_gid:
            self.select_pooled_unit(str(target_gid), source="queue")

    def _on_cluster_card_clicked(self, cluster_id: int) -> None:
        """Toggle side-by-side cluster comparison for the clicked cluster."""
        if self._comparison_cluster_id == cluster_id:
            self.clear_cluster_comparison()
        else:
            self.set_cluster_comparison(cluster_id)

    def set_cluster_comparison(self, cluster_id: int) -> None:
        """Open the side-by-side cluster comparison pane, populate ground-truth member waveforms and candidate suggestions."""
        if cluster_id not in self._cluster_cards or cluster_id not in self._cluster_card_order:
            self.clear_cluster_comparison(return_to_suggestion=False)
            return

        self._comparison_cluster_id = cluster_id
        if hasattr(self, "inspector_empty_state"):
            self.inspector_empty_state.setVisible(False)
        self.comparison_pane.setVisible(True)

        # Update card highlights
        for cid, card in self._cluster_cards.items():
            card.set_comparison_active(cid == cluster_id)

        cname = self._cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        color_hex = self._get_cluster_color(cluster_id)
        self.comparison_pane.setStyleSheet(CLUSTERS_COMPARISON_PANE_NEUTRAL_STYLE)

        member_gids = [
            gid for gid, rec in self._pooled_units_map.items()
            if rec.get("cluster_id") == cluster_id
        ]
        # Order Curated/Seeded members first, followed by Auto-assigned (or other) members
        display_gids = sorted(
            member_gids,
            key=lambda g: (
                0
                if self._unit_status_map.get(g, self._pooled_units_map[g].get("status", "Unassigned"))
                in ("Curated", "Seeded")
                else 1
            ),
        )

        self._update_inspector_header(cluster_id, cname, color_hex, member_gids)

        # Update marginal distributions with cluster overlay
        if hasattr(self, "marginal_widget"):
            cluster_df = pd.DataFrame([self._pooled_units_map[g] for g in member_gids]) if member_gids else pd.DataFrame()
            self.marginal_widget.update_cluster_distribution(cluster_df, color_hex, cluster_name=cname)

        if hasattr(self, "cluster_prob_bar"):
            self.cluster_prob_bar.set_inspected_cluster(cluster_id)

        self._refresh_embedding_and_probe_emphasis()
        self._refresh_list_panel()

        # Check if rendering the same cluster or switching
        if self._comparison_rendered_cid != cluster_id:
            while self.comparison_list_layout.count():
                item = self.comparison_list_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()
            self.comparison_viewers.clear()
            self._comparison_member_viewers_map.clear()
            self._comparison_rendered_cid = cluster_id

            self.lbl_members_section.setText(f"▼ CLUSTER MEMBERS (N={len(display_gids)})")
            if not display_gids:
                lbl_empty = QLabel("No member units in this cluster.")
                lbl_empty.setObjectName("lbl_comp_empty_members")
                lbl_empty.setStyleSheet(CLUSTERS_COMPARISON_EMPTY_LABEL_STYLE)
                self.comparison_list_layout.addWidget(lbl_empty)
            else:
                for gid in display_gids:
                    viewer = self._create_member_response_row(gid)
                    self.comparison_list_layout.addWidget(viewer)
                    self.comparison_viewers.append(viewer)
                    self._comparison_member_viewers_map[gid] = viewer
            self.comparison_list_layout.addStretch()
        else:
            # Incremental update for same cluster_id
            self.lbl_members_section.setText(f"▼ CLUSTER MEMBERS (N={len(display_gids)})")
            display_set = set(display_gids)

            # If empty placeholder label is present and now we have members, remove it
            for idx in range(self.comparison_list_layout.count()):
                item = self.comparison_list_layout.itemAt(idx)
                if item and item.widget() and item.widget().objectName() == "lbl_comp_empty_members":
                    w = item.widget()
                    self.comparison_list_layout.removeWidget(w)
                    w.deleteLater()
                    break

            # 1. Remove viewers for gids no longer in display_set
            for gid in list(self._comparison_member_viewers_map.keys()):
                if gid not in display_set:
                    viewer = self._comparison_member_viewers_map.pop(gid)
                    self.comparison_list_layout.removeWidget(viewer)
                    if viewer in self.comparison_viewers:
                        self.comparison_viewers.remove(viewer)
                    viewer.deleteLater()

            # 2. Add viewers for newly added gids
            insert_pos = max(0, self.comparison_list_layout.count() - 1)
            for gid in display_gids:
                if gid not in self._comparison_member_viewers_map:
                    viewer = self._create_member_response_row(gid)
                    self.comparison_list_layout.insertWidget(insert_pos, viewer)
                    self.comparison_viewers.append(viewer)
                    self._comparison_member_viewers_map[gid] = viewer
                    insert_pos += 1

            if not display_gids and self.comparison_list_layout.count() <= 1:
                lbl_empty = QLabel("No member units in this cluster.")
                lbl_empty.setObjectName("lbl_comp_empty_members")
                lbl_empty.setStyleSheet(CLUSTERS_COMPARISON_EMPTY_LABEL_STYLE)
                self.comparison_list_layout.insertWidget(0, lbl_empty)

        n_curated = sum(
            1 for g in member_gids
            if self._unit_status_map.get(g, self._pooled_units_map[g].get("status", "Unassigned")) in ("Curated", "Seeded")
        )
        self.lbl_comparison_title.setText(
            f"Comparing: {cname} (N={len(member_gids)} | {n_curated} Curated)"
        )
        self.lbl_comparison_title.setStyleSheet(build_comparison_title_style(color_hex))

        self.comparison_pane.setVisible(True)

        self._refresh_list_panel()
        self._update_inspector_member_highlights()
        self._hydrate_visible_comparison_members(hydrate_uncached=True)
        if display_gids:
            self._schedule_background_plot_prewarm(display_gids, prioritize_front=True)
        QTimer.singleShot(0, self._align_inspector_first_member_raster)

    def _update_inspector_header(
        self,
        cluster_id: int,
        cname: str,
        color_hex: str,
        member_gids: List[str],
    ) -> None:
        """Update the 46 px inspector header swatch, name, cluster ID, and latency summary."""
        if hasattr(self, "lbl_inspector_swatch"):
            self.lbl_inspector_swatch.setStyleSheet(build_inspector_swatch_style(color_hex))
        if hasattr(self, "lbl_inspector_name"):
            fm = QFontMetrics(self.lbl_inspector_name.font())
            elided = fm.elidedText(cname, Qt.TextElideMode.ElideRight, INSPECTOR_HEADER_NAME_MAX_WIDTH)
            self.lbl_inspector_name.setText(elided)
            self.lbl_inspector_name.setToolTip(cname)
        if hasattr(self, "lbl_inspector_cid"):
            self.lbl_inspector_cid.setText(f"C{cluster_id}")
        if hasattr(self, "lbl_inspector_summary"):
            valid_lats: List[float] = []
            for gid in member_gids:
                lat = self._pooled_units_map.get(gid, {}).get("latency_ms")
                if lat is not None and pd.notna(lat):
                    lat_f = float(lat)
                    if is_valid_metric_value(lat_f):
                        valid_lats.append(lat_f)
            if valid_lats:
                mean_lat = float(np.mean(valid_lats))
                self.lbl_inspector_summary.setText(
                    f"{len(member_gids)} members · mean latency {mean_lat:.1f} ms"
                )
            else:
                self.lbl_inspector_summary.setText(f"{len(member_gids)} members · mean latency —")

    def _align_inspector_first_member_raster(self) -> None:
        """Align the first inspector member's UnitResponsePlot horizontally with the focused unit's plot."""
        if not hasattr(self, "_focused_align_spacer") or not hasattr(self, "_inspector_align_spacer"):
            return
        if self._comparison_cluster_id is None or not self.comparison_viewers or not self.comparison_pane.isVisible():
            self._focused_align_spacer.setFixedHeight(0)
            self._inspector_align_spacer.setFixedHeight(0)
            return
        first_viewer = self.comparison_viewers[0]
        first_plot = getattr(first_viewer, "plot_slot", None) or getattr(first_viewer, "unit_response_plot", None)
        if first_plot is None or not hasattr(self, "unit_response_plot") or not hasattr(self, "centre_panel"):
            return
        if hasattr(self, "comparison_scroll_area") and self.comparison_scroll_area.verticalScrollBar().value() > 0:
            return
        focused_base_y = self.unit_response_plot.mapTo(self.centre_panel, QPoint(0, 0)).y() - self._focused_align_spacer.height()
        member_base_y = first_plot.mapTo(self.centre_panel, QPoint(0, 0)).y() - self._inspector_align_spacer.height()
        diff = member_base_y - focused_base_y
        if diff > 0:
            self._inspector_align_spacer.setFixedHeight(0)
            self._focused_align_spacer.setFixedHeight(diff)
        elif diff < 0:
            self._focused_align_spacer.setFixedHeight(0)
            self._inspector_align_spacer.setFixedHeight(-diff)
        else:
            self._focused_align_spacer.setFixedHeight(0)
            self._inspector_align_spacer.setFixedHeight(0)

    def _update_inspector_member_highlights(self) -> None:
        """Highlight the inspector member card matching the currently focused unit."""
        if not self._comparison_member_viewers_map:
            return
        focused = self._current_selected_gid
        for gid, viewer in self._comparison_member_viewers_map.items():
            is_active = bool(focused and gid == focused)
            if hasattr(viewer, "set_active"):
                viewer.set_active(is_active)
            else:
                viewer.setStyleSheet(
                    UNIT_RESPONSE_MEMBER_CARD_ACTIVE_STYLE if is_active else UNIT_RESPONSE_MEMBER_CARD_STYLE
                )

    def _on_inspector_member_clicked(self, gid: str) -> None:
        """Focus the clicked inspector member and record 'inspector' as the focus source (§ 10.1, § 10.2)."""
        self.select_pooled_unit(gid, source="inspector")

    def eventFilter(self, watched: Any, event: Any) -> bool:
        """Handle scroll viewport resize reflow and keyboard shortcuts across list/inspector widgets."""
        if hasattr(self, "clusters_scroll_area") and watched == self.clusters_scroll_area.viewport():
            if event.type() == QEvent.Type.Resize:
                new_cols = self._compute_card_grid_columns()
                if new_cols != getattr(self, "_current_grid_cols", None) and self._cluster_card_order:
                    self._relayout_cluster_cards()
                return False
        if hasattr(self, "comparison_scroll_area") and watched == self.comparison_scroll_area.viewport():
            if event.type() == QEvent.Type.Resize:
                self._hydrate_visible_comparison_members(hydrate_uncached=True)
                return False
        if event.type() == QEvent.Type.KeyPress and watched in (
            getattr(self, "table_pooled_units", None),
            getattr(self, "table_list_panel", None),
            getattr(self, "comparison_scroll_area", None),
        ):
            if QApplication.activePopupWidget() is None:
                is_text_input = self._is_text_input_widget(QApplication.focusWidget())
                if not is_text_input and self._handle_curation_key_event(event):
                    return True
        return super().eventFilter(watched, event)

    def clear_cluster_comparison(self, return_to_suggestion: bool = True) -> None:
        """Close side-by-side comparison pane, reset card borders, and clear cluster distribution overlay."""
        was_open = self._comparison_cluster_id is not None
        self._comparison_cluster_id = None
        self._comparison_rendered_cid = None
        self._comparison_member_viewers_map.clear()
        for card in self._cluster_cards.values():
            card.set_comparison_active(False)

        while self.comparison_list_layout.count():
            item = self.comparison_list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.comparison_viewers.clear()

        if hasattr(self, "marginal_widget"):
            self.marginal_widget.clear_cluster_distribution()

        if hasattr(self, "cluster_prob_bar"):
            self.cluster_prob_bar.set_inspected_cluster(None)

        if hasattr(self, "_focused_align_spacer"):
            self._focused_align_spacer.setFixedHeight(0)
        if hasattr(self, "_inspector_align_spacer"):
            self._inspector_align_spacer.setFixedHeight(0)

        self.comparison_pane.setVisible(False)
        if hasattr(self, "inspector_empty_state"):
            self.inspector_empty_state.setVisible(True)
        self._refresh_embedding_and_probe_emphasis()
        self._refresh_list_panel()

        if was_open and return_to_suggestion and self.active_learning_enabled:
            self.return_to_active_candidate()

    def _on_assign_to_comparison_cluster(self) -> None:
        """Assign the currently selected cell(s) / active candidate to the compared cluster."""
        if self._comparison_cluster_id is None:
            return
        cid = self._comparison_cluster_id

        target_gids: List[str] = []
        if hasattr(self, "table_pooled_units") and self.table_pooled_units.selectionModel() is not None:
            selected_model_rows = self.table_pooled_units.selectionModel().selectedRows()
            for model_idx in selected_model_rows:
                r = model_idx.row()
                gid_val = self._get_gid_for_row(r)
                if gid_val and gid_val in self._pooled_units_map:
                    target_gids.append(gid_val)

        if not target_gids:
            gid = self._current_selected_gid
            if gid is None and self.active_candidate is not None:
                gid = self.active_candidate.global_unit_id
            if gid is not None and gid in self._pooled_units_map:
                target_gids.append(gid)

        if target_gids:
            self.assign_candidate_to_cluster(cid, unit_id=target_gids)
            if not self.active_learning_enabled:
                self._advance_to_next_table_row(assigned_gid=target_gids[-1])

    def _on_cluster_merge_requested(self, src_id: int, tgt_id: int) -> None:
        """Callback when a cluster card is dropped onto another card to trigger merge confirmation."""
        self.merge_clusters(src_id, tgt_id, confirm=True)

    def _on_cluster_move_requested(self, src_id: int, tgt_id: int) -> None:
        """Callback when a cluster card is dropped onto another card to trigger reordering."""
        self.move_cluster_card(src_id, tgt_id)

    def prune_empty_clusters(self) -> List[int]:
        """Identify and remove any clusters that have 0 members across all pooled units.

        Returns:
            List of removed empty cluster IDs.
        """
        counts: Dict[int, int] = {}
        for rec in self._pooled_units_map.values():
            cid = rec.get("cluster_id", -1)
            if cid >= 0:
                counts[cid] = counts.get(cid, 0) + 1

        known_cids = set(self._cluster_card_order)
        known_cids.update(self._cluster_cards.keys())
        known_cids.update(self._cluster_names.keys())
        known_cids.update(self._cluster_colors.keys())
        if self._comparison_cluster_id is not None:
            known_cids.add(self._comparison_cluster_id)
        if self.current_clustering_result is not None:
            known_cids.update(self.current_clustering_result.cluster_summaries.keys())

        removed_cids = [cid for cid in known_cids if counts.get(cid, 0) == 0]
        if not removed_cids:
            return []

        # Close comparison pane if comparing one of the removed clusters
        if self._comparison_cluster_id is not None and self._comparison_cluster_id in removed_cids:
            self.clear_cluster_comparison(return_to_suggestion=False)

        # Remove from card order
        self._cluster_card_order = [cid for cid in self._cluster_card_order if cid not in removed_cids]

        # Remove card widgets and cleanup metadata
        for cid in removed_cids:
            card = self._cluster_cards.pop(cid, None)
            if card is not None:
                self.clusters_grid.removeWidget(card)
                card.deleteLater()
            self._cluster_names.pop(cid, None)
            self._cluster_colors.pop(cid, None)
            self._dismissed_cluster_candidates.pop(cid, None)
            if self.current_clustering_result is not None:
                self.current_clustering_result.cluster_summaries.pop(cid, None)

        self._relayout_cluster_cards()
        self._populate_cluster_filter_options()
        return removed_cids

    def move_cluster_card(self, src_id: int, tgt_id: int) -> bool:
        """Swap cluster card positions of src_id and tgt_id in the layout grid.

        Args:
            src_id: Cluster ID of the card being moved.
            tgt_id: Cluster ID of the card at the target drop position.

        Returns:
            True if moved successfully, False otherwise.
        """
        if src_id == tgt_id:
            return False
        if src_id not in self._cluster_card_order or tgt_id not in self._cluster_card_order:
            return False

        self._push_undo_snapshot()
        idx_src = self._cluster_card_order.index(src_id)
        idx_tgt = self._cluster_card_order.index(tgt_id)
        self._cluster_card_order[idx_src], self._cluster_card_order[idx_tgt] = (
            self._cluster_card_order[idx_tgt],
            self._cluster_card_order[idx_src],
        )

        self._relayout_cluster_cards()
        self.cluster_moved.emit(src_id, tgt_id)
        return True

    def merge_clusters(self, src_id: int, tgt_id: int, confirm: bool = True) -> bool:
        """Merge source cluster into target cluster with optional confirmation prompt.

        Args:
            src_id: Cluster ID of the source cluster to be merged and removed.
            tgt_id: Cluster ID of the target cluster receiving the units.
            confirm: If True, presents confirmation QMessageBox dialog.

        Returns:
            True if merged successfully, False if cancelled or invalid.
        """
        if self.current_clustering_result is None:
            return False

        summaries = self.current_clustering_result.cluster_summaries
        if src_id not in summaries or tgt_id not in summaries or src_id == tgt_id:
            return False

        n_src = summaries[src_id].count
        n_tgt = summaries[tgt_id].count
        name_src = self._cluster_names.get(src_id, f"Cluster {src_id}")
        name_tgt = self._cluster_names.get(tgt_id, f"Cluster {tgt_id}")

        if confirm:
            reply = QMessageBox.question(
                self,
                "Confirm Cluster Merge",
                f"Merge {name_src} (N={n_src}) -> {name_tgt} (N={n_tgt})?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False

        self._push_undo_snapshot()

        # 1. Reassign units in pooled records
        for record in self._pooled_rows:
            if record.get("cluster_id") == src_id:
                record["cluster_id"] = tgt_id

        for record in self._pooled_units_map.values():
            if record.get("cluster_id") == src_id:
                record["cluster_id"] = tgt_id

        # 2. Reassign units in cluster_labels array
        labels = self.current_clustering_result.cluster_labels
        labels[labels == src_id] = tgt_id

        # 3. Recompute cluster summaries (grand-average PSTH and SEM envelopes)
        df_active = self._get_active_pooled_dataframe()
        spike_map = self._get_cached_spike_data_map(df_active)
        psth_matrix, t_bins = self._get_cached_psth_matrix(df_active, spike_map)
        new_summaries = compute_cluster_summaries(
            df_active, labels, psth_matrix, t_bins, norm_mode=self.cluster_psth_norm
        )
        for c_id, summ in new_summaries.items():
            summ.color_hex = self._get_cluster_color(c_id)
        self.current_clustering_result.cluster_summaries = new_summaries

        # 4. Re-render cluster cards and 2D PCA scatter points
        self._render_cluster_cards(self.current_clustering_result)
        self._render_pca_scatter(self.current_clustering_result)

        # Update cluster column items across table
        self._update_all_pooled_cluster_items()

        # 5. Trigger active learning retrain and advance candidate
        self.refresh_active_learning()

        # Update cluster filter options and visibility
        self._populate_cluster_filter_options()

        self.cluster_merged.emit(src_id, tgt_id)
        return True

    def _optimistic_next_queue_candidate(self) -> Optional[ActiveLearningCandidate]:
        """Select the next eligible queue candidate from cached rankings while worker retrains."""
        cur = getattr(self, "active_candidate", None)
        cur_gid = cur.global_unit_id if cur is not None else None
        skipped = getattr(self, "_skipped_gids", set())

        has_unassigned = any(
            int(rec.get("cluster_id", -1)) < 0
            for rec in self._pooled_units_map.values()
        )

        def _is_eligible(gid: str) -> bool:
            if gid not in self._pooled_units_map:
                return False
            rec = self._pooled_units_map[gid]
            status = self._unit_status_map.get(gid, rec.get("status", "Unassigned"))
            if status in ("Curated", "Auto-assigned"):
                return False
            if has_unassigned and int(rec.get("cluster_id", -1)) >= 0:
                return False
            return True

        # Keep current candidate if still eligible and unskipped
        if cur_gid is not None and _is_eligible(cur_gid) and cur_gid not in skipped:
            return cur

        ranked = getattr(self, "ranked_candidates", []) or []
        for cand in ranked:
            if _is_eligible(cand.global_unit_id) and cand.global_unit_id not in skipped:
                return cand
        for cand in ranked:
            if cand.global_unit_id != cur_gid and _is_eligible(cand.global_unit_id):
                return cand
        for cand in ranked:
            if _is_eligible(cand.global_unit_id):
                return cand

        # If a unit was newly unassigned (not yet in stale ranked_candidates), focus it instead of drifting
        if has_unassigned:
            for gid, rec in self._pooled_units_map.items():
                if int(rec.get("cluster_id", -1)) < 0 and _is_eligible(gid) and gid not in skipped:
                    uid_val = rec.get("unit_id", 0)
                    try:
                        uid_int = int(uid_val)
                    except (TypeError, ValueError):
                        uid_int = 0
                    return ActiveLearningCandidate(
                        global_unit_id=str(gid),
                        entropy=0.0,
                        probabilities=dict(self._all_unit_probabilities.get(gid, {})),
                        feature_values={},
                        cluster_id=-1,
                        session_id=str(rec.get("session_id", "")),
                        unit_id=uid_int,
                    )
        return None

    def _invalidate_active_learning_worker(self) -> None:
        """Invalidate any in-flight ActiveLearningWorker so stale results cannot be applied."""
        self._worker_request_id = getattr(self, "_worker_request_id", 0) + 1
        worker = getattr(self, "_active_learning_worker", None)
        if worker is not None and worker.isRunning():
            try:
                worker.results_ready.disconnect()
            except Exception:
                pass

    def refresh_active_learning(self, synchronous: Optional[bool] = None) -> Optional[ActiveLearningCandidate]:
        """Evaluate Maximum Shannon Entropy on unassigned/boundary units and trigger auto-assignment."""
        if (
            self.current_clustering_result is not None
            and getattr(self.current_clustering_result, "unit_ids", None) is not None
            and len(self.current_clustering_result.unit_ids) == len(self._pooled_units_map)
        ):
            ordered_gids = [str(uid) for uid in self.current_clustering_result.unit_ids]
        else:
            ordered_gids = list(self._pooled_units_map.keys())

        pooled_records = []
        for gid in ordered_gids:
            rec = self._pooled_units_map.get(gid, {})
            r_copy = dict(rec)
            status_val = self._unit_status_map.get(gid, rec.get("status", "Unassigned"))
            r_copy["status"] = status_val
            r_copy["curated_status"] = status_val
            r_copy["cluster_id"] = rec.get("cluster_id", -1)
            pooled_records.append(r_copy)

        df = pd.DataFrame(pooled_records)
        if df.empty or len(df) == 0:
            self._invalidate_active_learning_worker()
            self.ranked_candidates = []
            self.active_candidate = None
            if len(self.loaded_experiments) == 0 and not self._is_mock:
                self.lbl_candidate.setText("No Experiment Selected")
            else:
                self.lbl_candidate.setText("No Units in Pool")
            self.cand_scatter.clear()
            for card in self._cluster_cards.values():
                card.set_candidate_probability(None)
            if hasattr(self, "cluster_prob_bar"):
                self.cluster_prob_bar.clear()
            self._refresh_list_panel()
            self._update_classification_progress()
            return None

        labels = (
            self.current_clustering_result.cluster_labels
            if self.current_clustering_result is not None
            else np.array([r.get("cluster_id", -1) for r in pooled_records], dtype=np.int32)
        )

        is_sync = bool(synchronous)

        feat_mat = (
            self.current_clustering_result.feature_matrix
            if (
                self.current_clustering_result is not None
                and getattr(self.current_clustering_result, "feature_matrix", None) is not None
                and len(self.current_clustering_result.feature_matrix) == len(df)
            )
            else None
        )

        if is_sync:
            spike_map = self._get_cached_spike_data_map(df)
            ranked, auto_assignments, confidences, all_probs, suggested_assignments = run_active_learning_cycle(
                df=df,
                cluster_labels=labels,
                unit_status_map=self._unit_status_map,
                threshold=self.auto_assign_threshold,
                max_distance_sigma=float(getattr(self, "al_max_distance_sigma", 3.5)),
                feature_config=self.feature_config,
                feature_matrix=feat_mat,
                skipped_gids=self._skipped_gids,
                n_estimators=max(10, int(getattr(self, "al_n_estimators", 100))),
                random_state=int(getattr(self, "al_random_state", 42)),
                spike_data_map=spike_map,
                psth_cache=self._unit_psth_cache,
                auto_assign_enabled=bool(getattr(self, "auto_assign_enabled", False)),
                pulse_duration_ms=self._get_pulse_duration_ms(),
            )
            self._apply_active_learning_results(ranked, auto_assignments, confidences, all_probs, suggested_assignments)
            return self.active_candidate
        else:
            # Run via asynchronous background worker QThread with cooperative request versioning
            self._invalidate_active_learning_worker()
            current_req_id = self._worker_request_id

            spike_map = self._get_cached_spike_data_map(df)
            # Parentless QThread retained until finished so C++ widget destruction never kills a live thread
            worker = ActiveLearningWorker(
                df=df.copy(),
                cluster_labels=np.array(labels, copy=True),
                unit_status_map=dict(self._unit_status_map),
                threshold=self.auto_assign_threshold,
                max_distance_sigma=float(getattr(self, "al_max_distance_sigma", 3.5)),
                feature_config=self.feature_config,
                feature_matrix=feat_mat.copy() if feat_mat is not None else None,
                skipped_gids=set(self._skipped_gids),
                random_state=int(getattr(self, "al_random_state", 42)),
                spike_data_map=dict(spike_map) if spike_map is not None else None,
                psth_cache=dict(self._unit_psth_cache),
                auto_assign_enabled=getattr(self, "auto_assign_enabled", False),
                n_estimators=max(10, int(getattr(self, "al_n_estimators", 100))),
                pulse_duration_ms=self._get_pulse_duration_ms(),
                parent=None,
            )

            self_ref = weakref.ref(self)

            def _on_worker_results(
                cands: List[ActiveLearningCandidate],
                autos: Dict[str, int],
                confs: Dict[str, float],
                all_probs: Optional[Dict[str, Dict[int, float]]] = None,
                suggested: Optional[Dict[str, int]] = None,
            ) -> None:
                owner = self_ref()
                if owner is None or getattr(owner, "_is_closing", False):
                    return
                try:
                    owner.isVisible()
                except RuntimeError:
                    return
                if getattr(owner, "_worker_request_id", 0) == current_req_id:
                    prev_active_gid = (
                        owner.active_candidate.global_unit_id if owner.active_candidate is not None else None
                    )
                    owner._apply_active_learning_results(cands, autos, confs, all_probs, suggested)
                    if (
                        owner.active_learning_enabled
                        and getattr(owner, "_focus_source", None) == "queue"
                        and not getattr(owner, "_is_applying_undo_redo", False)
                        and owner.active_candidate is not None
                        and (current_req_id, owner.active_candidate.global_unit_id)
                        != getattr(owner, "_skip_guard", None)
                        and (
                            owner.active_candidate.global_unit_id != prev_active_gid
                            or owner._current_selected_gid != owner.active_candidate.global_unit_id
                        )
                    ):
                        owner.select_pooled_unit(owner.active_candidate.global_unit_id, source="queue")

            def _cleanup_worker(w: ActiveLearningWorker = worker) -> None:
                _ACTIVE_AL_WORKERS.discard(w)
                owner = self_ref()
                if owner is not None:
                    owner._tracked_al_workers = [
                        tw for tw in getattr(owner, "_tracked_al_workers", []) if tw is not w
                    ]

            worker.results_ready.connect(_on_worker_results)
            worker.finished.connect(_cleanup_worker)
            _ACTIVE_AL_WORKERS.add(worker)
            self.active_candidate = self._optimistic_next_queue_candidate()
            self._tracked_al_workers = [
                tw for tw in getattr(self, "_tracked_al_workers", []) if tw.isRunning()
            ]
            self._tracked_al_workers.append(worker)
            self._active_learning_worker = worker
            worker.start()
            return self.active_candidate

    def _apply_active_learning_results(
        self,
        candidates: List[ActiveLearningCandidate],
        auto_assignments: Dict[str, int],
        confidences: Dict[str, float],
        all_probabilities: Optional[Dict[str, Dict[int, float]]] = None,
        suggested_assignments: Optional[Dict[str, int]] = None,
    ) -> None:
        """Apply active learning candidate queue and background auto-assignments."""
        self.ranked_candidates = candidates
        self._auto_confidences.update(confidences)
        if all_probabilities:
            self._all_unit_probabilities.update(all_probabilities)
        for cand in candidates:
            self._unit_candidates_map[cand.global_unit_id] = cand
            if cand.probabilities:
                self._all_unit_probabilities[cand.global_unit_id] = cand.probabilities

        if suggested_assignments is not None:
            self._suggested_auto_assignments = dict(suggested_assignments)
        elif getattr(self, "auto_assign_enabled", False):
            self._suggested_auto_assignments = {}

        # 1. Update auto-assigned units
        changed_gids: Set[str] = set()
        for gid, pred_cid in auto_assignments.items():
            if gid not in self._pooled_units_map:
                continue
            cur_status = self._unit_status_map.get(gid, "Unassigned")
            if cur_status in ("Curated", "Seeded"):
                continue  # Curated and Seeded units are not re-evaluated by the classifier!

            old_cid = self._pooled_units_map[gid].get("cluster_id", -1)
            if pred_cid >= 0:
                if old_cid != pred_cid or cur_status != "Auto-assigned":
                    self._pooled_units_map[gid]["cluster_id"] = pred_cid
                    self._unit_status_map[gid] = "Auto-assigned"
                    changed_gids.add(gid)
            else:
                if cur_status == "Auto-assigned":
                    self._pooled_units_map[gid]["cluster_id"] = -1
                    self._unit_status_map[gid] = "Unassigned"
                    changed_gids.add(gid)

        if changed_gids:
            for r in self._pooled_rows:
                gid = r.get("global_unit_id")
                if gid in changed_gids:
                    r["cluster_id"] = self._pooled_units_map[gid]["cluster_id"]
            self._update_all_pooled_cluster_items(target_gids=changed_gids)
            self._populate_cluster_filter_options()
            self._sync_clustering_result_and_cards()
            if self._comparison_cluster_id is not None:
                self.set_cluster_comparison(self._comparison_cluster_id)
            if self._current_selected_gid:
                self._update_probe_view_for_unit(self._current_selected_gid)
        elif self._comparison_cluster_id is not None and self._comparison_cluster_id in self._cluster_cards:
            self.set_cluster_comparison(self._comparison_cluster_id)

        # 2. Check if all units are assigned
        unassigned_count = sum(
            1 for rec in self._pooled_units_map.values()
            if rec.get("cluster_id", -1) < 0
        )
        if unassigned_count == 0 and len(self._pooled_units_map) > 0:
            curated_n = sum(1 for s in self._unit_status_map.values() if s == "Curated")
            auto_n = sum(1 for s in self._unit_status_map.values() if s == "Auto-assigned")
            seeded_n = sum(1 for s in self._unit_status_map.values() if s == "Seeded")
            if candidates:
                top_cand = candidates[0]
                self.active_candidate = top_cand
                self._display_candidate(top_cand)
                self.candidate_updated.emit(top_cand)
                sess_str, uid_str = _format_unit_display(top_cand.session_id, top_cand.unit_id, top_cand.global_unit_id)
                cid = top_cand.cluster_id
                if cid < 0 and top_cand.global_unit_id in self._pooled_units_map:
                    cid = self._pooled_units_map[top_cand.global_unit_id].get("cluster_id", -1)
                cname = self._cluster_names.get(cid, f"Cluster {cid}") if cid >= 0 else "Unassigned"
                self.lbl_candidate.setText(
                    f"🎉 All Units Classified! ({curated_n} Curated, {auto_n} Auto, {seeded_n} Seeded) • Session: {sess_str}  •  Unit: {uid_str}  •  Cluster: {cname}  (Entropy: H={top_cand.entropy:.2f})"
                )
            else:
                self.lbl_candidate.setText(
                    f"🎉 All Units Classified! ({curated_n} Curated, {auto_n} Auto-assigned, {seeded_n} Seeded)"
                )
                self.cand_scatter.clear()
                self.active_candidate = None
                for card in self._cluster_cards.values():
                    card.set_candidate_probability(None)
                if hasattr(self, "cluster_prob_bar"):
                    self.cluster_prob_bar.clear()
            self._update_classification_progress()
            self._refresh_list_panel()
            self._schedule_background_plot_prewarm()
            return

        # 3. Surface top candidate
        if candidates:
            top_cand = candidates[0]
            self.active_candidate = top_cand
            if self.active_learning_enabled:
                if self._focus_source == "queue" and not self._is_applying_undo_redo:
                    self._display_candidate(top_cand)
                else:
                    self._update_return_to_candidate_button()
                self.candidate_updated.emit(top_cand)
            else:
                self._update_return_to_candidate_button()
        else:
            self.active_candidate = None
            self.lbl_candidate.setText("Active Learning: All candidates curated or skipped.")
            self.cand_scatter.clear()
            for card in self._cluster_cards.values():
                card.set_candidate_probability(None)
            if hasattr(self, "cluster_prob_bar"):
                self.cluster_prob_bar.clear()
            self._update_return_to_candidate_button()

        self._update_classification_progress()
        self._refresh_list_panel()
        self._schedule_background_plot_prewarm()

    def assign_candidate_to_cluster(self, cluster_id: int, unit_id: Union[str, List[str], None] = None) -> bool:
        """Assign candidate or selected unit(s) to specified cluster and advance active learning loop.

        Args:
            cluster_id: Target cluster ID to assign unit(s) to (-1 for Unassigned).
            unit_id: Optional unit global ID or list of IDs to assign. If None, prioritizes all currently
                     selected units in the table, falling back to the active uncertainty candidate.

        Returns:
            True if at least one unit was successfully assigned and loop advanced, False otherwise.
        """
        target_gids: List[str] = []
        if isinstance(unit_id, list):
            for uid in unit_id:
                uid_str = str(uid)
                if uid_str in self._pooled_units_map:
                    target_gids.append(uid_str)
                else:
                    for k, rec in self._pooled_units_map.items():
                        if k == uid_str or str(rec.get("unit_id")) == uid_str:
                            target_gids.append(k)
                            break
        elif unit_id is not None:
            uid_str = str(unit_id)
            if uid_str in self._pooled_units_map:
                target_gids.append(uid_str)
            else:
                for k, rec in self._pooled_units_map.items():
                    if k == uid_str or str(rec.get("unit_id")) == uid_str:
                        target_gids.append(k)
                        break
        else:
            target_gids = self._resolve_selected_target_gids()

        # Deduplicate while preserving order
        seen = set()
        unique_gids = []
        for g in target_gids:
            if g not in seen and g in self._pooled_units_map:
                seen.add(g)
                unique_gids.append(g)

        if not unique_gids:
            return False

        if not self._is_applying_undo_redo:
            self._undo_stack.append(self._capture_curation_snapshot())
            self._redo_stack.clear()
            if hasattr(self, "btn_undo"):
                self.btn_undo.setEnabled(True)
            if hasattr(self, "btn_redo"):
                self.btn_redo.setEnabled(False)

        # Capture pre-action position in current focus source for § 10.2 Advance rule
        advance_ctx = self._capture_advance_context(unique_gids[-1])

        if cluster_id >= 0 and cluster_id not in self._cluster_card_order:
            self._cluster_card_order.append(cluster_id)

        cur_status = "Curated" if cluster_id >= 0 else "Unassigned"

        # 1. Update unit cluster assignment in internal records
        for gid in unique_gids:
            self._pooled_units_map[gid]["cluster_id"] = cluster_id
            self._pooled_units_map[gid]["status"] = cur_status
            self._pooled_units_map[gid]["curated_status"] = cur_status
            self._unit_status_map[gid] = cur_status
            for r in self._pooled_rows:
                if r.get("global_unit_id") == gid:
                    r["cluster_id"] = cluster_id
                    r["status"] = cur_status
                    r["curated_status"] = cur_status

            # Update loaded experiment if present
            sess = str(self._pooled_units_map[gid].get("session_id", ""))
            if sess in self.loaded_experiments:
                meta = self.loaded_experiments[sess]
                if meta.df is not None and not meta.df.empty:
                    if "cluster_id" not in meta.df.columns:
                        meta.df["cluster_id"] = -1
                    if "global_unit_id" in meta.df.columns:
                        meta.df.loc[meta.df["global_unit_id"] == gid, "cluster_id"] = cluster_id
                    elif "unit_id" in meta.df.columns and "unit_id" in self._pooled_units_map[gid]:
                        meta.df.loc[meta.df["unit_id"] == self._pooled_units_map[gid]["unit_id"], "cluster_id"] = cluster_id

            # 3. Mark candidate as curated by skipping it from future candidate queries
            if cluster_id >= 0:
                self._skipped_gids.add(gid)
            else:
                self._skipped_gids.discard(gid)

        # 2. Update cluster_labels in clustering result and synchronize summaries
        self._sync_clustering_result_and_cards()

        # Update cluster column items across table
        self._update_all_pooled_cluster_items(target_gids=set(unique_gids))

        # Update cluster filter options and refresh visibility
        self._populate_cluster_filter_options()

        # 4. Retrain Random Forest and advance to next highest-entropy candidate (when Active Learning is enabled)
        if self.active_learning_enabled:
            self.refresh_active_learning()
        else:
            self._update_classification_progress()
            self._update_focused_unit_header(self._current_selected_gid)
            self._update_return_to_candidate_button()

        # 5. If comparison pane is open, refresh member list
        if self._comparison_cluster_id is not None:
            if self._comparison_cluster_id in self._cluster_cards:
                self.set_cluster_comparison(self._comparison_cluster_id)
            else:
                self.clear_cluster_comparison(return_to_suggestion=False)

        # 6. WP7 § 10.2 Advance rule: continue in the list the focused unit came from
        if not self._is_applying_undo_redo:
            self._advance_after_action(advance_ctx, is_skip=False)

        self._schedule_background_plot_prewarm(unique_gids)
        self._preserve_cleared_selection = False

        return True

    def _capture_advance_context(self, anchor_gid: str) -> Dict[str, Any]:
        """Capture the current focus source and index before an assign/unassign/skip mutation."""
        source = getattr(self, "_focus_source", "queue")
        prev_idx = 0
        prev_table_row = -1

        if hasattr(self, "table_pooled_units"):
            for r in range(self.table_pooled_units.rowCount()):
                if self._get_gid_for_row(r) == anchor_gid:
                    prev_table_row = r
                    break

        if source == "inspector":
            gids = list(self._comparison_member_viewers_map.keys())
            if anchor_gid in gids:
                prev_idx = gids.index(anchor_gid)
        elif source == "list_candidates" and hasattr(self, "table_list_panel"):
            gids = []
            for r in range(self.table_list_panel.rowCount()):
                item = self.table_list_panel.item(r, 0)
                if item is not None and item.data(Qt.ItemDataRole.UserRole):
                    gids.append(str(item.data(Qt.ItemDataRole.UserRole)))
            if anchor_gid in gids:
                prev_idx = gids.index(anchor_gid)

        return {
            "source": source,
            "anchor_gid": anchor_gid,
            "prev_idx": prev_idx,
            "prev_table_row": prev_table_row,
        }

    def _advance_after_action(self, ctx: Dict[str, Any], is_skip: bool = False) -> None:
        """Advance focus according to WP7 § 10.2 ('continue in the list the unit came from')."""
        if getattr(self, "_preserve_cleared_selection", False):
            return
        source = ctx.get("source", "queue")
        anchor_gid = ctx.get("anchor_gid")
        prev_idx = int(ctx.get("prev_idx", 0))
        prev_table_row = int(ctx.get("prev_table_row", -1))

        if source == "inspector" and self._comparison_cluster_id is not None:
            remaining = list(self._comparison_member_viewers_map.keys())
            if remaining:
                next_idx = (prev_idx + 1) % len(remaining) if is_skip else min(prev_idx, len(remaining) - 1)
                self.select_pooled_unit(remaining[next_idx], source="inspector")
            return

        if source == "table_cluster" and hasattr(self, "table_pooled_units"):
            total_rows = self.table_pooled_units.rowCount()
            start_row = (prev_table_row + 1) if is_skip else max(0, prev_table_row)
            if prev_table_row >= 0 and self._get_gid_for_row(prev_table_row) == anchor_gid and not self.table_pooled_units.isRowHidden(prev_table_row):
                start_row = prev_table_row + 1

            next_row = -1
            for r in range(start_row, total_rows):
                if not self.table_pooled_units.isRowHidden(r):
                    next_row = r
                    break
            if next_row < 0:
                for r in range(min(prev_table_row - 1, total_rows - 1), -1, -1):
                    if not self.table_pooled_units.isRowHidden(r):
                        next_row = r
                        break
            if next_row >= 0:
                next_gid = self._get_gid_for_row(next_row)
                if next_gid:
                    self.select_pooled_unit(next_gid, source="table_cluster")
            else:
                self.table_pooled_units.blockSignals(True)
                self.table_pooled_units.clearSelection()
                self.table_pooled_units.setCurrentCell(-1, -1)
                self.table_pooled_units.blockSignals(False)
                self._current_selected_gid = None
            return

        if source == "list_candidates" and hasattr(self, "table_list_panel"):
            new_gids: List[str] = []
            for r in range(self.table_list_panel.rowCount()):
                item = self.table_list_panel.item(r, 0)
                if item is not None and item.data(Qt.ItemDataRole.UserRole):
                    new_gids.append(str(item.data(Qt.ItemDataRole.UserRole)))
            if new_gids:
                next_idx = (prev_idx + 1) % len(new_gids) if is_skip else min(prev_idx, len(new_gids) - 1)
                self.select_pooled_unit(new_gids[next_idx], source="list_candidates")
            return

        # Default: "queue"
        if self.active_learning_enabled:
            if self.active_candidate is not None:
                self.select_pooled_unit(self.active_candidate.global_unit_id, source="queue")
        elif hasattr(self, "table_pooled_units"):
            total_rows = self.table_pooled_units.rowCount()
            if total_rows == 0:
                return
            search_order = list(range(prev_table_row + 1, total_rows)) + list(range(0, max(0, prev_table_row)))
            for r in search_order:
                if self.table_pooled_units.isRowHidden(r):
                    continue
                gid = self._get_gid_for_row(r)
                if gid and self._pooled_units_map.get(gid, {}).get("cluster_id", -1) < 0:
                    self.select_pooled_unit(gid, source="queue")
                    return
            for r in search_order:
                if not self.table_pooled_units.isRowHidden(r):
                    gid = self._get_gid_for_row(r)
                    if gid:
                        self.select_pooled_unit(gid, source="queue")
                        return

    def skip_current_candidate(self) -> Optional[ActiveLearningCandidate]:
        """Skip current focused/candidate unit and advance in the current focus source (§ 10.2)."""
        target_gid = self._current_selected_gid or (
            self.active_candidate.global_unit_id if self.active_candidate is not None else None
        )
        if target_gid is None:
            return None

        if not self._is_applying_undo_redo:
            self._undo_stack.append(self._capture_curation_snapshot())
            self._redo_stack.clear()
            if hasattr(self, "btn_undo"):
                self.btn_undo.setEnabled(True)
            if hasattr(self, "btn_redo"):
                self.btn_redo.setEnabled(False)

        advance_ctx = self._capture_advance_context(target_gid)
        self._skipped_gids.add(target_gid)

        next_cand: Optional[ActiveLearningCandidate] = None
        if advance_ctx["source"] == "queue" and self.active_learning_enabled:
            self._skip_guard = (getattr(self, "_worker_request_id", 0) + 1, target_gid)
            next_cand = self.refresh_active_learning()
            if next_cand is not None:
                self.select_pooled_unit(next_cand.global_unit_id, source="queue")
        else:
            self._advance_after_action(advance_ctx, is_skip=True)
        return next_cand

    def _capture_curation_snapshot(self) -> Dict[str, Any]:
        """Capture unit assignments, skipped set, cluster metadata, and focused unit for Undo/Redo."""
        assignments = {
            gid: (
                int(rec.get("cluster_id", -1)) if rec.get("cluster_id") is not None else -1,
                self._unit_status_map.get(gid, rec.get("status", "Unassigned")),
            )
            for gid, rec in self._pooled_units_map.items()
        }
        focused_gid = self._current_selected_gid or (
            self.active_candidate.global_unit_id if self.active_candidate is not None else None
        )
        return {
            "assignments": assignments,
            "skipped_gids": set(self._skipped_gids),
            "cluster_names": dict(self._cluster_names),
            "cluster_colors": dict(self._cluster_colors),
            "cluster_hotkeys": dict(self._cluster_hotkeys),
            "cluster_card_order": list(self._cluster_card_order),
            "cluster_sort_mode": getattr(self, "_cluster_sort_mode", "id"),
            "focused_gid": focused_gid,
            "focus_source": getattr(self, "_focus_source", "queue"),
            "comparison_cluster_id": self._comparison_cluster_id,
        }

    def _restore_curation_snapshot(self, snap: Dict[str, Any]) -> None:
        """Restore a captured curation snapshot and refresh UI without pushing a new undo entry."""
        self._is_applying_undo_redo = True
        try:
            assignments: Dict[str, Tuple[int, str]] = snap["assignments"]
            for gid, (cid, status) in assignments.items():
                if gid in self._pooled_units_map:
                    self._pooled_units_map[gid]["cluster_id"] = cid
                    self._pooled_units_map[gid]["status"] = status
                    self._pooled_units_map[gid]["curated_status"] = status
                    self._unit_status_map[gid] = status

                    sess = str(self._pooled_units_map[gid].get("session_id", ""))
                    if sess in self.loaded_experiments:
                        meta = self.loaded_experiments[sess]
                        if meta.df is not None and not meta.df.empty:
                            if "cluster_id" not in meta.df.columns:
                                meta.df["cluster_id"] = -1
                            if "global_unit_id" in meta.df.columns:
                                meta.df.loc[meta.df["global_unit_id"] == gid, "cluster_id"] = cid
                            elif "unit_id" in meta.df.columns and "unit_id" in self._pooled_units_map[gid]:
                                meta.df.loc[
                                    meta.df["unit_id"] == self._pooled_units_map[gid]["unit_id"],
                                    "cluster_id",
                                ] = cid

            for r in self._pooled_rows:
                gid = r.get("global_unit_id")
                if gid in assignments:
                    cid, status = assignments[gid]
                    r["cluster_id"] = cid
                    r["status"] = status
                    r["curated_status"] = status

            self._skipped_gids = set(snap["skipped_gids"])
            self._cluster_names = dict(snap["cluster_names"])
            self._cluster_colors.clear()
            self._cluster_colors.update(snap["cluster_colors"])
            if hasattr(self, "cluster_color_service"):
                self.cluster_color_service.set_colors(dict(snap["cluster_colors"]))
            self._cluster_hotkeys = dict(snap["cluster_hotkeys"])
            self._cluster_card_order = list(snap["cluster_card_order"])
            self._cluster_sort_mode = snap.get("cluster_sort_mode", getattr(self, "_cluster_sort_mode", "id"))
            for sort_key, sort_act in getattr(self, "_sort_actions", {}).items():
                sort_act.blockSignals(True)
                sort_act.setChecked(sort_key == self._cluster_sort_mode)
                sort_act.blockSignals(False)

            self._sync_clustering_result_and_cards()
            self._update_all_pooled_cluster_items()
            self._populate_cluster_filter_options()

            if self.active_learning_enabled:
                self.refresh_active_learning()
            else:
                self._update_classification_progress()

            restored_source = snap.get("focus_source", "queue")
            self._focus_source = restored_source

            focused_gid = snap.get("focused_gid")
            if focused_gid and focused_gid in self._pooled_units_map:
                self.select_pooled_unit(focused_gid, source=restored_source)

            comp_cid = snap.get("comparison_cluster_id")
            if comp_cid is not None and comp_cid in self._cluster_cards:
                self.set_cluster_comparison(comp_cid)
            elif self._comparison_cluster_id is not None:
                self.clear_cluster_comparison(return_to_suggestion=False)
        finally:
            self._is_applying_undo_redo = False
            if hasattr(self, "btn_undo"):
                self.btn_undo.setEnabled(bool(self._undo_stack))
            if hasattr(self, "btn_redo"):
                self.btn_redo.setEnabled(bool(self._redo_stack))

    def undo(self) -> bool:
        """Undo the most recent curation action and restore unit assignments and focus."""
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._capture_curation_snapshot())
        snap = self._undo_stack.pop()
        self._restore_curation_snapshot(snap)
        self._set_dirty(True)
        return True

    def redo(self) -> bool:
        """Redo the most recently undone curation action."""
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._capture_curation_snapshot())
        snap = self._redo_stack.pop()
        self._restore_curation_snapshot(snap)
        self._set_dirty(True)
        return True

    def unassign_focused_unit(self) -> bool:
        """Unassign the currently focused unit (sets cluster_id=-1 and status='Unassigned')."""
        gid = self._current_selected_gid or (
            self.active_candidate.global_unit_id if self.active_candidate is not None else None
        )
        if not gid or gid not in self._pooled_units_map:
            return False
        rec = self._pooled_units_map[gid]
        cid = int(rec.get("cluster_id", -1)) if rec.get("cluster_id") is not None else -1
        if cid < 0:
            return False
        return self.assign_candidate_to_cluster(-1, unit_id=gid)

    def _handle_curation_key_event(self, event: Any) -> bool:
        """Process WP7 § 10.3 keyboard shortcuts; return True if handled."""
        key = event.key()
        modifiers = event.modifiers()

        # Esc: Close open inspector (and return focus to active-learning suggestion)
        if key == Qt.Key.Key_Escape and modifiers == Qt.KeyboardModifier.NoModifier:
            if self._comparison_cluster_id is not None:
                self.clear_cluster_comparison(return_to_suggestion=True)
                event.accept()
                return True
            return False

        # Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y: Undo / Redo
        if key == Qt.Key.Key_Z and (modifiers & Qt.KeyboardModifier.ControlModifier):
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                self.redo()
            else:
                self.undo()
            event.accept()
            return True

        if key == Qt.Key.Key_Y and (modifiers & Qt.KeyboardModifier.ControlModifier):
            self.redo()
            event.accept()
            return True

        # Check configured cluster hotkeys first (e.g. "1"-"9", "0", "Q", "W", "E", "R", "T")
        combo = event.keyCombination() if hasattr(event, "keyCombination") else None
        if combo is not None:
            seq_str = QKeySequence(combo).toString(QKeySequence.SequenceFormat.PortableText)
        else:
            seq_str = ""
        if not seq_str and modifiers == Qt.KeyboardModifier.NoModifier:
            seq_str = QKeySequence(key).toString(QKeySequence.SequenceFormat.PortableText)
        pressed_hk = normalize_hotkey(seq_str)

        if pressed_hk and self.current_clustering_result is not None:
            target_cid = None
            active_cids = list(self._cluster_card_order)
            for c in self.current_clustering_result.cluster_summaries.keys():
                if c not in active_cids:
                    active_cids.append(c)

            for cid in active_cids:
                if normalize_hotkey(self._get_cluster_hotkey(cid)) == pressed_hk:
                    target_cid = cid
                    break

            if target_cid is not None:
                self._on_cluster_hotkey_assign(target_cid)
                event.accept()
                return True

        # N: Create new cluster for candidate (N only; '+' is not a shortcut per § 10.3)
        if key == Qt.Key.Key_N and modifiers == Qt.KeyboardModifier.NoModifier:
            self.create_new_cluster_for_candidate()
            event.accept()
            return True

        # Space or S: Skip current unit - only without modifiers
        if key in (Qt.Key.Key_Space, Qt.Key.Key_S) and modifiers == Qt.KeyboardModifier.NoModifier:
            self.skip_current_candidate()
            event.accept()
            return True

        # Backspace: Unassign focused unit
        if key == Qt.Key.Key_Backspace and modifiers == Qt.KeyboardModifier.NoModifier:
            self.unassign_focused_unit()
            event.accept()
            return True

        # G: Return to suggested unit (only when Active Learning is enabled, § 11.1)
        if key == Qt.Key.Key_G and modifiers == Qt.KeyboardModifier.NoModifier:
            if self.active_learning_enabled:
                self.return_to_active_candidate()
                event.accept()
                return True
            return False

        # 0-9: Direct numeric fallback cluster assignment
        if Qt.Key.Key_0 <= key <= Qt.Key.Key_9 and modifiers == Qt.KeyboardModifier.NoModifier:
            if self.current_clustering_result is not None:
                self._on_cluster_hotkey(key - Qt.Key.Key_0)
                event.accept()
                return True

        return False

    def keyPressEvent(self, event: Any) -> None:
        """Handle keyboard shortcuts (§ 10.3)."""
        if self._handle_curation_key_event(event):
            return
        super().keyPressEvent(event)

    def _resolve_experiment_probe_and_unit_positions(
        self,
        session_id: str,
        folder_path: Path,
        df: pd.DataFrame,
        npz_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Resolve probe geometry and coordinates for all units in an experiment."""
        probe_positions: Optional[np.ndarray] = None

        # 1. Check npz_data if channel_positions exists
        if npz_data is not None and "channel_positions" in npz_data:
            try:
                arr = np.asarray(npz_data["channel_positions"], dtype=np.float64)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    probe_positions = arr
            except Exception:
                pass

        # 2. Check folder_path or parent for channel_positions.npy
        if probe_positions is None:
            pos_file = folder_path / "channel_positions.npy"
            if not pos_file.exists() and folder_path.parent != folder_path:
                parent_pos = folder_path.parent / "channel_positions.npy"
                if parent_pos.exists():
                    pos_file = parent_pos
            if pos_file.exists():
                try:
                    arr = np.load(pos_file)
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        probe_positions = arr.astype(np.float64)
                except Exception:
                    pass

        # 3. Fallback default probe (Neuropixels-like staggered 2-column, 64 sites)
        if probe_positions is None:
            n_sites = 64
            x_coords = np.array([16.0 if i % 2 == 0 else 48.0 for i in range(n_sites)], dtype=np.float64)
            y_coords = np.array([100.0 + (i // 2) * 20.0 for i in range(n_sites)], dtype=np.float64)
            probe_positions = np.column_stack((x_coords, y_coords))

        self._experiment_probes[session_id] = probe_positions

        # Resolve unit coordinates in df
        if df is not None and not df.empty:
            if "channel" not in df.columns:
                df["channel"] = 0
            if "depth" not in df.columns:
                df["depth"] = 0.0
            if "probe_x" not in df.columns:
                df["probe_x"] = 0.0

            # If depth or probe_x are 0 / NaN, map from probe_positions using channel
            for idx, row in df.iterrows():
                try:
                    ch = int(row.get("channel", 0)) if pd.notna(row.get("channel")) else 0
                except (ValueError, TypeError):
                    ch = 0
                d = float(row.get("depth", 0.0)) if pd.notna(row.get("depth")) else 0.0
                px = float(row.get("probe_x", 0.0)) if pd.notna(row.get("probe_x")) else 0.0

                if (d == 0.0 and px == 0.0) and 0 <= ch < len(probe_positions):
                    px = float(probe_positions[ch, 0])
                    d = float(probe_positions[ch, 1])

                df.at[idx, "channel"] = ch
                df.at[idx, "depth"] = d
                df.at[idx, "probe_x"] = px

        return probe_positions, df

    def _detect_shanks(self, sites: np.ndarray) -> List[Dict[str, float]]:
        """Detect individual shanks from electrode site positions by clustering x coordinates.

        Clusters site x positions with an inter-shank threshold of 80.0 µm. Single-shank probes
        with staggered columns (e.g. Neuropixels with x in [16, 48], gap 32 µm) cluster as 1 shank,
        while multi-shank arrays (e.g. spacing 250 µm) cleanly separate into distinct shanks.
        """
        if sites is None or len(sites) == 0:
            return []

        x_vals = sites[:, 0]
        unique_x = np.sort(np.unique(x_vals))
        if len(unique_x) == 0:
            return []

        # 1D clustering on unique x coordinates
        shank_x_groups: List[List[float]] = [[float(unique_x[0])]]
        for x in unique_x[1:]:
            if float(x) - shank_x_groups[-1][-1] > 80.0:
                shank_x_groups.append([float(x)])
            else:
                shank_x_groups[-1].append(float(x))

        shanks: List[Dict[str, float]] = []
        for idx, group in enumerate(shank_x_groups):
            mask = np.isin(x_vals, group)
            shank_sites = sites[mask]
            if len(shank_sites) == 0:
                continue

            x_min = float(np.min(shank_sites[:, 0]))
            x_max = float(np.max(shank_sites[:, 0]))
            y_min = float(np.min(shank_sites[:, 1]))
            y_max = float(np.max(shank_sites[:, 1]))
            center_x = (x_min + x_max) / 2.0
            # Width with padding, at least 30 µm
            width = max(30.0, (x_max - x_min) + 30.0)

            shanks.append({
                "shank_index": idx,
                "x_min": x_min,
                "x_max": x_max,
                "center_x": center_x,
                "y_min": y_min,
                "y_max": y_max,
                "width": width,
            })

        shanks.sort(key=lambda s: s["center_x"])
        for idx, s in enumerate(shanks):
            s["shank_index"] = idx

        return shanks

    def _init_probe_view(self) -> None:
        """Initialize probe view items with electrode sites, unit dots, and candidate marker."""
        for item in self._probe_shank_items + self._probe_shank_labels:
            try:
                self.probe_plot.removeItem(item)
            except Exception:
                pass
        self._probe_shank_items = []
        self._probe_shank_labels = []

        self.probe_sites_scatter = self.probe_widget.sites_scatter
        self.probe_units_scatter = self.probe_widget.units_scatter
        self.probe_scatter = self.probe_units_scatter
        try:
            self.probe_units_scatter.sigClicked.disconnect(self._on_probe_unit_clicked)
        except Exception:
            pass
        self.probe_units_scatter.sigClicked.connect(self._on_probe_unit_clicked)
        self.probe_cand_line = self.probe_widget.cand_line
        self.probe_cand_marker = self.probe_widget.cand_marker

    def _on_probe_unit_clicked(self, plot: Any, points: Any) -> None:
        """Interactive point selection callback: clicking a probe unit point selects the corresponding unit."""
        if points is None or len(points) == 0:
            return

        # If multiple overlapping points were clicked, cycle through them if current is selected
        curr_gid = getattr(self, "_current_selected_gid", None)
        target_gid = None
        if curr_gid and len(points) > 1:
            for idx, pt in enumerate(points):
                if pt.data() and str(pt.data()) == curr_gid:
                    target_gid = points[(idx + 1) % len(points)].data()
                    break
        if not target_gid:
            target_gid = points[0].data()

        if target_gid:
            self.select_pooled_unit(str(target_gid), source="queue")

    def _update_probe_marker(self, depth: float, x: float = 0.0, session_id: Optional[str] = None) -> None:
        """Update silicon probe depth marker to candidate's channel/depth."""
        if hasattr(self, "probe_cand_marker") and hasattr(self, "probe_cand_line"):
            self.probe_cand_marker.setData(x=[x], y=[depth])
            self.probe_cand_line.setValue(depth)

    def _update_probe_view_for_unit(self, gid: Optional[str]) -> None:
        """Update probe view to show the unit's session probe, session units, and marker."""
        if not gid or gid not in self._pooled_units_map:
            return

        rec = self._pooled_units_map[gid]
        sess_id = str(rec.get("session_id", ""))
        depth = float(rec.get("depth", 0.0))
        probe_x = float(rec.get("probe_x", 0.0))
        ch = int(rec.get("channel", 0)) if pd.notna(rec.get("channel")) else 0

        # Check if probe needs switching or initial load
        if sess_id not in self._experiment_probes:
            # Fallback probe
            n_sites = 64
            x_coords = np.array([16.0 if i % 2 == 0 else 48.0 for i in range(n_sites)], dtype=np.float64)
            y_coords = np.array([100.0 + (i // 2) * 20.0 for i in range(n_sites)], dtype=np.float64)
            self._experiment_probes[sess_id] = np.column_stack((x_coords, y_coords))

        sites = self._experiment_probes[sess_id]
        if sess_id not in self._probe_shanks_cache:
            self._probe_shanks_cache[sess_id] = self._detect_shanks(sites)
        shanks = self._probe_shanks_cache[sess_id]

        # Determine active shank
        shank_text = ""
        if shanks:
            active_shank = min(shanks, key=lambda s: abs(s["center_x"] - probe_x))
            s_idx = active_shank["shank_index"] + 1
            shank_text = f"Shank {s_idx} / {len(shanks)}"

        # Update info labels
        if hasattr(self, "lbl_probe_experiment"):
            self.lbl_probe_experiment.setText(f"Exp: {sess_id}")
        if hasattr(self, "lbl_probe_unit_info"):
            if shank_text:
                self.lbl_probe_unit_info.setText(f"{shank_text}  |  Ch {ch}  |  Depth: {depth:.0f} μm  |  X: {probe_x:.0f} μm")
            else:
                self.lbl_probe_unit_info.setText(f"Ch {ch}  |  Depth: {depth:.0f} μm  |  X: {probe_x:.0f} μm")

        self._current_probe_session_id = sess_id

        # Gather all pooled units belonging to this session
        session_units = [
            (u_gid, u_rec)
            for u_gid, u_rec in self._pooled_units_map.items()
            if str(u_rec.get("session_id", "")) == sess_id
        ]

        open_cid = self._comparison_cluster_id

        probe_units: List[Dict[str, Any]] = []
        bg_depths: List[float] = []
        cluster_depths: Dict[int, List[float]] = {}

        for u_gid, u_rec in session_units:
            ux = float(u_rec.get("probe_x", 0.0))
            uy = float(u_rec.get("depth", 0.0))
            ucid = int(u_rec.get("cluster_id", -1))
            bg_depths.append(uy)
            if ucid >= 0:
                cluster_depths.setdefault(ucid, []).append(uy)

            if open_cid is not None and open_cid >= 0:
                if ucid == open_cid:
                    color = self._get_cluster_color(ucid)
                    alpha = 255
                    priority = 1
                elif ucid >= 0:
                    color = self._get_cluster_color(ucid)
                    alpha = PROBE_DIM_ALPHA
                    priority = 0
                else:
                    color = COLOR_UNASSIGNED
                    alpha = PROBE_DIM_ALPHA
                    priority = 0
            else:
                if ucid >= 0:
                    color = self._get_cluster_color(ucid)
                    alpha = 255
                    priority = 1
                else:
                    color = COLOR_UNASSIGNED
                    alpha = 255
                    priority = 0

            cname = self._cluster_names.get(ucid, f"Cluster {ucid}") if ucid >= 0 else "Unassigned"
            u_num = u_rec.get("unit_id", "")
            probe_units.append({
                "id": u_gid,
                "probe_x": ux,
                "depth": uy,
                "color": color,
                "alpha": alpha,
                "priority": priority,
                "tooltip": f"u{int(u_num):04d} ({cname}) · Depth {uy:.0f} µm" if str(u_num).isdigit() else f"{u_gid} ({cname}) · Depth {uy:.0f} µm",
            })

        if open_cid is not None and open_cid >= 0:
            fg_kdes_arg: Optional[List[Tuple[List[float], str]]] = None
            fg_depths_arg: List[float] = cluster_depths.get(int(open_cid), [])
            fg_color_arg: str = self._get_cluster_color(int(open_cid))
        else:
            card_order = [int(c) for c in getattr(self, "_cluster_card_order", []) if int(c) in cluster_depths]
            remaining = [c for c in sorted(cluster_depths.keys()) if c not in card_order]
            ordered_cids = card_order + remaining
            fg_kdes_arg = [
                (cluster_depths[c], self._get_cluster_color(c))
                for c in ordered_cids
            ]
            fg_depths_arg = []
            fg_color_arg = COLOR_ACCENT

        self.probe_widget.set_probe_data(
            channel_positions=sites,
            units=probe_units,
            focused_id=gid,
            bg_kde_depths=bg_depths,
            fg_kde_depths=fg_depths_arg,
            fg_kde_color=fg_color_arg,
            fg_kdes=fg_kdes_arg,
        )
        self._probe_shank_items = list(self.probe_widget._shank_rect_items)

    def _update_marginal_distributions(self, feature_values: Dict[str, float]) -> None:
        """Update vertical indicator lines on population feature curves."""
        if hasattr(self, "marginal_widget"):
            self.marginal_widget.set_candidate_values(feature_values)

    def _get_suggested_counts_per_cluster(self) -> Dict[int, int]:
        """Compute count of putative suggested units per cluster ID."""
        counts: Dict[int, int] = {}
        threshold = getattr(self, "auto_assign_threshold", 0.85)
        if hasattr(self, "_all_unit_probabilities") and self._all_unit_probabilities:
            for gid, cluster_probs in self._all_unit_probabilities.items():
                if self._unit_status_map.get(gid, "Unassigned") in ("Curated", "Seeded", "Auto-assigned"):
                    continue
                if not cluster_probs:
                    continue
                best_cid_raw = max(cluster_probs, key=cluster_probs.get)
                best_prob = cluster_probs[best_cid_raw]
                try:
                    best_cid = int(best_cid_raw)
                except (ValueError, TypeError):
                    best_cid = -1
                if best_cid >= 0 and best_prob >= threshold:
                    dismissed = getattr(self, "_dismissed_cluster_candidates", {}).get(best_cid, set())
                    if gid not in dismissed:
                        counts[best_cid] = counts.get(best_cid, 0) + 1
            return counts

        # Fallback
        for gid, cid in getattr(self, "_suggested_auto_assignments", {}).items():
            if cid >= 0 and self._unit_status_map.get(gid, "Unassigned") not in ("Curated", "Seeded", "Auto-assigned"):
                dismissed = getattr(self, "_dismissed_cluster_candidates", {}).get(cid, set())
                if gid in dismissed:
                    continue
                counts[cid] = counts.get(cid, 0) + 1
        return counts

    def _update_all_cluster_card_suggested_badges(self) -> None:
        """Update warning badges on all cluster cards reflecting putative suggested cell counts."""
        suggested_counts = self._get_suggested_counts_per_cluster()
        threshold = getattr(self, "auto_assign_threshold", 0.85)
        for cid, card in self._cluster_cards.items():
            count = suggested_counts.get(cid, 0)
            card.set_suggested_count(count, threshold=threshold)

    def _update_classification_progress(self) -> None:
        """Calculate and update multi-segment classification progress bar."""
        if not hasattr(self, "progress_bar"):
            return
        curated_n = sum(1 for s in self._unit_status_map.values() if s == "Curated")
        auto_n = sum(1 for s in self._unit_status_map.values() if s == "Auto-assigned")
        seeded_n = sum(1 for s in self._unit_status_map.values() if s == "Seeded")
        total_pooled = len(self._pooled_units_map)
        unassigned_n = max(0, total_pooled - (curated_n + auto_n + seeded_n))

        suggested_counts = self._get_suggested_counts_per_cluster()
        suggested_n = sum(suggested_counts.values()) if not getattr(self, "auto_assign_enabled", False) else 0
        threshold = getattr(self, "auto_assign_threshold", 0.85)
        self.progress_bar.update_counts(
            curated=curated_n,
            auto_assigned=auto_n,
            seeded=seeded_n,
            unassigned=unassigned_n,
            suggested=suggested_n,
            threshold=threshold,
        )
        self.progress_updated.emit(curated_n, seeded_n, auto_n, suggested_n, total_pooled)
        self._update_all_cluster_card_suggested_badges()

    def _display_candidate(self, cand: ActiveLearningCandidate) -> None:
        """Update scatter indicator, probe marker, marginal distributions, probability bars, and preview for candidate."""
        sess_str, uid_str = _format_unit_display(cand.session_id, cand.unit_id, cand.global_unit_id)
        cid = cand.cluster_id
        if cid < 0 and cand.global_unit_id in self._pooled_units_map:
            cid = self._pooled_units_map[cand.global_unit_id].get("cluster_id", -1)
        cluster_str = self._cluster_names.get(cid, f"Cluster {cid}") if cid >= 0 else "Unassigned"
        self.lbl_candidate.setText(
            f"Session: {sess_str}  •  Unit: {uid_str}  •  Cluster: {cluster_str}  (Entropy: H={cand.entropy:.2f})"
        )
        self._update_focused_unit_header(cand.global_unit_id)

        # 1. Red ring on 2D scatter canvas
        if cand.global_unit_id in self._scatter_points_map:
            px, py, _ = self._scatter_points_map[cand.global_unit_id]
            self.cand_scatter.setData(x=[px], y=[py])
        elif self.current_clustering_result is not None:
            for i, u_id in enumerate(self.current_clustering_result.unit_ids):
                if u_id == cand.global_unit_id:
                    px = float(self.current_clustering_result.pca_coordinates[i, 0])
                    py = float(self.current_clustering_result.pca_coordinates[i, 1])
                    self.cand_scatter.setData(x=[px], y=[py])
                    break

        # 2. Silicon probe depth marker & dynamic probe view
        self._update_probe_view_for_unit(cand.global_unit_id)

        # 3. Feature marginal distributions
        self._update_marginal_distributions(cand.feature_values)

        # 4. Update Cluster Card probability badges and highlights
        probs = cand.probabilities or {}
        top_prob = max(probs.values()) if probs else 0.0
        for cid, card in self._cluster_cards.items():
            c_prob = probs.get(cid)
            # Mark is_top if probability is close to top or high
            is_top = False
            if c_prob is not None and top_prob > 0:
                if c_prob >= top_prob - 0.15 or c_prob >= 0.25:
                    is_top = True
            card.set_candidate_probability(c_prob, is_top=is_top)

        # 5. Update Cluster Candidate Probabilities Bar Chart
        if hasattr(self, "cluster_prob_bar"):
            self.cluster_prob_bar.set_probabilities(
                probabilities=probs,
                cluster_colors=self._cluster_colors,
                cluster_names=self._cluster_names,
                threshold=self.auto_assign_threshold,
                cluster_order=sorted(self._cluster_card_order),
            )

        # 6. Preview image or metadata in ImageViewer
        self._display_candidate_preview(cand.global_unit_id)

        # 7. Highlight in pooled units table
        self._highlight_candidate_table_row(cand.global_unit_id)

        # 8. Candidate PSTH waveform overlay across cluster cards
        self._update_cluster_cards_candidate_psth(cand.global_unit_id)

        # 9. Update Suggested Cell button state and inspector member highlight
        self._update_return_to_candidate_button()
        self._update_inspector_member_highlights()

    def _get_raster_and_psth_for_unit(self, gid: str) -> Optional[Dict[str, Any]]:
        """Retrieve or compute raster and PSTH histogram data for a unit."""
        if gid not in self._pooled_units_map:
            return None

        cache_key = (
            gid,
            (float(self.current_window_ms[0]), float(self.current_window_ms[1])),
            float(self.current_bin_size_ms),
            self.raster_order_by,
            bool(self.psth_smooth),
            float(self.psth_gaussian_sigma_ms),
        )
        psth_cache_key = (
            str(gid),
            (float(self.current_window_ms[0]), float(self.current_window_ms[1])),
            float(self.current_bin_size_ms),
            bool(self.psth_smooth),
            float(self.psth_gaussian_sigma_ms),
        )
        if cache_key in self._unit_raster_psth_cache:
            cached = self._unit_raster_psth_cache[cache_key]
            if psth_cache_key not in self._unit_psth_cache and cached.get("psth_rates") is not None:
                self._unit_psth_cache[psth_cache_key] = np.asarray(cached["psth_rates"], dtype=np.float64)
            return cached

        rec = self._pooled_units_map[gid]
        sess_id = str(rec.get("session_id", ""))
        unit_id = rec.get("unit_id")
        try:
            int_uid = int(unit_id) if unit_id is not None else None
        except (ValueError, TypeError):
            int_uid = None

        psth_data = None
        # Check if real spike times and pulse onsets exist for this session/unit
        if (
            sess_id in self._experiment_spikes
            and int_uid is not None
            and int_uid in self._experiment_spikes[sess_id]
            and sess_id in self._experiment_stimulus
        ):
            spikes = self._experiment_spikes[sess_id][int_uid]
            onsets = self._experiment_stimulus[sess_id]
            if len(spikes) > 0 and len(onsets) > 0:
                psth_data = compute_psth_and_raster(
                    spikes,
                    onsets,
                    window_ms=self.current_window_ms,
                    bin_size_ms=self.current_bin_size_ms,
                    order_by=self.raster_order_by,
                    smooth=self.psth_smooth,
                    gaussian_sigma_ms=self.psth_gaussian_sigma_ms,
                )

        if psth_data is not None:
            self._unit_raster_psth_cache[cache_key] = psth_data
            psth_rates = psth_data.get("psth_rates")
            if psth_rates is not None:
                self._unit_psth_cache[psth_cache_key] = psth_rates
        return psth_data

    def shutdown_background_workers(self) -> None:
        """Cancel pending background prewarm and active learning work cleanly before teardown."""
        self._is_closing = True
        self._reset_background_prewarm(clear_pixmaps=False)
        workers = list(getattr(self, "_tracked_al_workers", [])) + list(
            self.findChildren(ActiveLearningWorker)
        )
        seen_ids: Set[int] = set()
        for worker in workers:
            if id(worker) in seen_ids:
                continue
            seen_ids.add(id(worker))
            if worker.isRunning():
                try:
                    worker.results_ready.disconnect()
                except Exception:
                    pass
                worker.wait(5000)
            _ACTIVE_AL_WORKERS.discard(worker)

    def closeEvent(self, event: Any) -> None:
        """Shut down background workers cleanly before closing."""
        self.shutdown_background_workers()
        super().closeEvent(event)

    def _member_render_key(self, gid: str) -> Tuple[Any, ...]:
        """Return cache key covering all visual settings affecting member UnitResponsePlot rendering."""
        sess_id = str(self._pooled_units_map.get(gid, {}).get("session_id", ""))
        return (
            str(gid),
            (float(self.current_window_ms[0]), float(self.current_window_ms[1])),
            float(self.current_bin_size_ms),
            str(self.raster_order_by),
            float(self.raster_marker_size),
            bool(self.psth_smooth),
            float(self.psth_gaussian_sigma_ms),
            float(self._get_pulse_duration_ms(sess_id or None)),
        )

    def _get_or_render_member_pixmap(self, gid: str) -> Optional[QPixmap]:
        """Render or fetch cached HiDPI QPixmap using a single shared offscreen UnitResponsePlot."""
        if not gid or gid not in self._pooled_units_map:
            return None
        key = self._member_render_key(gid)
        cached = self._member_pixmap_cache.get(key)
        if cached is not None:
            return cached

        psth_data = self._get_raster_and_psth_for_unit(gid)
        if self._member_snapshot_plot is None:
            self._member_snapshot_plot = UnitResponsePlot(self, defer_initial_render=True)
            self._member_snapshot_plot.setVisible(False)
            self._member_snapshot_plot.ensurePolished()
        snap = self._member_snapshot_plot
        sess_id = str(self._pooled_units_map.get(gid, {}).get("session_id", ""))
        snap.set_psth_data(
            psth_data=psth_data,
            window_ms=self.current_window_ms,
            order_by=self.raster_order_by,
            marker_size=self.raster_marker_size,
            pulse_duration_ms=self._get_pulse_duration_ms(sess_id or None),
        )
        if snap.layout() is not None:
            snap.layout().activate()
        pix = snap.grab()
        if not pix.isNull():
            self._member_pixmap_cache[key] = pix
            return pix
        return None

    def _hydrate_visible_comparison_members(self, hydrate_uncached: bool = True) -> None:
        """Hydrate only Inspector member cards currently intersecting the scroll viewport (+buffer)."""
        viewers = getattr(self, "comparison_viewers", None)
        if not viewers or not hasattr(self, "comparison_scroll_area"):
            return
        list_widget = getattr(self, "comparison_list_widget", None)
        if list_widget is not None and list_widget.layout() is not None:
            list_widget.layout().activate()
            list_widget.adjustSize()

        vp = self.comparison_scroll_area.viewport()
        vp_h = vp.height() if vp is not None else 0
        scroll_y = self.comparison_scroll_area.verticalScrollBar().value()

        # Before Qt positions all rows vertically, only the first card sits at y=0 while subsequent
        # cards also report (y=0, h=0). Wait for distinct row positions before spatial culling.
        has_layout = vp_h > 20 and (
            len(viewers) == 1
            or (len(viewers) > 1 and viewers[1].y() > 0 and viewers[1].height() > 0)
        )
        if not has_layout:
            for viewer in viewers[:4]:
                if hasattr(viewer, "hydrate_if_cached") and hasattr(viewer, "hydrate"):
                    if not viewer.hydrate_if_cached() and hydrate_uncached:
                        viewer.hydrate()
            if self.comparison_pane.isVisible() and len(viewers) > 4:
                QTimer.singleShot(0, self._hydrate_visible_comparison_members)
            return

        y_min = scroll_y - 240
        y_max = scroll_y + vp_h + 240
        needs_deferred = False
        for viewer in viewers:
            h = viewer.height()
            if h <= 0:
                continue
            top = viewer.y()
            bot = top + h
            if bot < y_min or top > y_max:
                continue
            if hasattr(viewer, "hydrate_if_cached") and hasattr(viewer, "hydrate"):
                if not viewer.hydrate_if_cached():
                    if hydrate_uncached:
                        viewer.hydrate()
                    else:
                        needs_deferred = True

        if needs_deferred and not hydrate_uncached:
            self._scroll_hydrate_timer.start(16)

    def _on_inspector_scrolled(self, _value: int) -> None:
        """Immediately apply cached pixmaps on scroll; defer uncached renders until brief pause."""
        self._hydrate_visible_comparison_members(hydrate_uncached=False)

    def _on_scroll_hydrate_timeout(self) -> None:
        """Render any uncached visible member cards once scroll velocity pauses."""
        self._hydrate_visible_comparison_members(hydrate_uncached=True)

    def _reset_background_prewarm(self, clear_pixmaps: bool = False) -> None:
        """Invalidate pending prewarm queues/workers and optionally clear rendered pixmaps."""
        self._prewarm_generation += 1
        self._prewarm_queue.clear()
        self._prewarm_data_queue.clear()
        self._prewarm_worker_active = False
        if hasattr(self, "_prewarm_timer") and self._prewarm_timer.isActive():
            self._prewarm_timer.stop()
        if hasattr(self, "_scroll_hydrate_timer") and self._scroll_hydrate_timer.isActive():
            self._scroll_hydrate_timer.stop()
        if clear_pixmaps:
            self._member_pixmap_cache.clear()

    def _kick_prewarm_data_worker(self) -> None:
        """Dispatch next pure-NumPy compute_psth_and_raster batch to QThreadPool."""
        if self._prewarm_worker_active or not self._prewarm_data_queue:
            return

        batch_items: List[Tuple[Any, ...]] = []
        win_ms = (float(self.current_window_ms[0]), float(self.current_window_ms[1]))
        bin_ms = float(self.current_bin_size_ms)
        order_by = str(self.raster_order_by)
        smooth = bool(self.psth_smooth)
        sigma_ms = float(self.psth_gaussian_sigma_ms)

        while self._prewarm_data_queue and len(batch_items) < 8:
            gid = self._prewarm_data_queue.pop(0)
            rec = self._pooled_units_map.get(gid)
            if rec is None:
                continue
            cache_key = (gid, win_ms, bin_ms, order_by, smooth, sigma_ms)
            if cache_key in self._unit_raster_psth_cache:
                continue
            sess_id = str(rec.get("session_id", ""))
            unit_id = rec.get("unit_id")
            try:
                int_uid = int(unit_id) if unit_id is not None else None
            except (ValueError, TypeError):
                int_uid = None
            if (
                sess_id in self._experiment_spikes
                and int_uid is not None
                and int_uid in self._experiment_spikes[sess_id]
                and sess_id in self._experiment_stimulus
            ):
                spikes = self._experiment_spikes[sess_id][int_uid]
                onsets = self._experiment_stimulus[sess_id]
                if len(spikes) > 0 and len(onsets) > 0:
                    psth_key = (str(gid), win_ms, bin_ms, smooth, sigma_ms)
                    batch_items.append((cache_key, psth_key, spikes, onsets, win_ms, bin_ms, order_by, smooth, sigma_ms))

        if not batch_items:
            if self._prewarm_data_queue:
                self._kick_prewarm_data_worker()
            return

        self._prewarm_worker_active = True
        runnable = _PrewarmBatchRunnable(self._prewarm_generation, batch_items, self._prewarm_signals)
        QThreadPool.globalInstance().start(runnable)

    def _continue_prewarm_queues(self) -> None:
        """Resume any queued background raster/PSTH or pixmap prewarm work if not closing."""
        if getattr(self, "_is_closing", False):
            return
        if self._prewarm_data_queue:
            self._kick_prewarm_data_worker()
        if self._prewarm_queue and not self._prewarm_timer.isActive():
            self._prewarm_timer.start(10)

    def _on_prewarm_batch_ready(self, generation: int, results: List[Tuple[Any, Any, Dict[str, Any]]]) -> None:
        """Store worker-computed raster/PSTH results on the GUI thread and continue warming."""
        self._prewarm_worker_active = False
        if generation != self._prewarm_generation:
            self._continue_prewarm_queues()
            return
        for cache_key, psth_key, psth_data in results:
            if cache_key not in self._unit_raster_psth_cache:
                self._unit_raster_psth_cache[cache_key] = psth_data
            psth_rates = psth_data.get("psth_rates")
            if psth_rates is not None and psth_key not in self._unit_psth_cache:
                self._unit_psth_cache[psth_key] = psth_rates
            gid = str(cache_key[0])
            if self._member_render_key(gid) not in self._member_pixmap_cache and gid not in self._prewarm_queue:
                self._prewarm_queue.append(gid)
        self._continue_prewarm_queues()

    def _schedule_background_plot_prewarm(
        self,
        gids: Optional[List[str]] = None,
        prioritize_front: bool = False,
    ) -> None:
        """Enqueue background raster/PSTH computation and idle HiDPI member pixmap warming."""
        if gids is None:
            target_gids = [str(r.get("global_unit_id")) for r in self._pooled_rows if r.get("global_unit_id")]
        else:
            target_gids = [str(g) for g in gids if str(g).strip()]

        win_ms = (float(self.current_window_ms[0]), float(self.current_window_ms[1]))
        bin_ms = float(self.current_bin_size_ms)
        order_by = str(self.raster_order_by)
        smooth = bool(self.psth_smooth)
        sigma_ms = float(self.psth_gaussian_sigma_ms)

        data_to_add: List[str] = []
        pix_to_add: List[str] = []
        existing_data = set(self._prewarm_data_queue)
        existing_pix = set(self._prewarm_queue)

        for gid in target_gids:
            cache_key = (gid, win_ms, bin_ms, order_by, smooth, sigma_ms)
            if cache_key not in self._unit_raster_psth_cache:
                if gid not in existing_data:
                    data_to_add.append(gid)
                    existing_data.add(gid)
            elif self._member_render_key(gid) not in self._member_pixmap_cache:
                if gid not in existing_pix:
                    pix_to_add.append(gid)
                    existing_pix.add(gid)

        if prioritize_front:
            if data_to_add:
                self._prewarm_data_queue = data_to_add + [g for g in self._prewarm_data_queue if g not in data_to_add]
            if pix_to_add:
                self._prewarm_queue = pix_to_add + [g for g in self._prewarm_queue if g not in pix_to_add]
        else:
            self._prewarm_data_queue.extend(data_to_add)
            self._prewarm_queue.extend(pix_to_add)

        if self._prewarm_data_queue:
            self._kick_prewarm_data_worker()
        if self._prewarm_queue and not self._prewarm_timer.isActive():
            self._prewarm_timer.start(10)

    def _process_next_prewarm_unit(self) -> None:
        """Render one idle-slice HiDPI member pixmap without blocking GUI interaction."""
        if not self._prewarm_queue:
            return

        gid = self._prewarm_queue.pop(0)
        self._get_or_render_member_pixmap(gid)

        if self._prewarm_queue:
            self._prewarm_timer.start(10)

    def _get_pulse_duration_ms(self, session_id: Optional[str] = None) -> float:
        """Return inferred stimulation pulse duration in ms for a session (or across loaded sessions)."""
        if session_id:
            if hasattr(self, "_experiment_pulse_durations") and session_id in self._experiment_pulse_durations:
                return float(self._experiment_pulse_durations[session_id])
            meta = self.loaded_experiments.get(session_id)
            if meta is not None:
                if hasattr(meta, "pulse_duration_ms") and getattr(meta, "pulse_duration_ms", None) is not None:
                    return float(meta.pulse_duration_ms)
                if meta.df is not None and not meta.df.empty:
                    for col in ("pulse_duration_ms", "pulse_duration", "stim_duration_ms"):
                        if col in meta.df.columns and pd.notna(meta.df[col].iloc[0]):
                            return float(meta.df[col].iloc[0])

        durs: List[float] = []
        if hasattr(self, "_experiment_pulse_durations") and self._experiment_pulse_durations:
            durs.extend(float(v) for v in self._experiment_pulse_durations.values() if v > 0)
        for meta in self.loaded_experiments.values():
            if hasattr(meta, "pulse_duration_ms") and getattr(meta, "pulse_duration_ms", None) is not None:
                durs.append(float(meta.pulse_duration_ms))
            elif meta.df is not None and not meta.df.empty:
                for col in ("pulse_duration_ms", "pulse_duration", "stim_duration_ms"):
                    if col in meta.df.columns and pd.notna(meta.df[col].iloc[0]):
                        durs.append(float(meta.df[col].iloc[0]))
                        break
        if durs:
            return round(float(np.median(durs)), 3)
        return DEFAULT_PULSE_DURATION_MS

    def _create_member_response_row(self, gid: str) -> QFrame:
        """Create a lightweight Inspector member row (hydrated lazily for visible viewport)."""
        rec = self._pooled_units_map.get(gid, {})
        sess_str, uid_str = _format_unit_display(
            rec.get("session_id", ""),
            rec.get("unit_id"),
            gid,
        )
        lat = rec.get("latency_ms")
        rel = rec.get("reliability")
        lat_str = f"{float(lat):.1f} ms" if lat is not None and pd.notna(lat) and float(lat) < 500 else "—"
        rel_str = f"{float(rel):.2f}" if rel is not None and pd.notna(rel) else "—"

        short_sess = _short_session_id(sess_str)
        card = _MemberRowFrame(
            member_gid=gid,
            owner=self,
            unit_title=f"Unit {uid_str} · {short_sess}",
            unit_tooltip=f"Unit {uid_str} · {sess_str}",
            meta_text=f"{lat_str} · {rel_str}",
            on_click=self._on_inspector_member_clicked,
            parent=self,
        )
        card.set_active(bool(self._current_selected_gid and gid == self._current_selected_gid))
        return card

    def _populate_unit_response_for_unit(self, plot_widget: UnitResponsePlot, gid: str) -> None:
        """Populate a UnitResponsePlot instance live from spike/PSTH data (no PNGs)."""
        psth_data = self._get_raster_and_psth_for_unit(gid)
        sess_id = str(self._pooled_units_map.get(gid, {}).get("session_id", ""))
        plot_widget.set_psth_data(
            psth_data=psth_data,
            window_ms=self.current_window_ms,
            order_by=self.raster_order_by,
            marker_size=self.raster_marker_size,
            pulse_duration_ms=self._get_pulse_duration_ms(sess_id or None),
        )

    def _populate_image_viewer_for_unit(self, viewer: Any, gid: str) -> None:
        """Compatibility wrapper delegating live unit rendering to UnitResponsePlot."""
        target = getattr(viewer, "unit_response_plot", viewer)
        if isinstance(target, UnitResponsePlot):
            self._populate_unit_response_for_unit(target, gid)

    def _display_candidate_preview(self, gid: str) -> None:
        """Display focused unit raster/PSTH live via UnitResponsePlot."""
        if hasattr(self.unit_response_plot, "set_psth_tooltip"):
            self.unit_response_plot.set_psth_tooltip("Firing rate (Hz)")
        self._populate_unit_response_for_unit(self.unit_response_plot, gid)

    def _get_psth_for_unit(self, gid: str) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve or compute time_bins_ms and psth_hz for a given unit global ID."""
        if gid not in self._pooled_units_map:
            return np.array([]), np.array([])
        cache_key = (
            gid,
            (float(self.current_window_ms[0]), float(self.current_window_ms[1])),
            float(self.current_bin_size_ms),
            bool(self.psth_smooth),
            float(self.psth_gaussian_sigma_ms),
        )
        if hasattr(self, "_unit_psth_cache") and cache_key in self._unit_psth_cache:
            rates = self._unit_psth_cache[cache_key]
            start_ms, end_ms = self.current_window_ms
            step_ms = self.current_bin_size_ms
            t_bins = np.arange(start_ms + step_ms / 2.0, end_ms, step_ms)
            return t_bins, rates

        rec = self._pooled_units_map[gid]
        df_single = pd.DataFrame([rec])
        spike_map = self._get_spike_data_map(df_single)
        psth_matrix, t_bins = generate_psth_matrix(
            df_single,
            time_window_ms=self.current_window_ms,
            bin_size_ms=self.current_bin_size_ms,
            spike_data_map=spike_map,
            psth_cache=self._unit_psth_cache,
            smooth=self.psth_smooth,
            gaussian_sigma_ms=self.psth_gaussian_sigma_ms,
        )
        if len(psth_matrix) > 0:
            return t_bins, psth_matrix[0]
        return np.array([]), np.array([])

    def _update_cluster_cards_candidate_psth(self, gid: Optional[str]) -> None:
        """Update candidate PSTH curve overlay across all active cluster cards."""
        if not getattr(self, "overlay_focused_unit_enabled", True) or not gid or gid not in self._pooled_units_map:
            for card in self._cluster_cards.values():
                card.set_candidate_psth(np.array([]), np.array([]))
            return

        t_bins, rates = self._get_psth_for_unit(gid)
        for card in self._cluster_cards.values():
            card.set_candidate_psth(t_bins, rates)

    def _highlight_candidate_table_row(self, gid: str) -> None:
        """Select row corresponding to candidate without triggering re-clustering."""
        selected_cid = self.combo_filter_cluster.currentData()
        if selected_cid is not None and selected_cid != -2:
            return
        if not self.table_pooled_units.selectionModel().hasSelection() and self._current_selected_gid is None:
            return

        for row in range(self.table_pooled_units.rowCount()):
            if self._get_gid_for_row(row) == gid:
                if self.table_pooled_units.isRowHidden(row):
                    return
                self.table_pooled_units.blockSignals(True)
                self._select_single_pooled_row(row)
                self.table_pooled_units.blockSignals(False)
                self._current_selected_gid = gid
                self._highlight_focused_table_row(gid)
                break

    def _on_save_pooled_curation(self) -> None:
        """Handle save action from shortcut or button."""
        saved_path = self.save_project()
        if saved_path:
            QMessageBox.information(
                self,
                "Save Multi-Session Curation",
                f"Multi-Session Curation saved successfully.\n\n"
                f"Project Manifest: {saved_path.name}\n"
                f"Directory: {saved_path.parent}",
            )

    def save_project(self, project_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """Save current clustering project manifest and pooled curated units CSV.

        Args:
            project_dir: Target directory. If None, uses self.current_project_dir if set,
                         otherwise prompts user via prompt_save_project(save_as=True).

        Returns:
            Path to clustering_project.yaml if successfully saved, None if cancelled or failed.
        """
        if project_dir is None:
            if self.current_project_dir is not None:
                target_dir = self.current_project_dir
            else:
                return self.prompt_save_project(save_as=True)
        else:
            target_dir = Path(project_dir)

        target_dir = target_dir.resolve()

        # 1. Collect experiments
        exp_refs: List[ExperimentRef] = []
        for meta in self.loaded_experiments.values():
            portable = compute_portable_path(target_dir, meta.folder_path)
            exp_refs.append(
                ExperimentRef(
                    session_id=meta.session_id,
                    path=portable,
                    protocol_name=meta.protocol_name,
                    total_units=meta.total_units,
                    tagged_units=meta.tagged_units,
                )
            )

        # 2. Collect model parameters
        model_params = serialize_model_parameters(
            feature_config=self.feature_config,
            embedding_params=self.embedding_params,
            ap_params=self.ap_params,
            clustering_params=self.active_clustering_params,
            window_ms=self.current_window_ms,
            bin_size_ms=self.current_bin_size_ms,
            raster_order_by=self.raster_order_by,
            raster_marker_size=self.raster_marker_size,
            auto_assign_threshold=self.auto_assign_threshold,
            auto_assign_enabled=self.auto_assign_enabled,
            active_learning_enabled=self.active_learning_enabled,
            psth_smooth=self.psth_smooth,
            psth_gaussian_sigma_ms=self.psth_gaussian_sigma_ms,
            cluster_psth_norm=self.cluster_psth_norm,
            al_n_estimators=self.al_n_estimators,
            al_random_state=self.al_random_state,
            al_max_distance_sigma=self.al_max_distance_sigma,
            overlay_focused_unit_enabled=self.overlay_focused_unit_enabled,
            shared_y_axis_enabled=self.shared_y_axis_enabled,
            cluster_sort_mode=self._cluster_sort_mode,
        )

        # 3. Collect cluster definitions
        clusters_state = {
            "names": {int(k): str(v) for k, v in self._cluster_names.items()},
            "colors": {int(k): str(v) for k, v in self._cluster_colors.items()},
            "hotkeys": {int(k): str(v) for k, v in getattr(self, "_cluster_hotkeys", {}).items()},
            "card_order": [int(c) for c in self._cluster_card_order],
        }

        # 4. Collect active learning state
        al_state = {
            "skipped_gids": list(self._skipped_gids),
            "dismissed_cluster_candidates": {
                int(k): list(v) for k, v in self._dismissed_cluster_candidates.items()
            },
            "active_learning_enabled": self.active_learning_enabled,
            "auto_assign_enabled": self.auto_assign_enabled,
            "auto_assign_threshold": self.auto_assign_threshold,
        }

        # 5. Collect unit assignments and construct pooled_df
        unit_assignments: List[UnitAssignmentRecord] = []
        rows_for_df: List[Dict[str, Any]] = []

        for r in self._pooled_rows:
            gid = str(r.get("global_unit_id", ""))
            sess = str(r.get("session_id", ""))
            uid = int(r.get("unit_id", 0))
            cid = int(self._pooled_units_map.get(gid, {}).get("cluster_id", r.get("cluster_id", -1)))
            status = self._unit_status_map.get(gid, "Unassigned")
            auto_conf = self._auto_confidences.get(gid)

            rec = UnitAssignmentRecord(
                global_unit_id=gid,
                session_id=sess,
                unit_id=uid,
                cluster_id=cid,
                curated_status=status,
                auto_confidence=auto_conf,
            )
            unit_assignments.append(rec)

            df_row = dict(r)
            df_row["cluster_id"] = cid
            df_row["curated_status"] = status
            df_row["auto_confidence"] = auto_conf
            rows_for_df.append(df_row)

        pooled_df = pd.DataFrame(rows_for_df) if rows_for_df else None

        # 6. Save project data
        project_data = ClusteringProjectData(
            schema_version="1.0",
            project_type="optocurator_clustering_project",
            experiments=exp_refs,
            model_parameters=model_params,
            clusters=clusters_state,
            active_learning_state=al_state,
            unit_assignments=unit_assignments,
        )

        try:
            yaml_path = save_clustering_project(target_dir, project_data, pooled_df=pooled_df)
            self.current_project_dir = target_dir
            self._set_dirty(False)
            return yaml_path
        except Exception as err:
            QMessageBox.critical(self, "Save Project Failed", f"Failed to save clustering project:\n{err}")
            return None

    def prompt_save_project(self, save_as: bool = False) -> Optional[Path]:
        """Prompt user for a directory to save the clustering project.

        Args:
            save_as: If True or if current_project_dir is unset, always prompts user for a directory.

        Returns:
            Path to saved clustering_project.yaml or None if cancelled.
        """
        start_dir = str(self.current_project_dir) if self.current_project_dir else ""
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Clustering Project Folder to Save",
            start_dir,
        )
        if not selected:
            return None
        return self.save_project(selected)

    def prompt_load_project(self) -> bool:
        """Prompt user to select a clustering project directory and load it.

        Returns:
            True if project was successfully loaded, False otherwise.
        """
        start_dir = str(self.current_project_dir) if self.current_project_dir else ""
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Clustering Project Folder to Open (containing clustering_project.yaml)",
            start_dir,
        )
        if not selected:
            return False
        return self.load_project(selected)

    def load_project(self, project_dir: Optional[Union[str, Path]] = None) -> bool:
        """Load and restore an entire clustering project from disk.

        Args:
            project_dir: Directory containing clustering_project.yaml, or None to prompt.

        Returns:
            True on success, False on cancellation or failure.
        """
        if project_dir is None:
            return self.prompt_load_project()

        proj_path = Path(project_dir).resolve()
        manifest_dir = proj_path.parent if proj_path.is_file() else proj_path

        try:
            data = load_clustering_project(proj_path)
        except ProjectError as err:
            QMessageBox.critical(self, "Load Project Error", str(err))
            return False
        except Exception as err:
            QMessageBox.critical(
                self,
                "Failed to Load Project",
                f"An error occurred while loading project manifest:\n{err}",
            )
            return False

        # Reset existing experiments and workspace state
        self.clear_all_experiments()
        self._skipped_gids.clear()
        self._dismissed_cluster_candidates.clear()

        # Ingest each experiment using resolve_experiment_path and self.add_experiment_folder()
        for exp_ref in data.experiments:
            try:
                resolved_folder = resolve_experiment_path(manifest_dir, exp_ref.path)
            except MissingExperimentDirectoryError as err:
                QMessageBox.critical(
                    self,
                    "Experiment Directory Not Found",
                    f"Could not load experiment '{exp_ref.session_id}':\n{err}",
                )
                return False

            success = self.add_experiment_folder(resolved_folder, _defer_refresh=True)
            if not success:
                QMessageBox.warning(
                    self,
                    "Experiment Ingestion Warning",
                    f"Experiment '{exp_ref.session_id}' at '{resolved_folder}' could not be loaded into the pool.",
                )

        if self.loaded_experiments:
            self._invalidate_clustering_caches()
            self._invalidate_probe_caches()
            self._refresh_pooled_units_table()
            self._refresh_experiments_table()
            if hasattr(self, "marginal_widget"):
                self.marginal_widget.update_distributions(self._get_active_pooled_dataframe())

        # Reconstruct model parameters
        params_dict = deserialize_model_parameters(data.model_parameters)
        self.feature_config = params_dict["feature_config"]
        self._feature_window_customized = (
            tuple(self.feature_config.time_window_ms) != self._default_feature_time_window_ms()
        )
        self.embedding_params = params_dict["embedding_params"]
        self.ap_params = params_dict["ap_params"]
        self.active_clustering_params = params_dict["clustering_params"]
        self.current_window_ms = params_dict["window_ms"]
        self.current_bin_size_ms = params_dict["bin_size_ms"]
        self.raster_order_by = params_dict["raster_order_by"]
        self.raster_marker_size = params_dict["raster_marker_size"]
        self.auto_assign_threshold = params_dict["auto_assign_threshold"]
        self.auto_assign_enabled = params_dict["auto_assign_enabled"]
        self.active_learning_enabled = params_dict["active_learning_enabled"]
        self.psth_smooth = params_dict.get("psth_smooth", True)
        self.psth_gaussian_sigma_ms = params_dict.get("psth_gaussian_sigma_ms", 2.0)
        self.cluster_psth_norm = params_dict.get("cluster_psth_norm", "probability")
        self.al_n_estimators = int(params_dict.get("al_n_estimators", 100))
        self.al_random_state = int(params_dict.get("al_random_state", 42))
        self.al_max_distance_sigma = float(params_dict.get("al_max_distance_sigma", 3.5))
        self.overlay_focused_unit_enabled = bool(params_dict.get("overlay_focused_unit_enabled", True))
        self.shared_y_axis_enabled = bool(params_dict.get("shared_y_axis_enabled", False))
        self._cluster_sort_mode = str(params_dict.get("cluster_sort_mode", "id"))
        if self.active_clustering_params is not None:
            self.active_clustering_params.cluster_psth_norm = self.cluster_psth_norm

        if hasattr(self, "combo_psth_norm"):
            self.combo_psth_norm.blockSignals(True)
            norm_idx = {"peak": 0, "probability": 1, "raw": 2}.get(self.cluster_psth_norm, 1)
            self.combo_psth_norm.setCurrentIndex(norm_idx)
            self.combo_psth_norm.blockSignals(False)

        if hasattr(self, "combo_embedding_method"):
            self.combo_embedding_method.blockSignals(True)
            emb_idx = self.combo_embedding_method.findData(str(self.embedding_params.method).upper())
            if emb_idx >= 0:
                self.combo_embedding_method.setCurrentIndex(emb_idx)
            self.combo_embedding_method.blockSignals(False)

        if hasattr(self, "chk_smooth"):
            self.chk_smooth.blockSignals(True)
            self.chk_smooth.setChecked(self.psth_smooth)
            self.chk_smooth.blockSignals(False)
        if hasattr(self, "spin_sigma"):
            self.spin_sigma.blockSignals(True)
            self.spin_sigma.setValue(self.psth_gaussian_sigma_ms)
            self.spin_sigma.blockSignals(False)
        if hasattr(self, "slider_threshold"):
            self.slider_threshold.blockSignals(True)
            self.slider_threshold.setValue(int(round(self.auto_assign_threshold * 100)))
            self.slider_threshold.blockSignals(False)
        if hasattr(self, "chk_auto_assign"):
            self.chk_auto_assign.blockSignals(True)
            self.chk_auto_assign.setChecked(self.auto_assign_enabled)
            self.chk_auto_assign.blockSignals(False)
        if hasattr(self, "act_overlay_focused_unit"):
            self.act_overlay_focused_unit.blockSignals(True)
            self.act_overlay_focused_unit.setChecked(self.overlay_focused_unit_enabled)
            self.act_overlay_focused_unit.blockSignals(False)
        if hasattr(self, "act_shared_y_axis"):
            self.act_shared_y_axis.blockSignals(True)
            self.act_shared_y_axis.setChecked(self.shared_y_axis_enabled)
            self.act_shared_y_axis.blockSignals(False)
        for sort_key, sort_act in getattr(self, "_sort_actions", {}).items():
            sort_act.blockSignals(True)
            sort_act.setChecked(sort_key == self._cluster_sort_mode)
            sort_act.blockSignals(False)
        self._update_active_learning_button_style()

        # Restore cluster definitions
        self._cluster_names.clear()
        self._cluster_colors.clear()
        self._cluster_hotkeys.clear()
        self._cluster_card_order.clear()

        names = data.clusters.get("names", {})
        for k, v in names.items():
            try:
                self._cluster_names[int(k)] = str(v)
            except (ValueError, TypeError):
                pass

        colors = data.clusters.get("colors", {})
        for k, v in colors.items():
            try:
                self._cluster_colors[int(k)] = str(v)
            except (ValueError, TypeError):
                pass

        hotkeys = data.clusters.get("hotkeys", {})
        for k, v in hotkeys.items():
            try:
                self._cluster_hotkeys[int(k)] = normalize_hotkey(str(v))
            except (ValueError, TypeError):
                pass

        card_order = data.clusters.get("card_order", [])
        for cid in card_order:
            try:
                self._cluster_card_order.append(int(cid))
            except (ValueError, TypeError):
                pass

        # Restore unit assignments and curation status
        assignments_map = {rec.global_unit_id: rec for rec in data.unit_assignments}
        for gid, rec in assignments_map.items():
            if gid in self._pooled_units_map:
                self._pooled_units_map[gid]["cluster_id"] = rec.cluster_id
            self._unit_status_map[gid] = rec.curated_status
            if rec.auto_confidence is not None:
                self._auto_confidences[gid] = rec.auto_confidence
            elif gid in self._auto_confidences:
                del self._auto_confidences[gid]

        for r in self._pooled_rows:
            gid = str(r.get("global_unit_id", ""))
            if gid in assignments_map:
                r["cluster_id"] = assignments_map[gid].cluster_id

        # Synchronize cluster_id in meta.df for loaded experiments
        for meta in self.loaded_experiments.values():
            if hasattr(meta, "df") and meta.df is not None and not meta.df.empty:
                if "cluster_id" not in meta.df.columns:
                    meta.df["cluster_id"] = -1
                for gid, rec in assignments_map.items():
                    if "global_unit_id" in meta.df.columns:
                        meta.df.loc[meta.df["global_unit_id"] == gid, "cluster_id"] = rec.cluster_id
                    if "unit_id" in meta.df.columns and rec.session_id == meta.session_id:
                        meta.df.loc[meta.df["unit_id"] == rec.unit_id, "cluster_id"] = rec.cluster_id

        # Restore active learning queue state
        self._skipped_gids = set(data.active_learning_state.get("skipped_gids", []))
        raw_dismissed = data.active_learning_state.get("dismissed_cluster_candidates", {})
        self._dismissed_cluster_candidates.clear()
        for k, v in raw_dismissed.items():
            try:
                self._dismissed_cluster_candidates[int(k)] = set(v)
            except (ValueError, TypeError):
                pass

        # Ensure all assigned cluster IDs are present in _cluster_card_order
        for r in self._pooled_rows:
            cid = r.get("cluster_id", -1)
            if cid >= 0 and cid not in self._cluster_card_order:
                self._cluster_card_order.append(cid)

        # Trigger full re-rendering
        self._recompute_and_render_clusters()
        if self.shared_y_axis_enabled:
            self._apply_shared_y_axis()
        self._refresh_cluster_shortcuts()
        self._update_all_pooled_cluster_items()
        self._populate_cluster_filter_options()
        self.refresh_active_learning()

        if self.table_pooled_units.rowCount() > 0:
            self.table_pooled_units.blockSignals(True)
            self.table_pooled_units.clearSelection()
            self.table_pooled_units.selectRow(0)
            self.table_pooled_units.blockSignals(False)
            self._on_pooled_unit_selected()

        self.current_project_dir = manifest_dir
        self._set_dirty(False)
        self.experiments_changed.emit()
        return True

    def save_clustering_project(self, project_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """Save current clustering project to disk and clear dirty state."""
        return self.save_project(project_dir=project_dir)

    def open_clustering_project(self, project_dir: Optional[Union[str, Path]] = None) -> bool:
        """Open a clustering project folder from disk and clear dirty state."""
        return self.load_project(project_dir=project_dir)

    def new_clustering_project(self, project_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """Initialize a new clustering project in the given (or chosen) folder."""
        if project_dir is None:
            selected = QFileDialog.getExistingDirectory(
                self,
                "Select Folder for New Clustering Project",
                str(self.current_project_dir) if self.current_project_dir else "",
            )
            if not selected:
                return None
            target_dir = Path(selected).resolve()
        else:
            target_dir = Path(project_dir).resolve()

        target_dir.mkdir(parents=True, exist_ok=True)
        self.clear_all_experiments()
        self._skipped_gids.clear()
        self._dismissed_cluster_candidates.clear()
        self.reset_to_unassigned()
        self.current_project_dir = target_dir
        saved = self.save_project(target_dir)
        self._set_dirty(False)
        return target_dir if saved is not None else None

    def close_clustering_project(self) -> None:
        """Close the current clustering project and reset workspace to clean empty state."""
        self._invalidate_active_learning_worker()
        self.clear_all_experiments()
        self._skipped_gids.clear()
        self._dismissed_cluster_candidates.clear()
        self.reset_to_unassigned()
        self.current_project_dir = None
        self._set_dirty(False)

    def remove_clustering_experiment(
        self,
        session_id: Optional[str] = None,
        confirm: bool = True,
    ) -> bool:
        """Remove an experiment from the project pool with optional confirmation."""
        target_sid = session_id
        if target_sid is None:
            selected_rows = self.table_exps.selectionModel().selectedRows()
            if selected_rows:
                item = self.table_exps.item(selected_rows[0].row(), 0)
                if item:
                    target_sid = item.text()
            elif self.table_exps.currentRow() >= 0:
                item = self.table_exps.item(self.table_exps.currentRow(), 0)
                if item:
                    target_sid = item.text()
            elif self.loaded_experiments:
                sids = list(self.loaded_experiments.keys())
                chosen, ok = QInputDialog.getItem(
                    self,
                    "Remove Experiment",
                    "Select experiment to remove:",
                    sids,
                    0,
                    False,
                )
                if not ok or not chosen:
                    return False
                target_sid = chosen

        if confirm and target_sid and target_sid in self.loaded_experiments:
            reply = QMessageBox.question(
                self,
                "Remove Experiment",
                f"Remove experiment '{target_sid}' and its units from this clustering project?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False

        return self.remove_selected_experiment(target_sid)


    def _select_single_pooled_row(self, row: int) -> None:
        """Select a single row in table_pooled_units without inheriting keyboard Shift/Ctrl modifiers."""
        sel_model = self.table_pooled_units.selectionModel()
        if sel_model is not None:
            idx = self.table_pooled_units.model().index(row, 0)
            sel_model.setCurrentIndex(
                idx,
                QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows,
            )
        else:
            self.table_pooled_units.clearSelection()
            self.table_pooled_units.selectRow(row)

    def select_pooled_unit(self, gid: str, source: Optional[str] = None) -> bool:
        """Select a unit in the pooled units table by global_unit_id."""
        if source is not None:
            self._focus_source = source
        for row in range(self.table_pooled_units.rowCount()):
            if self._get_gid_for_row(row) == gid:
                self._programmatic_unit_select = True
                try:
                    self.table_pooled_units.blockSignals(True)
                    self._select_single_pooled_row(row)
                    self.table_pooled_units.blockSignals(False)
                    self._on_pooled_unit_selected()
                finally:
                    self._programmatic_unit_select = False
                return True
        return False

    def add_experiment_folder(self, folder_path: Optional[Union[str, Path]] = None, _defer_refresh: bool = False) -> bool:
        """Add and validate a Phase 1 curated experiment directory into the multi-experiment pool.

        Args:
            folder_path: Directory containing <session_id>_opto_curated.csv and <session_id>_plots/.
                         If None, prompts user via QFileDialog.

        Returns:
            True if experiment was successfully validated and registered, False otherwise.
        """
        if folder_path is None:
            selected = QFileDialog.getExistingDirectory(
                self,
                "Select Curated Experiment Folder (Must contain <session_id>_opto_curated.csv)",
                "",
            )
            if not selected:
                return False
            folder = Path(selected)
        else:
            folder = Path(folder_path)

        progress = QProgressDialog(
            "Validating Phase 1 curation export...",
            None,
            0,
            100,
            self,
        )
        progress.setWindowTitle("Loading Experiment")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(True)
        progress.setStyleSheet(PROGRESS_DIALOG_STYLE)
        progress.setValue(10)
        QApplication.processEvents()

        try:
            try:
                meta = validate_phase1_export(folder)
            except MissingPhase1ExportError as err:
                progress.close()
                QMessageBox.warning(
                    self,
                    "Missing Phase 1 Export",
                    str(err),
                )
                return False
            except Exception as err:
                progress.close()
                QMessageBox.critical(
                    self,
                    "Failed to Ingest Experiment",
                    f"An error occurred while inspecting directory '{folder.name}':\n{err}",
                )
                return False

            if meta.tagged_units == 0:
                QMessageBox.information(
                    self,
                    "No Tagged Units",
                    f"Curated experiment '{meta.session_id}' contains no optotagged units (0 tagged).\n\n"
                    "Phase 2 response clustering exclusively analyzes optotagged units, so no cells "
                    "from this recording were added to the pool.",
                )

            # Step 1b: (25%) "Loading spike timing and stimulus pulses..."
            progress.setValue(25)
            progress.setLabelText("Loading spike timing and stimulus pulses...")
            QApplication.processEvents()

            spikes_dict, pulse_onsets, pulse_offsets, npz_dict = load_phase1_session_arrays(meta)

            # Resolve experiment probe geometry and unit coordinates
            probe_pos, resolved_df = self._resolve_experiment_probe_and_unit_positions(
                session_id=meta.session_id,
                folder_path=meta.folder_path,
                df=meta.df,
                npz_data=npz_dict if npz_dict else None,
            )
            meta.df = resolved_df

            # Step 2: (40%) "Populating pooled units..."
            progress.setValue(40)
            progress.setLabelText("Populating pooled units...")
            QApplication.processEvents()

            # Transition away from mock state upon first real dataset load
            if self._is_mock:
                self.loaded_experiments.clear()
                self._pooled_rows.clear()
                self._pooled_units_map.clear()
                self._experiment_spikes.clear()
                self._experiment_stimulus.clear()
                self._experiment_pulse_durations.clear()
                self._unit_psth_cache.clear()
                self._unit_raster_psth_cache.clear()
                self._reset_background_prewarm(clear_pixmaps=True)
                self._is_mock = False

            if spikes_dict:
                self._experiment_spikes[meta.session_id] = spikes_dict
            if pulse_onsets is not None:
                self._experiment_stimulus[meta.session_id] = pulse_onsets
            inferred_dur_ms = infer_pulse_duration_ms(
                pulse_onsets,
                pulse_offsets,
                default_ms=DEFAULT_PULSE_DURATION_MS,
            )
            if inferred_dur_ms == DEFAULT_PULSE_DURATION_MS and meta.df is not None and not meta.df.empty:
                for col in ("pulse_duration_ms", "pulse_duration", "stim_duration_ms"):
                    if col in meta.df.columns and pd.notna(meta.df[col].iloc[0]):
                        inferred_dur_ms = float(meta.df[col].iloc[0])
                        break
            self._experiment_pulse_durations[meta.session_id] = inferred_dur_ms
            meta.pulse_duration_ms = inferred_dur_ms

            self.loaded_experiments[meta.session_id] = meta
            if not self._feature_window_customized:
                self.feature_config.time_window_ms = self._default_feature_time_window_ms()

            if not _defer_refresh:
                self._invalidate_clustering_caches()
                self._invalidate_probe_caches()

                self._refresh_pooled_units_table()
                self._refresh_experiments_table()
                if hasattr(self, "marginal_widget"):
                    self.marginal_widget.update_distributions(self._get_active_pooled_dataframe())

                # Step 3: (70%) "Computing embeddings & active learning..."
                progress.setValue(70)
                progress.setLabelText("Computing embeddings & active learning...")
                QApplication.processEvents()

                # Run or update clustering across newly pooled units
                if self.table_pooled_units.rowCount() > 0:
                    if self._start_with_clusters:
                        self.run_clustering(self.active_clustering_params)
                    else:
                        self.reset_to_unassigned()

                    # Step 4: (90%) "Rendering preview..."
                    progress.setValue(90)
                    progress.setLabelText("Rendering preview...")
                    QApplication.processEvents()

                    self.table_pooled_units.blockSignals(True)
                    self.table_pooled_units.clearSelection()
                    self.table_pooled_units.selectRow(0)
                    self.table_pooled_units.blockSignals(False)
                    self._on_pooled_unit_selected()

                self._set_dirty(True)
                self.experiment_added.emit(meta.session_id)

            # Step 5: (100%) Close dialog
            progress.setValue(100)
            QApplication.processEvents()
            return True
        finally:
            progress.close()
            progress.deleteLater()

    def remove_selected_experiment(self, session_id: Optional[str] = None) -> bool:
        """Drop selected experiment from multi-experiment pool and clean state.

        Args:
            session_id: Optional session ID string. If not provided, removes selected row from table_exps.

        Returns:
            True if an experiment was removed, False otherwise.
        """
        if session_id is None:
            selected_rows = self.table_exps.selectionModel().selectedRows()
            if not selected_rows:
                curr_row = self.table_exps.currentRow()
                if curr_row < 0:
                    QMessageBox.information(
                        self,
                        "No Experiment Selected",
                        "Please select an experiment from the active experiments table to remove.",
                    )
                    return False
                row = curr_row
            else:
                row = selected_rows[0].row()

            item = self.table_exps.item(row, 0)
            if not item:
                return False

            session_id = item.text()
        if session_id in self.loaded_experiments:
            del self.loaded_experiments[session_id]
            if session_id in self._experiment_probes:
                del self._experiment_probes[session_id]
            if self._current_probe_session_id == session_id:
                self._current_probe_session_id = None
                self._probe_shank_items.clear()
                if hasattr(self, "probe_widget"):
                    self.probe_widget.clear_probe()
            if session_id in self._experiment_spikes:
                del self._experiment_spikes[session_id]
            if session_id in self._experiment_stimulus:
                del self._experiment_stimulus[session_id]
            if session_id in self._experiment_pulse_durations:
                del self._experiment_pulse_durations[session_id]
            self._unit_psth_cache = {
                k: v for k, v in self._unit_psth_cache.items()
                if not str(k[0]).startswith(session_id)
            }
            self._unit_raster_psth_cache = {
                k: v for k, v in self._unit_raster_psth_cache.items()
                if not str(k[0]).startswith(session_id)
            }
            self._member_pixmap_cache = {
                k: v for k, v in self._member_pixmap_cache.items()
                if not str(k[0]).startswith(session_id)
            }
            self._prewarm_generation += 1
            self._set_dirty(True)
            self.experiment_removed.emit(session_id)
            self._invalidate_clustering_caches()
            self._invalidate_probe_caches()
            if not self._feature_window_customized:
                self.feature_config.time_window_ms = self._default_feature_time_window_ms()

        self._refresh_pooled_units_table()
        self._refresh_experiments_table()
        if hasattr(self, "marginal_widget"):
            self.marginal_widget.update_distributions(self._get_active_pooled_dataframe())

        if len(self.loaded_experiments) == 0:
            self._invalidate_active_learning_worker()
            self.ranked_candidates = []
            self.active_candidate = None
            self.clear_cluster_comparison()
            self.image_viewer.clear_preview()
            self.lbl_candidate.setText("No Experiment Selected")
            self.scatter_item.clear()
            self.cand_scatter.clear()
            while self.clusters_grid.count():
                item = self.clusters_grid.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            for card in self._cluster_cards.values():
                card.set_candidate_probability(None)
            if hasattr(self, "cluster_prob_bar"):
                self.cluster_prob_bar.clear()
            self._refresh_list_panel()
            self._update_classification_progress()
        elif self.table_pooled_units.rowCount() > 0:
            if self._start_with_clusters:
                self.run_clustering(self.active_clustering_params)
            else:
                self.reset_to_unassigned()
            self.table_pooled_units.blockSignals(True)
            self.table_pooled_units.clearSelection()
            self.table_pooled_units.selectRow(0)
            self.table_pooled_units.blockSignals(False)
            self._on_pooled_unit_selected()
            self._update_classification_progress()

        return True

    def clear_all_experiments(self) -> None:
        """Clear all loaded experiments and reset Phase 2 pooled state."""
        self._invalidate_active_learning_worker()
        self._invalidate_clustering_caches()
        self._invalidate_probe_caches()
        self.loaded_experiments.clear()
        self._experiment_probes.clear()
        self._current_probe_session_id = None
        self._probe_shank_items.clear()
        self._probe_shank_labels.clear()
        if hasattr(self, "probe_widget"):
            self.probe_widget.clear_probe()
        self._experiment_spikes.clear()
        self._experiment_stimulus.clear()
        self._experiment_pulse_durations.clear()
        self._feature_window_customized = False
        self.feature_config.time_window_ms = self._default_feature_time_window_ms()
        self._unit_psth_cache.clear()
        self._unit_raster_psth_cache.clear()
        self._reset_background_prewarm(clear_pixmaps=True)
        self._pooled_rows.clear()
        self._pooled_units_map.clear()
        self._unit_status_map.clear()
        self._auto_confidences.clear()
        self._all_unit_probabilities.clear()
        self._unit_candidates_map.clear()
        self._is_mock = False
        self.table_exps.setRowCount(0)
        self.table_pooled_units.setRowCount(0)
        self.lbl_pool_count.setText("0 units")
        self.lbl_candidate.setText("No Experiment Selected")
        self.image_viewer.clear_preview()
        self._populate_cluster_filter_options()
        self.scatter_item.clear()
        self.cand_scatter.clear()
        self.reset_to_unassigned()
        self.experiments_changed.emit()

    def _get_gid_for_row(self, row: int) -> str:
        """Get the global unit ID associated with a row in table_pooled_units."""
        if row < 0 or row >= self.table_pooled_units.rowCount():
            return ""
        item = self.table_pooled_units.item(row, 0)
        if item is not None:
            gid = item.data(Qt.ItemDataRole.UserRole)
            if gid is not None and str(gid).strip():
                return str(gid)
        if hasattr(self, "_pooled_rows") and 0 <= row < len(self._pooled_rows):
            return str(self._pooled_rows[row].get("global_unit_id", ""))
        return item.text().strip() if item is not None else ""

    def _update_cluster_column_for_row(self, row: int, cluster_id: int) -> None:
        """Update column 2 (Cluster) and column 6 (Status) for a given row in-place."""
        if row < 0 or row >= self.table_pooled_units.rowCount():
            return
        if cluster_id < 0:
            display_text = "—"
            full_name = "Unassigned"
            color_hex = COLOR_TEXT_MUTED
            swatch_icon = QIcon()
            sort_key = -1
        else:
            full_name = self._cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            swatch_hex = self._get_cluster_color(cluster_id)
            color_hex = self._get_cluster_text_color(cluster_id)
            swatch_icon = _make_swatch_icon(swatch_hex, 10)
            display_text = full_name
            sort_key = cluster_id

        # Column 2: Cluster (full cluster name in readable colour, no swatch icon)
        existing_item = self.table_pooled_units.item(row, 2)
        if isinstance(existing_item, NumericTableWidgetItem):
            existing_item.setText(display_text)
            existing_item.setIcon(QIcon())
            existing_item.sort_key = sort_key
            existing_item.setData(Qt.ItemDataRole.UserRole, sort_key)
            existing_item.setForeground(QColor(color_hex))
            existing_item.setToolTip(full_name)
        else:
            item = NumericTableWidgetItem(display_text, sort_key=sort_key)
            item.setData(Qt.ItemDataRole.UserRole, sort_key)
            item.setForeground(QColor(color_hex))
            item.setToolTip(full_name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_pooled_units.setItem(row, 2, item)

        # Column 6: Status (icon-only + tooltip)
        gid = self._get_gid_for_row(row)
        status = self._unit_status_map.get(gid, "Unassigned")
        if cluster_id < 0:
            status = "Unassigned"
        status_icon = _make_status_icon(status, 14)

        existing_status_item = self.table_pooled_units.item(row, 6)
        if isinstance(existing_status_item, NumericTableWidgetItem):
            existing_status_item.setText("")
            existing_status_item.setIcon(status_icon)
            existing_status_item.sort_key = status
            existing_status_item.setData(Qt.ItemDataRole.UserRole, status)
            existing_status_item.setToolTip(status)
            existing_status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            status_item = NumericTableWidgetItem("", sort_key=status)
            status_item.setIcon(status_icon)
            status_item.setData(Qt.ItemDataRole.UserRole, status)
            status_item.setToolTip(status)
            status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_pooled_units.setItem(row, 6, status_item)

    def _populate_pooled_table_row(self, r_idx: int, r: Dict[str, Any]) -> None:
        """Populate all 7 WP4 columns (Unit, Session, Cluster, Lat (ms), Rel., SALT p, Status) for one row."""
        gid = str(r.get("global_unit_id", ""))
        sess = str(r.get("session_id", ""))
        uid = r.get("unit_id", 0)
        lat = r.get("latency_ms")
        rel = r.get("reliability")
        salt_p = r.get("salt_p_value")
        cid = int(r.get("cluster_id", -1))

        # Col 0: Unit (numeric sort, stores gid in UserRole for _get_gid_for_row)
        raw_uid = None if (uid is None or pd.isna(uid) or str(uid).strip() == gid) else uid
        sess_str, uid_str = _format_unit_display(sess, raw_uid, gid)
        if not uid_str or uid_str == gid:
            _, parsed_uid = _format_unit_display("", None, gid)
            uid_str = parsed_uid or "0"
        sort_uid = int(uid_str) if str(uid_str).lstrip("-").isdigit() else 0
        unit_item = NumericTableWidgetItem(uid_str, sort_key=sort_uid)
        unit_item.setData(Qt.ItemDataRole.UserRole, gid)
        unit_item.setFlags(unit_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table_pooled_units.setItem(r_idx, 0, unit_item)

        # Col 1: Session (short ID + full session name in tooltip)
        full_sess = sess_str or sess
        sess_item = QTableWidgetItem(_short_session_id(full_sess))
        sess_item.setToolTip(full_sess)
        sess_item.setFlags(sess_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table_pooled_units.setItem(r_idx, 1, sess_item)

        mono_font = make_mono_font(FONT_SIZE_BODY)

        # Col 3: Lat (ms) (right-aligned, 1 dp, tabular)
        lat_str = f"{float(lat):.1f}" if pd.notna(lat) and is_valid_metric_value(lat) else "N/A"
        lat_key = float(lat) if pd.notna(lat) else MISSING_METRIC_SENTINEL
        lat_item = NumericTableWidgetItem(lat_str, sort_key=lat_key)
        lat_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lat_item.setFont(mono_font)
        lat_item.setFlags(lat_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table_pooled_units.setItem(r_idx, 3, lat_item)

        # Col 4: Rel. (right-aligned, 2 dp, tabular)
        rel_str = f"{float(rel):.2f}" if pd.notna(rel) else "0.00"
        rel_key = float(rel) if pd.notna(rel) else 0.0
        rel_item = NumericTableWidgetItem(rel_str, sort_key=rel_key)
        rel_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        rel_item.setFont(mono_font)
        rel_item.setFlags(rel_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table_pooled_units.setItem(r_idx, 4, rel_item)

        # Col 5: SALT p (right-aligned, <= 0.01 resolution floor + exact value in tooltip)
        if pd.notna(salt_p):
            salt_val = float(salt_p)
            salt_str = "≤ 0.01" if salt_val <= SALT_P_RESOLUTION_FLOOR + 1e-9 else f"{salt_val:.2f}"
            salt_key = salt_val
            salt_tip = f"SALT p = {salt_val:.4f}"
        else:
            salt_str = "1.00"
            salt_key = 1.0
            salt_tip = "SALT p = 1.0000"
        salt_item = NumericTableWidgetItem(salt_str, sort_key=salt_key)
        salt_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        salt_item.setFont(mono_font)
        salt_item.setToolTip(salt_tip)
        salt_item.setFlags(salt_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table_pooled_units.setItem(r_idx, 5, salt_item)

        # Col 2 (Cluster) & Col 6 (Status)
        self._update_cluster_column_for_row(r_idx, cid)

    def _highlight_focused_table_row(self, focused_gid: Optional[str]) -> None:
        """Highlight the focused unit row with COLOR_ACCENT_BG and semibold font."""
        accent_brush = QBrush(QColor(COLOR_ACCENT_BG))
        default_brush = QBrush()
        for r in range(self.table_pooled_units.rowCount()):
            is_focused = bool(focused_gid) and (self._get_gid_for_row(r) == focused_gid)
            for c in range(self.table_pooled_units.columnCount()):
                item = self.table_pooled_units.item(r, c)
                if item is None:
                    continue
                item.setBackground(accent_brush if is_focused else default_brush)
                font = item.font()
                font.setWeight(QFont.Weight.DemiBold if is_focused else QFont.Weight.Normal)
                item.setFont(font)

    def _update_all_pooled_cluster_items(self, target_gids: Optional[Set[str]] = None) -> None:
        """Update column 2 (Cluster) and column 6 (Status) for rows in table_pooled_units based on _pooled_units_map."""
        self.table_pooled_units.setUpdatesEnabled(False)
        self.table_pooled_units.blockSignals(True)
        was_sorting = self.table_pooled_units.isSortingEnabled()
        self.table_pooled_units.setSortingEnabled(False)
        try:
            total_rows = self.table_pooled_units.rowCount()
            for row in range(total_rows):
                gid = self._get_gid_for_row(row)
                if gid and (target_gids is None or gid in target_gids):
                    cid = self._pooled_units_map.get(gid, {}).get("cluster_id", -1)
                    self._update_cluster_column_for_row(row, cid)
            self.table_pooled_units.setSortingEnabled(was_sorting)
        finally:
            self.table_pooled_units.blockSignals(False)
            self.table_pooled_units.setUpdatesEnabled(True)

        if getattr(self, "_current_selected_gid", None):
            self._update_probe_view_for_unit(self._current_selected_gid)
        elif getattr(self, "active_candidate", None):
            self._update_probe_view_for_unit(self.active_candidate.global_unit_id)

    def _refresh_experiments_table(self) -> None:
        """Refresh table_exps with current loaded_experiments."""
        self.table_exps.setSortingEnabled(False)
        self.table_exps.setRowCount(len(self.loaded_experiments))
        for row, (session_id, meta) in enumerate(self.loaded_experiments.items()):
            self.table_exps.setItem(row, 0, QTableWidgetItem(meta.session_id))
            self.table_exps.setItem(row, 1, QTableWidgetItem(meta.protocol_name))
            cells_str = f"{meta.tagged_units} tagged"
            self.table_exps.setItem(row, 2, NumericTableWidgetItem(cells_str, sort_key=meta.tagged_units))
        self.table_exps.setSortingEnabled(True)
        self.experiments_changed.emit()

    def _refresh_pooled_units_table(self) -> None:
        """Refresh table_pooled_units with tagged units only from all active experiments."""
        self.table_pooled_units.setUpdatesEnabled(False)
        self.table_pooled_units.blockSignals(True)
        was_sorting = self.table_pooled_units.isSortingEnabled()
        self.table_pooled_units.setSortingEnabled(False)
        try:
            self._pooled_rows.clear()
            self._pooled_units_map.clear()

            for meta in self.loaded_experiments.values():
                tagged_df = meta.tagged_df
                if tagged_df is not None and not tagged_df.empty:
                    for record in tagged_df.to_dict(orient="records"):
                        gid = str(record.get("global_unit_id", ""))
                        if "cluster_id" not in record:
                            record["cluster_id"] = -1
                        self._pooled_rows.append(record)
                        self._pooled_units_map[gid] = record

            self.table_pooled_units.setRowCount(len(self._pooled_rows))
            for r_idx, r in enumerate(self._pooled_rows):
                self._populate_pooled_table_row(r_idx, r)

            self.table_pooled_units.setSortingEnabled(was_sorting)
        finally:
            self.table_pooled_units.blockSignals(False)
            self.table_pooled_units.setUpdatesEnabled(True)

        if getattr(self, "_current_selected_gid", None):
            self._update_probe_view_for_unit(self._current_selected_gid)
        elif getattr(self, "active_candidate", None):
            self._update_probe_view_for_unit(self.active_candidate.global_unit_id)

        self._highlight_focused_table_row(getattr(self, "_current_selected_gid", None))
        self.lbl_pooled_units_badge.setText(f"{len(self._pooled_rows)} units")
        self._populate_cluster_filter_options()
        if hasattr(self, "marginal_widget"):
            self.marginal_widget.update_distributions(self._get_active_pooled_dataframe())

    def _set_pooled_table_rows(
        self,
        rows_data: List[Any],
    ) -> None:
        """Helper to populate table_pooled_units with tagged units only (used in mock state)."""
        tagged_rows = [r for r in rows_data if r[6] is True]
        self.table_pooled_units.setUpdatesEnabled(False)
        self.table_pooled_units.blockSignals(True)
        was_sorting = self.table_pooled_units.isSortingEnabled()
        self.table_pooled_units.setSortingEnabled(False)
        try:
            self.table_pooled_units.setRowCount(len(tagged_rows))
            self._pooled_rows.clear()
            self._pooled_units_map.clear()

            for r_idx, r_item in enumerate(tagged_rows):
                gid, sess, uid, lat, rel, salt_p, is_tagged = r_item[:7]
                ch = int(r_item[7]) if len(r_item) > 7 else int(uid) % 32
                depth = float(r_item[8]) if len(r_item) > 8 else float(100.0 + (int(uid) % 32) * 50.0)
                probe_x = float(r_item[9]) if len(r_item) > 9 else float(16.0 if (int(uid) % 2 == 0) else 48.0)
                record = {
                    "global_unit_id": gid,
                    "session_id": sess,
                    "unit_id": uid,
                    "latency_ms": lat,
                    "jitter_ms": 1.2,
                    "reliability": rel,
                    "salt_p_value": salt_p,
                    "is_tagged": True,
                    "cluster_id": -1,
                    "channel": ch,
                    "depth": depth,
                    "probe_x": probe_x,
                }
                self._pooled_rows.append(record)
                self._pooled_units_map[gid] = record
                self._populate_pooled_table_row(r_idx, record)

            self.table_pooled_units.setSortingEnabled(was_sorting)
        finally:
            self.table_pooled_units.blockSignals(False)
            self.table_pooled_units.setUpdatesEnabled(True)

        if getattr(self, "_current_selected_gid", None):
            self._update_probe_view_for_unit(self._current_selected_gid)
        elif getattr(self, "active_candidate", None):
            self._update_probe_view_for_unit(self.active_candidate.global_unit_id)

        self._highlight_focused_table_row(getattr(self, "_current_selected_gid", None))
        self.lbl_pooled_units_badge.setText(f"{len(tagged_rows)} units")
        self._populate_cluster_filter_options()
        if hasattr(self, "marginal_widget"):
            self.marginal_widget.update_distributions(self._get_active_pooled_dataframe())

    def _on_pooled_unit_selected(self) -> None:
        """Handle selection change in pooled units table to update pre-rendered preview, scatter marker, probe, marginals, and probabilities."""
        selected_rows = self.table_pooled_units.selectionModel().selectedRows()
        if not selected_rows:
            self._current_selected_gid = None
            self._highlight_focused_table_row(None)
            if hasattr(self, "btn_comp_assign"):
                self.btn_comp_assign.setText("✓ Assign Focused")
            if hasattr(self, "btn_assign_candidate"):
                self.btn_assign_candidate.setText("✓ Assign Focused")
            return

        n_selected = len(selected_rows)
        assign_text = f"✓ Assign Selected ({n_selected})" if n_selected > 1 else "✓ Assign Focused"
        if hasattr(self, "btn_comp_assign"):
            self.btn_comp_assign.setText(assign_text)
        if hasattr(self, "btn_assign_candidate"):
            self.btn_assign_candidate.setText(assign_text)

        row = selected_rows[0].row()

        gid = self._get_gid_for_row(row)
        if not gid:
            return

        self._current_selected_gid = gid
        if not self._programmatic_unit_select:
            flt = self.combo_filter_cluster.currentData() if hasattr(self, "combo_filter_cluster") else None
            if isinstance(flt, int) and flt >= 0:
                self._focus_source = "table_cluster"
            else:
                self._focus_source = "queue"
        self._highlight_focused_table_row(gid)

        # 1. Scatter indicator
        if gid in self._scatter_points_map:
            px, py, _ = self._scatter_points_map[gid]
            self.cand_scatter.setData(x=[px], y=[py])
        elif self.current_clustering_result is not None:
            for i, u_id in enumerate(self.current_clustering_result.unit_ids):
                if u_id == gid:
                    px = float(self.current_clustering_result.pca_coordinates[i, 0])
                    py = float(self.current_clustering_result.pca_coordinates[i, 1])
                    self.cand_scatter.setData(x=[px], y=[py])
                    break

        # 2. Probe marker & marginal distributions
        rec = self._pooled_units_map.get(gid, {})
        self._update_probe_view_for_unit(gid)

        feat_vals = {}
        if gid in self._unit_candidates_map:
            feat_vals = self._unit_candidates_map[gid].feature_values
        else:
            for k in ("latency_ms", "jitter_ms", "reliability", "salt_p_value"):
                if k in rec:
                    try:
                        feat_vals[k] = float(rec[k])
                    except (ValueError, TypeError):
                        pass
        if feat_vals:
            self._update_marginal_distributions(feat_vals)

        # 3. Update Cluster Probability Bar and Card Badges
        probs = self._all_unit_probabilities.get(gid)
        if not probs and gid in self._unit_candidates_map:
            probs = self._unit_candidates_map[gid].probabilities
        if probs:
            if hasattr(self, "cluster_prob_bar"):
                self.cluster_prob_bar.set_probabilities(
                    probabilities=probs,
                    cluster_colors=self._cluster_colors,
                    cluster_names=self._cluster_names,
                    threshold=self.auto_assign_threshold,
                    cluster_order=sorted(self._cluster_card_order),
                )
            top_prob = max(probs.values()) if probs else 0.0
            for cid, card in self._cluster_cards.items():
                c_prob = probs.get(cid)
                is_top = (c_prob is not None and top_prob > 0 and (c_prob >= top_prob - 0.15 or c_prob >= 0.25))
                card.set_candidate_probability(c_prob, is_top=is_top)
        else:
            if hasattr(self, "cluster_prob_bar"):
                self.cluster_prob_bar.clear()
            for card in self._cluster_cards.values():
                card.set_candidate_probability(None)

        # 4. Preview image or metadata in ImageViewer
        self._display_candidate_preview(gid)

        # 5. Update candidate PSTH overlay on cluster cards
        self._update_cluster_cards_candidate_psth(gid)

        # 6. Update Suggested Cell button state, Focused Unit header, and inspector member highlight
        self._update_focused_unit_header(gid)
        self._update_return_to_candidate_button()
        self._update_inspector_member_highlights()

        # 7. Update Status Label
        rec = self._pooled_units_map.get(gid, {})
        sess_id = rec.get("session_id", "")
        unit_id = rec.get("unit_id")
        sess_str, uid_str = _format_unit_display(sess_id, unit_id, gid)
        cid = rec.get("cluster_id", -1)
        cname = self._cluster_names.get(cid, f"Cluster {cid}") if cid >= 0 else "Unassigned"
        status = self._unit_status_map.get(gid, "Unassigned")
        if n_selected > 1:
            self.lbl_candidate.setText(
                f"{n_selected} Cells Selected  •  Primary: Session {sess_str}  •  Unit {uid_str}  •  Cluster: {cname}"
            )
        elif self.active_candidate and self.active_candidate.global_unit_id == gid:
            self.lbl_candidate.setText(
                f"Session: {sess_str}  •  Unit: {uid_str}  •  Cluster: {cname}  (Entropy: H={self.active_candidate.entropy:.2f})"
            )
        else:
            self.lbl_candidate.setText(
                f"Session: {sess_str}  •  Unit: {uid_str}  •  Cluster: {cname} ({status})"
            )

        self.unit_selected.emit(gid)

    def _show_pooled_units_context_menu(self, pos: QPoint) -> None:
        """Display right-click context menu on pooled units table for batch cluster assignment."""
        if not hasattr(self, "table_pooled_units") or self.table_pooled_units.selectionModel() is None:
            return

        selected_model_rows = self.table_pooled_units.selectionModel().selectedRows()
        clicked_row = self.table_pooled_units.rowAt(pos.y())
        if clicked_row >= 0:
            selected_row_indices = [idx.row() for idx in selected_model_rows]
            if clicked_row not in selected_row_indices:
                self.table_pooled_units.clearSelection()
                self.table_pooled_units.selectRow(clicked_row)
                selected_model_rows = self.table_pooled_units.selectionModel().selectedRows()

        target_gids: List[str] = []
        for model_idx in selected_model_rows:
            r = model_idx.row()
            gid = self._get_gid_for_row(r)
            if gid and gid in self._pooled_units_map:
                target_gids.append(gid)

        if not target_gids:
            return

        menu = QMenu(self)
        n = len(target_gids)
        title_str = f"Assign {n} Selected Cell{'s' if n != 1 else ''} To:"
        header_act = menu.addAction(title_str)
        header_act.setEnabled(False)
        menu.addSeparator()

        active_cids: List[int] = list(self._cluster_card_order)
        if self.current_clustering_result:
            for cid in self.current_clustering_result.cluster_summaries:
                if cid not in active_cids:
                    active_cids.append(cid)

        for cid in active_cids:
            cname = self._cluster_names.get(cid, f"Cluster {cid}")
            hotkey_val = self._get_cluster_hotkey(cid)
            hotkey_str = f" (Hotkey: {hotkey_val})" if hotkey_val else ""
            act_text = f"{cname}{hotkey_str}"

            color_hex = self._get_cluster_color(cid)

            pix = QPixmap(14, 14)
            pix.fill(QColor(color_hex))
            icon = QIcon(pix)

            act = menu.addAction(icon, act_text)
            act.triggered.connect(lambda checked, c=cid, g=list(target_gids): self.assign_candidate_to_cluster(c, unit_id=g))

        menu.addSeparator()
        act_new = menu.addAction("➕ New Cluster (Hotkey: N)")
        act_new.triggered.connect(lambda: self.create_new_cluster_for_candidate())

        act_unassign = menu.addAction("Unassign (-1)")
        act_unassign.triggered.connect(lambda checked, g=list(target_gids): self.assign_candidate_to_cluster(-1, unit_id=g))

        menu.exec(self.table_pooled_units.viewport().mapToGlobal(pos))

    def _load_mock_data(self) -> None:
        """Populate initial mock experiments, run initial clustering, and configure 2D projection."""
        self._is_mock = True
        # Mock Experiments
        mock_exps = [
            ("01942_74R_20261019", "OPTS03", "49 tagged"),
            ("02034_83L_20261020", "OPTS03", "28 tagged"),
            ("02035_83R_20261021", "OPTS03", "35 tagged"),
        ]
        self.table_exps.setRowCount(len(mock_exps))
        for row, (exp, proto, cells) in enumerate(mock_exps):
            self.table_exps.setItem(row, 0, QTableWidgetItem(exp))
            self.table_exps.setItem(row, 1, QTableWidgetItem(proto))
            sort_val = int(cells.split()[0]) if cells.split()[0].isdigit() else row
            self.table_exps.setItem(row, 2, NumericTableWidgetItem(cells, sort_key=sort_val))

        # Setup 3 distinct probe geometries for the mock sessions:
        # 1. 01942_74R_20261019: Neuropixels 2-column staggered array (64 sites)
        s1_sites = np.array(
            [[16.0 if i % 2 == 0 else 48.0, 100.0 + (i // 2) * 20.0] for i in range(64)],
            dtype=np.float64,
        )
        self._experiment_probes["01942_74R_20261019"] = s1_sites

        # 2. 02034_83L_20261020: Dual-shank array (shank 1 at x=0, shank 2 at x=250)
        s2_sites = np.array(
            [[0.0, 100.0 + i * 40.0] for i in range(32)] + [[250.0, 100.0 + i * 40.0] for i in range(32)],
            dtype=np.float64,
        )
        self._experiment_probes["02034_83L_20261020"] = s2_sites

        # 3. 02035_83R_20261021: 4-shank probe (shanks at x=0, 250, 500, 750 µm with 16 sites each, 64 sites total)
        s3_sites = np.array(
            [[0.0, 100.0 + i * 50.0] for i in range(16)]
            + [[250.0, 100.0 + i * 50.0] for i in range(16)]
            + [[500.0, 100.0 + i * 50.0] for i in range(16)]
            + [[750.0, 100.0 + i * 50.0] for i in range(16)],
            dtype=np.float64,
        )
        self._experiment_probes["02035_83R_20261021"] = s3_sites

        # Mock Pooled Units (Only confirmed tagged units in Phase 2)
        mock_units = [
            ("01942_74R_20261019_u0001", "01942_74R_20261019", 1, 3.4, 0.92, 0.0008, True, 8, 240.0, 16.0),
            ("01942_74R_20261019_u0004", "01942_74R_20261019", 4, 4.1, 0.88, 0.0015, True, 15, 380.0, 48.0),
            ("02034_83L_20261020_u0002", "02034_83L_20261020", 2, 2.9, 0.95, 0.0003, True, 5, 300.0, 0.0),
            ("02034_83L_20261020_u0007", "02034_83L_20261020", 7, 5.2, 0.74, 0.0062, True, 22, 980.0, 0.0),
            ("02034_83L_20261020_u0008", "02034_83L_20261020", 8, 5.5, 0.71, 0.0070, True, 42, 500.0, 250.0),
            ("02035_83R_20261021_u0003", "02035_83R_20261021", 3, 3.8, 0.89, 0.0011, True, 4, 300.0, 0.0),
            ("02035_83R_20261021_u0009", "02035_83R_20261021", 9, 8.5, 0.58, 0.0350, True, 20, 300.0, 250.0),
            ("02035_83R_20261021_u0010", "02035_83R_20261021", 10, 8.8, 0.55, 0.0380, True, 36, 300.0, 500.0),
            ("02035_83R_20261021_u0013", "02035_83R_20261021", 13, 3.2, 0.93, 0.0007, True, 52, 300.0, 750.0),
        ]

        # Populate deterministic synthetic spikes and stimulus onsets BEFORE table selection & clustering
        self._experiment_spikes.clear()
        self._experiment_stimulus.clear()
        self._unit_psth_cache.clear()
        self._unit_raster_psth_cache.clear()
        self._invalidate_clustering_caches()
        rng = np.random.RandomState(42)
        n_trials = 40
        mock_onsets = np.arange(1.0, 1.0 + n_trials * 0.5, 0.5, dtype=np.float64)
        for exp_id, _, _ in mock_exps:
            self._experiment_stimulus[exp_id] = mock_onsets.copy()
            self._experiment_spikes[exp_id] = {}

        for _gid, sess_id, uid, lat_ms, rel, _salt, _tag, _ch, _depth, _px in mock_units:
            jit_ms = 1.2
            spikes_list: List[float] = []
            for t_pulse in mock_onsets:
                n_bg = int(rng.poisson(2))
                if n_bg > 0:
                    spikes_list.extend((t_pulse + rng.uniform(-0.1, 0.1, size=n_bg)).tolist())
                if rng.rand() < rel:
                    ev_offset_s = float(rng.normal(lat_ms / 1000.0, jit_ms / 1000.0))
                    spikes_list.append(t_pulse + max(0.0005, ev_offset_s))
            self._experiment_spikes.setdefault(sess_id, {})[int(uid)] = np.sort(
                np.asarray(spikes_list, dtype=np.float64)
            )

        self._set_pooled_table_rows(mock_units)

        if self._start_with_clusters:
            # Run clustering on initial mock units
            self.run_clustering(self.active_clustering_params)
            for gid, rec in self._pooled_units_map.items():
                if rec.get("cluster_id", -1) >= 0:
                    self._unit_status_map[gid] = "Seeded"
                else:
                    self._unit_status_map[gid] = "Unassigned"
            self._update_all_pooled_cluster_items()
            self._populate_cluster_filter_options()
        else:
            self.reset_to_unassigned()

        # Initialize probe view with first unit if available
        if self._pooled_rows:
            first_gid = self._pooled_rows[0].get("global_unit_id")
            self._update_probe_view_for_unit(first_gid)

        self._undo_stack.clear()
        self._redo_stack.clear()
        if hasattr(self, "btn_undo"):
            self.btn_undo.setEnabled(False)
        if hasattr(self, "btn_redo"):
            self.btn_redo.setEnabled(False)
        self._set_dirty(False)
        self._schedule_background_plot_prewarm()
        self.experiments_changed.emit()

