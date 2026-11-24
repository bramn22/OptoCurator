"""Optotagged Cell Detection Workspace for single-session analysis and curation."""

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetrics, QKeySequence, QPainter, QPalette, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from optocurator.analysis.clustering import COLOR_UNASSIGNED
from optocurator.analysis.likelihood import CriterionSpec, compute_session_likelihoods
from optocurator.analysis.metrics import (
    compute_psth_and_raster,
    compute_session_metrics,
)
from optocurator.analysis.precompute import batch_precompute_plots
from optocurator.analysis.waveform import (
    extract_evoked_and_spontaneous_snippets,
    extract_session_waveforms,
)
from optocurator.io.nwb import (
    NwbSpikeReader,
    NwbStimulusReader,
    inspect_nwb,
    load_nwb_probe_geometry,
)
from optocurator.io.phy import PhyKilosortReader
from optocurator.io.pulses import MatNpyStimulusReader
from optocurator.models import (
    DEFAULT_SAMPLING_RATE_HZ,
    MISSING_METRIC_SENTINEL,
    StatsParameters,
    StimulusProtocol,
    UnitData,
    UnitMetrics,
    is_valid_metric_value,
)
from optocurator.ui.detection_widgets import (
    DetectionCriterionRow,
    DetectionPairplotWidget,
    DetectionWaveformPanel,
    install_combo_chevron,
)
from optocurator.ui.plots import (
    DEFAULT_PULSE_DURATION_MS,
    NumericTableWidgetItem,
    SiliconProbeWidget,
    infer_pulse_duration_ms,
    render_psth,
    render_raster,
)
from optocurator.ui.splitters import ResettableSplitter
from optocurator.ui.theme import (
    CLUSTERS_SECTION_SUBHEADER_STYLE,
    COLOR_ACCENT,
    COLOR_ACCENT_BG,
    COLOR_BORDER_STRONG,
    COLOR_DANGER,
    COLOR_DIVIDER,
    COLOR_PANEL,
    COLOR_SUCCESS,
    COLOR_SURFACE_MUTED,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_SECONDARY,
    DETECTION_COLUMN_PANEL_STYLE,
    DETECTION_COMBO_STYLE,
    DETECTION_CRITERIA_HELP_STYLE,
    DETECTION_NEXT_LIKELY_BTN_STYLE,
    DETECTION_OVERRIDE_BTN_STYLE,
    DETECTION_PLOT_LEFT_MARGIN,
    DETECTION_PROBE_LEGEND_HTML,
    DETECTION_PROBE_LEGEND_STYLE,
    DETECTION_PSTH_SIZE,
    DETECTION_RASTER_SIZE,
    DETECTION_ROW_TAGGED_BG,
    DETECTION_SAVE_BTN_STYLE,
    DETECTION_STATUS_CHIP_TAGGED_STYLE,
    DETECTION_STATUS_CHIP_UNTAGGED_STYLE,
    DETECTION_SUBPANEL_META_STYLE,
    DETECTION_SUBPANEL_TITLE_STYLE,
    DETECTION_SUBTLE_BTN_STYLE,
    DETECTION_SUMMARY_BOX_STYLE,
    DETECTION_SUMMARY_LABEL_STYLE,
    DETECTION_SUMMARY_VALUE_STYLE,
    DETECTION_TABLE_STYLE,
    DETECTION_UNIT_META_STYLE,
    DETECTION_UNIT_TITLE_STYLE,
    DETECTION_WINDOW_INPUT_STYLE,
    DIALOG_LINE_EDIT_STYLE,
    DIALOG_MUTED_LABEL_STYLE,
    PROGRESS_DIALOG_STYLE,
    TOOLBAR_LABEL_STYLE,
    TOOLBAR_SPIN_STYLE,
)

_COL_UNIT = 0
_COL_LAT = 1
_COL_JIT = 2
_COL_REL = 3
_COL_SALT = 4
_COL_WF = 5
_COL_LIK = 6
_COL_STATUS = 7

_DEFAULT_CRITERIA: Dict[str, Tuple[bool, float]] = {
    "latency": (True, 8.0),
    "jitter": (True, 2.0),
    "rel": (True, 0.60),
    "salt": (True, 0.05),
    "wf": (True, 0.90),
}

_LIKELY_THRESHOLD = 0.30
_WINDOW_MIN_SPAN_MS = 10.0
_WINDOW_MIN_START_MS = -500.0
_WINDOW_MAX_END_MS = 1000.0


class _FilterStatusComboBox(QComboBox):
    """QComboBox with backward-compatible lookup for legacy filter names."""

    _ALIASES = {
        "Tagged Only": "Tagged",
        "Untagged Only": "Untagged",
        "All Units": "All units",
    }

    def findText(self, text: str, flags: Qt.MatchFlag = Qt.MatchFlag.MatchExactly) -> int:  # noqa: N802
        mapped = self._ALIASES.get(text, text)
        return super().findText(mapped, flags)

    def setCurrentText(self, text: str) -> None:  # noqa: N802
        mapped = self._ALIASES.get(text, text)
        super().setCurrentText(mapped)


def _paint_table_cell_bg(
    painter: QPainter,
    option: QStyleOptionViewItem,
    index: Any,
) -> None:
    """Paint row background matching selection, focused, or tagged tint."""
    if option.state & QStyle.StateFlag.State_Selected:
        painter.fillRect(option.rect, QColor(COLOR_ACCENT_BG))
    else:
        bg = index.data(Qt.ItemDataRole.BackgroundRole)
        if isinstance(bg, QBrush) and bg.style() != Qt.BrushStyle.NoBrush:
            painter.fillRect(option.rect, bg)
        else:
            painter.fillRect(option.rect, QColor(COLOR_PANEL))
    painter.setPen(QColor(COLOR_DIVIDER))
    painter.drawLine(option.rect.bottomLeft(), option.rect.bottomRight())


class _LikelihoodDelegate(QStyledItemDelegate):
    """Draw a compact horizontal bar next to the percentage text in the Likelihood column."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: Any) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        _paint_table_cell_bg(painter, option, index)

        lik_raw = index.data(Qt.ItemDataRole.UserRole)
        try:
            lik = max(0.0, min(1.0, float(lik_raw))) if lik_raw is not None else 0.0
        except (TypeError, ValueError):
            lik = 0.0

        rect = option.rect
        bar_w = 22.0
        bar_h = 4.0
        bar_x = float(rect.left()) + 4.0
        bar_y = float(rect.center().y()) - bar_h * 0.5

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLOR_DIVIDER))
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 2.0, 2.0)

        if lik > 0.0:
            fill_w = max(2.0, bar_w * lik)
            painter.setBrush(QColor(COLOR_ACCENT))
            painter.drawRoundedRect(QRectF(bar_x, bar_y, fill_w, bar_h), 2.0, 2.0)

        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "0%")
        font_val = index.data(Qt.ItemDataRole.FontRole)
        if isinstance(font_val, QFont):
            painter.setFont(font_val)
        else:
            painter.setFont(option.font)
        painter.setPen(QColor(COLOR_TEXT))
        text_left = bar_x + bar_w + 4.0
        text_rect = QRectF(
            text_left,
            float(rect.top()),
            max(0.0, float(rect.right()) - text_left - 4.0),
            float(rect.height()),
        )
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            text,
        )
        painter.restore()


class _StatusDelegate(QStyledItemDelegate):
    """Draw green check / hollow grey circle plus grey 'M' when manually overridden."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: Any) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        _paint_table_cell_bg(painter, option, index)

        is_tagged = bool(index.data(Qt.ItemDataRole.UserRole))
        is_manual = bool(index.data(Qt.ItemDataRole.UserRole + 1))

        rect = option.rect
        font_val = index.data(Qt.ItemDataRole.FontRole)
        base_font = font_val if isinstance(font_val, QFont) else option.font

        symbol = "✓" if is_tagged else "○"
        sym_color = QColor(COLOR_SUCCESS if is_tagged else COLOR_TEXT_MUTED)

        if is_manual:
            sym_rect = QRectF(float(rect.left()), float(rect.top()), float(rect.width()) * 0.52, float(rect.height()))
            m_rect = QRectF(float(rect.left()) + float(rect.width()) * 0.50, float(rect.top()), float(rect.width()) * 0.45, float(rect.height()))
        else:
            sym_rect = QRectF(rect)
            m_rect = QRectF()

        sym_font = QFont(base_font)
        sym_font.setBold(is_tagged)
        painter.setFont(sym_font)
        painter.setPen(sym_color)
        painter.drawText(sym_rect, int(Qt.AlignmentFlag.AlignCenter), symbol)

        if is_manual:
            m_font = QFont(base_font)
            m_font.setBold(False)
            painter.setFont(m_font)
            painter.setPen(QColor(COLOR_TEXT_MUTED))
            painter.drawText(m_rect, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), "M")

        painter.restore()


def _attach_criterion_compat(row: DetectionCriterionRow) -> QLineEdit:
    """Attach QDoubleSpinBox-like `.value()` and `.setValue()` methods to a criterion edit box."""
    edit = row.edit_threshold

    def _value() -> float:
        return row.threshold()

    def _set_value(val: Union[float, int, str]) -> None:
        try:
            fval = float(val)
        except (ValueError, TypeError):
            return
        if row._is_log:
            fval = max(0.0, min(row._max_val, fval))
        else:
            fval = max(row._min_val, min(row._max_val, fval))
        row._threshold = fval
        edit.blockSignals(True)
        edit.setText(row._format_threshold(fval))
        edit.blockSignals(False)
        row.threshold_changed.emit(fval)

    edit.value = _value  # type: ignore[attr-defined]
    edit.setValue = _set_value  # type: ignore[attr-defined]
    return edit


class DetectionView(QWidget):
    """Workspace for optotagged cell detection, threshold tuning, and single-session curation."""

    request_home = pyqtSignal()
    request_clustering_view = pyqtSignal()
    view_settings_changed = pyqtSignal(tuple, str)
    dirty_changed = pyqtSignal(bool)
    session_changed = pyqtSignal()
    protocols_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None, load_mock: bool = False) -> None:
        super().__init__(parent)
        self.setObjectName("detectionWorkspace")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)

        # Readers and session state
        self.phy_reader = PhyKilosortReader()
        self.stim_reader = MatNpyStimulusReader()

        self.current_folder: Optional[Path] = None
        self.current_nwb_path: Optional[Path] = None
        self.session_id: str = "None"
        self.units: List[UnitData] = []
        self.units_by_id: Dict[int, UnitData] = {}
        self.protocols: Dict[str, StimulusProtocol] = {}
        self.current_protocol: Optional[StimulusProtocol] = None
        self.metrics: Dict[int, UnitMetrics] = {}
        self.manual_overrides: Dict[int, bool] = {}

        # Analytical metric hyperparameters
        self.stats_params: StatsParameters = StatsParameters.default()
        self._stats_params_customized: bool = False

        # Visualization configuration state
        self.current_window_ms: Tuple[float, float] = (-30.0, 50.0)
        self.raster_order_by: str = "presentation"  # "presentation" or "latency"
        self.raster_marker_size: float = 3.0
        self.psth_smooth: bool = True
        self.psth_gaussian_sigma_ms: float = 2.0

        # Curation & undo state
        self._is_dirty: bool = False
        self._focused_unit_id: Optional[int] = None
        self._criteria_undo_stack: List[Dict[str, Any]] = []
        self._criteria_state: Dict[str, Tuple[bool, float]] = dict(_DEFAULT_CRITERIA)
        self._updating_criteria_ui: bool = False
        self._channel_positions: Optional[np.ndarray] = None

        self._init_ui()
        self._init_shortcuts()
        self._update_criteria_visuals()
        self._update_summary_box()

        if load_mock:
            self._load_mock_data()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Hidden compatibility widgets (synced with MainWindow header)
        self.btn_load_folder = QPushButton("Load Experiment Folder…", self)
        self.btn_load_folder.hide()
        self.btn_load_folder.clicked.connect(lambda: self.load_experiment_folder())

        self.lbl_session = QLabel(f"Session: {self.session_id}", self)
        self.lbl_session.hide()

        self.combo_proto = QComboBox(self)
        self.combo_proto.hide()
        self.combo_proto.currentIndexChanged.connect(self._on_protocol_selected)

        # Hidden compatibility PSTH smooth/sigma controls (driven via View -> Configure visualizations…)
        self.chk_smooth = QCheckBox("Smooth", self)
        self.chk_smooth.setChecked(True)
        self.chk_smooth.hide()
        self.chk_smooth.toggled.connect(self.set_psth_smooth)

        self.spin_sigma = QDoubleSpinBox(self)
        self.spin_sigma.setRange(0.2, 20.0)
        self.spin_sigma.setSingleStep(0.5)
        self.spin_sigma.setValue(2.0)
        self.spin_sigma.setSuffix(" ms")
        self.spin_sigma.setStyleSheet(TOOLBAR_SPIN_STYLE)
        self.spin_sigma.hide()
        self.spin_sigma.valueChanged.connect(self.set_psth_gaussian_sigma)

        # 3-region ResettableSplitter (§ 13.1, § 14 WP9)
        self.main_splitter = ResettableSplitter(
            Qt.Orientation.Horizontal,
            parent=self,
            default_sizes=[380, 1120, 420],
            settings_key="detection/splitter_sizes",
        )
        main_layout.addWidget(self.main_splitter)

        self.left_col = self._build_left_column()
        self.center_col = self._build_center_column()
        self.right_col = self._build_right_column()

        self.main_splitter.addWidget(self.left_col)
        self.main_splitter.addWidget(self.center_col)
        self.main_splitter.addWidget(self.right_col)

        if not self.main_splitter.restore_from_settings():
            self.main_splitter.setSizes([380, 1120, 420])

    def _build_left_column(self) -> QWidget:
        """Build Left Column: Units table, filter row, Next likely button, and Probe (§ 13.4)."""
        col = QWidget(self)
        col.setObjectName("detectionColumnPanel")
        col.setStyleSheet(DETECTION_COLUMN_PANEL_STYLE)
        col.setMinimumWidth(280)

        layout = QVBoxLayout(col)
        layout.setContentsMargins(12, 10, 10, 10)
        layout.setSpacing(8)

        # Filter row: "Show" + combo_filter_status + lbl_unit_summary + Next likely button
        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.setSpacing(6)

        lbl_show = QLabel("Show", col)
        lbl_show.setStyleSheet(TOOLBAR_LABEL_STYLE)
        filter_row.addWidget(lbl_show)

        self.combo_filter_status = _FilterStatusComboBox(col)
        self.combo_filter_status.addItems([
            "All units",
            "Tagged",
            "Untagged",
            "Passing current criteria",
            "Changed manually",
            "Likely but untagged",
        ])
        self.combo_filter_status.setToolTip("Filter displayed units by tagging status or criteria")
        self.combo_filter_status.setStyleSheet(DETECTION_COMBO_STYLE)
        install_combo_chevron(self.combo_filter_status)
        self.combo_filter_status.currentIndexChanged.connect(self._apply_status_filter)
        filter_row.addWidget(self.combo_filter_status, stretch=1)

        self.lbl_unit_summary = QLabel("0 units", col)
        self.lbl_unit_summary.setStyleSheet(DIALOG_MUTED_LABEL_STYLE)
        filter_row.addWidget(self.lbl_unit_summary)

        layout.addLayout(filter_row)

        # 8-column units table
        self.table_units = QTableWidget(col)
        self.table_units.setColumnCount(8)
        headers = ["Unit", "Lat.", "Jit.", "Rel.", "SALT p", "WF r", "Likelihood", "Status"]
        header_tooltips = [
            "Unit ID",
            "Latency (ms)",
            "Jitter (ms)",
            "Reliability",
            "SALT p-value",
            "Waveform correlation (r)",
            "Likelihood of being tagged",
            "Tagging status",
        ]
        self.table_units.setHorizontalHeaderLabels(headers)
        for idx, tip in enumerate(header_tooltips):
            item = self.table_units.horizontalHeaderItem(idx)
            if item is not None:
                item.setToolTip(tip)

        self.table_units.verticalHeader().setVisible(False)
        self.table_units.verticalHeader().setDefaultSectionSize(22)
        self.table_units.setShowGrid(False)
        self.table_units.setStyleSheet(DETECTION_TABLE_STYLE)
        h_header = self.table_units.horizontalHeader()
        h_header.setMinimumSectionSize(30)
        h_header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        h_header.setStretchLastSection(True)
        col_widths = {0: 54, 1: 39, 2: 37, 3: 40, 4: 49, 5: 37, 6: 59, 7: 40}
        for col_idx, width in col_widths.items():
            self.table_units.setColumnWidth(col_idx, width)
        self.table_units.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table_units.setItemDelegateForColumn(_COL_LIK, _LikelihoodDelegate(self.table_units))
        self.table_units.setItemDelegateForColumn(_COL_STATUS, _StatusDelegate(self.table_units))
        h_header.setSectionsClickable(True)
        h_header.setSortIndicatorShown(True)
        self.table_units.setSortingEnabled(True)
        h_header.sectionClicked.connect(self._on_header_section_clicked)
        self.table_units.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_units.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table_units.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_units.itemSelectionChanged.connect(self._on_unit_selected)
        self.table_units.itemDoubleClicked.connect(self._on_table_double_clicked)
        layout.addWidget(self.table_units, stretch=1)

        # Probe section header + Silicon Probe + colour key to its right
        probe_section = QVBoxLayout()
        probe_section.setContentsMargins(0, 4, 0, 0)
        probe_section.setSpacing(4)

        lbl_probe_hdr = QLabel("Probe", col)
        lbl_probe_hdr.setStyleSheet(CLUSTERS_SECTION_SUBHEADER_STYLE)
        probe_section.addWidget(lbl_probe_hdr)

        probe_row = QHBoxLayout()
        probe_row.setContentsMargins(0, 0, 0, 0)
        probe_row.setSpacing(10)

        self.probe_widget = SiliconProbeWidget(col)
        self.probe_widget.units_scatter.sigClicked.connect(self._on_probe_unit_clicked)
        probe_row.addWidget(self.probe_widget)

        probe_key_col = QVBoxLayout()
        probe_key_col.setContentsMargins(0, 6, 0, 6)
        probe_key_col.setSpacing(4)
        self.lbl_probe_key = QLabel(DETECTION_PROBE_LEGEND_HTML, col)
        self.lbl_probe_key.setStyleSheet(DETECTION_PROBE_LEGEND_STYLE)
        probe_key_col.addWidget(self.lbl_probe_key)
        probe_key_col.addStretch()

        probe_row.addLayout(probe_key_col)
        probe_row.addStretch()
        probe_section.addLayout(probe_row)
        layout.addLayout(probe_section)

        return col

    def _build_center_column(self) -> QWidget:
        """Build Centre Column: Focused Unit header, display controls, Raster+PSTH, Waveforms, Pairplot (§ 13.5)."""
        col = QWidget(self)
        col.setObjectName("detectionColumnPanel")
        col.setStyleSheet(DETECTION_COLUMN_PANEL_STYLE)
        col.setMinimumWidth(560)

        layout = QVBoxLayout(col)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        # Header: title + meta + status chip + Mark as tagged/untagged
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(10)

        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(2)

        self.lbl_focused_unit_title = QLabel("No unit selected", col)
        self.lbl_focused_unit_title.setStyleSheet(DETECTION_UNIT_TITLE_STYLE)
        title_col.addWidget(self.lbl_focused_unit_title)

        self.lbl_focused_unit_meta = QLabel("Load an experiment folder to inspect units", col)
        self.lbl_focused_unit_meta.setStyleSheet(DETECTION_UNIT_META_STYLE)
        title_col.addWidget(self.lbl_focused_unit_meta)

        header_row.addLayout(title_col)
        header_row.addStretch()

        self.lbl_focused_status = QLabel("○ Untagged", col)
        self.lbl_focused_status.setStyleSheet(DETECTION_STATUS_CHIP_UNTAGGED_STYLE)
        header_row.addWidget(self.lbl_focused_status)

        self.btn_toggle_tag = QPushButton("Mark as tagged", col)
        self.btn_toggle_tag.setStyleSheet(DETECTION_OVERRIDE_BTN_STYLE)
        self.btn_toggle_tag.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_tag.setToolTip("Manually toggle tagged status for focused unit (Shortcut: T)")
        self.btn_toggle_tag.clicked.connect(lambda: self._toggle_selected_unit_tag())
        header_row.addWidget(self.btn_toggle_tag)

        layout.addLayout(header_row)

        # Display controls row (display-only)
        ctrl_row = QHBoxLayout()
        ctrl_row.setContentsMargins(0, 0, 0, 0)
        ctrl_row.setSpacing(6)

        lbl_win = QLabel("Window", col)
        lbl_win.setStyleSheet(DETECTION_SUBPANEL_META_STYLE)
        lbl_win.setToolTip("Display only; does not change latency, jitter, reliability, or SALT p metrics.")
        ctrl_row.addWidget(lbl_win)

        self.edit_win_start = QLineEdit("-30", col)
        self.edit_win_start.setFixedWidth(48)
        self.edit_win_start.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.edit_win_start.setStyleSheet(DETECTION_WINDOW_INPUT_STYLE)
        self.edit_win_start.setToolTip("Display window start (ms, display only; does not change metrics).")
        self.edit_win_start.editingFinished.connect(self._on_window_text_edited)
        self.edit_win_start.returnPressed.connect(self._on_window_text_edited)
        ctrl_row.addWidget(self.edit_win_start)

        lbl_to = QLabel("to", col)
        lbl_to.setStyleSheet(DETECTION_SUBPANEL_META_STYLE)
        ctrl_row.addWidget(lbl_to)

        self.edit_win_end = QLineEdit("+50", col)
        self.edit_win_end.setFixedWidth(48)
        self.edit_win_end.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.edit_win_end.setStyleSheet(DETECTION_WINDOW_INPUT_STYLE)
        self.edit_win_end.setToolTip("Display window end (ms, display only; does not change metrics).")
        self.edit_win_end.editingFinished.connect(self._on_window_text_edited)
        self.edit_win_end.returnPressed.connect(self._on_window_text_edited)
        ctrl_row.addWidget(self.edit_win_end)

        lbl_ms = QLabel("ms", col)
        lbl_ms.setStyleSheet(DETECTION_SUBPANEL_META_STYLE)
        ctrl_row.addWidget(lbl_ms)

        ctrl_row.addSpacing(12)

        lbl_order = QLabel("Pulse order", col)
        lbl_order.setStyleSheet(DETECTION_SUBPANEL_META_STYLE)
        lbl_order.setToolTip("Display only; does not change metrics.")
        ctrl_row.addWidget(lbl_order)

        self.combo_order = QComboBox(col)
        self.combo_order.addItem("Presentation order", "presentation")
        self.combo_order.addItem("By first-spike latency", "latency")
        self.combo_order.setStyleSheet(DETECTION_COMBO_STYLE)
        install_combo_chevron(self.combo_order)
        self.combo_order.setToolTip("Pulse trial ordering in the raster plot (display only).")
        self.combo_order.currentIndexChanged.connect(self._on_combo_order_changed)
        ctrl_row.addWidget(self.combo_order)

        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # Plot area: two side-by-side columns (left: Raster + PSTH, right: Waveforms + Pairplot)
        # Plot area inside a scroll guard so fixed plots never clip each other at 1366x768
        plot_scroll = QScrollArea(col)
        plot_scroll.setWidgetResizable(True)
        plot_scroll.setFrameShape(QFrame.Shape.NoFrame)
        plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        plot_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        plot_container = QWidget(plot_scroll)
        plot_area = QHBoxLayout(plot_container)
        plot_area.setContentsMargins(0, 4, 8, 0)
        plot_area.setSpacing(14)

        pg.setConfigOption("background", COLOR_PANEL)
        pg.setConfigOption("foreground", COLOR_TEXT_SECONDARY)
        pg.setConfigOptions(antialias=True)

        raster_w = DETECTION_RASTER_SIZE[0] + DETECTION_PLOT_LEFT_MARGIN
        total_plot_h = DETECTION_RASTER_SIZE[1] + DETECTION_PSTH_SIZE[1] + 28
        self.plot_canvas = pg.GraphicsLayoutWidget(col)
        self.plot_canvas.setBackground(COLOR_PANEL)
        self.plot_canvas.setFixedSize(raster_w, total_plot_h)
        self.plot_canvas.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.raster_plot = self.plot_canvas.addPlot(row=0, col=0)
        self.raster_plot.setTitle(None)
        self.raster_plot.getPlotItem = lambda: self.raster_plot

        self.psth_plot = self.plot_canvas.addPlot(row=1, col=0)
        self.psth_plot.getPlotItem = lambda: self.psth_plot

        self.raster_plot.getAxis("left").setWidth(DETECTION_PLOT_LEFT_MARGIN)
        self.psth_plot.getAxis("left").setWidth(DETECTION_PLOT_LEFT_MARGIN)
        self.psth_plot.setXLink(self.raster_plot)
        self.raster_plot.hideAxis("bottom")
        self.raster_plot.setXRange(self.current_window_ms[0], self.current_window_ms[1], padding=0.0)
        self.raster_plot.getViewBox().disableAutoRange(pg.ViewBox.XAxis)

        self.plot_canvas.ci.layout.setSpacing(4)
        self.plot_canvas.ci.layout.setRowStretchFactor(0, DETECTION_RASTER_SIZE[1])
        self.plot_canvas.ci.layout.setRowStretchFactor(1, DETECTION_PSTH_SIZE[1])

        left_plot_col = QVBoxLayout()
        left_plot_col.setContentsMargins(0, 0, 0, 0)
        left_plot_col.addWidget(self.plot_canvas)
        left_plot_col.addStretch()
        plot_area.addLayout(left_plot_col)

        right_plot_col = QVBoxLayout()
        right_plot_col.setContentsMargins(0, 0, 0, 0)
        right_plot_col.setSpacing(12)

        self.waveform_panel = DetectionWaveformPanel(col)
        self.waveform_panel.setVisible(False)
        right_plot_col.addWidget(self.waveform_panel)

        self.pairplot_widget = DetectionPairplotWidget(col)
        self.pairplot_widget.unit_clicked.connect(self._focus_unit_by_id)
        right_plot_col.addWidget(self.pairplot_widget)
        right_plot_col.addStretch()

        plot_area.addLayout(right_plot_col)
        plot_area.addStretch()

        plot_scroll.setWidget(plot_container)
        layout.addWidget(plot_scroll, stretch=1)
        return col

    def _build_right_column(self) -> QWidget:
        """Build Right Column: Tagging criteria, 5 DetectionCriterionRows, summary box, Save button (§ 13.6)."""
        col = QWidget(self)
        col.setObjectName("detectionColumnPanel")
        col.setStyleSheet(DETECTION_COLUMN_PANEL_STYLE)
        col.setMinimumWidth(340)

        layout = QVBoxLayout(col)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header row: "Tagging criteria" + Undo + Reset to defaults
        hdr_row = QHBoxLayout()
        hdr_row.setContentsMargins(0, 0, 0, 0)
        hdr_row.setSpacing(6)

        lbl_title = QLabel("Tagging criteria", col)
        lbl_title.setStyleSheet(DETECTION_SUBPANEL_TITLE_STYLE)
        hdr_row.addWidget(lbl_title)
        hdr_row.addStretch()

        self.btn_criteria_undo = QPushButton("Undo", col)
        self.btn_criteria_undo.setStyleSheet(DETECTION_SUBTLE_BTN_STYLE)
        self.btn_criteria_undo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_criteria_undo.setToolTip("Undo last criteria change (Ctrl+Z)")
        self.btn_criteria_undo.setEnabled(False)
        self.btn_criteria_undo.clicked.connect(self.undo_criteria_change)
        hdr_row.addWidget(self.btn_criteria_undo)

        self.btn_criteria_reset = QPushButton("Reset to defaults", col)
        self.btn_criteria_reset.setStyleSheet(DETECTION_SUBTLE_BTN_STYLE)
        self.btn_criteria_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_criteria_reset.setToolTip("Reset all 5 tagging criteria to default values")
        self.btn_criteria_reset.clicked.connect(self.reset_criteria_to_defaults)
        hdr_row.addWidget(self.btn_criteria_reset)

        layout.addLayout(hdr_row)

        # Help line
        lbl_help = QLabel("Units passing every enabled criterion are tagged. Changes apply immediately.", col)
        lbl_help.setWordWrap(True)
        lbl_help.setStyleSheet(DETECTION_CRITERIA_HELP_STYLE)
        layout.addWidget(lbl_help)

        # Align top of first criterion row with top edge of raster plot_canvas
        layout.addSpacing(16)

        # Five DetectionCriterionRow widgets
        self.row_latency = DetectionCriterionRow(
            "Max. latency",
            operator="≤",
            unit="ms",
            axis_range=(0.0, 20.0),
            ticks=[0, 5, 10, 15, 20],
            default_threshold=8.0,
            min_val=0.1,
            max_val=100.0,
            decimals=1,
            parent=col,
        )
        self.row_jitter = DetectionCriterionRow(
            "Max. jitter",
            operator="≤",
            unit="ms",
            axis_range=(0.0, 6.0),
            ticks=[0, 2, 4, 6],
            default_threshold=2.0,
            min_val=0.01,
            max_val=50.0,
            decimals=1,
            parent=col,
        )
        self.row_rel = DetectionCriterionRow(
            "Min. reliability",
            operator="≥",
            unit="",
            axis_range=(0.0, 1.0),
            ticks=[0, 0.25, 0.5, 0.75, 1],
            default_threshold=0.60,
            min_val=0.0,
            max_val=1.0,
            decimals=2,
            parent=col,
        )
        self.row_salt = DetectionCriterionRow(
            "Max. SALT p",
            operator="≤",
            unit="",
            axis_range=(0.001, 1.0),
            ticks=[0.001, 0.01, 0.1, 1],
            default_threshold=0.05,
            min_val=0.0001,
            max_val=1.0,
            decimals=4,
            is_log=True,
            parent=col,
        )
        self.row_wf = DetectionCriterionRow(
            "Min. waveform r",
            operator="≥",
            unit="",
            axis_range=(0.4, 1.0),
            ticks=[0.4, 0.6, 0.8, 1],
            default_threshold=0.90,
            min_val=0.0,
            max_val=1.0,
            decimals=2,
            parent=col,
        )
        self.row_wf.setToolTip(
            "Units without extractable waveforms are not tested on waveform correlation."
        )

        self._criterion_rows: Dict[str, DetectionCriterionRow] = {
            "latency": self.row_latency,
            "jitter": self.row_jitter,
            "rel": self.row_rel,
            "salt": self.row_salt,
            "wf": self.row_wf,
        }

        for row_widget in self._criterion_rows.values():
            layout.addWidget(row_widget)
            row_widget.enabled_changed.connect(self._on_criteria_ui_changed)
            row_widget.threshold_changed.connect(self._on_criteria_ui_changed)

        # Backward-compat aliases
        self.chk_latency = self.row_latency.chk_enabled
        self.edit_latency = _attach_criterion_compat(self.row_latency)
        self.spin_latency = self.edit_latency

        self.chk_jitter = self.row_jitter.chk_enabled
        self.edit_jitter = _attach_criterion_compat(self.row_jitter)
        self.spin_jitter = self.edit_jitter

        self.chk_rel = self.row_rel.chk_enabled
        self.edit_rel = _attach_criterion_compat(self.row_rel)
        self.spin_rel = self.edit_rel

        self.chk_salt = self.row_salt.chk_enabled
        self.edit_salt = _attach_criterion_compat(self.row_salt)
        self.spin_salt = self.edit_salt

        self.chk_wf = self.row_wf.chk_enabled
        self.edit_wf = _attach_criterion_compat(self.row_wf)
        self.spin_wf = self.edit_wf

        self.chk_wv = self.chk_salt
        self.spin_wv = self.spin_salt

        layout.addSpacing(16)

        # Summary box (4 clean rows, 16 px below last criterion row)
        self.summary_box = QFrame(col)
        self.summary_box.setObjectName("detectionSummaryBox")
        self.summary_box.setStyleSheet(DETECTION_SUMMARY_BOX_STYLE)
        sum_layout = QVBoxLayout(self.summary_box)
        sum_layout.setContentsMargins(10, 8, 10, 8)
        sum_layout.setSpacing(4)

        def _make_summary_row(
            label_text: str,
            dot_color: Optional[str] = None,
        ) -> Tuple[QHBoxLayout, QLabel]:
            r = QHBoxLayout()
            r.setContentsMargins(0, 0, 0, 0)
            r.setSpacing(6)
            if dot_color is not None:
                dot = QLabel("●", self.summary_box)
                dot_pal = dot.palette()
                dot_pal.setColor(QPalette.ColorRole.WindowText, QColor(dot_color))
                dot.setPalette(dot_pal)
                r.addWidget(dot)
            lbl = QLabel(label_text, self.summary_box)
            lbl.setStyleSheet(DETECTION_SUMMARY_LABEL_STYLE)
            val = QLabel("0", self.summary_box)
            val.setStyleSheet(DETECTION_SUMMARY_VALUE_STYLE)
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            r.addWidget(lbl)
            r.addStretch()
            r.addWidget(val)
            return r, val

        r_tagged, self.lbl_summary_tagged = _make_summary_row("Tagged", COLOR_SUCCESS)
        sum_layout.addLayout(r_tagged)

        r_untagged, self.lbl_summary_untagged = _make_summary_row("Untagged", COLOR_TEXT_MUTED)
        sum_layout.addLayout(r_untagged)

        divider = QFrame(self.summary_box)
        divider.setFrameShape(QFrame.Shape.NoFrame)
        divider.setFixedHeight(1)
        divider.setAutoFillBackground(True)
        div_pal = divider.palette()
        div_pal.setColor(QPalette.ColorRole.Window, QColor(COLOR_DIVIDER))
        divider.setPalette(div_pal)
        sum_layout.addWidget(divider)

        r_total, self.lbl_summary_total = _make_summary_row("Units in session")
        sum_layout.addLayout(r_total)

        r_manual, self.lbl_summary_manual = _make_summary_row("Changed manually")
        sum_layout.addLayout(r_manual)

        # Hidden backward-compat summary label
        self.lbl_summary_counts = QLabel(
            "Tagged: <b>0</b> · Untagged: <b>0</b><br>"
            "Units in session: <b>0</b> · Changed manually: <b>0</b>",
            self.summary_box,
        )
        self.lbl_summary_counts.hide()
        sum_layout.addWidget(self.lbl_summary_counts)
        layout.addWidget(self.summary_box)

        layout.addSpacing(12)

        # Primary Save button below summary box
        save_row = QHBoxLayout()
        save_row.setContentsMargins(0, 0, 0, 0)
        save_row.addStretch()

        self.btn_save = QPushButton("Save", col)
        self.btn_save.setStyleSheet(DETECTION_SAVE_BTN_STYLE)
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.setToolTip("Save curated detection session CSV and tagged spikes (Ctrl+S)")
        self.btn_save.clicked.connect(
            lambda: self.export_curated_session(show_dialog=True, precompute_pngs=False)
        )
        save_row.addWidget(self.btn_save)
        layout.addLayout(save_row)

        layout.addStretch()

        self.btn_export = self.btn_save
        return col

    def _init_shortcuts(self) -> None:
        self._shortcut_toggle_tag = QShortcut(QKeySequence("T"), self)
        self._shortcut_toggle_tag.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._shortcut_toggle_tag.activated.connect(self._toggle_selected_unit_tag)

        self._shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self._shortcut_undo.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._shortcut_undo.activated.connect(self.undo_criteria_change)

    # --- Layout & Dirty State ---

    def reset_default_layout(self) -> None:
        """Reset 3-column detection splitter to default widths."""
        if hasattr(self, "main_splitter"):
            self.main_splitter.reset_to_default()

    def _set_dirty(self, dirty: bool) -> None:
        dirty_bool = bool(dirty)
        if self._is_dirty != dirty_bool:
            self._is_dirty = dirty_bool
            self.dirty_changed.emit(dirty_bool)

    # --- Criteria State & Undo ---

    def _read_criteria_from_ui(self) -> Dict[str, Tuple[bool, float]]:
        return {
            key: (row.is_enabled(), row.threshold())
            for key, row in self._criterion_rows.items()
        }

    def _write_criteria_to_ui(self, state: Dict[str, Tuple[bool, float]]) -> None:
        self._updating_criteria_ui = True
        try:
            for key, (enabled, thresh) in state.items():
                row = self._criterion_rows.get(key)
                if row is None:
                    continue
                row.chk_enabled.blockSignals(True)
                row.chk_enabled.setChecked(bool(enabled))
                row.chk_enabled.blockSignals(False)
                row._enabled = bool(enabled)
                row._threshold = float(thresh)
                row.edit_threshold.blockSignals(True)
                row.edit_threshold.setText(row._format_threshold(float(thresh)))
                row.edit_threshold.blockSignals(False)
                row._apply_enabled_visuals(bool(enabled))
            self._criteria_state = {k: (bool(v[0]), float(v[1])) for k, v in state.items()}
        finally:
            self._updating_criteria_ui = False

    def _push_detection_undo_snapshot(self) -> None:
        """Push current criteria thresholds and manual overrides onto the Detection undo stack."""
        self._criteria_undo_stack.append({
            "criteria": dict(self._criteria_state),
            "manual_overrides": dict(self.manual_overrides),
        })
        self.btn_criteria_undo.setEnabled(len(self._criteria_undo_stack) > 0)

    def _on_criteria_ui_changed(self, *_args: Any) -> None:
        if self._updating_criteria_ui:
            return
        new_state = self._read_criteria_from_ui()
        if new_state != self._criteria_state:
            self._push_detection_undo_snapshot()
            self._criteria_state = new_state
        self._apply_thresholds()
        if self.units or self.table_units.rowCount() > 0:
            self._set_dirty(True)

    def undo_criteria_change(self) -> None:
        """Step back to the previous tagging criteria and manual-override state."""
        if not self._criteria_undo_stack:
            return
        prev = self._criteria_undo_stack.pop()
        self.btn_criteria_undo.setEnabled(len(self._criteria_undo_stack) > 0)
        if "criteria" in prev and "manual_overrides" in prev:
            prev_criteria = prev["criteria"]
            self.manual_overrides.clear()
            self.manual_overrides.update(prev["manual_overrides"])
        else:
            prev_criteria = prev
        self._write_criteria_to_ui(prev_criteria)
        self._apply_thresholds()
        if self.units or self.table_units.rowCount() > 0:
            self._set_dirty(True)

    def reset_criteria_to_defaults(self) -> None:
        """Reset all 5 criteria to default thresholds and enabled state."""
        new_state = dict(_DEFAULT_CRITERIA)
        if new_state != self._criteria_state:
            self._push_detection_undo_snapshot()
        self._write_criteria_to_ui(new_state)
        self._apply_thresholds()
        if self.units or self.table_units.rowCount() > 0:
            self._set_dirty(True)

    # --- Real Data Loading and Protocol Management ---

    def load_nwb_file(
        self,
        nwb_path: Optional[Union[str, Path, bool]] = None,
        probe_name: Optional[str] = None,
        qc_only: Optional[bool] = None,
        expand_pulse_trains: Optional[bool] = None,
    ) -> bool:
        """Load an NWB recording file (e.g. Allen OpenScope Psycode) into DetectionView."""
        if nwb_path is None or isinstance(nwb_path, bool):
            selected, _ = QFileDialog.getOpenFileName(
                self,
                "Select NWB Recording File",
                "",
                "NWB Files (*.nwb);;All Files (*)",
            )
            if not selected:
                return False
            nwb_path = Path(selected)
        else:
            nwb_path = Path(nwb_path)

        if not nwb_path.is_file():
            QMessageBox.warning(self, "Invalid NWB File", f"The NWB file does not exist: {nwb_path}")
            return False

        try:
            summary = inspect_nwb(nwb_path)
        except Exception as exc:
            QMessageBox.critical(self, "NWB Inspection Error", f"Failed to inspect NWB file:\n{exc}")
            return False

        if summary.n_total_units == 0:
            QMessageBox.warning(self, "No Units Found", f"No sorted units found in: {nwb_path}")
            return False

        if probe_name is None or qc_only is None or expand_pulse_trains is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("Import NWB Session")
            dlg.setMinimumWidth(380)
            form = QFormLayout(dlg)

            lbl_sess = QLabel(summary.session_id, dlg)
            form.addRow("Session:", lbl_sess)

            combo_probe = QComboBox(dlg)
            for p_name, total_cnt in summary.probe_unit_counts.items():
                qc_cnt = summary.probe_qc_counts.get(p_name, total_cnt)
                combo_probe.addItem(f"{p_name} ({qc_cnt} QC / {total_cnt} total)", p_name)
            if len(summary.probe_unit_counts) > 1:
                total_qc = sum(summary.probe_qc_counts.values())
                combo_probe.addItem(f"All probes ({total_qc} QC / {summary.n_total_units} total)", "All")
            if probe_name is not None:
                idx = combo_probe.findData(probe_name)
                if idx >= 0:
                    combo_probe.setCurrentIndex(idx)
            form.addRow("Probe:", combo_probe)

            chk_qc = QCheckBox("Only default_qc == True units", dlg)
            chk_qc.setChecked(True if qc_only is None else bool(qc_only))
            form.addRow("", chk_qc)

            chk_expand = QCheckBox("Expand pulse trains into individual pulses", dlg)
            chk_expand.setChecked(True if expand_pulse_trains is None else bool(expand_pulse_trains))
            form.addRow("", chk_expand)

            if summary.protocols:
                proto_txt = ", ".join(f"{p.display_name} ({p.n_intervals})" for p in summary.protocols)
            else:
                proto_txt = "None detected"
            lbl_protos = QLabel(proto_txt, dlg)
            lbl_protos.setWordWrap(True)
            form.addRow("Protocols:", lbl_protos)

            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
                parent=dlg,
            )
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)
            form.addRow(buttons)

            if dlg.exec() != QDialog.DialogCode.Accepted:
                return False

            probe_name = str(combo_probe.currentData() or "ProbeA")
            qc_only = chk_qc.isChecked()
            expand_pulse_trains = chk_expand.isChecked()

        progress = QProgressDialog("Loading NWB session...", None, 0, 100, self)
        progress.setWindowTitle("Loading NWB File")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(True)
        progress.setStyleSheet(PROGRESS_DIALOG_STYLE)
        progress.setValue(10)
        progress.setLabelText(f"Reading units and spike times for {probe_name}...")
        QApplication.processEvents()

        try:
            sess_label = summary.session_id
            if probe_name and probe_name != "All":
                sess_label = f"{summary.session_id}_{probe_name}"

            reader = NwbSpikeReader(probe_name=probe_name, qc_only=bool(qc_only))
            units = reader.load(nwb_path, session_id=sess_label)
            if not units:
                progress.close()
                QMessageBox.warning(
                    self,
                    "No Units Found",
                    f"No matching units found for probe '{probe_name}' (qc_only={qc_only}).",
                )
                return False

            progress.setValue(35)
            progress.setLabelText(f"Loaded {len(units)} units. Loading optotagging protocols...")
            QApplication.processEvents()

            self.current_folder = nwb_path.parent
            self.current_nwb_path = nwb_path
            self.session_id = units[0].session_id
            self.units = units
            self.units_by_id = {u.unit_id: u for u in units}
            self.metrics.clear()
            self.manual_overrides.clear()
            self._criteria_undo_stack.clear()
            self.btn_criteria_undo.setEnabled(False)
            self._focused_unit_id = None
            self._disk_waveform_unit_ids = set()
            self.lbl_session.setText(f"Session: {self.session_id}")
            self.lbl_session.setToolTip(str(nwb_path))

            channel_pos, _ = load_nwb_probe_geometry(nwb_path, probe_name=probe_name)
            self._channel_positions = channel_pos

            stim_reader = NwbStimulusReader(expand_pulse_trains=bool(expand_pulse_trains))
            protos = stim_reader.load_all(nwb_path)
            self.protocols = {}
            self.combo_proto.blockSignals(True)
            self.combo_proto.clear()
            if protos:
                for proto in protos:
                    self.protocols[proto.name] = proto
                    self.combo_proto.addItem(f"{proto.name} ({len(proto.onsets)} pulses)", proto.name)
                self.current_protocol = protos[0]
            else:
                self.current_protocol = None
            self.combo_proto.blockSignals(False)
            self._apply_protocol_default_stats_params(reset_customized=True)

            progress.setValue(50)
            progress.setLabelText("Computing first-spike latency, jitter & reliability metrics...")
            QApplication.processEvents()

            def on_salt_progress(done: int, total: int) -> None:
                pct = int(55 + 30 * (done / max(1, total)))
                progress.setValue(pct)
                progress.setLabelText(f"Calculating parallel SALT tests ({done}/{total} units)...")
                QApplication.processEvents()

            self._recompute_session_metrics(progress_callback=on_salt_progress)

            progress.setValue(88)
            progress.setLabelText("Populating detected units table...")
            QApplication.processEvents()
            self._populate_table()
            self._set_dirty(False)
            self.session_changed.emit()
            self.protocols_changed.emit()

            progress.setValue(100)
            progress.setLabelText("NWB session loaded successfully!")
            QApplication.processEvents()
            progress.close()
            return True
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "NWB Loading Error", f"Failed to load NWB file:\n{str(e)}")
            return False
        finally:
            progress.close()

    def load_experiment_folder(
        self, folder_path: Optional[Union[str, Path, bool]] = None
    ) -> bool:
        """Load experiment folder containing Kilosort/Phy files and stimulus pulse files."""
        if folder_path is None or isinstance(folder_path, bool):
            folder = QFileDialog.getExistingDirectory(
                self, "Select Recording Session Folder (Kilosort/Phy)"
            )
            if not folder:
                return False
            folder_path = Path(folder)
        else:
            folder_path = Path(folder_path)

        if not folder_path.is_dir():
            QMessageBox.warning(self, "Invalid Directory", f"The directory does not exist: {folder_path}")
            return False

        progress = QProgressDialog(
            "Loading experiment session...",
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
        progress.setValue(5)
        progress.setLabelText("Reading spike times and cluster assignments from Phy / Kilosort...")
        QApplication.processEvents()

        try:
            units = self.phy_reader.load(folder_path)
            if not units:
                progress.close()
                QMessageBox.warning(self, "No Units Found", f"No units found in: {folder_path}")
                return False

            progress.setValue(30)
            progress.setLabelText(f"Loaded {len(units)} units. Discovering stimulus protocols...")
            QApplication.processEvents()

            self.current_folder = folder_path
            self.session_id = units[0].session_id
            self.units = units
            self.units_by_id = {u.unit_id: u for u in units}
            self.metrics.clear()
            self.manual_overrides.clear()
            self._criteria_undo_stack.clear()
            self.btn_criteria_undo.setEnabled(False)
            self._focused_unit_id = None
            self.lbl_session.setText(f"Session: {self.session_id}")
            self.lbl_session.setToolTip(str(folder_path))

            # Try loading channel_positions.npy for probe geometry
            self._channel_positions = None
            for cp_cand in (folder_path / "channel_positions.npy", folder_path.parent / "channel_positions.npy"):
                if cp_cand.exists():
                    try:
                        arr = np.load(cp_cand)
                        if arr.ndim == 2 and arr.shape[1] >= 2:
                            self._channel_positions = arr[:, :2]
                            break
                    except Exception:
                        pass

            # Discover stimulus protocols
            pulse_files = self.stim_reader.discover(folder_path)
            self.protocols = {}
            self.combo_proto.blockSignals(True)
            self.combo_proto.clear()

            if pulse_files:
                for p_file in pulse_files:
                    proto = self.stim_reader.load(p_file, sampling_rate=units[0].sampling_rate)
                    self.protocols[proto.name] = proto
                    self.combo_proto.addItem(f"{proto.name} ({len(proto.onsets)} pulses)", proto.name)
                self.current_protocol = next(iter(self.protocols.values()))
            else:
                self.current_protocol = None

            self.combo_proto.blockSignals(False)
            self._apply_protocol_default_stats_params(reset_customized=True)

            pre_none_ids = {u.unit_id for u in self.units if u.waveforms is None}
            if self.current_protocol is not None:
                extract_session_waveforms(
                    Path(folder_path),
                    self.units,
                    self.current_protocol.onsets,
                    self.stats_params,
                )
            self._disk_waveform_unit_ids = {
                u.unit_id for u in self.units if u.unit_id in pre_none_ids and u.waveforms is not None
            }

            progress.setValue(45)
            progress.setLabelText("Computing first-spike latency, jitter & reliability metrics...")
            QApplication.processEvents()

            def on_salt_progress(done: int, total: int) -> None:
                pct = int(50 + 35 * (done / max(1, total)))
                progress.setValue(pct)
                progress.setLabelText(f"Calculating parallel SALT tests ({done}/{total} units)...")
                QApplication.processEvents()

            self._recompute_session_metrics(progress_callback=on_salt_progress)

            progress.setValue(88)
            progress.setLabelText("Populating detected units table...")
            QApplication.processEvents()
            self._populate_table()
            self._set_dirty(False)
            self.session_changed.emit()
            self.protocols_changed.emit()

            progress.setValue(100)
            progress.setLabelText("Session loaded successfully!")
            QApplication.processEvents()
            progress.close()
            return True

        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Loading Error", f"Failed to load experiment data:\n{str(e)}")
            return False
        finally:
            progress.close()

    def close_session(self) -> None:
        """Clear loaded Detection session and reset UI state."""
        self.current_folder = None
        self.current_nwb_path = None
        self.session_id = "None"
        self.units = []
        self.units_by_id = {}
        self.protocols.clear()
        self.current_protocol = None
        self.metrics.clear()
        self.manual_overrides.clear()
        self.stats_params = StatsParameters.default()
        self._stats_params_customized = False
        self._focused_unit_id = None
        self._channel_positions = None
        self._criteria_undo_stack.clear()
        self.btn_criteria_undo.setEnabled(False)

        self.lbl_session.setText("Session: None")
        self.combo_proto.blockSignals(True)
        self.combo_proto.clear()
        self.combo_proto.blockSignals(False)

        self.table_units.setRowCount(0)
        self.raster_plot.clear()
        self.psth_plot.clear()
        self.waveform_panel.setVisible(False)
        self.pairplot_widget.set_data([], [], [], [], [])
        self.probe_widget.sites_scatter.setData(x=[], y=[])
        self.probe_widget.units_scatter.setData(x=[], y=[])
        self.probe_widget.cand_marker.setData(x=[], y=[])

        self.lbl_focused_unit_title.setText("No unit selected")
        self.lbl_focused_unit_meta.setText("Load an experiment folder to inspect units")
        self.lbl_focused_status.setText("○ Untagged")
        self.lbl_focused_status.setStyleSheet(DETECTION_STATUS_CHIP_UNTAGGED_STYLE)
        self.btn_toggle_tag.setText("Mark as tagged")

        self._update_criteria_visuals()
        self._update_summary_box()
        self._update_unit_summary_count()
        self._set_dirty(False)
        self.session_changed.emit()
        self.protocols_changed.emit()

    def _on_protocol_selected(self, index: int) -> None:
        """Handle protocol change in combo box."""
        if index < 0 or not self.protocols:
            return
        proto_name = self.combo_proto.currentData()
        if proto_name in self.protocols:
            self.current_protocol = self.protocols[proto_name]
            self._apply_protocol_default_stats_params(reset_customized=False)
            progress = QProgressDialog(
                f"Recalculating metrics for protocol '{proto_name}'...",
                None,
                0,
                100,
                self,
            )
            progress.setWindowTitle("Updating Protocol")
            progress.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress.setMinimumDuration(0)
            progress.setValue(10)
            QApplication.processEvents()
            try:
                def on_salt_progress(done: int, total: int) -> None:
                    pct = int(20 + 65 * (done / max(1, total)))
                    progress.setValue(pct)
                    progress.setLabelText(f"Calculating parallel SALT tests ({done}/{total} units)...")
                    QApplication.processEvents()

                self._reextract_disk_waveforms()
                self._recompute_session_metrics(progress_callback=on_salt_progress)
                progress.setValue(90)
                progress.setLabelText("Updating units table...")
                QApplication.processEvents()
                self._populate_table()
                self.protocols_changed.emit()
            finally:
                progress.close()

    def _reextract_disk_waveforms(self) -> None:
        """Re-extract raw binary waveforms for disk-backed units after protocol/stats updates."""
        disk_ids = getattr(self, "_disk_waveform_unit_ids", None)
        if not disk_ids or self.current_folder is None or self.current_protocol is None:
            return
        targets = [u for u in self.units if u.unit_id in disk_ids]
        if not targets:
            return
        for u in targets:
            u.waveforms = None
        extract_session_waveforms(
            Path(self.current_folder),
            targets,
            self.current_protocol.onsets,
            self.stats_params,
        )

    def _default_stats_params_for_protocol(self, pulse_dur_ms: float) -> StatsParameters:
        """Return default StatsParameters with post-pulse windows spanning [0, pulse_dur_ms]."""
        end_ms = float(pulse_dur_ms)
        if not math.isfinite(end_ms) or end_ms <= 0.0:
            end_ms = float(DEFAULT_PULSE_DURATION_MS)
        return StatsParameters(
            reliability_window_start_ms=0.0,
            reliability_window_end_ms=end_ms,
            latency_window_start_ms=0.0,
            latency_window_end_ms=end_ms,
            jitter_window_start_ms=0.0,
            jitter_window_end_ms=end_ms,
            salt_window_start_ms=0.0,
            salt_window_end_ms=end_ms,
            wf_evoked_start_ms=0.0,
            wf_evoked_end_ms=end_ms,
        )

    def default_stats_parameters(self) -> StatsParameters:
        """Return protocol-matched default StatsParameters for the currently active protocol."""
        return self._default_stats_params_for_protocol(self._get_current_pulse_duration_ms())

    def _apply_protocol_default_stats_params(self, reset_customized: bool = False) -> None:
        """Update stats_params to match the active protocol pulse duration unless customized."""
        if reset_customized:
            self._stats_params_customized = False
        if not self._stats_params_customized:
            self.stats_params = self.default_stats_parameters()

    def update_stats_parameters(self, new_params: StatsParameters) -> None:
        """Update analytical metric hyperparameters and recompute metrics if session is loaded."""
        self.stats_params = new_params
        self._stats_params_customized = True
        if self.units and self.current_protocol is not None:
            progress = QProgressDialog(
                "Recalculating metrics with updated stats parameters...",
                None,
                0,
                100,
                self,
            )
            progress.setWindowTitle("Updating Stats Parameters")
            progress.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress.setMinimumDuration(0)
            progress.setValue(10)
            QApplication.processEvents()
            try:
                def on_salt_progress(done: int, total: int) -> None:
                    pct = int(20 + 65 * (done / max(1, total)))
                    progress.setValue(pct)
                    progress.setLabelText(f"Calculating parallel SALT tests ({done}/{total} units)...")
                    QApplication.processEvents()

                self._reextract_disk_waveforms()
                self._recompute_session_metrics(progress_callback=on_salt_progress)
                progress.setValue(90)
                progress.setLabelText("Updating units table...")
                QApplication.processEvents()
                self._populate_table()
                selected_rows = self.table_units.selectionModel().selectedRows()
                if selected_rows:
                    self._on_unit_selected()
            finally:
                progress.close()

    def _build_likelihood_criteria(self) -> List[CriterionSpec]:
        """Build CriterionSpec list from the 5 live tagging criterion rows."""
        return [
            CriterionSpec(
                name="latency_ms",
                threshold=self.row_latency.threshold(),
                direction="le",
                enabled=self.row_latency.is_enabled(),
            ),
            CriterionSpec(
                name="jitter_ms",
                threshold=self.row_jitter.threshold(),
                direction="le",
                enabled=self.row_jitter.is_enabled(),
            ),
            CriterionSpec(
                name="reliability",
                threshold=self.row_rel.threshold(),
                direction="ge",
                enabled=self.row_rel.is_enabled(),
            ),
            CriterionSpec(
                name="salt_p_value",
                threshold=self.row_salt.threshold(),
                direction="le",
                enabled=self.row_salt.is_enabled(),
                is_log=True,
            ),
            CriterionSpec(
                name="waveform_correlation",
                threshold=self.row_wf.threshold(),
                direction="ge",
                enabled=self.row_wf.is_enabled(),
            ),
        ]

    def _recompute_session_metrics(
        self, progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """Compute metrics for all units using currently active protocol, thresholds, and stats parameters."""
        if not self.units or self.current_protocol is None:
            return

        use_lat = self.row_latency.is_enabled()
        use_jit = self.row_jitter.is_enabled()
        use_rel = self.row_rel.is_enabled()
        use_salt = self.row_salt.is_enabled()

        max_lat = self.row_latency.threshold() if use_lat else float("inf")
        max_jit = self.row_jitter.threshold() if use_jit else float("inf")
        min_rel = self.row_rel.threshold() if use_rel else float("-inf")
        max_salt_p = self.row_salt.threshold() if use_salt else float("inf")

        metrics_list = compute_session_metrics(
            self.units,
            self.current_protocol,
            max_latency_ms=max_lat,
            max_jitter_ms=max_jit,
            min_reliability=min_rel,
            max_salt_p=max_salt_p,
            stats_params=self.stats_params,
            progress_callback=progress_callback,
        )

        self.metrics = {m.unit_id: m for m in metrics_list}

        for uid, override_state in self.manual_overrides.items():
            if uid in self.metrics:
                self.metrics[uid].is_tagged = override_state

        compute_session_likelihoods(
            list(self.metrics.values()),
            self._build_likelihood_criteria(),
        )

        self._apply_thresholds()

    # --- Threshold Evaluation & Table Population ---

    def _passes_criteria(self, unit_id: int) -> bool:
        """Return whether unit passes all currently enabled criteria (ignoring manual overrides)."""
        m = self.metrics.get(unit_id)
        if not m:
            return False
        u = self.units_by_id.get(unit_id)

        passes_lat = (not self.row_latency.is_enabled()) or (
            np.isfinite(m.latency_ms) and m.latency_ms <= self.row_latency.threshold()
        )
        passes_jit = (not self.row_jitter.is_enabled()) or (
            np.isfinite(m.jitter_ms) and m.jitter_ms <= self.row_jitter.threshold()
        )
        passes_rel = (not self.row_rel.is_enabled()) or (
            np.isfinite(m.reliability) and m.reliability >= self.row_rel.threshold()
        )
        passes_salt = (not self.row_salt.is_enabled()) or (
            np.isfinite(m.salt_p_value) and m.salt_p_value <= self.row_salt.threshold()
        )

        # Units without extractable waveforms are not tested on WF r (WP11)
        has_wf = (
            u is not None
            and u.waveforms is not None
            and np.isfinite(m.waveform_correlation)
        )
        passes_wf = (
            (not self.row_wf.is_enabled())
            or (not has_wf)
            or (m.waveform_correlation >= self.row_wf.threshold())
        )
        return bool(passes_lat and passes_jit and passes_rel and passes_salt and passes_wf)

    def _populate_table(self) -> None:
        """Fill 8-column QTableWidget with computed unit metrics."""
        self.table_units.setSortingEnabled(False)
        self.table_units.blockSignals(True)
        self.table_units.setRowCount(len(self.units))

        use_lat = self.row_latency.is_enabled()
        use_jit = self.row_jitter.is_enabled()
        use_rel = self.row_rel.is_enabled()
        use_salt = self.row_salt.is_enabled()
        use_wf = self.row_wf.is_enabled()

        max_lat = self.row_latency.threshold()
        max_jit = self.row_jitter.threshold()
        min_rel = self.row_rel.threshold()
        max_salt = self.row_salt.threshold()
        min_wf = self.row_wf.threshold()

        danger_brush = QBrush(QColor(COLOR_DANGER))
        default_brush = QBrush(QColor(COLOR_TEXT))
        muted_brush = QBrush(QColor(COLOR_TEXT_MUTED))
        success_brush = QBrush(QColor(COLOR_SUCCESS))

        for row, unit in enumerate(self.units):
            uid = unit.unit_id
            m = self.metrics.get(uid)
            uid_str = f"u{uid:04d}"

            lat_valid = bool(m and is_valid_metric_value(m.latency_ms))
            lat_val = m.latency_ms if lat_valid else float("inf")
            lat_str = f"{m.latency_ms:.1f}" if lat_valid else "—"

            jit_valid = bool(m and is_valid_metric_value(m.jitter_ms))
            jit_val = m.jitter_ms if jit_valid else float("inf")
            jit_str = f"{m.jitter_ms:.1f}" if jit_valid else "—"

            rel_val = m.reliability if (m and np.isfinite(m.reliability)) else 0.0
            rel_str = f"{rel_val:.2f}"

            salt_val = m.salt_p_value if (m and np.isfinite(m.salt_p_value)) else 1.0
            salt_str = "≤ 0.01" if salt_val <= 0.01 else f"{salt_val:.3f}"

            wf_avail = (
                unit.waveforms is not None
                and m is not None
                and np.isfinite(m.waveform_correlation)
            )
            wf_val = float(m.waveform_correlation) if wf_avail and m is not None else -float("inf")
            wf_str = f"{wf_val:.2f}" if wf_avail else "—"

            lik_val = float(getattr(m, "likelihood", 0.0)) if m else 0.0
            if not np.isfinite(lik_val):
                lik_val = 0.0
            lik_str = f"{int(round(lik_val * 100))}%"

            is_tagged = bool(m.is_tagged) if m else False
            is_manual = uid in self.manual_overrides
            if is_tagged:
                status_str = "✓ M" if is_manual else "✓"
                status_brush = success_brush
                status_key = 0
            else:
                status_str = "○ M" if is_manual else "○"
                status_brush = muted_brush
                status_key = 1

            it_unit = NumericTableWidgetItem(uid_str, sort_key=uid)
            it_unit.setData(Qt.ItemDataRole.UserRole, uid)
            it_unit.setTextAlignment(int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter))

            it_lat = NumericTableWidgetItem(lat_str, sort_key=lat_val)
            it_lat.setTextAlignment(int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))
            if not lat_valid:
                it_lat.setForeground(muted_brush)
            elif use_lat and lat_val > max_lat:
                it_lat.setForeground(danger_brush)
            else:
                it_lat.setForeground(default_brush)

            it_jit = NumericTableWidgetItem(jit_str, sort_key=jit_val)
            it_jit.setTextAlignment(int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))
            if not jit_valid:
                it_jit.setForeground(muted_brush)
            elif use_jit and jit_val > max_jit:
                it_jit.setForeground(danger_brush)
            else:
                it_jit.setForeground(default_brush)

            it_rel = NumericTableWidgetItem(rel_str, sort_key=rel_val)
            it_rel.setTextAlignment(int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))
            it_rel.setForeground(danger_brush if (use_rel and rel_val < min_rel) else default_brush)

            it_salt = NumericTableWidgetItem(salt_str, sort_key=salt_val)
            it_salt.setTextAlignment(int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))
            it_salt.setToolTip(f"p = {salt_val:.5g}")
            it_salt.setForeground(danger_brush if (use_salt and salt_val > max_salt) else default_brush)

            it_wf = NumericTableWidgetItem(wf_str, sort_key=wf_val)
            it_wf.setTextAlignment(int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))
            if not wf_avail:
                it_wf.setForeground(muted_brush)
            elif use_wf and wf_val < min_wf:
                it_wf.setForeground(danger_brush)
            else:
                it_wf.setForeground(default_brush)

            it_lik = NumericTableWidgetItem(lik_str, sort_key=lik_val)
            it_lik.setTextAlignment(int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))
            it_lik.setData(Qt.ItemDataRole.UserRole, lik_val)
            it_lik.setForeground(default_brush)

            it_status = NumericTableWidgetItem(status_str, sort_key=status_key)
            it_status.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
            it_status.setForeground(status_brush)
            it_status.setData(Qt.ItemDataRole.UserRole, is_tagged)
            it_status.setData(Qt.ItemDataRole.UserRole + 1, is_manual)

            self.table_units.setItem(row, _COL_UNIT, it_unit)
            self.table_units.setItem(row, _COL_LAT, it_lat)
            self.table_units.setItem(row, _COL_JIT, it_jit)
            self.table_units.setItem(row, _COL_REL, it_rel)
            self.table_units.setItem(row, _COL_SALT, it_salt)
            self.table_units.setItem(row, _COL_WF, it_wf)
            self.table_units.setItem(row, _COL_LIK, it_lik)
            self.table_units.setItem(row, _COL_STATUS, it_status)

            self._style_table_row(row, is_tagged=is_tagged, is_focused=(uid == self._focused_unit_id))

        self.table_units.blockSignals(False)
        self.table_units.setSortingEnabled(True)

        header = self.table_units.horizontalHeader()
        if header.isSortIndicatorShown():
            self.table_units.sortItems(header.sortIndicatorSection(), header.sortIndicatorOrder())

        self._apply_status_filter()
        self._update_criteria_visuals()
        self._update_summary_box()
        self._update_pairplot()
        self._update_probe_widget()

        if len(self.units) > 0:
            target_row = 0
            for r in range(self.table_units.rowCount()):
                if self._is_row_tagged(r):
                    target_row = r
                    break
            self.table_units.clearSelection()
            self.table_units.selectRow(target_row)
            self._on_unit_selected()

    def _row_unit_id(self, row: int) -> int:
        uid_item = self.table_units.item(row, _COL_UNIT)
        if uid_item is None:
            return row
        data_uid = uid_item.data(Qt.ItemDataRole.UserRole)
        if isinstance(data_uid, int):
            return data_uid
        try:
            return int(uid_item.text().replace("u", ""))
        except ValueError:
            return row

    def _is_row_tagged(self, row: int) -> bool:
        status_item = self.table_units.item(row, _COL_STATUS)
        if status_item is None:
            return False
        role_val = status_item.data(Qt.ItemDataRole.UserRole)
        if isinstance(role_val, bool):
            return role_val
        txt = status_item.text()
        return txt.startswith("✓") or txt == "Tagged"

    def _style_table_row(self, row: int, is_tagged: bool, is_focused: bool) -> None:
        if is_focused:
            bg_brush = QBrush(QColor(COLOR_ACCENT_BG))
        elif is_tagged:
            bg_brush = QBrush(QColor(DETECTION_ROW_TAGGED_BG))
        else:
            bg_brush = QBrush()

        for c in range(self.table_units.columnCount()):
            item = self.table_units.item(row, c)
            if item is None:
                continue
            item.setBackground(bg_brush)
            f = item.font()
            f.setBold(is_focused)
            item.setFont(f)

    def _apply_thresholds(self) -> None:
        """Dynamically evaluate units against active threshold cutoffs."""
        if self.metrics:
            compute_session_likelihoods(
                list(self.metrics.values()),
                self._build_likelihood_criteria(),
            )

        use_lat = self.row_latency.is_enabled()
        use_jit = self.row_jitter.is_enabled()
        use_rel = self.row_rel.is_enabled()
        use_salt = self.row_salt.is_enabled()
        use_wf = self.row_wf.is_enabled()

        max_lat = self.row_latency.threshold()
        max_jit = self.row_jitter.threshold()
        min_rel = self.row_rel.threshold()
        max_salt_p = self.row_salt.threshold()
        min_wf = self.row_wf.threshold()

        danger_brush = QBrush(QColor(COLOR_DANGER))
        default_brush = QBrush(QColor(COLOR_TEXT))
        muted_brush = QBrush(QColor(COLOR_TEXT_MUTED))
        success_brush = QBrush(QColor(COLOR_SUCCESS))

        if self.table_units.rowCount() == 0:
            self._update_unit_summary_count()
            self._update_criteria_visuals()
            self._update_summary_box()
            self._update_pairplot()
            self._update_probe_widget()
            return

        was_sorting = self.table_units.isSortingEnabled()
        self.table_units.setSortingEnabled(False)
        self.table_units.blockSignals(True)

        for row in range(self.table_units.rowCount()):
            int_uid = self._row_unit_id(row)
            m = self.metrics.get(int_uid)
            u = self.units_by_id.get(int_uid)

            if m is not None:
                if int_uid in self.manual_overrides:
                    is_tagged = self.manual_overrides[int_uid]
                else:
                    is_tagged = self._passes_criteria(int_uid)
                m.is_tagged = is_tagged

                lat_val = m.latency_ms if np.isfinite(m.latency_ms) else float("inf")
                jit_val = m.jitter_ms if np.isfinite(m.jitter_ms) else float("inf")
                rel_val = m.reliability if np.isfinite(m.reliability) else 0.0
                salt_val = m.salt_p_value if np.isfinite(m.salt_p_value) else 1.0
                wf_avail = u is not None and u.waveforms is not None and np.isfinite(m.waveform_correlation)
                wf_val = float(m.waveform_correlation) if wf_avail else float("nan")
            else:
                it_lat_txt = self.table_units.item(row, _COL_LAT).text()
                lat_val = float(it_lat_txt) if it_lat_txt not in ("N/A", "—") else MISSING_METRIC_SENTINEL
                it_jit_txt = self.table_units.item(row, _COL_JIT).text()
                jit_val = float(it_jit_txt) if it_jit_txt not in ("N/A", "—") else MISSING_METRIC_SENTINEL
                rel_val = float(self.table_units.item(row, _COL_REL).text())
                salt_item = self.table_units.item(row, _COL_SALT)
                salt_val = salt_item.sort_key if isinstance(salt_item, NumericTableWidgetItem) else 1.0
                wf_avail = False
                wf_val = float("nan")
                if int_uid in self.manual_overrides:
                    is_tagged = self.manual_overrides[int_uid]
                else:
                    passes_lat = (not use_lat) or (lat_val <= max_lat)
                    passes_jit = (not use_jit) or (jit_val <= max_jit)
                    passes_rel = (not use_rel) or (rel_val >= min_rel)
                    passes_salt = (not use_salt) or (salt_val <= max_salt_p)
                    is_tagged = bool(passes_lat and passes_jit and passes_rel and passes_salt)

            # Update failing cell colors (missing values stay muted, never red)
            lat_missing = not is_valid_metric_value(lat_val)
            self.table_units.item(row, _COL_LAT).setForeground(
                muted_brush if lat_missing else (danger_brush if (use_lat and lat_val > max_lat) else default_brush)
            )
            jit_missing = not is_valid_metric_value(jit_val)
            self.table_units.item(row, _COL_JIT).setForeground(
                muted_brush if jit_missing else (danger_brush if (use_jit and jit_val > max_jit) else default_brush)
            )
            self.table_units.item(row, _COL_REL).setForeground(
                danger_brush if (use_rel and rel_val < min_rel) else default_brush
            )
            self.table_units.item(row, _COL_SALT).setForeground(
                danger_brush if (use_salt and salt_val > max_salt_p) else default_brush
            )
            if wf_avail:
                self.table_units.item(row, _COL_WF).setForeground(
                    danger_brush if (use_wf and wf_val < min_wf) else default_brush
                )
            else:
                self.table_units.item(row, _COL_WF).setForeground(muted_brush)

            if m is not None:
                lik_val = float(getattr(m, "likelihood", 0.0))
                if not np.isfinite(lik_val):
                    lik_val = 0.0
                lik_item = self.table_units.item(row, _COL_LIK)
                if lik_item is not None:
                    lik_item.setText(f"{int(round(lik_val * 100))}%")
                    lik_item.setData(Qt.ItemDataRole.UserRole, lik_val)
                    if isinstance(lik_item, NumericTableWidgetItem):
                        lik_item.sort_key = lik_val

            is_manual = int_uid in self.manual_overrides
            status_item = self.table_units.item(row, _COL_STATUS)
            if is_tagged:
                status_item.setText("✓ M" if is_manual else "✓")
                status_item.setForeground(success_brush)
                status_item.setData(Qt.ItemDataRole.UserRole, True)
                status_item.setData(Qt.ItemDataRole.UserRole + 1, is_manual)
                if isinstance(status_item, NumericTableWidgetItem):
                    status_item.sort_key = 0
            else:
                status_item.setText("○ M" if is_manual else "○")
                status_item.setForeground(muted_brush)
                status_item.setData(Qt.ItemDataRole.UserRole, False)
                status_item.setData(Qt.ItemDataRole.UserRole + 1, is_manual)
                if isinstance(status_item, NumericTableWidgetItem):
                    status_item.sort_key = 1

            self._style_table_row(row, is_tagged=is_tagged, is_focused=(int_uid == self._focused_unit_id))

        self.table_units.blockSignals(False)
        self.table_units.setSortingEnabled(was_sorting)

        header = self.table_units.horizontalHeader()
        if was_sorting and header.isSortIndicatorShown():
            sort_col = header.sortIndicatorSection()
            if sort_col in (_COL_STATUS, _COL_LIK):
                self.table_units.sortItems(sort_col, header.sortIndicatorOrder())

        self._apply_status_filter()
        self._update_focused_unit_header()
        self._update_criteria_visuals()
        self._update_summary_box()
        self._update_pairplot()
        self._update_probe_widget()

    def _toggle_selected_unit_tag(self, *args: Any, **kwargs: Any) -> None:
        """Manually toggle tagged/untagged status for the selected unit."""
        selected_rows = self.table_units.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        int_uid = self._row_unit_id(row)
        currently_tagged = self._is_row_tagged(row)
        new_tagged = not currently_tagged

        self._push_detection_undo_snapshot()
        self.manual_overrides[int_uid] = new_tagged
        if int_uid in self.metrics:
            self.metrics[int_uid].is_tagged = new_tagged

        status_item = self.table_units.item(row, _COL_STATUS)
        if status_item is not None:
            status_item.setData(Qt.ItemDataRole.UserRole + 1, True)
            if new_tagged:
                status_item.setText("✓ M")
                status_item.setForeground(QBrush(QColor(COLOR_SUCCESS)))
                status_item.setData(Qt.ItemDataRole.UserRole, True)
                if isinstance(status_item, NumericTableWidgetItem):
                    status_item.sort_key = 0
            else:
                status_item.setText("○ M")
                status_item.setForeground(QBrush(QColor(COLOR_TEXT_MUTED)))
                status_item.setData(Qt.ItemDataRole.UserRole, False)
                if isinstance(status_item, NumericTableWidgetItem):
                    status_item.sort_key = 1

        self._style_table_row(row, is_tagged=new_tagged, is_focused=(int_uid == self._focused_unit_id))

        header = self.table_units.horizontalHeader()
        if header.isSortIndicatorShown() and header.sortIndicatorSection() == _COL_STATUS:
            self.table_units.sortItems(_COL_STATUS, header.sortIndicatorOrder())

        self._apply_status_filter()
        self._update_focused_unit_header()
        self._update_summary_box()
        self._update_pairplot()
        self._update_probe_widget()
        self._set_dirty(True)

    # --- Sorting & Filtering ---

    def _on_header_section_clicked(self, logical_index: int) -> None:
        header = self.table_units.horizontalHeader()
        order = header.sortIndicatorOrder()
        self.table_units.sortItems(logical_index, order)
        self._apply_status_filter()

    def sort_by_column(self, col: int, order: Optional[Qt.SortOrder] = None) -> None:
        """Programmatically sort unit table by column, toggling order if already active."""
        header = self.table_units.horizontalHeader()
        if order is None:
            if header.sortIndicatorSection() == col:
                order = (
                    Qt.SortOrder.DescendingOrder
                    if header.sortIndicatorOrder() == Qt.SortOrder.AscendingOrder
                    else Qt.SortOrder.AscendingOrder
                )
            else:
                order = Qt.SortOrder.AscendingOrder
        header.setSortIndicator(col, order)
        self.table_units.sortItems(col, order)
        self._apply_status_filter()

    def _apply_status_filter(self) -> None:
        """Filter table rows based on active status filter selection."""
        filter_text = self.combo_filter_status.currentText()
        for row in range(self.table_units.rowCount()):
            int_uid = self._row_unit_id(row)
            is_tagged = self._is_row_tagged(row)
            m = self.metrics.get(int_uid)
            lik = float(getattr(m, "likelihood", 0.0)) if m else 0.0

            if filter_text in ("Tagged", "Tagged Only"):
                hide = not is_tagged
            elif filter_text in ("Untagged", "Untagged Only"):
                hide = is_tagged
            elif filter_text == "Passing current criteria":
                hide = not self._passes_criteria(int_uid)
            elif filter_text == "Changed manually":
                hide = int_uid not in self.manual_overrides
            elif filter_text == "Likely but untagged":
                hide = is_tagged or (lik < _LIKELY_THRESHOLD)
            else:
                hide = False
            self.table_units.setRowHidden(row, hide)

        selected_rows = self.table_units.selectionModel().selectedRows()
        curr_row = selected_rows[0].row() if selected_rows else self.table_units.currentRow()
        if curr_row >= 0 and self.table_units.isRowHidden(curr_row):
            first_visible = -1
            for r in range(self.table_units.rowCount()):
                if not self.table_units.isRowHidden(r):
                    first_visible = r
                    break
            self.table_units.clearSelection()
            if first_visible >= 0:
                self.table_units.selectRow(first_visible)
                self._on_unit_selected()

        self._update_unit_summary_count()

    def _update_unit_summary_count(self) -> None:
        total = self.table_units.rowCount()
        if total == 0:
            self.lbl_unit_summary.setText("0 units")
            return
        visible_count = sum(1 for r in range(total) if not self.table_units.isRowHidden(r))
        if visible_count < total:
            self.lbl_unit_summary.setText(f"{visible_count} of {total} units")
        else:
            unit_word = "unit" if total == 1 else "units"
            self.lbl_unit_summary.setText(f"{total} {unit_word}")

    def _focus_unit_by_id(self, unit_id: int) -> None:
        """Select and focus the specified unit ID in the units table."""
        for row in range(self.table_units.rowCount()):
            if self._row_unit_id(row) == int(unit_id):
                if self.table_units.isRowHidden(row):
                    self.combo_filter_status.setCurrentIndex(0)
                self.table_units.clearSelection()
                self.table_units.selectRow(row)
                self.table_units.scrollToItem(self.table_units.item(row, _COL_UNIT))
                self._on_unit_selected()
                break

    def _on_probe_unit_clicked(self, _scatter: Any, points: Any) -> None:
        if points is not None and len(points) > 0:
            uid = points[0].data()
            if uid is not None:
                self._focus_unit_by_id(int(uid))

    def _on_table_double_clicked(self, _item: QTableWidgetItem) -> None:
        self._toggle_selected_unit_tag()

    # --- Focused Unit & Plot Rendering ---

    def _update_focused_unit_header(self) -> None:
        uid = self._focused_unit_id
        if uid is None:
            self.lbl_focused_unit_title.setText("No unit selected")
            self.lbl_focused_unit_meta.setText("Load an experiment folder to inspect units")
            self.lbl_focused_status.setText("○ Untagged")
            self.lbl_focused_status.setStyleSheet(DETECTION_STATUS_CHIP_UNTAGGED_STYLE)
            self.lbl_focused_status.setToolTip("")
            self.btn_toggle_tag.setText("Mark as tagged")
            return

        self.lbl_focused_unit_title.setText(f"Unit u{uid:04d}")
        u = self.units_by_id.get(uid)
        if u is not None:
            shank = int(getattr(u, "shank", 0))
            self.lbl_focused_unit_meta.setText(
                f"{self.session_id} · ch {u.channel} · shank {shank} · {u.depth:.0f} µm"
            )
        else:
            self.lbl_focused_unit_meta.setText(f"{self.session_id}")

        m = self.metrics.get(uid)
        is_tagged = bool(m.is_tagged) if m else False
        if m is None:
            for r in range(self.table_units.rowCount()):
                if self._row_unit_id(r) == uid:
                    is_tagged = self._is_row_tagged(r)
                    break

        is_manual = uid in self.manual_overrides
        suffix = " (manual)" if is_manual else ""
        if is_tagged:
            self.lbl_focused_status.setText(f"● Tagged{suffix}")
            self.lbl_focused_status.setStyleSheet(DETECTION_STATUS_CHIP_TAGGED_STYLE)
            self.lbl_focused_status.setToolTip(
                "Manually marked as tagged (overrides criteria)."
                if is_manual
                else "Tagged because all enabled criteria pass."
            )
            self.btn_toggle_tag.setText("Mark as untagged")
        else:
            self.lbl_focused_status.setText(f"○ Untagged{suffix}")
            self.lbl_focused_status.setStyleSheet(DETECTION_STATUS_CHIP_UNTAGGED_STYLE)
            self.lbl_focused_status.setToolTip(
                "Manually marked as untagged (overrides criteria)."
                if is_manual
                else "Untagged: fails one or more enabled criteria."
            )
            self.btn_toggle_tag.setText("Mark as tagged")

    def _update_criteria_visuals(self) -> None:
        """Refresh all 5 DetectionCriterionRow KDEs, pass counts, and focused readouts."""
        uids = [u.unit_id for u in self.units]
        n_total = len(uids)
        focused_m = self.metrics.get(self._focused_unit_id) if self._focused_unit_id is not None else None
        focused_u = self.units_by_id.get(self._focused_unit_id) if self._focused_unit_id is not None else None

        lat_vals = [
            self.metrics[uid].latency_ms
            for uid in uids
            if uid in self.metrics
            and is_valid_metric_value(self.metrics[uid].latency_ms)
        ]
        jit_vals = [
            self.metrics[uid].jitter_ms
            for uid in uids
            if uid in self.metrics
            and is_valid_metric_value(self.metrics[uid].jitter_ms)
        ]
        rel_vals = [self.metrics[uid].reliability for uid in uids if uid in self.metrics]
        salt_vals = [self.metrics[uid].salt_p_value for uid in uids if uid in self.metrics]
        wf_vals = [
            self.metrics[uid].waveform_correlation
            for uid in uids
            if uid in self.metrics
            and self.units_by_id[uid].waveforms is not None
            and np.isfinite(self.metrics[uid].waveform_correlation)
        ]

        lat_t = self.row_latency.threshold()
        jit_t = self.row_jitter.threshold()
        rel_t = self.row_rel.threshold()
        salt_t = self.row_salt.threshold()
        wf_t = self.row_wf.threshold()

        lat_focused = (
            focused_m.latency_ms
            if (focused_m and is_valid_metric_value(focused_m.latency_ms))
            else None
        )
        jit_focused = (
            focused_m.jitter_ms
            if (focused_m and is_valid_metric_value(focused_m.jitter_ms))
            else None
        )

        self.row_latency.set_population_data(
            lat_vals,
            threshold=lat_t,
            enabled=self.row_latency.is_enabled(),
            focused_value=lat_focused,
            n_pass=sum(1 for v in lat_vals if np.isfinite(v) and v <= lat_t),
            n_total=n_total,
        )
        self.row_jitter.set_population_data(
            jit_vals,
            threshold=jit_t,
            enabled=self.row_jitter.is_enabled(),
            focused_value=jit_focused,
            n_pass=sum(1 for v in jit_vals if np.isfinite(v) and v <= jit_t),
            n_total=n_total,
        )
        self.row_rel.set_population_data(
            rel_vals,
            threshold=rel_t,
            enabled=self.row_rel.is_enabled(),
            focused_value=focused_m.reliability if focused_m else None,
            n_pass=sum(1 for v in rel_vals if np.isfinite(v) and v >= rel_t),
            n_total=n_total,
        )
        self.row_salt.set_population_data(
            salt_vals,
            threshold=salt_t,
            enabled=self.row_salt.is_enabled(),
            focused_value=focused_m.salt_p_value if focused_m else None,
            n_pass=sum(1 for v in salt_vals if np.isfinite(v) and v <= salt_t),
            n_total=n_total,
        )
        if focused_m is not None and np.isfinite(focused_m.salt_p_value):
            salt_val_f = float(focused_m.salt_p_value)
            self.row_salt.lbl_focused_value.setText("≤ 0.01" if salt_val_f <= 0.01 else f"{salt_val_f:.3f}")
        self.row_wf.set_population_data(
            wf_vals,
            threshold=wf_t,
            enabled=self.row_wf.is_enabled(),
            focused_value=(
                focused_m.waveform_correlation
                if (focused_m and focused_u and focused_u.waveforms is not None and np.isfinite(focused_m.waveform_correlation))
                else None
            ),
            n_pass=sum(1 for v in wf_vals if np.isfinite(v) and v >= wf_t),
            n_total=len(wf_vals),
        )

    def _update_summary_box(self) -> None:
        total = self.table_units.rowCount()
        tagged = sum(1 for r in range(total) if self._is_row_tagged(r))
        untagged = total - tagged
        manual = len(self.manual_overrides)
        self.lbl_summary_tagged.setText(str(tagged))
        self.lbl_summary_untagged.setText(str(untagged))
        self.lbl_summary_total.setText(str(total))
        self.lbl_summary_manual.setText(str(manual))
        self.lbl_summary_counts.setText(
            f"Tagged: <b>{tagged}</b> · Untagged: <b>{untagged}</b><br>"
            f"Units in session: <b>{total}</b> · Changed manually: <b>{manual}</b>"
        )

    def _update_pairplot(self) -> None:
        uids: List[int] = []
        lats: List[float] = []
        jits: List[float] = []
        rels: List[float] = []
        tags: List[bool] = []
        for u in self.units:
            m = self.metrics.get(u.unit_id)
            if m is None:
                continue
            uids.append(u.unit_id)
            lats.append(m.latency_ms if is_valid_metric_value(m.latency_ms) else float("nan"))
            jits.append(m.jitter_ms if is_valid_metric_value(m.jitter_ms) else float("nan"))
            rels.append(m.reliability)
            tags.append(bool(m.is_tagged))

        self.pairplot_widget.set_data(
            unit_ids=uids,
            latencies=lats,
            jitters=jits,
            reliabilities=rels,
            is_tagged=tags,
            focused_unit_id=self._focused_unit_id,
            lat_max=self.row_latency.threshold(),
            lat_enabled=self.row_latency.is_enabled(),
            jit_max=self.row_jitter.threshold(),
            jit_enabled=self.row_jitter.is_enabled(),
            rel_min=self.row_rel.threshold(),
            rel_enabled=self.row_rel.is_enabled(),
        )

    def _update_probe_widget(self) -> None:
        if not self.units:
            self.probe_widget.sites_scatter.setData(x=[], y=[])
            self.probe_widget.units_scatter.setData(x=[], y=[])
            self.probe_widget.cand_marker.setData(x=[], y=[])
            self.probe_widget.cand_line.setVisible(False)
            return

        if self._channel_positions is not None and len(self._channel_positions) > 0:
            self.probe_widget.sites_scatter.setData(
                x=self._channel_positions[:, 0],
                y=self._channel_positions[:, 1],
            )
        else:
            seen_pos = {(float(u.probe_x), float(u.depth)) for u in self.units}
            if seen_pos:
                sx = [p[0] for p in seen_pos]
                sy = [p[1] for p in seen_pos]
                self.probe_widget.sites_scatter.setData(x=sx, y=sy)

        spots: List[Dict[str, Any]] = []
        focused_pos: Optional[Tuple[float, float]] = None
        for u in self.units:
            m = self.metrics.get(u.unit_id)
            is_tagged = bool(m.is_tagged) if m else False
            is_focused = u.unit_id == self._focused_unit_id
            px = float(u.probe_x)
            py = float(u.depth)
            if is_focused:
                focused_pos = (px, py)
            spots.append({
                "pos": (px, py),
                "size": 10 if is_tagged else 6,
                "brush": pg.mkBrush(COLOR_SUCCESS if is_tagged else COLOR_UNASSIGNED),
                "pen": pg.mkPen(COLOR_ACCENT, width=2.0) if is_focused else pg.mkPen(COLOR_PANEL, width=0.6),
                "data": u.unit_id,
            })

        self.probe_widget.units_scatter.setData(spots=spots)

        if focused_pos is not None:
            self.probe_widget.cand_marker.setSymbol("o")
            self.probe_widget.cand_marker.setSize(13)
            self.probe_widget.cand_marker.setBrush(pg.mkBrush(0, 0, 0, 0))
            self.probe_widget.cand_marker.setPen(pg.mkPen(COLOR_ACCENT, width=2.0))
            self.probe_widget.cand_marker.setData(x=[focused_pos[0]], y=[focused_pos[1]])
            self.probe_widget.cand_marker.setVisible(True)
            self.probe_widget.cand_line.setValue(focused_pos[1])
            self.probe_widget.cand_line.setVisible(False)
        else:
            self.probe_widget.cand_marker.setVisible(False)
            self.probe_widget.cand_line.setVisible(False)

    def _on_window_text_edited(self) -> None:
        """Parse display-only window start and end from text boxes; ignore invalid ranges."""
        text_start = self.edit_win_start.text().strip().replace("+", "")
        text_end = self.edit_win_end.text().strip().replace("+", "")
        try:
            val_start = float(text_start)
            val_end = float(text_end)
        except ValueError:
            self._sync_window_text_boxes()
            return

        if (
            val_start >= _WINDOW_MIN_START_MS
            and val_end <= _WINDOW_MAX_END_MS
            and val_end > val_start + _WINDOW_MIN_SPAN_MS
        ):
            self.set_visualized_window((val_start, val_end))
        else:
            self._sync_window_text_boxes()

    def _sync_window_text_boxes(self) -> None:
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

    def _on_combo_order_changed(self, index: int) -> None:
        data = self.combo_order.itemData(index)
        if isinstance(data, str):
            self.set_pulse_ordering(data)

    def set_visualized_window(self, window_ms: Tuple[float, float]) -> None:
        """Set the active display window in milliseconds (display only)."""
        self.current_window_ms = (float(window_ms[0]), float(window_ms[1]))
        self._sync_window_text_boxes()
        self._on_unit_selected()
        self.view_settings_changed.emit(self.current_window_ms, self.raster_order_by)

    def set_pulse_ordering(self, order_by: str) -> None:
        """Set optogenetic pulse trial ordering mode ("presentation" or "latency")."""
        if order_by not in ("presentation", "latency"):
            return
        self.raster_order_by = order_by

        self.combo_order.blockSignals(True)
        for idx in range(self.combo_order.count()):
            if self.combo_order.itemData(idx) == order_by:
                self.combo_order.setCurrentIndex(idx)
                break
        self.combo_order.blockSignals(False)

        self._on_unit_selected()
        self.view_settings_changed.emit(self.current_window_ms, self.raster_order_by)

    def set_psth_smooth(self, smooth: bool) -> None:
        """Enable or disable smooth firing rate PSTH rendering."""
        self.psth_smooth = bool(smooth)
        if hasattr(self, "chk_smooth") and self.chk_smooth.isChecked() != self.psth_smooth:
            self.chk_smooth.blockSignals(True)
            self.chk_smooth.setChecked(self.psth_smooth)
            self.chk_smooth.blockSignals(False)
        if hasattr(self, "spin_sigma"):
            self.spin_sigma.setEnabled(self.psth_smooth)
        self._on_unit_selected()

    def set_psth_gaussian_sigma(self, sigma_ms: float) -> None:
        """Set Gaussian kernel sigma in ms for smooth firing rate PSTH."""
        self.psth_gaussian_sigma_ms = max(0.2, min(20.0, float(sigma_ms)))
        if hasattr(self, "spin_sigma") and abs(self.spin_sigma.value() - self.psth_gaussian_sigma_ms) > 1e-3:
            self.spin_sigma.blockSignals(True)
            self.spin_sigma.setValue(self.psth_gaussian_sigma_ms)
            self.spin_sigma.blockSignals(False)
        self._on_unit_selected()

    def set_raster_marker_size(self, size: float) -> None:
        """Set raster plot scatter marker size in pixels and re-render."""
        self.raster_marker_size = max(0.1, float(size))
        self._on_unit_selected()
        self.view_settings_changed.emit(self.current_window_ms, self.raster_order_by)

    def prompt_custom_marker_size(self) -> None:
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

    def prompt_custom_window(self) -> None:
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
            if w_end > w_start + _WINDOW_MIN_SPAN_MS:
                self.set_visualized_window((w_start, w_end))
            else:
                QMessageBox.warning(
                    self,
                    "Invalid Window",
                    f"Window end must be more than {_WINDOW_MIN_SPAN_MS:.0f} ms after start.",
                )
        else:
            self._sync_window_text_boxes()

    def prompt_visualization_settings(self) -> None:
        """Open a compact visualization settings dialog for Detection plots."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Configure Visualizations")
        dialog.setFixedWidth(340)
        vbox = QVBoxLayout(dialog)

        form = QFormLayout()
        chk_smooth = QCheckBox("Smooth PSTH firing rate", dialog)
        chk_smooth.setChecked(self.psth_smooth)
        form.addRow("PSTH mode:", chk_smooth)

        spin_sigma = QDoubleSpinBox(dialog)
        spin_sigma.setRange(0.2, 20.0)
        spin_sigma.setSingleStep(0.5)
        spin_sigma.setValue(self.psth_gaussian_sigma_ms)
        spin_sigma.setSuffix(" ms")
        spin_sigma.setEnabled(self.psth_smooth)
        chk_smooth.toggled.connect(spin_sigma.setEnabled)
        form.addRow("Gaussian sigma:", spin_sigma)

        spin_marker = QDoubleSpinBox(dialog)
        spin_marker.setRange(0.5, 30.0)
        spin_marker.setSingleStep(0.5)
        spin_marker.setValue(self.raster_marker_size)
        spin_marker.setSuffix(" px")
        form.addRow("Raster marker size:", spin_marker)

        vbox.addLayout(form)
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        vbox.addWidget(btn_box)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.set_psth_smooth(chk_smooth.isChecked())
            self.set_psth_gaussian_sigma(spin_sigma.value())
            self.set_raster_marker_size(spin_marker.value())

    def _on_unit_selected(self) -> None:
        """Render spike raster, PSTH, waveform panel, pairplot, and probe when a unit is focused."""
        selected_rows = self.table_units.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        int_uid = self._row_unit_id(row)
        prev_focused = self._focused_unit_id
        self._focused_unit_id = int_uid

        if prev_focused != int_uid:
            self.table_units.blockSignals(True)
            for r in range(self.table_units.rowCount()):
                r_uid = self._row_unit_id(r)
                if r_uid == prev_focused or r_uid == int_uid:
                    self._style_table_row(
                        r,
                        is_tagged=self._is_row_tagged(r),
                        is_focused=(r_uid == int_uid),
                    )
            self.table_units.blockSignals(False)

        self._update_focused_unit_header()
        self._update_criteria_visuals()
        self._update_pairplot()
        self._update_probe_widget()

        raster_y_label = "Pulse (by first spike)" if self.raster_order_by == "latency" else "Pulse"

        unit = self.units_by_id.get(int_uid)
        if unit is None:
            unit = next((u for u in self.units if u.unit_id == int_uid), None)

        if unit is not None and self.current_protocol is not None:
            pulse_onsets = self.current_protocol.onsets
            psth_data = compute_psth_and_raster(
                unit.spike_times,
                pulse_onsets,
                window_ms=self.current_window_ms,
                order_by=self.raster_order_by,
                smooth=self.psth_smooth,
                gaussian_sigma_ms=self.psth_gaussian_sigma_ms,
            )

            pulse_dur_ms = self._get_current_pulse_duration_ms()
            render_raster(
                self.raster_plot,
                raster_x=psth_data["raster_x"],
                raster_y=psth_data["raster_y"],
                n_trials=psth_data["n_trials"],
                title=None,
                x_range=self.current_window_ms,
                order_by=self.raster_order_by,
                show_x_axis=False,
                marker_size=self.raster_marker_size,
                y_axis_width=DETECTION_PLOT_LEFT_MARGIN,
                pulse_duration_ms=pulse_dur_ms,
                y_label=raster_y_label,
            )

            render_psth(
                self.psth_plot,
                psth_bins=psth_data["psth_bins"],
                psth_rates=psth_data["psth_rates"],
                baseline_mean=0.0,
                baseline_3sigma=0.0,
                title=None,
                x_range=self.current_window_ms,
                show_x_axis=True,
                y_axis_width=DETECTION_PLOT_LEFT_MARGIN,
                is_smooth=self.psth_smooth,
                pulse_duration_ms=pulse_dur_ms,
                x_label="Time from pulse onset (ms)",
                y_label="Rate (Hz)",
            )

            # Waveform panel visibility (§ 13.5.4 & § 13.9)
            if unit is not None and unit.waveforms is not None and self.current_protocol is not None:
                m = self.metrics.get(int_uid)
                r_val = m.waveform_correlation if (m and np.isfinite(m.waveform_correlation)) else None
                snippets = extract_evoked_and_spontaneous_snippets(
                    unit.waveforms,
                    unit.spike_times,
                    self.current_protocol.onsets,
                    evoked_window_ms=(
                        self.stats_params.wf_evoked_start_ms,
                        self.stats_params.wf_evoked_end_ms,
                    ),
                    baseline_window_ms=(
                        self.stats_params.wf_baseline_start_ms,
                        self.stats_params.wf_baseline_end_ms,
                    ),
                    max_spikes_per_group=self.stats_params.wf_max_spikes,
                )
                if snippets is not None:
                    evoked_snips, spont_snips = snippets
                    time_ms = np.linspace(
                        self.stats_params.wf_window_start_ms,
                        self.stats_params.wf_window_end_ms,
                        evoked_snips.shape[-1],
                    )
                    self.waveform_panel.set_waveforms(
                        spont_snippets=spont_snips,
                        evoked_snippets=evoked_snips,
                        time_ms=time_ms,
                        r_value=r_val,
                        window_ms=(
                            self.stats_params.wf_window_start_ms,
                            self.stats_params.wf_window_end_ms,
                        ),
                    )
                    self.waveform_panel.setVisible(True)
                else:
                    wf_arr = np.asarray(unit.waveforms, dtype=np.float64)
                    if wf_arr.ndim == 2 and wf_arr.shape[0] > 0 and wf_arr.shape[-1] > 0:
                        wf_arr = wf_arr[np.all(np.isfinite(wf_arr), axis=1)]
                    if (
                        wf_arr.ndim == 2
                        and wf_arr.shape[0] > 0
                        and wf_arr.shape[-1] > 0
                    ):
                        time_ms = np.linspace(
                            self.stats_params.wf_window_start_ms,
                            self.stats_params.wf_window_end_ms,
                            wf_arr.shape[-1],
                        )
                        self.waveform_panel.set_waveforms(
                            spont_snippets=wf_arr,
                            evoked_snippets=None,
                            time_ms=time_ms,
                            r_value=r_val,
                            window_ms=(
                                self.stats_params.wf_window_start_ms,
                                self.stats_params.wf_window_end_ms,
                            ),
                        )
                        self.waveform_panel.setVisible(True)
                    else:
                        self.waveform_panel.setVisible(False)
            else:
                self.waveform_panel.setVisible(False)
        else:
            self.waveform_panel.setVisible(False)
            self.raster_plot.clear()
            self.psth_plot.clear()
            self.raster_plot.setXRange(self.current_window_ms[0], self.current_window_ms[1], padding=0.0)
            self.raster_plot.getViewBox().disableAutoRange(pg.ViewBox.XAxis)
            return

        self.raster_plot.setXRange(self.current_window_ms[0], self.current_window_ms[1], padding=0.0)
        self.raster_plot.getViewBox().disableAutoRange(pg.ViewBox.XAxis)

    def _get_current_pulse_duration_ms(self) -> float:
        """Infer active protocol stimulation duration in ms from onset/offset timestamps."""
        proto = getattr(self, "current_protocol", None)
        if proto is None:
            return DEFAULT_PULSE_DURATION_MS
        return infer_pulse_duration_ms(
            getattr(proto, "onsets", None),
            getattr(proto, "offsets", None),
            default_ms=DEFAULT_PULSE_DURATION_MS,
        )

    # --- Session Export ---

    def export_curated_session(
        self,
        export_dir: Optional[Union[str, Path, bool]] = None,
        show_dialog: bool = True,
        only_tagged_plots: bool = True,
        precompute_pngs: bool = False,
    ) -> Optional[Path]:
        """Export current session curation metadata to <session_id>_opto_curated.csv and cached spike arrays."""
        if not self.units:
            if show_dialog:
                QMessageBox.warning(
                    self,
                    "No Units to Export",
                    "No spike-sorted units are currently loaded in the Detection workspace.\n\n"
                    "Please load an experiment folder or NWB file before saving.",
                )
            return None

        if export_dir is None or isinstance(export_dir, bool):
            if self.current_folder is not None:
                export_dir = self.current_folder
            else:
                export_dir = Path.cwd()
        else:
            export_dir = Path(export_dir)

        export_dir.mkdir(parents=True, exist_ok=True)
        csv_filename = f"{self.session_id}_opto_curated.csv"
        csv_path = export_dir / csv_filename
        plots_dir = export_dir / f"{self.session_id}_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        arrays_dir = export_dir / f"{self.session_id}_arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)

        progress = QProgressDialog(
            "Exporting session and pre-rendering plots...",
            None,
            0,
            100,
            self,
        )
        progress.setWindowTitle("Exporting Session")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(True)
        progress.setStyleSheet(PROGRESS_DIALOG_STYLE)
        progress.setValue(10)
        QApplication.processEvents()

        try:
            rows: List[Dict[str, Any]] = []

            for u in self.units:
                m = self.metrics.get(u.unit_id)
                if not m:
                    continue
                plot_rel = f"{self.session_id}_plots/{m.global_id}_psth_raster.png"
                rows.append({
                    "global_unit_id": m.global_id,
                    "session_id": m.session_id,
                    "unit_id": m.unit_id,
                    "latency_ms": round(m.latency_ms, 3) if is_valid_metric_value(m.latency_ms) else np.nan,
                    "jitter_ms": round(m.jitter_ms, 3) if is_valid_metric_value(m.jitter_ms) else np.nan,
                    "reliability": round(m.reliability, 4),
                    "waveform_correlation": round(m.waveform_correlation, 4),
                    "salt_p_value": round(m.salt_p_value, 5),
                    "is_tagged": m.is_tagged,
                    "cluster_id": m.cluster_id,
                    "curated_label": m.curated_label,
                    "channel": int(getattr(u, "channel", 0)),
                    "depth": float(getattr(u, "depth", 0.0)),
                    "probe_x": float(getattr(u, "probe_x", 0.0)),
                    "plot_path": plot_rel,
                })

            if precompute_pngs and self.current_protocol:
                batch_precompute_plots(
                    units=self.units,
                    stimulus=self.current_protocol,
                    output_dir=plots_dir,
                    only_tagged=only_tagged_plots,
                    metrics_map=self.metrics,
                )

            spikes_export: Dict[str, Any] = {}
            for u in self.units:
                m = self.metrics.get(u.unit_id)
                if m and m.is_tagged and u.spike_times is not None:
                    spikes_export[f"unit_{u.unit_id}"] = u.spike_times

            if self.current_protocol:
                if hasattr(self.current_protocol, "onsets") and self.current_protocol.onsets is not None:
                    spikes_export["pulse_onsets"] = self.current_protocol.onsets
                elif hasattr(self.current_protocol, "pulse_times") and self.current_protocol.pulse_times is not None:
                    spikes_export["pulse_onsets"] = self.current_protocol.pulse_times
                if hasattr(self.current_protocol, "offsets") and self.current_protocol.offsets is not None:
                    spikes_export["pulse_offsets"] = self.current_protocol.offsets
                proto_name = str(getattr(self.current_protocol, "name", "") or "").strip()
                if proto_name:
                    spikes_export["protocol_name"] = np.array(proto_name)

            if hasattr(self, "current_folder") and self.current_folder:
                cp_path = Path(self.current_folder) / "channel_positions.npy"
                if not cp_path.exists() and Path(self.current_folder).parent != Path(self.current_folder):
                    parent_cp = Path(self.current_folder).parent / "channel_positions.npy"
                    if parent_cp.exists():
                        cp_path = parent_cp
                if cp_path.exists():
                    try:
                        spikes_export["channel_positions"] = np.load(cp_path)
                    except Exception:
                        pass
            if "channel_positions" not in spikes_export:
                chan_pos = getattr(self, "_channel_positions", None)
                if chan_pos is None:
                    chan_pos = getattr(self, "channel_positions", None)
                if chan_pos is not None:
                    spikes_export["channel_positions"] = np.asarray(chan_pos)

            if spikes_export:
                np.savez_compressed(arrays_dir / f"{self.session_id}_spikes.npz", **spikes_export)
                np.savez_compressed(export_dir / f"{self.session_id}_spikes.npz", **spikes_export)

            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
        finally:
            progress.close()

        self._set_dirty(False)

        tagged_count = int(df["is_tagged"].sum()) if ("is_tagged" in df.columns and not df.empty) else 0
        total_count = len(df)
        num_plots = len(list(plots_dir.glob("*.png")))

        if show_dialog:
            QMessageBox.information(
                self,
                "Session Export Complete",
                f"Successfully exported Phase 1 curation session!\n\n"
                f"File: {csv_filename}\n"
                f"Plots Directory: {plots_dir.name}/ ({num_plots} pre-rendered plots)\n"
                f"Total Units: {total_count} ({tagged_count} tagged)",
            )

        return csv_path

    def _load_mock_data(self) -> None:
        """Populate initial mock units with realistic spikes, waveforms, metrics, and likelihoods."""
        if self.session_id == "None":
            self.session_id = "demo_mouseA_20261021"
            self.lbl_session.setText(f"Session: {self.session_id}")

        rng = np.random.default_rng(20261021)
        sr = DEFAULT_SAMPLING_RATE_HZ
        n_pulses = 30
        onsets = np.arange(1.0, 1.0 + n_pulses * 0.5, 0.5)
        offsets = onsets + 0.020
        proto = StimulusProtocol(onsets=onsets, offsets=offsets, name="default")
        self.protocols = {"default": proto}
        self.current_protocol = proto
        self._stats_params_customized = False
        self.stats_params = StatsParameters.default()
        self.combo_proto.blockSignals(True)
        self.combo_proto.clear()
        self.combo_proto.addItem("default", "default")
        self.combo_proto.setEnabled(True)
        self.combo_proto.blockSignals(False)
        self.protocols_changed.emit()

        # Realistic biphasic AP template (48 samples covering [-1.0, 2.0] ms)
        t_wf = np.linspace(-1.0, 2.0, 48)
        ap_template = (
            -90.0 * np.exp(-((t_wf - 0.0) / 0.22) ** 2)
            + 32.0 * np.exp(-((t_wf - 0.45) / 0.35) ** 2)
        )

        # (unit_id, lat_s, jit_s, rel, has_wf, ch, depth, probe_x, wf_noise_evoked)
        mock_specs = [
            (1, 0.0042, 0.0008, 0.85, True, 2, 120.0, 16.0, 6.0),
            (2, 0.0068, 0.0012, 0.72, True, 4, 160.0, 32.0, 8.0),
            (3, 0.0145, 0.0030, 0.25, True, 6, 200.0, 16.0, 28.0),
            (4, 0.0051, 0.0010, 0.68, True, 8, 240.0, 32.0, 7.0),
            (5, 0.0175, 0.0020, 0.20, False, 10, 280.0, 16.0, 10.0),
        ]

        units: List[UnitData] = []
        for uid, lat_s, jit_s, rel, has_wf, ch, depth, probe_x, ev_noise in mock_specs:
            spikes: List[float] = []
            n_evoked_target = max(2, int(round(rel * len(onsets))))
            evoked_pulses = sorted(
                rng.choice(len(onsets), size=n_evoked_target, replace=False).tolist()
            )
            ev_offsets = rng.normal(0.0, jit_s, size=n_evoked_target)
            ev_offsets -= np.mean(ev_offsets)
            ev_lat_map = {
                p_idx: float(np.clip(lat_s + off, 0.001, 0.019))
                for p_idx, off in zip(evoked_pulses, ev_offsets)
            }
            for idx_p, t_p in enumerate(onsets):
                n_spont = int(rng.integers(1, 4))
                for _ in range(n_spont):
                    spikes.append(float(t_p + rng.uniform(-0.045, -0.008)))
                if idx_p in ev_lat_map:
                    spikes.append(float(t_p + ev_lat_map[idx_p]))

            spike_times = np.sort(np.asarray(spikes, dtype=np.float64))
            if has_wf:
                n_spk = len(spike_times)
                wfs = np.tile(ap_template, (n_spk, 1)) + rng.normal(0.0, 6.0, size=(n_spk, 48))
                if uid == 3:
                    ev_mask = np.zeros(n_spk, dtype=bool)
                    for t_p in onsets:
                        ev_mask |= (spike_times >= t_p) & (spike_times <= t_p + 0.020)
                    wfs[ev_mask] = 0.4 * ap_template + rng.normal(
                        0.0, ev_noise, size=(int(np.sum(ev_mask)), 48)
                    )
                wfs = wfs.astype(np.float32)
            else:
                wfs = None

            units.append(
                UnitData(
                    session_id=self.session_id,
                    unit_id=uid,
                    spike_times=spike_times,
                    sampling_rate=sr,
                    channel=ch,
                    depth=depth,
                    probe_x=probe_x,
                    waveforms=wfs,
                )
            )

        self.units = units
        self.units_by_id = {u.unit_id: u for u in units}
        self._channel_positions = np.array(
            [[16.0 if (i % 2 == 0) else 32.0, 100.0 + i * 20.0] for i in range(12)],
            dtype=np.float64,
        )
        self.manual_overrides.clear()
        self._recompute_session_metrics()
        self._populate_table()
