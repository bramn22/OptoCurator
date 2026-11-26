"""WP8 § 12 Clustering Settings & Confirmation Modal Dialogs."""

from typing import Any, Callable, Dict, Optional, Tuple
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from optocurator.analysis.clustering import FeatureConfiguration
from optocurator.ui.dialog_feature_settings import FeatureSettingsDialog, format_feature_summary
from optocurator.ui.theme import (
    CLUSTERS_TOOLBAR_SLIDER_STYLE,
    DIALOG_DANGER_BUTTON_STYLE,
    DIALOG_INFO_BANNER_STYLE,
    DIALOG_MODAL_STYLE,
    DIALOG_NOTE_STYLE,
    DIALOG_PRIMARY_BUTTON_STYLE,
    DIALOG_RADIO_DANGER_STYLE,
    DIALOG_RADIO_SECONDARY_STYLE,
    DIALOG_SECONDARY_BUTTON_STYLE,
    DIALOG_SUBTITLE_STYLE,
    DIALOG_VALUE_BADGE_STYLE,
)


class ConfidenceThresholdDialog(QDialog):
    """Modal dialog for configuring auto-assignment confidence threshold (50-99%)."""

    def __init__(
        self,
        current_threshold: float = 0.85,
        auto_assign_enabled: bool = False,
        unassigned_max_probs: Optional[Dict[str, float]] = None,
        on_live_update: Optional[Callable[[int, bool], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._unassigned_max_probs: Dict[str, float] = dict(unassigned_max_probs or {})
        self._on_live_update = on_live_update

        init_pct = max(50, min(99, int(round(float(current_threshold) * 100))))
        self._init_pct = init_pct
        self._init_auto_assign = bool(auto_assign_enabled)

        self.setWindowTitle("Confidence Threshold")
        self.setFixedWidth(420)
        self.setModal(True)
        self.setStyleSheet(DIALOG_MODAL_STYLE)

        self._init_ui(init_pct, self._init_auto_assign)
        self._update_readout(init_pct)

    def _init_ui(self, init_pct: int, auto_assign_enabled: bool) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        lbl_info = QLabel(
            "Units whose top cluster probability reaches this threshold are flagged on cluster cards "
            "and eligible for auto-assignment.",
            self,
        )
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet(DIALOG_SUBTITLE_STYLE)
        layout.addWidget(lbl_info)

        slider_row = QHBoxLayout()
        slider_row.setSpacing(12)

        self.slider_threshold = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_threshold.setRange(50, 99)
        self.slider_threshold.setValue(init_pct)
        self.slider_threshold.setStyleSheet(CLUSTERS_TOOLBAR_SLIDER_STYLE)
        slider_row.addWidget(self.slider_threshold, stretch=1)

        self.lbl_threshold_value = QLabel(f"{init_pct}%", self)
        self.lbl_threshold_value.setStyleSheet(DIALOG_VALUE_BADGE_STYLE)
        self.lbl_threshold_value.setMinimumWidth(46)
        self.lbl_threshold_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        slider_row.addWidget(self.lbl_threshold_value)
        layout.addLayout(slider_row)

        self.lbl_above_count = QLabel("", self)
        self.lbl_above_count.setStyleSheet(DIALOG_INFO_BANNER_STYLE)
        layout.addWidget(self.lbl_above_count)

        self.chk_auto_assign = QCheckBox("Auto-assign unassigned units meeting threshold", self)
        self.chk_auto_assign.setChecked(auto_assign_enabled)
        layout.addWidget(self.chk_auto_assign)

        self.slider_threshold.valueChanged.connect(self._on_slider_changed)
        self.chk_auto_assign.toggled.connect(self._on_auto_assign_toggled)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch()

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.setStyleSheet(DIALOG_SECONDARY_BUTTON_STYLE)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        self.btn_update = QPushButton("Update threshold", self)
        self.btn_update.setStyleSheet(DIALOG_PRIMARY_BUTTON_STYLE)
        self.btn_update.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_update.setDefault(True)
        self.btn_update.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_update)

        layout.addLayout(btn_row)

    def _count_above_threshold(self, pct: int) -> int:
        thresh = pct / 100.0
        return sum(1 for p in self._unassigned_max_probs.values() if p >= thresh)

    def _update_readout(self, pct: int) -> None:
        self.lbl_threshold_value.setText(f"{pct}%")
        count = self._count_above_threshold(pct)
        noun = "unit" if count == 1 else "units"
        self.lbl_above_count.setText(
            f"<b>{count}</b> unassigned {noun} at or above <b>{pct}%</b> confidence."
        )

    def _on_slider_changed(self, pct: int) -> None:
        self._update_readout(pct)
        if self._on_live_update is not None:
            self._on_live_update(pct, self.chk_auto_assign.isChecked())

    def _on_auto_assign_toggled(self, checked: bool) -> None:
        if self._on_live_update is not None:
            self._on_live_update(self.slider_threshold.value(), bool(checked))

    def get_threshold_percent(self) -> int:
        return int(self.slider_threshold.value())

    def get_threshold(self) -> float:
        return float(self.slider_threshold.value()) / 100.0

    def is_auto_assign_enabled(self) -> bool:
        return bool(self.chk_auto_assign.isChecked())


class ActiveLearningSettingsDialog(QDialog):
    """Modal dialog exposing supported Active Learning & Random Forest classifier settings."""

    def __init__(
        self,
        feature_config: Optional[FeatureConfiguration] = None,
        threshold: float = 0.85,
        n_estimators: int = 100,
        random_state: int = 42,
        max_distance_sigma: float = 3.5,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.feature_config = feature_config or FeatureConfiguration()
        self._threshold = max(0.50, min(0.99, float(threshold)))
        self._n_estimators = max(10, int(n_estimators))
        self._random_state = int(random_state)
        self._max_distance_sigma = float(max_distance_sigma)

        self.setWindowTitle("Active Learning Settings")
        self.setFixedWidth(490)
        self.setModal(True)
        self.setStyleSheet(DIALOG_MODAL_STYLE)

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        intro = QLabel(
            "Configure the Random Forest classifier, feature space, and uncertainty ranking used for "
            "Active Learning and Auto-assignment.",
            self,
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(DIALOG_SUBTITLE_STYLE)
        layout.addWidget(intro)

        # 1. Classifier Features & Scaling
        box_feat = QGroupBox("1. Classifier Features & Scaling", self)
        layout_feat = QVBoxLayout(box_feat)
        layout_feat.setContentsMargins(12, 14, 12, 12)
        layout_feat.setSpacing(8)

        self.lbl_feat_summary = QLabel("", self)
        self.lbl_feat_summary.setWordWrap(True)
        self.lbl_feat_summary.setStyleSheet(DIALOG_INFO_BANNER_STYLE)
        self._update_feature_summary()
        layout_feat.addWidget(self.lbl_feat_summary)

        feat_row = QHBoxLayout()
        feat_row.addStretch()

        self.btn_edit_features = QPushButton("Edit Features...", self)
        self.btn_edit_features.setStyleSheet(DIALOG_SECONDARY_BUTTON_STYLE)
        self.btn_edit_features.clicked.connect(self._on_edit_features)
        feat_row.addWidget(self.btn_edit_features)
        layout_feat.addLayout(feat_row)
        layout.addWidget(box_feat)

        # 2. Random Forest & Active Learning Parameters
        box_rf = QGroupBox("2. Random Forest & Active Learning", self)
        form_rf = QFormLayout(box_rf)
        form_rf.setContentsMargins(12, 14, 12, 12)
        form_rf.setSpacing(8)

        self.spin_n_estimators = QSpinBox(self)
        self.spin_n_estimators.setRange(10, 1000)
        self.spin_n_estimators.setSingleStep(10)
        self.spin_n_estimators.setValue(self._n_estimators)
        form_rf.addRow("Random Forest Trees (n_estimators):", self.spin_n_estimators)

        self.spin_max_distance_sigma = QDoubleSpinBox(self)
        self.spin_max_distance_sigma.setRange(0.5, 20.0)
        self.spin_max_distance_sigma.setSingleStep(0.5)
        self.spin_max_distance_sigma.setDecimals(1)
        self.spin_max_distance_sigma.setValue(self._max_distance_sigma)
        self.spin_max_distance_sigma.setSuffix(" σ")
        self.spin_max_distance_sigma.setToolTip(
            "Maximum standardized distance from cluster centroid permitted for auto-assignment."
        )
        form_rf.addRow("Max Centroid Distance:", self.spin_max_distance_sigma)

        self.spin_threshold = QSpinBox(self)
        self.spin_threshold.setRange(50, 99)
        self.spin_threshold.setSuffix("%")
        self.spin_threshold.setValue(int(round(self._threshold * 100)))
        form_rf.addRow("Confidence Threshold:", self.spin_threshold)

        self.spin_random_seed = QSpinBox(self)
        self.spin_random_seed.setRange(0, 999999)
        self.spin_random_seed.setValue(self._random_state)
        form_rf.addRow("Random Seed:", self.spin_random_seed)

        layout.addWidget(box_rf)

        # Action Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch()

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.setStyleSheet(DIALOG_SECONDARY_BUTTON_STYLE)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        self.btn_apply = QPushButton("Apply and retrain", self)
        self.btn_apply.setStyleSheet(DIALOG_PRIMARY_BUTTON_STYLE)
        self.btn_apply.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_apply.setDefault(True)
        self.btn_apply.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_apply)

        layout.addLayout(btn_row)

    def _update_feature_summary(self) -> None:
        summary = format_feature_summary(self.feature_config)
        self.lbl_feat_summary.setText(f"<b>Classifier Features:</b> {summary}")

    def _on_edit_features(self) -> None:
        dlg = FeatureSettingsDialog(self.feature_config, parent=self)
        if dlg.exec():
            self.feature_config = dlg.get_feature_configuration()
            self._update_feature_summary()

    def get_feature_configuration(self) -> FeatureConfiguration:
        return self.feature_config

    def get_threshold(self) -> float:
        return float(self.spin_threshold.value()) / 100.0

    def get_n_estimators(self) -> int:
        return int(self.spin_n_estimators.value())

    def get_random_state(self) -> int:
        return int(self.spin_random_seed.value())

    def get_max_distance_sigma(self) -> float:
        return float(self.spin_max_distance_sigma.value())


class VisualizationSettingsDialog(QDialog):
    """Modal dialog for display-only clustering raster and PSTH visualization settings."""

    def __init__(
        self,
        window_ms: Tuple[float, float] = (-30.0, 50.0),
        bin_size_ms: float = 0.5,
        pulse_order: str = "presentation",
        psth_smooth: bool = True,
        psth_sigma_ms: float = 1.0,
        norm_mode: str = "peak",
        marker_size: float = 3.0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Visualization Settings")
        self.setFixedWidth(440)
        self.setModal(True)
        self.setStyleSheet(DIALOG_MODAL_STYLE)

        self._init_ui(
            window_ms=window_ms,
            bin_size_ms=bin_size_ms,
            pulse_order=pulse_order,
            psth_smooth=psth_smooth,
            psth_sigma_ms=psth_sigma_ms,
            norm_mode=norm_mode,
            marker_size=marker_size,
        )

    def _init_ui(
        self,
        window_ms: Tuple[float, float],
        bin_size_ms: float,
        pulse_order: str,
        psth_smooth: bool,
        psth_sigma_ms: float,
        norm_mode: str,
        marker_size: float,
    ) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        lbl_note = QLabel(
            "Display only — changing these settings updates rasters and PSTHs without altering "
            "computed metrics, clustering, or classifier features.",
            self,
        )
        lbl_note.setWordWrap(True)
        lbl_note.setStyleSheet(DIALOG_SUBTITLE_STYLE)
        layout.addWidget(lbl_note)

        form = QFormLayout()
        form.setSpacing(8)

        self.spin_window_start = QDoubleSpinBox(self)
        self.spin_window_start.setRange(-500.0, 0.0)
        self.spin_window_start.setSingleStep(5.0)
        self.spin_window_start.setSuffix(" ms")
        self.spin_window_start.setValue(float(window_ms[0]))
        form.addRow("Window Onset:", self.spin_window_start)

        self.spin_window_end = QDoubleSpinBox(self)
        self.spin_window_end.setRange(5.0, 500.0)
        self.spin_window_end.setSingleStep(5.0)
        self.spin_window_end.setSuffix(" ms")
        self.spin_window_end.setValue(float(window_ms[1]))
        form.addRow("Window Offset:", self.spin_window_end)

        self.spin_bin_width = QDoubleSpinBox(self)
        self.spin_bin_width.setRange(0.1, 20.0)
        self.spin_bin_width.setSingleStep(0.1)
        self.spin_bin_width.setDecimals(1)
        self.spin_bin_width.setSuffix(" ms")
        self.spin_bin_width.setValue(float(bin_size_ms))
        form.addRow("Bin Width:", self.spin_bin_width)

        self.combo_pulse_order = QComboBox(self)
        self.combo_pulse_order.addItem("Presentation order", "presentation")
        self.combo_pulse_order.addItem("By first-spike latency", "latency")
        idx_order = self.combo_pulse_order.findData(pulse_order)
        if idx_order >= 0:
            self.combo_pulse_order.setCurrentIndex(idx_order)
        form.addRow("Pulse Order:", self.combo_pulse_order)

        self.chk_smooth = QCheckBox("Gaussian smoothing", self)
        self.chk_smooth.setChecked(bool(psth_smooth))
        form.addRow("Smoothing:", self.chk_smooth)

        self.spin_sigma = QDoubleSpinBox(self)
        self.spin_sigma.setRange(0.2, 20.0)
        self.spin_sigma.setSingleStep(0.5)
        self.spin_sigma.setSuffix(" ms")
        self.spin_sigma.setValue(float(psth_sigma_ms))
        self.spin_sigma.setEnabled(bool(psth_smooth))
        self.chk_smooth.toggled.connect(self.spin_sigma.setEnabled)
        form.addRow("Smoothing σ:", self.spin_sigma)

        self.combo_norm = QComboBox(self)
        self.combo_norm.addItem("Normalized [0–1]", "peak")
        self.combo_norm.addItem("Firing Rate (Hz)", "raw")
        self.combo_norm.addItem("Spike Probability", "probability")
        idx_norm = self.combo_norm.findData(norm_mode)
        if idx_norm >= 0:
            self.combo_norm.setCurrentIndex(idx_norm)
        form.addRow("Normalisation:", self.combo_norm)

        self.spin_marker_size = QDoubleSpinBox(self)
        self.spin_marker_size.setRange(1.0, 15.0)
        self.spin_marker_size.setSingleStep(0.5)
        self.spin_marker_size.setValue(float(marker_size))
        form.addRow("Raster Marker Size:", self.spin_marker_size)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch()

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.setStyleSheet(DIALOG_SECONDARY_BUTTON_STYLE)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        self.btn_apply = QPushButton("Apply", self)
        self.btn_apply.setStyleSheet(DIALOG_PRIMARY_BUTTON_STYLE)
        self.btn_apply.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_apply.setDefault(True)
        self.btn_apply.clicked.connect(self._on_accept)
        btn_row.addWidget(self.btn_apply)

        layout.addLayout(btn_row)

    def _on_accept(self) -> None:
        if self.spin_window_start.value() >= self.spin_window_end.value():
            QMessageBox.warning(self, "Invalid Window", "Window onset must be less than window offset.")
            return
        self.accept()

    def get_settings(self) -> Dict[str, Any]:
        return {
            "window_ms": (float(self.spin_window_start.value()), float(self.spin_window_end.value())),
            "bin_size_ms": float(self.spin_bin_width.value()),
            "pulse_order": str(self.combo_pulse_order.currentData() or "presentation"),
            "psth_smooth": bool(self.chk_smooth.isChecked()),
            "psth_sigma_ms": float(self.spin_sigma.value()),
            "norm_mode": str(self.combo_norm.currentData() or "peak"),
            "marker_size": float(self.spin_marker_size.value()),
        }


class ResetClusteringDialog(QDialog):
    """Modal confirmation dialog for resetting clustering (auto-assigned only vs everything)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Reset Clustering")
        self.setFixedWidth(400)
        self.setModal(True)
        self.setStyleSheet(DIALOG_MODAL_STYLE)

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        lbl_info = QLabel("Choose which unit assignments to reset:", self)
        lbl_info.setStyleSheet(DIALOG_SUBTITLE_STYLE)
        layout.addWidget(lbl_info)

        self.radio_auto_only = QRadioButton("Auto-assigned units only", self)
        self.radio_auto_only.setStyleSheet(DIALOG_RADIO_SECONDARY_STYLE)
        self.radio_auto_only.setChecked(True)

        self.radio_everything = QRadioButton("Everything including curated", self)
        self.radio_everything.setStyleSheet(DIALOG_RADIO_DANGER_STYLE)

        self.choice_group = QButtonGroup(self)
        self.choice_group.addButton(self.radio_auto_only)
        self.choice_group.addButton(self.radio_everything)

        layout.addWidget(self.radio_auto_only)
        layout.addWidget(self.radio_everything)

        lbl_note = QLabel("You can undo this action from the toolbar or Ctrl+Z.", self)
        lbl_note.setStyleSheet(DIALOG_NOTE_STYLE)
        layout.addWidget(lbl_note)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch()

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.setStyleSheet(DIALOG_SECONDARY_BUTTON_STYLE)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        self.btn_reset = QPushButton("Reset", self)
        self.btn_reset.setStyleSheet(DIALOG_DANGER_BUTTON_STYLE)
        self.btn_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_reset)

        layout.addLayout(btn_row)

    def reset_mode(self) -> str:
        """Return 'auto_only' or 'everything'."""
        return "auto_only" if self.radio_auto_only.isChecked() else "everything"


class RevertAutoAssignmentsDialog(QDialog):
    """Modal confirmation dialog for reverting all auto-assigned units to unassigned."""

    def __init__(self, auto_count: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.auto_count = int(auto_count)
        self.setWindowTitle("Revert Auto-Assignments")
        self.setFixedWidth(400)
        self.setModal(True)
        self.setStyleSheet(DIALOG_MODAL_STYLE)

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(14)

        lbl_msg = QLabel(
            f"Return all {self.auto_count} auto-assigned units to unassigned? "
            "Curated assignments are not affected.",
            self,
        )
        lbl_msg.setWordWrap(True)
        lbl_msg.setStyleSheet(DIALOG_SUBTITLE_STYLE)
        layout.addWidget(lbl_msg)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch()

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.setStyleSheet(DIALOG_SECONDARY_BUTTON_STYLE)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        self.btn_revert = QPushButton("Revert", self)
        self.btn_revert.setStyleSheet(DIALOG_DANGER_BUTTON_STYLE)
        self.btn_revert.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_revert.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_revert)

        layout.addLayout(btn_row)
