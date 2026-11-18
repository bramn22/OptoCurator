"""Feature Configuration Modal Dialog for Phase 2 Clustering and Active Learning."""

from typing import Optional, Tuple
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from optocurator.analysis.clustering import FeatureConfiguration
from optocurator.ui.theme import (
    DIALOG_ACCENT_CHECKBOX_STYLE,
    DIALOG_BOLD_LABEL_STYLE,
    DIALOG_GRID_HEADER_STYLE,
    DIALOG_MODAL_STYLE,
    DIALOG_PRIMARY_BUTTON_STYLE,
    DIALOG_SECONDARY_BUTTON_STYLE,
    DIALOG_SUBTITLE_STYLE,
)


class WeightLineEdit(QLineEdit):
    """Clean numeric text box for feature weight input supporting both text() and value()/setValue()."""

    def __init__(self, initial_val: float = 1.0, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedWidth(68)
        self.setValue(initial_val)

    def value(self) -> float:
        try:
            v = float(self.text().strip())
            return max(0.0, v)
        except (ValueError, TypeError):
            return 1.0

    def setValue(self, val: float) -> None:
        self.setText(f"{float(val):g}")


def format_feature_summary(config: FeatureConfiguration) -> str:
    """Generate a concise human-readable summary of active features, weights, and PSTH settings."""
    parts = []
    if config.use_latency:
        parts.append(f"Latency ({config.weight_latency:g}×)")
    if config.use_jitter:
        parts.append(f"Jitter ({config.weight_jitter:g}×)")
    if config.use_reliability:
        parts.append(f"Reliability ({config.weight_reliability:g}×)")
    if config.use_salt:
        salt_lbl = "-log10(SALT)" if config.salt_log10_transform else "SALT p"
        parts.append(f"{salt_lbl} ({config.weight_salt:g}×)")
    if config.use_baseline_rate:
        parts.append(f"Baseline ({config.weight_baseline_rate:g}×)")
    if config.use_peak_rate:
        parts.append(f"Peak ({config.weight_peak_rate:g}×)")
    if config.use_peak_response_rate:
        parts.append(f"Peak Resp ({config.weight_peak_response_rate:g}×)")
    if config.use_time_to_response:
        parts.append(f"Time-to-Resp ({config.weight_time_to_response:g}×, {config.response_std_threshold:g}SD)")
    if config.use_psth_profile:
        w0, w1 = config.time_window_ms
        norm = config.psth_normalization.replace("_", " ")
        smooth_str = f", smooth σ={config.psth_gaussian_sigma_ms:g}ms" if getattr(config, "psth_smooth", False) else ""
        pca_str = f", PCA {config.psth_pca_variance_explained:.0%}" if getattr(config, "psth_use_pca", False) else ""
        parts.append(f"PSTH [{w0:+.0f}, {w1:+.0f} ms] ({config.weight_psth_profile:g}×, {norm}{smooth_str}{pca_str})")

    if not parts:
        return "No features selected (PSTH fallback)"
    return " + ".join(parts)


class FeatureSettingsDialog(QDialog):
    """Modal dialog allowing users to configure active features and weights for clustering and classification."""

    def __init__(
        self,
        current_config: Optional[FeatureConfiguration] = None,
        parent: Optional[QWidget] = None,
        default_time_window_ms: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__(parent)
        self.config = current_config or FeatureConfiguration()
        self._default_time_window_ms: Optional[Tuple[float, float]] = (
            (float(default_time_window_ms[0]), float(default_time_window_ms[1]))
            if default_time_window_ms is not None
            else None
        )
        self.setWindowTitle("Feature Space & Response Profiling Settings")
        self.setFixedWidth(540)
        self.setModal(True)

        self.setStyleSheet(DIALOG_MODAL_STYLE)

        self._init_ui()
        self._populate_fields(self.config)

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 16, 18, 16)
        main_layout.setSpacing(12)

        intro = QLabel(
            "Select features and assign relative weights used for 2D embedding (PCA/UMAP), "
            "Affinity Propagation seeding, response clustering, and active learning classification:"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(DIALOG_SUBTITLE_STYLE)
        main_layout.addWidget(intro)

        top_opts = QHBoxLayout()
        top_opts.setSpacing(12)
        self.chk_same_as_al = QCheckBox("Same as active learning")
        self.chk_same_as_al.setChecked(True)
        self.chk_same_as_al.setToolTip("Use the same feature set for seeding, unsupervised clustering, and active learning.")
        top_opts.addWidget(self.chk_same_as_al)
        top_opts.addStretch()
        top_opts.addWidget(QLabel("Scaling:"))
        self.combo_scaling = QComboBox()
        self.combo_scaling.addItem("Standardized (z-score)", "zscore")
        self.combo_scaling.setToolTip("Features are standardized (zero mean, unit variance) before clustering.")
        top_opts.addWidget(self.combo_scaling)
        main_layout.addLayout(top_opts)

        # Scroll area for clean containment
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # 1. Analytical & Summary Metrics Group
        grp_metrics = QGroupBox("Analytical & Summary Response Metrics")
        grid_metrics = QGridLayout(grp_metrics)
        grid_metrics.setContentsMargins(12, 14, 12, 12)
        grid_metrics.setSpacing(6)
        grid_metrics.setColumnStretch(0, 1)
        grid_metrics.setColumnStretch(1, 0)

        # Header row
        lbl_feat = QLabel("Feature")
        lbl_feat.setStyleSheet(DIALOG_GRID_HEADER_STYLE)
        lbl_w = QLabel("Weight")
        lbl_w.setStyleSheet(DIALOG_GRID_HEADER_STYLE)
        grid_metrics.addWidget(lbl_feat, 0, 0)
        grid_metrics.addWidget(lbl_w, 0, 1, alignment=Qt.AlignmentFlag.AlignRight)

        def _make_weight_spin(val: float = 1.0) -> WeightLineEdit:
            return WeightLineEdit(val)

        # Row 1: Latency
        self.chk_latency = QCheckBox("First-Spike Onset Latency (latency_ms)")
        self.spin_w_latency = _make_weight_spin()
        self.edit_weight_latency = self.spin_w_latency
        self.chk_latency.toggled.connect(self.spin_w_latency.setEnabled)
        grid_metrics.addWidget(self.chk_latency, 1, 0)
        grid_metrics.addWidget(self.spin_w_latency, 1, 1)

        # Row 2: Jitter
        self.chk_jitter = QCheckBox("First-Spike Latency Jitter / IQR (jitter_ms)")
        self.spin_w_jitter = _make_weight_spin()
        self.edit_weight_jitter = self.spin_w_jitter
        self.chk_jitter.toggled.connect(self.spin_w_jitter.setEnabled)
        grid_metrics.addWidget(self.chk_jitter, 2, 0)
        grid_metrics.addWidget(self.spin_w_jitter, 2, 1)

        # Row 3: Reliability
        self.chk_reliability = QCheckBox("Evoked Spike Reliability (fraction of pulses)")
        self.spin_w_reliability = _make_weight_spin()
        self.edit_weight_reliability = self.spin_w_reliability
        self.chk_reliability.toggled.connect(self.spin_w_reliability.setEnabled)
        grid_metrics.addWidget(self.chk_reliability, 3, 0)
        grid_metrics.addWidget(self.spin_w_reliability, 3, 1)

        # Row 4: SALT p-value + log10 transform
        row_salt_layout = QHBoxLayout()
        row_salt_layout.setContentsMargins(0, 0, 0, 0)
        row_salt_layout.setSpacing(6)
        self.chk_salt = QCheckBox("SALT Test Significance (salt_p_value)")
        self.chk_salt_log10 = QCheckBox("-log10(p)")
        self.chk_salt_log10.setToolTip("Transform p-value to -log10(p) for linear significance separation")
        self.chk_salt_log10.setStyleSheet(DIALOG_ACCENT_CHECKBOX_STYLE)
        row_salt_layout.addWidget(self.chk_salt)
        row_salt_layout.addWidget(self.chk_salt_log10)
        row_salt_layout.addStretch()
        self.spin_w_salt = _make_weight_spin()
        self.edit_weight_salt = self.spin_w_salt
        self.chk_salt.toggled.connect(self.spin_w_salt.setEnabled)
        self.chk_salt.toggled.connect(self.chk_salt_log10.setEnabled)
        grid_metrics.addLayout(row_salt_layout, 4, 0)
        grid_metrics.addWidget(self.spin_w_salt, 4, 1)

        # Row 5: Baseline Firing Rate
        self.chk_baseline_rate = QCheckBox("Baseline Firing Rate (Hz, t < 0 ms)")
        self.spin_w_baseline = _make_weight_spin()
        self.edit_weight_baseline_rate = self.spin_w_baseline
        self.chk_baseline_rate.toggled.connect(self.spin_w_baseline.setEnabled)
        grid_metrics.addWidget(self.chk_baseline_rate, 5, 0)
        grid_metrics.addWidget(self.spin_w_baseline, 5, 1)

        # Row 6: Peak Evoked Rate
        self.chk_peak_rate = QCheckBox("Peak Evoked Firing Rate (Hz, t >= 0 ms)")
        self.spin_w_peak_rate = _make_weight_spin()
        self.edit_weight_peak_rate = self.spin_w_peak_rate
        self.chk_peak_rate.toggled.connect(self.spin_w_peak_rate.setEnabled)
        grid_metrics.addWidget(self.chk_peak_rate, 6, 0)
        grid_metrics.addWidget(self.spin_w_peak_rate, 6, 1)

        # Row 7: Peak Response Rate
        self.chk_peak_response_rate = QCheckBox(
            "Peak Response Rate (max ±Hz deviation from baseline in pulse window)"
        )
        self.spin_w_peak_response_rate = _make_weight_spin()
        self.chk_peak_response_rate.toggled.connect(self.spin_w_peak_response_rate.setEnabled)
        grid_metrics.addWidget(self.chk_peak_response_rate, 7, 0)
        grid_metrics.addWidget(self.spin_w_peak_response_rate, 7, 1)

        # Row 8: Time to Response + SD Threshold
        row_ttr_layout = QHBoxLayout()
        row_ttr_layout.setContentsMargins(0, 0, 0, 0)
        row_ttr_layout.setSpacing(6)
        self.chk_time_to_response = QCheckBox("Time to Response (first ±SD crossing)")
        self.spin_response_std_threshold = QDoubleSpinBox()
        self.spin_response_std_threshold.setRange(0.5, 10.0)
        self.spin_response_std_threshold.setSingleStep(0.5)
        self.spin_response_std_threshold.setDecimals(1)
        self.spin_response_std_threshold.setSuffix(" SD")
        self.spin_response_std_threshold.setValue(2.0)
        self.spin_response_std_threshold.setToolTip(
            "Baseline standard deviation threshold (0.5–10.0 SD) for first response crossing"
        )
        row_ttr_layout.addWidget(self.chk_time_to_response)
        row_ttr_layout.addWidget(self.spin_response_std_threshold)
        row_ttr_layout.addStretch()
        self.spin_w_time_to_response = _make_weight_spin()
        self.chk_time_to_response.toggled.connect(self.spin_w_time_to_response.setEnabled)
        self.chk_time_to_response.toggled.connect(self.spin_response_std_threshold.setEnabled)
        grid_metrics.addLayout(row_ttr_layout, 8, 0)
        grid_metrics.addWidget(self.spin_w_time_to_response, 8, 1)

        layout.addWidget(grp_metrics)

        # 2. Temporal PSTH Profile Group
        grp_psth = QGroupBox("Full Temporal Waveform Profile (PSTH)")
        vbox_psth = QVBoxLayout(grp_psth)
        vbox_psth.setContentsMargins(12, 14, 12, 12)
        vbox_psth.setSpacing(8)

        psth_header = QHBoxLayout()
        self.chk_psth = QCheckBox("Include Full Peri-Stimulus Time Histogram (PSTH) Profile")
        self.chk_psth.setStyleSheet(DIALOG_BOLD_LABEL_STYLE)
        self.spin_w_psth = _make_weight_spin()
        self.edit_weight_psth = self.spin_w_psth
        psth_header.addWidget(self.chk_psth)
        psth_header.addStretch()
        psth_header.addWidget(QLabel("Weight:"))
        psth_header.addWidget(self.spin_w_psth)
        vbox_psth.addLayout(psth_header)

        form_psth = QFormLayout()
        form_psth.setSpacing(6)
        form_psth.setContentsMargins(18, 4, 4, 4)

        self.combo_psth_norm = QComboBox()
        self.combo_psth_norm.addItem("Area-Normalized (Probability distribution, sum = 1)", "area_normalized")
        self.combo_psth_norm.addItem("Peak-Normalized [0–1] (Waveform shape only)", "peak_normalized")
        self.combo_psth_norm.addItem("Raw Firing Rate (Hz)", "raw")
        form_psth.addRow("PSTH Normalization:", self.combo_psth_norm)

        start_ms, end_ms = self.config.time_window_ms
        self.spin_start = QDoubleSpinBox()
        self.spin_start.setRange(-500.0, 50.0)
        self.spin_start.setValue(start_ms)
        self.spin_start.setSuffix(" ms")
        self.spin_start.setSingleStep(5.0)
        form_psth.addRow("PSTH Window Start:", self.spin_start)

        self.spin_end = QDoubleSpinBox()
        self.spin_end.setRange(5.0, 500.0)
        self.spin_end.setValue(end_ms)
        self.spin_end.setSuffix(" ms")
        self.spin_end.setSingleStep(5.0)
        form_psth.addRow("PSTH Window Stop:", self.spin_end)

        self.spin_bin = QDoubleSpinBox()
        self.spin_bin.setRange(0.5, 20.0)
        self.spin_bin.setValue(self.config.bin_size_ms)
        self.spin_bin.setSuffix(" ms")
        self.spin_bin.setSingleStep(0.5)
        form_psth.addRow("PSTH Bin Width:", self.spin_bin)

        self.chk_psth_smooth = QCheckBox("Smooth Firing Rate (Elephant Gaussian Kernel)")
        self.chk_psth_smooth.setChecked(getattr(self.config, "psth_smooth", True))
        self.chk_psth_smooth.setToolTip("Compute smooth continuous firing rates using Elephant GaussianKernel")
        form_psth.addRow("PSTH Smoothing:", self.chk_psth_smooth)

        self.spin_psth_sigma = QDoubleSpinBox()
        self.spin_psth_sigma.setRange(0.2, 20.0)
        self.spin_psth_sigma.setValue(getattr(self.config, "psth_gaussian_sigma_ms", 2.0))
        self.spin_psth_sigma.setSuffix(" ms")
        self.spin_psth_sigma.setSingleStep(0.5)
        self.spin_psth_sigma.setToolTip("Gaussian kernel standard deviation σ in milliseconds")
        form_psth.addRow("Gaussian Kernel σ:", self.spin_psth_sigma)

        self.chk_psth_smooth.toggled.connect(self.spin_psth_sigma.setEnabled)

        self.chk_psth_use_pca = QCheckBox("Apply PCA to PSTH traces")
        self.chk_psth_use_pca.setChecked(getattr(self.config, "psth_use_pca", True))
        self.chk_psth_use_pca.setToolTip("Compress PSTH bins with PCA to retain the target variance fraction")
        form_psth.addRow("PSTH Compression:", self.chk_psth_use_pca)

        self.spin_psth_pca_var = QDoubleSpinBox()
        self.spin_psth_pca_var.setRange(0.50, 0.99)
        self.spin_psth_pca_var.setSingleStep(0.05)
        self.spin_psth_pca_var.setDecimals(2)
        self.spin_psth_pca_var.setValue(getattr(self.config, "psth_pca_variance_explained", 0.80))
        self.spin_psth_pca_var.setToolTip("Fraction of PSTH variance explained by retained principal components")
        form_psth.addRow("PCA Variance Explained:", self.spin_psth_pca_var)

        self.chk_psth_use_pca.toggled.connect(
            lambda checked: self.spin_psth_pca_var.setEnabled(self.chk_psth.isChecked() and checked)
        )

        vbox_psth.addLayout(form_psth)
        layout.addWidget(grp_psth)

        self.chk_psth.toggled.connect(self._on_psth_toggled)

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # 3. Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 8, 0, 0)

        self.btn_reset = QPushButton("Reset to Defaults")
        self.btn_reset.setStyleSheet(DIALOG_SECONDARY_BUTTON_STYLE)
        self.btn_reset.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addStretch()

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Apply")
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setStyleSheet(DIALOG_PRIMARY_BUTTON_STYLE)
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setStyleSheet(DIALOG_SECONDARY_BUTTON_STYLE)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        btn_layout.addWidget(self.button_box)

        main_layout.addLayout(btn_layout)

    def _on_psth_toggled(self, checked: bool) -> None:
        """Enable or disable PSTH time window and normalization inputs based on checkbox."""
        self.spin_w_psth.setEnabled(checked)
        self.combo_psth_norm.setEnabled(checked)
        self.spin_start.setEnabled(checked)
        self.spin_end.setEnabled(checked)
        self.spin_bin.setEnabled(checked)
        self.chk_psth_smooth.setEnabled(checked)
        self.spin_psth_sigma.setEnabled(checked and self.chk_psth_smooth.isChecked())
        self.chk_psth_use_pca.setEnabled(checked)
        self.spin_psth_pca_var.setEnabled(checked and self.chk_psth_use_pca.isChecked())

    def _populate_fields(self, c: FeatureConfiguration) -> None:
        """Populate GUI widgets from FeatureConfiguration."""
        self.chk_latency.setChecked(c.use_latency)
        self.spin_w_latency.setValue(c.weight_latency)
        self.spin_w_latency.setEnabled(c.use_latency)

        self.chk_jitter.setChecked(c.use_jitter)
        self.spin_w_jitter.setValue(c.weight_jitter)
        self.spin_w_jitter.setEnabled(c.use_jitter)

        self.chk_reliability.setChecked(c.use_reliability)
        self.spin_w_reliability.setValue(c.weight_reliability)
        self.spin_w_reliability.setEnabled(c.use_reliability)

        self.chk_salt.setChecked(c.use_salt)
        self.chk_salt_log10.setChecked(c.salt_log10_transform)
        self.spin_w_salt.setValue(c.weight_salt)
        self.spin_w_salt.setEnabled(c.use_salt)
        self.chk_salt_log10.setEnabled(c.use_salt)

        self.chk_baseline_rate.setChecked(c.use_baseline_rate)
        self.spin_w_baseline.setValue(c.weight_baseline_rate)
        self.spin_w_baseline.setEnabled(c.use_baseline_rate)

        self.chk_peak_rate.setChecked(c.use_peak_rate)
        self.spin_w_peak_rate.setValue(c.weight_peak_rate)
        self.spin_w_peak_rate.setEnabled(c.use_peak_rate)

        self.chk_peak_response_rate.setChecked(c.use_peak_response_rate)
        self.spin_w_peak_response_rate.setValue(c.weight_peak_response_rate)
        self.spin_w_peak_response_rate.setEnabled(c.use_peak_response_rate)

        self.chk_time_to_response.setChecked(c.use_time_to_response)
        self.spin_response_std_threshold.setValue(c.response_std_threshold)
        self.spin_w_time_to_response.setValue(c.weight_time_to_response)
        self.spin_w_time_to_response.setEnabled(c.use_time_to_response)
        self.spin_response_std_threshold.setEnabled(c.use_time_to_response)

        self.chk_psth.setChecked(c.use_psth_profile)
        self.spin_w_psth.setValue(c.weight_psth_profile)

        # Normalization combo
        idx = self.combo_psth_norm.findData(c.psth_normalization)
        self.combo_psth_norm.setCurrentIndex(idx if idx >= 0 else 0)

        w0, w1 = c.time_window_ms
        self.spin_start.setValue(w0)
        self.spin_end.setValue(w1)
        self.spin_bin.setValue(c.bin_size_ms)
        self.chk_psth_smooth.setChecked(getattr(c, "psth_smooth", True))
        self.spin_psth_sigma.setValue(getattr(c, "psth_gaussian_sigma_ms", 2.0))
        self.chk_psth_use_pca.setChecked(getattr(c, "psth_use_pca", True))
        self.spin_psth_pca_var.setValue(getattr(c, "psth_pca_variance_explained", 0.80))
        self._on_psth_toggled(c.use_psth_profile)

    def reset_to_defaults(self) -> None:
        """Public alias to reset dialog values to default configuration."""
        self._reset_defaults()

    def _reset_defaults(self) -> None:
        """Reset checkboxes and spinboxes to default publication values."""
        defaults = FeatureConfiguration()
        if self._default_time_window_ms is not None:
            defaults.time_window_ms = self._default_time_window_ms
        self._populate_fields(defaults)

    def get_feature_configuration(self) -> FeatureConfiguration:
        """Return the compiled FeatureConfiguration based on dialog inputs."""
        norm_mode = self.combo_psth_norm.currentData() or "area_normalized"
        return FeatureConfiguration(
            use_latency=self.chk_latency.isChecked(),
            weight_latency=float(self.spin_w_latency.value()),
            use_jitter=self.chk_jitter.isChecked(),
            weight_jitter=float(self.spin_w_jitter.value()),
            use_reliability=self.chk_reliability.isChecked(),
            weight_reliability=float(self.spin_w_reliability.value()),
            use_salt=self.chk_salt.isChecked(),
            salt_log10_transform=self.chk_salt_log10.isChecked(),
            weight_salt=float(self.spin_w_salt.value()),
            use_baseline_rate=self.chk_baseline_rate.isChecked(),
            weight_baseline_rate=float(self.spin_w_baseline.value()),
            use_peak_rate=self.chk_peak_rate.isChecked(),
            weight_peak_rate=float(self.spin_w_peak_rate.value()),
            use_peak_response_rate=self.chk_peak_response_rate.isChecked(),
            weight_peak_response_rate=float(self.spin_w_peak_response_rate.value()),
            use_time_to_response=self.chk_time_to_response.isChecked(),
            weight_time_to_response=float(self.spin_w_time_to_response.value()),
            response_std_threshold=float(self.spin_response_std_threshold.value()),
            use_psth_profile=self.chk_psth.isChecked(),
            weight_psth_profile=float(self.spin_w_psth.value()),
            psth_normalization=str(norm_mode),
            time_window_ms=(float(self.spin_start.value()), float(self.spin_end.value())),
            bin_size_ms=float(self.spin_bin.value()),
            psth_smooth=self.chk_psth_smooth.isChecked(),
            psth_gaussian_sigma_ms=float(self.spin_psth_sigma.value()),
            psth_use_pca=self.chk_psth_use_pca.isChecked(),
            psth_pca_variance_explained=float(self.spin_psth_pca_var.value()),
        )
