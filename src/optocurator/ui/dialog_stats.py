"""Modal dialog for inspecting metric calculation methods and configuring hyperparameters."""

from dataclasses import replace
from typing import Optional
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from optocurator.models import StatsParameters
from optocurator.ui.theme import (
    DIALOG_DESCRIPTION_STYLE,
    DIALOG_GROUP_BOX_STYLE,
    DIALOG_LINE_EDIT_STYLE,
    DIALOG_MODAL_STYLE,
    DIALOG_PRIMARY_BUTTON_STYLE,
    DIALOG_SCROLL_AREA_STYLE,
    DIALOG_SECTION_HEADER_STYLE,
    DIALOG_SUBTITLE_STYLE,
    DIALOG_TITLE_STYLE,
)


class StatsParametersDialog(QDialog):
    """Modal dialog presenting metric calculation explanations and hyperparameter text inputs."""

    def __init__(
        self,
        current_params: Optional[StatsParameters] = None,
        parent: Optional[QWidget] = None,
        default_params: Optional[StatsParameters] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stats Parameters — Metric Hyperparameters & Formulations")
        self.resize(680, 700)
        self.setMinimumSize(560, 520)
        self.setStyleSheet(DIALOG_MODAL_STYLE)
        self._default_params: StatsParameters = (
            default_params if default_params is not None else StatsParameters.default()
        )
        self.params: StatsParameters = (
            current_params if current_params is not None else replace(self._default_params)
        )

        self._init_ui()
        self._populate_fields(self.params)

    def _init_ui(self) -> None:
        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(18, 18, 18, 16)
        dialog_layout.setSpacing(12)

        # Header Title and Subtitle
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        lbl_title = QLabel("Statistical Metric Parameters")
        lbl_title.setStyleSheet(DIALOG_TITLE_STYLE)
        lbl_desc = QLabel(
            "Configure analytical windows, trial limits, and test hyperparameters for each metric separately. "
            "Each section includes a concise explanation of how that metric is calculated."
        )
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet(DIALOG_SUBTITLE_STYLE)
        header_layout.addWidget(lbl_title)
        header_layout.addWidget(lbl_desc)
        dialog_layout.addLayout(header_layout)

        # Scroll Area for Metric Sections
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(DIALOG_SCROLL_AREA_STYLE)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(14, 14, 14, 14)
        scroll_layout.setSpacing(16)

        line_edit_style = DIALOG_LINE_EDIT_STYLE

        # Section 1: Response Reliability
        group_rel = QGroupBox("1. Response Reliability")
        group_rel.setStyleSheet(DIALOG_GROUP_BOX_STYLE)
        layout_rel = QVBoxLayout(group_rel)
        layout_rel.setSpacing(8)

        desc_rel = QLabel(
            "• Calculation: Proportion of evaluated optical stimulus pulses that elicit at least one spike within "
            "the designated post-stimulus reliability window:\n"
            "  Reliability = (Evoked Pulses) / (Evaluated Pulses) ∈ [0.0, 1.0].\n"
            "• Measures the probability that photostimulation reliably triggers action potential firing (typically ≥ 0.60 for tagged units)."
        )
        desc_rel.setWordWrap(True)
        desc_rel.setStyleSheet(DIALOG_DESCRIPTION_STYLE)
        layout_rel.addWidget(desc_rel)

        form_rel = QFormLayout()
        form_rel.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form_rel.setSpacing(8)

        self.edit_rel_start = QLineEdit()
        self.edit_rel_start.setStyleSheet(line_edit_style)
        self.edit_rel_start.setToolTip("Start of post-pulse reliability evaluation window in ms (e.g. 0.0)")
        form_rel.addRow("Reliability Window Start (ms):", self.edit_rel_start)

        self.edit_rel_end = QLineEdit()
        self.edit_rel_end.setStyleSheet(line_edit_style)
        self.edit_rel_end.setToolTip("End of post-pulse reliability evaluation window in ms (e.g. 20.0)")
        form_rel.addRow("Reliability Window End (ms):", self.edit_rel_end)

        self.edit_rel_trials = QLineEdit()
        self.edit_rel_trials.setStyleSheet(line_edit_style)
        self.edit_rel_trials.setToolTip("Number of stimulus pulses to evaluate ('All' for entire protocol, or integer e.g. 50)")
        form_rel.addRow("Evaluated Pulse Count (Trials):", self.edit_rel_trials)

        layout_rel.addLayout(form_rel)
        scroll_layout.addWidget(group_rel)

        # Section 2: First-Spike Latency
        group_lat = QGroupBox("2. First-Spike Latency")
        group_lat.setStyleSheet(DIALOG_GROUP_BOX_STYLE)
        layout_lat = QVBoxLayout(group_lat)
        layout_lat.setSpacing(8)

        desc_lat = QLabel(
            "• Calculation: Median time (ms) from optical pulse onset (t=0) to the earliest evoked action potential "
            "across active trials within the latency evaluation window.\n"
            "• Short latencies (< 8–10 ms) indicate direct opsin-mediated activation rather than multi-synaptic network transmission."
        )
        desc_lat.setWordWrap(True)
        desc_lat.setStyleSheet(DIALOG_DESCRIPTION_STYLE)
        layout_lat.addWidget(desc_lat)

        form_lat = QFormLayout()
        form_lat.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form_lat.setSpacing(8)

        self.edit_lat_start = QLineEdit()
        self.edit_lat_start.setStyleSheet(line_edit_style)
        self.edit_lat_start.setToolTip("Start of post-pulse latency evaluation window in ms (e.g. 0.0)")
        form_lat.addRow("Latency Window Start (ms):", self.edit_lat_start)

        self.edit_lat_end = QLineEdit()
        self.edit_lat_end.setStyleSheet(line_edit_style)
        self.edit_lat_end.setToolTip("End of post-pulse latency evaluation window in ms (e.g. 20.0)")
        form_lat.addRow("Latency Window End (ms):", self.edit_lat_end)

        self.edit_lat_trials = QLineEdit()
        self.edit_lat_trials.setStyleSheet(line_edit_style)
        self.edit_lat_trials.setToolTip("Number of stimulus pulses to evaluate ('All' for entire protocol, or integer e.g. 50)")
        form_lat.addRow("Evaluated Pulse Count (Trials):", self.edit_lat_trials)

        layout_lat.addLayout(form_lat)
        scroll_layout.addWidget(group_lat)

        # Section 3: First-Spike Jitter
        group_jit = QGroupBox("3. First-Spike Jitter")
        group_jit.setStyleSheet(DIALOG_GROUP_BOX_STYLE)
        layout_jit = QVBoxLayout(group_jit)
        layout_jit.setSpacing(8)

        desc_jit = QLabel(
            "• Calculation: Standard deviation of first-spike latencies (ms) across evoked trials within the jitter evaluation window.\n"
            "• Measures temporal fidelity and spike-timing precision; directly tagged units typically exhibit sub-millisecond or low jitter (≤ 2.0 ms)."
        )
        desc_jit.setWordWrap(True)
        desc_jit.setStyleSheet(DIALOG_DESCRIPTION_STYLE)
        layout_jit.addWidget(desc_jit)

        form_jit = QFormLayout()
        form_jit.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form_jit.setSpacing(8)

        self.edit_jit_start = QLineEdit()
        self.edit_jit_start.setStyleSheet(line_edit_style)
        self.edit_jit_start.setToolTip("Start of post-pulse jitter evaluation window in ms (e.g. 0.0)")
        form_jit.addRow("Jitter Window Start (ms):", self.edit_jit_start)

        self.edit_jit_end = QLineEdit()
        self.edit_jit_end.setStyleSheet(line_edit_style)
        self.edit_jit_end.setToolTip("End of post-pulse jitter evaluation window in ms (e.g. 20.0)")
        form_jit.addRow("Jitter Window End (ms):", self.edit_jit_end)

        self.edit_jit_trials = QLineEdit()
        self.edit_jit_trials.setStyleSheet(line_edit_style)
        self.edit_jit_trials.setToolTip("Number of stimulus pulses to evaluate ('All' for entire protocol, or integer e.g. 50)")
        form_jit.addRow("Evaluated Pulse Count (Trials):", self.edit_jit_trials)

        layout_jit.addLayout(form_jit)
        scroll_layout.addWidget(group_jit)

        # Section 4: SALT (Stimulus-Associated Spike Latency Test)
        group_salt = QGroupBox("4. Stimulus-Associated Spike Latency Test (SALT)")
        group_salt.setStyleSheet(DIALOG_GROUP_BOX_STYLE)
        layout_salt = QVBoxLayout(group_salt)
        layout_salt.setSpacing(8)

        desc_salt = QLabel(
            "• Calculation: Non-parametric statistical test (Kvitsiani et al., Nature 2013) that tests whether post-stimulus "
            "first-spike latencies are stimulus-locked vs. spontaneous baseline intervals. Uses Jensen-Shannon divergence (Information Radius) "
            "and permutation testing to compute the significance p-value."
        )
        desc_salt.setWordWrap(True)
        desc_salt.setStyleSheet(DIALOG_DESCRIPTION_STYLE)
        layout_salt.addWidget(desc_salt)

        form_salt = QFormLayout()
        form_salt.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form_salt.setSpacing(8)

        self.edit_salt_start = QLineEdit()
        self.edit_salt_start.setStyleSheet(line_edit_style)
        self.edit_salt_start.setToolTip(
            f"Test window start relative to pulse onset in ms (default: {self._default_params.salt_window_start_ms:.1f})"
        )
        form_salt.addRow("SALT Test Window Start (ms):", self.edit_salt_start)

        self.edit_salt_end = QLineEdit()
        self.edit_salt_end.setStyleSheet(line_edit_style)
        self.edit_salt_end.setToolTip(
            f"Test window end relative to pulse onset in ms (default: {self._default_params.salt_window_end_ms:.1f})"
        )
        form_salt.addRow("SALT Test Window End (ms):", self.edit_salt_end)

        self.edit_salt_base_start = QLineEdit()
        self.edit_salt_base_start.setStyleSheet(line_edit_style)
        self.edit_salt_base_start.setToolTip(
            f"Pre-pulse baseline window start in ms (default: {self._default_params.salt_baseline_start_ms:.1f})"
        )
        form_salt.addRow("Baseline Interval Start (ms):", self.edit_salt_base_start)

        self.edit_salt_base_end = QLineEdit()
        self.edit_salt_base_end.setStyleSheet(line_edit_style)
        self.edit_salt_base_end.setToolTip(
            f"Pre-pulse baseline window end in ms (default: {self._default_params.salt_baseline_end_ms:.1f})"
        )
        form_salt.addRow("Baseline Interval End (ms):", self.edit_salt_base_end)

        self.edit_salt_bins = QLineEdit()
        self.edit_salt_bins.setStyleSheet(line_edit_style)
        self.edit_salt_bins.setToolTip(
            f"Number of latency histogram bins in the test interval (default: {self._default_params.salt_n_bins})"
        )
        form_salt.addRow("Latency Bins (Count):", self.edit_salt_bins)

        self.edit_salt_perms = QLineEdit()
        self.edit_salt_perms.setStyleSheet(line_edit_style)
        self.edit_salt_perms.setToolTip(
            f"Number of permutation iterations to construct the null distribution (default: {self._default_params.salt_n_permutations})"
        )
        form_salt.addRow("Permutation Iterations:", self.edit_salt_perms)

        # Artifact Exclusion Window
        self.chk_salt_exclusion = QCheckBox("Enable Artifact Exclusion Window (Onsets & Offsets)")
        self.chk_salt_exclusion.setStyleSheet(DIALOG_SECTION_HEADER_STYLE)
        self.chk_salt_exclusion.setToolTip("Exclude time window around pulse onsets and offsets where the signal was artificially flattened or corrupted by optical artifacts (e.g. -1.0 to +1.0 ms)")
        form_salt.addRow("", self.chk_salt_exclusion)

        self.edit_salt_excl_start = QLineEdit("-1.0")
        self.edit_salt_excl_start.setStyleSheet(line_edit_style)
        self.edit_salt_excl_start.setToolTip(
            f"Artifact exclusion start in ms relative to pulse onset and offset (default: {self._default_params.salt_exclusion_window_start_ms:.1f})"
        )
        form_salt.addRow("Exclusion Window Start (ms):", self.edit_salt_excl_start)

        self.edit_salt_excl_end = QLineEdit("1.0")
        self.edit_salt_excl_end.setStyleSheet(line_edit_style)
        self.edit_salt_excl_end.setToolTip(
            f"Artifact exclusion end in ms relative to pulse onset and offset (default: {self._default_params.salt_exclusion_window_end_ms:.1f})"
        )
        form_salt.addRow("Exclusion Window End (ms):", self.edit_salt_excl_end)

        self.chk_salt_exclusion.toggled.connect(self.edit_salt_excl_start.setEnabled)
        self.chk_salt_exclusion.toggled.connect(self.edit_salt_excl_end.setEnabled)

        layout_salt.addLayout(form_salt)
        scroll_layout.addWidget(group_salt)

        # Section 5: Waveform Correlation
        group_wf = QGroupBox("5. Waveform Correlation")
        group_wf.setStyleSheet(DIALOG_GROUP_BOX_STYLE)
        layout_wf = QVBoxLayout(group_wf)
        layout_wf.setSpacing(8)

        desc_wf = QLabel(
            "• Calculation: Pearson correlation (r) between the mean light-evoked spike waveform "
            "(spikes inside the evoked window) and the mean spontaneous waveform (spikes inside the baseline window) "
            "on the unit's peak channel."
        )
        desc_wf.setWordWrap(True)
        desc_wf.setStyleSheet(DIALOG_DESCRIPTION_STYLE)
        layout_wf.addWidget(desc_wf)

        form_wf = QFormLayout()
        form_wf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form_wf.setSpacing(8)

        self.edit_wf_evoked_start = QLineEdit()
        self.edit_wf_evoked_start.setStyleSheet(line_edit_style)
        self.edit_wf_evoked_start.setToolTip(
            f"Start of light-evoked waveform window relative to pulse onset in ms (default: {self._default_params.wf_evoked_start_ms:.1f})"
        )
        form_wf.addRow("Evoked Window Start (ms):", self.edit_wf_evoked_start)

        self.edit_wf_evoked_end = QLineEdit()
        self.edit_wf_evoked_end.setStyleSheet(line_edit_style)
        self.edit_wf_evoked_end.setToolTip(
            f"End of light-evoked waveform window relative to pulse onset in ms (default: {self._default_params.wf_evoked_end_ms:.1f})"
        )
        form_wf.addRow("Evoked Window End (ms):", self.edit_wf_evoked_end)

        self.edit_wf_baseline_start = QLineEdit()
        self.edit_wf_baseline_start.setStyleSheet(line_edit_style)
        self.edit_wf_baseline_start.setToolTip(
            f"Start of spontaneous baseline waveform window in ms (default: {self._default_params.wf_baseline_start_ms:.1f})"
        )
        form_wf.addRow("Baseline Window Start (ms):", self.edit_wf_baseline_start)

        self.edit_wf_baseline_end = QLineEdit()
        self.edit_wf_baseline_end.setStyleSheet(line_edit_style)
        self.edit_wf_baseline_end.setToolTip(
            f"End of spontaneous baseline waveform window in ms (default: {self._default_params.wf_baseline_end_ms:.1f})"
        )
        form_wf.addRow("Baseline Window End (ms):", self.edit_wf_baseline_end)

        self.edit_wf_window_start = QLineEdit()
        self.edit_wf_window_start.setStyleSheet(line_edit_style)
        self.edit_wf_window_start.setToolTip(
            f"Start of extracted spike waveform snippet window in ms (default: {self._default_params.wf_window_start_ms:.1f})"
        )
        self.edit_wf_window_end = QLineEdit()
        self.edit_wf_window_end.setStyleSheet(line_edit_style)
        self.edit_wf_window_end.setToolTip(
            f"End of extracted spike waveform snippet window in ms (default: {self._default_params.wf_window_end_ms:.1f})"
        )
        wf_win_row = QHBoxLayout()
        wf_win_row.setSpacing(6)
        wf_win_row.addWidget(self.edit_wf_window_start)
        wf_win_row.addWidget(QLabel("to"))
        wf_win_row.addWidget(self.edit_wf_window_end)
        form_wf.addRow("Waveform window (ms):", wf_win_row)

        self.edit_wf_max_spikes = QLineEdit()
        self.edit_wf_max_spikes.setStyleSheet(line_edit_style)
        self.edit_wf_max_spikes.setToolTip(
            f"Maximum number of spikes per group (evoked / spontaneous) for waveform extraction and plotting (default: {self._default_params.wf_max_spikes})"
        )
        form_wf.addRow("Max spikes per group:", self.edit_wf_max_spikes)

        layout_wf.addLayout(form_wf)
        scroll_layout.addWidget(group_wf)

        scroll_area.setWidget(scroll_widget)
        dialog_layout.addWidget(scroll_area)

        # Button Bar
        btn_bar = QHBoxLayout()
        btn_reset = QPushButton("Reset Defaults")
        btn_reset.setObjectName("secondaryButton")
        btn_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_reset.clicked.connect(self._reset_defaults)
        btn_bar.addWidget(btn_reset)

        btn_bar.addStretch()

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setObjectName("secondaryButton")
        btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_cancel.clicked.connect(self.reject)
        btn_bar.addWidget(btn_cancel)

        self.btn_apply = QPushButton("Apply and recompute")
        self.btn_apply.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_apply.setStyleSheet(DIALOG_PRIMARY_BUTTON_STYLE)
        self.btn_apply.clicked.connect(self._on_apply_clicked)
        btn_bar.addWidget(self.btn_apply)

        dialog_layout.addLayout(btn_bar)

    def _populate_fields(self, p: StatsParameters) -> None:
        """Fill text inputs from StatsParameters instance."""
        # Reliability
        self.edit_rel_start.setText(f"{p.reliability_window_start_ms:.1f}")
        self.edit_rel_end.setText(f"{p.reliability_window_end_ms:.1f}")
        self.edit_rel_trials.setText("All" if p.reliability_trials is None else str(p.reliability_trials))

        # Latency
        self.edit_lat_start.setText(f"{p.latency_window_start_ms:.1f}")
        self.edit_lat_end.setText(f"{p.latency_window_end_ms:.1f}")
        self.edit_lat_trials.setText("All" if p.latency_trials is None else str(p.latency_trials))

        # Jitter
        self.edit_jit_start.setText(f"{p.jitter_window_start_ms:.1f}")
        self.edit_jit_end.setText(f"{p.jitter_window_end_ms:.1f}")
        self.edit_jit_trials.setText("All" if p.jitter_trials is None else str(p.jitter_trials))

        # SALT
        self.edit_salt_start.setText(f"{p.salt_window_start_ms:.1f}")
        self.edit_salt_end.setText(f"{p.salt_window_end_ms:.1f}")
        self.edit_salt_base_start.setText(f"{p.salt_baseline_start_ms:.1f}")
        self.edit_salt_base_end.setText(f"{p.salt_baseline_end_ms:.1f}")
        self.edit_salt_bins.setText(str(p.salt_n_bins))
        self.edit_salt_perms.setText(str(p.salt_n_permutations))
        self.chk_salt_exclusion.setChecked(p.salt_exclusion_enabled)
        self.edit_salt_excl_start.setText(f"{p.salt_exclusion_window_start_ms:.1f}")
        self.edit_salt_excl_end.setText(f"{p.salt_exclusion_window_end_ms:.1f}")
        self.edit_salt_excl_start.setEnabled(p.salt_exclusion_enabled)
        self.edit_salt_excl_end.setEnabled(p.salt_exclusion_enabled)

        # Waveform correlation
        self.edit_wf_evoked_start.setText(f"{p.wf_evoked_start_ms:.1f}")
        self.edit_wf_evoked_end.setText(f"{p.wf_evoked_end_ms:.1f}")
        self.edit_wf_baseline_start.setText(f"{p.wf_baseline_start_ms:.1f}")
        self.edit_wf_baseline_end.setText(f"{p.wf_baseline_end_ms:.1f}")
        self.edit_wf_window_start.setText(f"{p.wf_window_start_ms:.1f}")
        self.edit_wf_window_end.setText(f"{p.wf_window_end_ms:.1f}")
        self.edit_wf_max_spikes.setText(str(p.wf_max_spikes))

    def _reset_defaults(self) -> None:
        """Reset text inputs to configured default parameters."""
        self._populate_fields(self._default_params)

    def _on_apply_clicked(self) -> None:
        """Validate input text boxes and construct new StatsParameters."""
        try:
            # 1. Reliability
            rel_start = float(self.edit_rel_start.text().strip())
            rel_end = float(self.edit_rel_end.text().strip())
            if rel_start >= rel_end:
                QMessageBox.warning(self, "Invalid Reliability Window", "Reliability window start must be less than window end.")
                return

            rel_trials_text = self.edit_rel_trials.text().strip().lower()
            if rel_trials_text in ("all", "", "none"):
                rel_trials = None
            else:
                rel_trials = int(rel_trials_text)
                if rel_trials <= 0:
                    QMessageBox.warning(self, "Invalid Trial Count", "Reliability evaluated pulse count must be a positive integer or 'All'.")
                    return

            # 2. Latency
            lat_start = float(self.edit_lat_start.text().strip())
            lat_end = float(self.edit_lat_end.text().strip())
            if lat_start >= lat_end:
                QMessageBox.warning(self, "Invalid Latency Window", "Latency window start must be less than window end.")
                return

            lat_trials_text = self.edit_lat_trials.text().strip().lower()
            if lat_trials_text in ("all", "", "none"):
                lat_trials = None
            else:
                lat_trials = int(lat_trials_text)
                if lat_trials <= 0:
                    QMessageBox.warning(self, "Invalid Trial Count", "Latency evaluated pulse count must be a positive integer or 'All'.")
                    return

            # 3. Jitter
            jit_start = float(self.edit_jit_start.text().strip())
            jit_end = float(self.edit_jit_end.text().strip())
            if jit_start >= jit_end:
                QMessageBox.warning(self, "Invalid Jitter Window", "Jitter window start must be less than window end.")
                return

            jit_trials_text = self.edit_jit_trials.text().strip().lower()
            if jit_trials_text in ("all", "", "none"):
                jit_trials = None
            else:
                jit_trials = int(jit_trials_text)
                if jit_trials <= 0:
                    QMessageBox.warning(self, "Invalid Trial Count", "Jitter evaluated pulse count must be a positive integer or 'All'.")
                    return

            # 4. SALT
            salt_start = float(self.edit_salt_start.text().strip())
            salt_end = float(self.edit_salt_end.text().strip())
            if salt_start >= salt_end:
                QMessageBox.warning(self, "Invalid SALT Window", "SALT test window start must be less than test window end.")
                return

            salt_base_start = float(self.edit_salt_base_start.text().strip())
            salt_base_end = float(self.edit_salt_base_end.text().strip())
            if salt_base_start >= salt_base_end:
                QMessageBox.warning(self, "Invalid SALT Baseline", "SALT baseline start must be less than baseline end.")
                return

            salt_bins = int(self.edit_salt_bins.text().strip())
            if salt_bins < 2:
                QMessageBox.warning(self, "Invalid SALT Bins", "SALT latency bins must be at least 2.")
                return

            salt_perms = int(self.edit_salt_perms.text().strip())
            if salt_perms < 10:
                QMessageBox.warning(self, "Invalid SALT Permutations", "Permutation iterations must be at least 10.")
                return

            salt_excl_enabled = self.chk_salt_exclusion.isChecked()
            salt_excl_start = float(self.edit_salt_excl_start.text().strip())
            salt_excl_end = float(self.edit_salt_excl_end.text().strip())
            if salt_excl_enabled and salt_excl_start >= salt_excl_end:
                QMessageBox.warning(
                    self,
                    "Invalid Exclusion Window",
                    "Artifact exclusion window start must be less than exclusion window end.",
                )
                return

            # 5. Waveform Correlation
            wf_evoked_start = float(self.edit_wf_evoked_start.text().strip())
            wf_evoked_end = float(self.edit_wf_evoked_end.text().strip())
            if wf_evoked_start >= wf_evoked_end:
                QMessageBox.warning(
                    self,
                    "Invalid Evoked Waveform Window",
                    "Evoked waveform window start must be less than window end.",
                )
                return

            wf_baseline_start = float(self.edit_wf_baseline_start.text().strip())
            wf_baseline_end = float(self.edit_wf_baseline_end.text().strip())
            if wf_baseline_start >= wf_baseline_end:
                QMessageBox.warning(
                    self,
                    "Invalid Baseline Waveform Window",
                    "Baseline waveform window start must be less than window end.",
                )
                return

            wf_window_start = float(self.edit_wf_window_start.text().strip())
            wf_window_end = float(self.edit_wf_window_end.text().strip())
            if wf_window_start >= wf_window_end:
                QMessageBox.warning(
                    self,
                    "Invalid Waveform Window",
                    "Waveform window start must be less than window end.",
                )
                return

            wf_max_spikes = int(self.edit_wf_max_spikes.text().strip())
            if wf_max_spikes < 1:
                QMessageBox.warning(
                    self,
                    "Invalid Max Spikes",
                    "Max spikes per group must be an integer >= 1.",
                )
                return

        except ValueError as err:
            QMessageBox.warning(self, "Invalid Numerical Value", f"Please check all text inputs for valid numbers.\n\nError: {err}")
            return

        self.params = StatsParameters(
            reliability_window_start_ms=rel_start,
            reliability_window_end_ms=rel_end,
            reliability_trials=rel_trials,
            latency_window_start_ms=lat_start,
            latency_window_end_ms=lat_end,
            latency_trials=lat_trials,
            jitter_window_start_ms=jit_start,
            jitter_window_end_ms=jit_end,
            jitter_trials=jit_trials,
            salt_window_start_ms=salt_start,
            salt_window_end_ms=salt_end,
            salt_baseline_start_ms=salt_base_start,
            salt_baseline_end_ms=salt_base_end,
            salt_n_bins=salt_bins,
            salt_n_permutations=salt_perms,
            salt_exclusion_enabled=salt_excl_enabled,
            salt_exclusion_window_start_ms=salt_excl_start,
            salt_exclusion_window_end_ms=salt_excl_end,
            wf_evoked_start_ms=wf_evoked_start,
            wf_evoked_end_ms=wf_evoked_end,
            wf_baseline_start_ms=wf_baseline_start,
            wf_baseline_end_ms=wf_baseline_end,
            wf_window_start_ms=wf_window_start,
            wf_window_end_ms=wf_window_end,
            wf_max_spikes=wf_max_spikes,
        )
        self.accept()

    def get_parameters(self) -> StatsParameters:
        """Return the validated StatsParameters."""
        return self.params
