"""Affinity Propagation Cluster Seeding Modal Configuration Dialog."""

from typing import Optional
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from optocurator.analysis.clustering import AffinityPropagationParameters, FeatureConfiguration
from optocurator.ui.dialog_feature_settings import FeatureSettingsDialog, format_feature_summary
from optocurator.ui.theme import (
    DIALOG_INFO_BANNER_STYLE,
    DIALOG_MODAL_STYLE,
    DIALOG_NOTE_STYLE,
    DIALOG_PRIMARY_BUTTON_STYLE,
    DIALOG_RADIO_DANGER_STYLE,
    DIALOG_RADIO_SECONDARY_STYLE,
    DIALOG_SECONDARY_BUTTON_STYLE,
    DIALOG_SUBTITLE_STYLE,
)


class SeedClustersDialog(QDialog):
    """Modal dialog allowing users to configure Affinity Propagation parameters for cluster seeding."""

    def __init__(
        self,
        current_params: Optional[AffinityPropagationParameters] = None,
        feature_config: Optional[FeatureConfiguration] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.params = current_params or AffinityPropagationParameters()
        self.feature_config = feature_config or FeatureConfiguration()
        self.setWindowTitle("Seed Clusters (Affinity Propagation)")
        self.setFixedWidth(480)
        self.setModal(True)

        self.setStyleSheet(DIALOG_MODAL_STYLE)

        self._init_ui()
        self._populate_fields(self.params)

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 16, 18, 16)
        main_layout.setSpacing(14)

        intro = QLabel(
            "<b>Affinity propagation chooses the number of clusters automatically.</b> "
            "Units near cluster exemplars are seeded, while distant or ambiguous units remain unassigned."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(DIALOG_SUBTITLE_STYLE)
        main_layout.addWidget(intro)

        # 1. Feature Representation Space Group
        box_feat = QGroupBox("1. Active Feature Space & Weights")
        layout_feat = QVBoxLayout(box_feat)
        layout_feat.setContentsMargins(12, 14, 12, 12)
        layout_feat.setSpacing(8)

        self.chk_use_embedding_features = QCheckBox("Use the current embedding features")
        self.chk_use_embedding_features.setChecked(True)
        self.chk_use_embedding_features.toggled.connect(
            lambda checked: self.btn_edit_features.setEnabled(not checked)
        )
        layout_feat.addWidget(self.chk_use_embedding_features)

        self.lbl_feat_summary = QLabel()
        self.lbl_feat_summary.setWordWrap(True)
        self.lbl_feat_summary.setStyleSheet(DIALOG_INFO_BANNER_STYLE)
        self._update_feature_summary_label()
        layout_feat.addWidget(self.lbl_feat_summary)

        btn_feat_row = QHBoxLayout()
        btn_feat_row.addStretch()
        self.btn_edit_features = QPushButton("Edit Features & Weights...")
        self.btn_edit_features.setStyleSheet(DIALOG_SECONDARY_BUTTON_STYLE)
        self.btn_edit_features.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_edit_features.setEnabled(False)
        self.btn_edit_features.clicked.connect(self._on_edit_features)
        btn_feat_row.addWidget(self.btn_edit_features)
        layout_feat.addLayout(btn_feat_row)

        main_layout.addWidget(box_feat)

        # 2. Affinity Propagation Parameters Group
        box_ap = QGroupBox("2. Affinity Propagation Hyperparameters")
        layout_ap = QFormLayout(box_ap)
        layout_ap.setContentsMargins(12, 14, 12, 12)
        layout_ap.setSpacing(8)

        self.spin_damping = QDoubleSpinBox()
        self.spin_damping.setRange(0.50, 0.99)
        self.spin_damping.setSingleStep(0.05)
        self.spin_damping.setValue(0.80)
        layout_ap.addRow("Damping Factor (0.50 – 0.99):", self.spin_damping)

        self.spin_percentile = QDoubleSpinBox()
        self.spin_percentile.setRange(5.0, 100.0)
        self.spin_percentile.setSingleStep(5.0)
        self.spin_percentile.setSuffix("%")
        self.spin_percentile.setValue(50.0)
        layout_ap.addRow("Exemplar Cutoff Percentile:", self.spin_percentile)

        lbl_pct_help = QLabel("Units within this distance percentile of their cluster exemplar are seeded.")
        lbl_pct_help.setStyleSheet(DIALOG_NOTE_STYLE)
        layout_ap.addRow("", lbl_pct_help)

        self.spin_min_size = QSpinBox()
        self.spin_min_size.setRange(1, 20)
        self.spin_min_size.setValue(2)
        layout_ap.addRow("Minimum Cluster Size:", self.spin_min_size)

        pref_row = QHBoxLayout()
        pref_row.setContentsMargins(0, 0, 0, 0)
        pref_row.setSpacing(8)
        self.chk_pref_auto = QCheckBox("Auto (Median)")
        self.chk_pref_auto.setChecked(True)
        self.spin_preference = QDoubleSpinBox()
        self.spin_preference.setRange(-10000.0, 1000.0)
        self.spin_preference.setSingleStep(5.0)
        self.spin_preference.setDecimals(2)
        self.spin_preference.setValue(-10.0)
        self.spin_preference.setEnabled(False)
        self.chk_pref_auto.toggled.connect(lambda auto: self.spin_preference.setEnabled(not auto))
        pref_row.addWidget(self.chk_pref_auto)
        pref_row.addWidget(self.spin_preference, stretch=1)
        layout_ap.addRow("Preference:", pref_row)

        lbl_pref_help = QLabel(
            "Changing Preference can change the number of clusters: higher (less negative) values "
            "produce more clusters, lower (more negative) values produce fewer clusters."
        )
        lbl_pref_help.setWordWrap(True)
        lbl_pref_help.setStyleSheet(DIALOG_NOTE_STYLE)
        layout_ap.addRow("", lbl_pref_help)

        self.chk_preserve_curated = QCheckBox("Preserve manually curated unit assignments")
        self.chk_preserve_curated.setChecked(True)
        self.chk_preserve_curated.setVisible(False)
        layout_ap.addRow("", self.chk_preserve_curated)

        main_layout.addWidget(box_ap)

        # 2b. Existing Clusters Handling
        box_existing = QGroupBox("3. Existing Clusters")
        layout_existing = QVBoxLayout(box_existing)
        layout_existing.setContentsMargins(12, 14, 12, 12)
        layout_existing.setSpacing(6)

        self.radio_keep_curated = QRadioButton("Keep curated + add seeds")
        self.radio_keep_curated.setStyleSheet(DIALOG_RADIO_SECONDARY_STYLE)
        self.radio_replace_all = QRadioButton("Replace all")
        self.radio_replace_all.setStyleSheet(DIALOG_RADIO_DANGER_STYLE)
        self.existing_group = QButtonGroup(self)
        self.existing_group.addButton(self.radio_keep_curated)
        self.existing_group.addButton(self.radio_replace_all)
        self.radio_keep_curated.setChecked(True)
        self.radio_keep_curated.toggled.connect(self.chk_preserve_curated.setChecked)
        self.chk_preserve_curated.toggled.connect(
            lambda c: self.radio_keep_curated.setChecked(c) if c else self.radio_replace_all.setChecked(True)
        )
        layout_existing.addWidget(self.radio_keep_curated)
        layout_existing.addWidget(self.radio_replace_all)
        main_layout.addWidget(box_existing)

        # 3. Action Buttons
        btn_box = QHBoxLayout()
        btn_box.setSpacing(10)

        self.btn_reset = QPushButton("Reset Defaults")
        self.btn_reset.setStyleSheet(DIALOG_SECONDARY_BUTTON_STYLE)
        self.btn_reset.clicked.connect(self._reset_defaults)
        btn_box.addWidget(self.btn_reset)
        btn_box.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(DIALOG_SECONDARY_BUTTON_STYLE)
        self.btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(self.btn_cancel)

        self.btn_run = QPushButton("Seed clusters")
        self.btn_run.setStyleSheet(DIALOG_PRIMARY_BUTTON_STYLE)
        self.btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run.clicked.connect(self.accept)
        btn_box.addWidget(self.btn_run)

        main_layout.addLayout(btn_box)

    def _update_feature_summary_label(self) -> None:
        """Update feature summary string display."""
        summary = format_feature_summary(self.feature_config)
        self.lbl_feat_summary.setText(f"<b>Active Features:</b> {summary}")

    def _on_edit_features(self) -> None:
        """Open feature settings dialog to edit feature configuration in-place."""
        dlg = FeatureSettingsDialog(self.feature_config, parent=self)
        if dlg.exec():
            self.feature_config = dlg.get_feature_configuration()
            self._update_feature_summary_label()

    def _populate_fields(self, p: AffinityPropagationParameters) -> None:
        """Populate GUI widgets from AffinityPropagationParameters."""
        self.spin_damping.setValue(p.damping)
        self.spin_percentile.setValue(p.similarity_percentile)
        self.spin_min_size.setValue(p.min_cluster_size)
        is_auto = p.preference is None
        self.chk_pref_auto.setChecked(is_auto)
        self.spin_preference.setEnabled(not is_auto)
        if p.preference is not None:
            self.spin_preference.setValue(float(p.preference))
        else:
            self.spin_preference.setValue(-10.0)
        self.chk_preserve_curated.setChecked(p.preserve_curated)

    def _reset_defaults(self) -> None:
        """Reset controls to default parameters."""
        defaults = AffinityPropagationParameters()
        self._populate_fields(defaults)

    def get_parameters(self) -> AffinityPropagationParameters:
        """Extract user-configured parameters into AffinityPropagationParameters dataclass."""
        pref: Optional[float] = None if self.chk_pref_auto.isChecked() else float(self.spin_preference.value())
        return AffinityPropagationParameters(
            damping=float(self.spin_damping.value()),
            similarity_percentile=float(self.spin_percentile.value()),
            min_cluster_size=int(self.spin_min_size.value()),
            preference=pref,
            preserve_curated=self.chk_preserve_curated.isChecked(),
        )

    def get_feature_configuration(self) -> FeatureConfiguration:
        """Return the active FeatureConfiguration."""
        return self.feature_config
