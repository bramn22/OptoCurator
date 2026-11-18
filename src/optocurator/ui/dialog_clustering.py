"""Response Clustering & 2D Embedding Configuration Modal Dialogs."""

from typing import Optional
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from optocurator.analysis.clustering import (
    ClusteringParameters,
    EmbeddingParameters,
    FeatureConfiguration,
)
from optocurator.ui.dialog_feature_settings import FeatureSettingsDialog, format_feature_summary
from optocurator.ui.theme import (
    DIALOG_INFO_BANNER_STYLE,
    DIALOG_MODAL_STYLE,
    DIALOG_PRIMARY_BUTTON_STYLE,
    DIALOG_RADIO_DANGER_STYLE,
    DIALOG_RADIO_SECONDARY_STYLE,
    DIALOG_SECONDARY_BUTTON_STYLE,
    DIALOG_SUBTITLE_STYLE,
)


class HierarchicalClusteringDialog(QDialog):
    """Modal dialog allowing researchers to configure and run response clustering on pooled units."""

    def __init__(
        self,
        current_params: Optional[ClusteringParameters] = None,
        feature_config: Optional[FeatureConfiguration] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.params = current_params or ClusteringParameters()
        self.feature_config = feature_config or self.params.feature_config or FeatureConfiguration()
        self.setWindowTitle("Response Clustering Configuration")
        self.setFixedWidth(490)
        self.setModal(True)

        self.setStyleSheet(DIALOG_MODAL_STYLE)

        self._init_ui()
        self._populate_fields(self.params)

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 16, 18, 16)
        main_layout.setSpacing(12)

        # Header description
        lbl_info = QLabel(
            "<b>Unsupervised clustering of units.</b> Partition pooled optotagged units into response types "
            "using the selected algorithm, feature representation, and parameters."
        )
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet(DIALOG_SUBTITLE_STYLE)
        main_layout.addWidget(lbl_info)

        # 1. Feature Representation Space Group
        box_feat = QGroupBox("1. Active Feature Space & Weights")
        layout_feat = QVBoxLayout(box_feat)
        layout_feat.setContentsMargins(12, 14, 12, 12)
        layout_feat.setSpacing(8)

        self.chk_use_embedding_features = QCheckBox("Use the current embedding features")
        self.chk_use_embedding_features.setChecked(True)
        self.chk_use_embedding_features.toggled.connect(self._on_use_embedding_features_toggled)
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

        # 2. Algorithm Selector & Parameters Group
        box_algo = QGroupBox("2. Clustering Algorithm & Parameters")
        layout_algo = QVBoxLayout(box_algo)
        layout_algo.setContentsMargins(12, 14, 12, 12)
        layout_algo.setSpacing(8)

        form_algo_choice = QFormLayout()
        self.combo_algorithm = QComboBox()
        self.combo_algorithm.addItem("Agglomerative Hierarchical", "hierarchical")
        self.combo_algorithm.addItem("K-Means Clustering", "kmeans")
        self.combo_algorithm.addItem("Gaussian Mixture Model (GMM)", "gmm")
        self.combo_algorithm.currentIndexChanged.connect(self._on_algorithm_changed)
        form_algo_choice.addRow("Algorithm:", self.combo_algorithm)
        layout_algo.addLayout(form_algo_choice)

        # Stack of algorithm parameter panels
        self.stack_algo = QStackedWidget()

        # Panel 0: Hierarchical
        panel_hier = QWidget()
        vbox_hier = QVBoxLayout(panel_hier)
        vbox_hier.setContentsMargins(0, 0, 0, 0)
        vbox_hier.setSpacing(8)

        form_metric = QFormLayout()
        self.combo_metric = QComboBox()
        self.combo_metric.addItems(["euclidean", "cosine", "correlation", "cityblock"])
        form_metric.addRow("Distance Metric:", self.combo_metric)

        self.combo_linkage = QComboBox()
        self.combo_linkage.addItems(["ward", "complete", "average", "single"])
        form_metric.addRow("Linkage Criterion:", self.combo_linkage)
        self.combo_linkage.currentTextChanged.connect(self._on_linkage_changed)
        self.combo_metric.currentTextChanged.connect(self._on_metric_changed)
        vbox_hier.addLayout(form_metric)

        cutoff_box = QGroupBox("Cutoff Strategy")
        cutoff_layout = QVBoxLayout(cutoff_box)
        cutoff_layout.setContentsMargins(8, 8, 8, 8)
        cutoff_layout.setSpacing(6)

        row_k = QHBoxLayout()
        self.radio_k = QRadioButton("Number of Clusters (k):")
        self.spin_k = QSpinBox()
        self.spin_k.setRange(2, 50)
        self.spin_k.setValue(4)
        row_k.addWidget(self.radio_k)
        row_k.addSpacing(8)
        row_k.addWidget(self.spin_k)
        row_k.addStretch()
        cutoff_layout.addLayout(row_k)

        row_d = QHBoxLayout()
        self.radio_dist = QRadioButton("Distance Threshold (d_cut):")
        self.spin_dist = QDoubleSpinBox()
        self.spin_dist.setRange(0.05, 50.0)
        self.spin_dist.setSingleStep(0.1)
        self.spin_dist.setDecimals(2)
        self.spin_dist.setValue(1.50)
        self.spin_dist.setEnabled(False)
        row_d.addWidget(self.radio_dist)
        row_d.addSpacing(8)
        row_d.addWidget(self.spin_dist)
        row_d.addStretch()
        cutoff_layout.addLayout(row_d)

        self.cutoff_group = QButtonGroup(self)
        self.cutoff_group.addButton(self.radio_k)
        self.cutoff_group.addButton(self.radio_dist)
        self.radio_k.setChecked(True)
        self.radio_k.toggled.connect(self._on_cutoff_toggled)

        vbox_hier.addWidget(cutoff_box)
        self.stack_algo.addWidget(panel_hier)

        # Panel 1: K-Means
        panel_kmeans = QWidget()
        form_kmeans = QFormLayout(panel_kmeans)
        form_kmeans.setContentsMargins(0, 0, 0, 0)
        form_kmeans.setSpacing(6)

        self.spin_k_kmeans = QSpinBox()
        self.spin_k_kmeans.setRange(2, 50)
        self.spin_k_kmeans.setValue(4)
        form_kmeans.addRow("Number of Clusters (k):", self.spin_k_kmeans)

        self.combo_kmeans_init = QComboBox()
        self.combo_kmeans_init.addItems(["k-means++", "random"])
        form_kmeans.addRow("Initialization:", self.combo_kmeans_init)

        self.spin_kmeans_n_init = QSpinBox()
        self.spin_kmeans_n_init.setRange(1, 50)
        self.spin_kmeans_n_init.setValue(10)
        form_kmeans.addRow("Number of Initializations (n_init):", self.spin_kmeans_n_init)

        self.spin_kmeans_max_iter = QSpinBox()
        self.spin_kmeans_max_iter.setRange(50, 1000)
        self.spin_kmeans_max_iter.setSingleStep(50)
        self.spin_kmeans_max_iter.setValue(300)
        form_kmeans.addRow("Max Iterations:", self.spin_kmeans_max_iter)

        self.stack_algo.addWidget(panel_kmeans)

        # Panel 2: GMM
        panel_gmm = QWidget()
        form_gmm = QFormLayout(panel_gmm)
        form_gmm.setContentsMargins(0, 0, 0, 0)
        form_gmm.setSpacing(6)

        self.spin_k_gmm = QSpinBox()
        self.spin_k_gmm.setRange(2, 50)
        self.spin_k_gmm.setValue(4)
        form_gmm.addRow("Number of Components (k):", self.spin_k_gmm)

        self.combo_gmm_covar = QComboBox()
        self.combo_gmm_covar.addItems(["full", "tied", "diag", "spherical"])
        form_gmm.addRow("Covariance Type:", self.combo_gmm_covar)

        self.spin_gmm_reg = QDoubleSpinBox()
        self.spin_gmm_reg.setRange(1e-9, 1.0)
        self.spin_gmm_reg.setDecimals(6)
        self.spin_gmm_reg.setValue(1e-6)
        form_gmm.addRow("Covariance Regularization:", self.spin_gmm_reg)

        self.spin_gmm_max_iter = QSpinBox()
        self.spin_gmm_max_iter.setRange(20, 1000)
        self.spin_gmm_max_iter.setSingleStep(20)
        self.spin_gmm_max_iter.setValue(100)
        form_gmm.addRow("Max Iterations:", self.spin_gmm_max_iter)

        self.stack_algo.addWidget(panel_gmm)

        layout_algo.addWidget(self.stack_algo)
        main_layout.addWidget(box_algo)

        # 3. Existing Clusters Handling
        box_existing = QGroupBox("3. Existing Clusters")
        layout_existing = QVBoxLayout(box_existing)
        layout_existing.setContentsMargins(12, 14, 12, 12)
        layout_existing.setSpacing(6)

        self.radio_keep_curated = QRadioButton("Keep curated + cluster the rest")
        self.radio_keep_curated.setStyleSheet(DIALOG_RADIO_SECONDARY_STYLE)
        self.radio_replace_all = QRadioButton("Replace all")
        self.radio_replace_all.setStyleSheet(DIALOG_RADIO_DANGER_STYLE)
        self.existing_group = QButtonGroup(self)
        self.existing_group.addButton(self.radio_keep_curated)
        self.existing_group.addButton(self.radio_replace_all)
        self.radio_keep_curated.setChecked(True)
        layout_existing.addWidget(self.radio_keep_curated)
        layout_existing.addWidget(self.radio_replace_all)
        main_layout.addWidget(box_existing)

        # Dialog Buttons
        btn_box = QHBoxLayout()
        btn_box.setSpacing(10)
        btn_box.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(DIALOG_SECONDARY_BUTTON_STYLE)
        self.btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(self.btn_cancel)

        self.btn_run = QPushButton("Run clustering")
        self.btn_run.setStyleSheet(DIALOG_PRIMARY_BUTTON_STYLE)
        self.btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run.clicked.connect(self.accept)
        btn_box.addWidget(self.btn_run)

        main_layout.addLayout(btn_box)

    def _on_use_embedding_features_toggled(self, checked: bool) -> None:
        """Enable custom feature editing only when not using current embedding features."""
        self.btn_edit_features.setEnabled(not checked)

    def keep_curated(self) -> bool:
        """Return True if curated assignments should be preserved."""
        return self.radio_keep_curated.isChecked()

    def _update_feature_summary_label(self) -> None:
        """Update active feature space summary string."""
        summary = format_feature_summary(self.feature_config)
        self.lbl_feat_summary.setText(f"<b>Active Features:</b> {summary}")

    def _on_edit_features(self) -> None:
        """Open FeatureSettingsDialog to modify features in-place."""
        dlg = FeatureSettingsDialog(self.feature_config, parent=self)
        if dlg.exec():
            self.feature_config = dlg.get_feature_configuration()
            self._update_feature_summary_label()

    def _on_algorithm_changed(self, index: int) -> None:
        """Switch active parameter panel when algorithm changes."""
        self.stack_algo.setCurrentIndex(index)

    def _on_linkage_changed(self, text: str) -> None:
        """Ensure Euclidean metric is enforced when Ward linkage is chosen."""
        if text.lower() == "ward":
            self.combo_metric.setCurrentText("euclidean")
            self.combo_metric.setEnabled(False)
        else:
            self.combo_metric.setEnabled(True)

    def _on_metric_changed(self, text: str) -> None:
        """Ensure linkage is switched away from Ward if a non-Euclidean metric is chosen."""
        if text.lower() != "euclidean" and self.combo_linkage.currentText().lower() == "ward":
            self.combo_linkage.setCurrentText("average")

    def _on_cutoff_toggled(self, checked: bool) -> None:
        """Toggle between target cluster count (k) and distance threshold."""
        self.spin_k.setEnabled(checked)
        self.spin_dist.setEnabled(not checked)

    def _populate_fields(self, p: ClusteringParameters) -> None:
        """Populate GUI widgets from current ClusteringParameters."""
        # Algorithm
        idx_algo = self.combo_algorithm.findData(getattr(p, "algorithm", "hierarchical").lower())
        if idx_algo >= 0:
            self.combo_algorithm.setCurrentIndex(idx_algo)

        # Hierarchical
        idx_metric = self.combo_metric.findText(p.metric.lower())
        if idx_metric >= 0:
            self.combo_metric.setCurrentIndex(idx_metric)

        idx_link = self.combo_linkage.findText(p.linkage.lower())
        if idx_link >= 0:
            self.combo_linkage.setCurrentIndex(idx_link)

        if p.cutoff_mode == "distance":
            self.radio_dist.setChecked(True)
            self.spin_dist.setValue(p.distance_threshold)
        else:
            self.radio_k.setChecked(True)
            self.spin_k.setValue(p.n_clusters)

        # K-Means
        self.spin_k_kmeans.setValue(p.n_clusters)
        idx_k_init = self.combo_kmeans_init.findText(getattr(p, "kmeans_init", "k-means++"))
        if idx_k_init >= 0:
            self.combo_kmeans_init.setCurrentIndex(idx_k_init)
        self.spin_kmeans_n_init.setValue(getattr(p, "kmeans_n_init", 10))
        self.spin_kmeans_max_iter.setValue(getattr(p, "kmeans_max_iter", 300))

        # GMM
        self.spin_k_gmm.setValue(p.n_clusters)
        idx_gmm_cov = self.combo_gmm_covar.findText(getattr(p, "gmm_covariance_type", "full"))
        if idx_gmm_cov >= 0:
            self.combo_gmm_covar.setCurrentIndex(idx_gmm_cov)
        self.spin_gmm_reg.setValue(getattr(p, "gmm_reg_covar", 1e-6))
        self.spin_gmm_max_iter.setValue(getattr(p, "gmm_max_iter", 100))

    def get_parameters(self) -> ClusteringParameters:
        """Extract user-configured parameters into a ClusteringParameters dataclass."""
        algo = self.combo_algorithm.currentData() or "hierarchical"

        cutoff = "k" if self.radio_k.isChecked() else "distance"
        if algo == "kmeans":
            k_val = self.spin_k_kmeans.value()
        elif algo == "gmm":
            k_val = self.spin_k_gmm.value()
        else:
            k_val = self.spin_k.value()

        return ClusteringParameters(
            metric=self.combo_metric.currentText().lower(),
            linkage=self.combo_linkage.currentText().lower(),
            cutoff_mode=cutoff,
            n_clusters=k_val,
            distance_threshold=self.spin_dist.value(),
            feature_config=self.feature_config,
            algorithm=str(algo),
            kmeans_init=self.combo_kmeans_init.currentText(),
            kmeans_n_init=self.spin_kmeans_n_init.value(),
            kmeans_max_iter=self.spin_kmeans_max_iter.value(),
            gmm_covariance_type=self.combo_gmm_covar.currentText(),
            gmm_reg_covar=self.spin_gmm_reg.value(),
            gmm_max_iter=self.spin_gmm_max_iter.value(),
        )

    def get_feature_configuration(self) -> FeatureConfiguration:
        """Return active FeatureConfiguration."""
        return self.feature_config


# Backward compatibility alias
ClusteringDialog = HierarchicalClusteringDialog


class EmbeddingSettingsDialog(QDialog):
    """Modal dialog allowing users to configure 2D dimensionality reduction (PCA or UMAP)."""

    def __init__(
        self,
        current_params: Optional[EmbeddingParameters] = None,
        feature_config: Optional[FeatureConfiguration] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.params = current_params or EmbeddingParameters()
        self.feature_config = feature_config or FeatureConfiguration()
        self.setWindowTitle("2D Embedding Configuration (PCA / UMAP)")
        self.setFixedWidth(460)
        self.setModal(True)

        self.setStyleSheet(DIALOG_MODAL_STYLE)

        self._init_ui()
        self._populate_fields(self.params)

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        intro = QLabel(
            "<b>Configure 2D dimensionality reduction space</b> for scatter exploration. "
            "Choose between linear Principal Component Analysis (PCA) and non-linear "
            "Uniform Manifold Approximation and Projection (UMAP)."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(DIALOG_SUBTITLE_STYLE)
        layout.addWidget(intro)

        # 1. Feature Representation Space Group
        box_feat = QGroupBox("1. Active Feature Space & Weights")
        layout_feat = QVBoxLayout(box_feat)
        layout_feat.setContentsMargins(12, 14, 12, 12)
        layout_feat.setSpacing(8)

        self.lbl_feat_summary = QLabel()
        self.lbl_feat_summary.setWordWrap(True)
        self.lbl_feat_summary.setStyleSheet(DIALOG_INFO_BANNER_STYLE)
        self._update_feature_summary_label()
        layout_feat.addWidget(self.lbl_feat_summary)

        btn_feat_row = QHBoxLayout()
        btn_feat_row.addStretch()
        self.btn_edit_features = QPushButton("⚙️ Edit Features & Weights...")
        self.btn_edit_features.setStyleSheet(DIALOG_SECONDARY_BUTTON_STYLE)
        self.btn_edit_features.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_edit_features.clicked.connect(self._on_edit_features)
        btn_feat_row.addWidget(self.btn_edit_features)
        layout_feat.addLayout(btn_feat_row)

        layout.addWidget(box_feat)

        # 2. Embedding Method & Hyperparameters Group
        box_embed = QGroupBox("2. Embedding Method & Hyperparameters")
        layout_embed = QVBoxLayout(box_embed)
        layout_embed.setContentsMargins(12, 14, 12, 12)
        layout_embed.setSpacing(8)

        form_method = QFormLayout()
        self.combo_method = QComboBox()
        self.combo_method.addItem("Principal Component Analysis (PCA)", "PCA")
        self.combo_method.addItem("Uniform Manifold Approximation and Projection (UMAP)", "UMAP")
        self.combo_method.currentIndexChanged.connect(self._on_method_changed)
        form_method.addRow("Method:", self.combo_method)

        self.combo_scaling = QComboBox()
        self.combo_scaling.addItem("Standardized (z-score)", "zscore")
        self.combo_scaling.setToolTip("Features are standardized (zero mean, unit variance) before projection.")
        form_method.addRow("Feature Scaling:", self.combo_scaling)

        self.combo_color_by = QComboBox()
        self.combo_color_by.addItem("Cluster", "cluster")
        self.combo_color_by.addItem("Status", "status")
        self.combo_color_by.addItem("Session", "session")
        form_method.addRow("Colour Points By:", self.combo_color_by)

        layout_embed.addLayout(form_method)

        self.stack_method = QStackedWidget()

        # Panel 0: PCA
        panel_pca = QWidget()
        form_pca = QFormLayout(panel_pca)
        form_pca.setContentsMargins(0, 0, 0, 0)
        form_pca.setSpacing(6)

        self.spin_pca_x = QSpinBox()
        self.spin_pca_x.setRange(1, 10)
        self.spin_pca_x.setValue(1)
        form_pca.addRow("X-Axis Component:", self.spin_pca_x)

        self.spin_pca_y = QSpinBox()
        self.spin_pca_y.setRange(1, 10)
        self.spin_pca_y.setValue(2)
        form_pca.addRow("Y-Axis Component:", self.spin_pca_y)

        self.chk_pca_whiten = QCheckBox("Whiten components (unit variance)")
        form_pca.addRow("", self.chk_pca_whiten)
        self.stack_method.addWidget(panel_pca)

        # Panel 1: UMAP
        panel_umap = QWidget()
        form_umap = QFormLayout(panel_umap)
        form_umap.setContentsMargins(0, 0, 0, 0)
        form_umap.setSpacing(6)

        self.spin_umap_neighbors = QSpinBox()
        self.spin_umap_neighbors.setRange(2, 200)
        self.spin_umap_neighbors.setValue(15)
        form_umap.addRow("Number of Neighbors (n_neighbors):", self.spin_umap_neighbors)

        self.spin_umap_min_dist = QDoubleSpinBox()
        self.spin_umap_min_dist.setRange(0.00, 0.99)
        self.spin_umap_min_dist.setSingleStep(0.05)
        self.spin_umap_min_dist.setValue(0.10)
        form_umap.addRow("Minimum Distance (min_dist):", self.spin_umap_min_dist)

        self.combo_umap_metric = QComboBox()
        self.combo_umap_metric.addItems(["euclidean", "cosine", "correlation", "manhattan"])
        form_umap.addRow("Distance Metric:", self.combo_umap_metric)

        self.stack_method.addWidget(panel_umap)
        layout_embed.addWidget(self.stack_method)
        layout.addWidget(box_embed)

        # Dialog Buttons
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

        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setStyleSheet(DIALOG_PRIMARY_BUTTON_STYLE)
        self.btn_apply.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_apply.clicked.connect(self.accept)
        btn_box.addWidget(self.btn_apply)

        layout.addLayout(btn_box)

    def _update_feature_summary_label(self) -> None:
        """Update active feature space summary string."""
        summary = format_feature_summary(self.feature_config)
        self.lbl_feat_summary.setText(f"<b>Active Features:</b> {summary}")

    def _on_edit_features(self) -> None:
        """Open FeatureSettingsDialog to modify features in-place."""
        dlg = FeatureSettingsDialog(self.feature_config, parent=self)
        if dlg.exec():
            self.feature_config = dlg.get_feature_configuration()
            self._update_feature_summary_label()

    def _on_method_changed(self, index: int) -> None:
        """Switch active parameter stack when method changes."""
        self.stack_method.setCurrentIndex(index)

    def _populate_fields(self, p: EmbeddingParameters) -> None:
        """Populate GUI widgets from EmbeddingParameters."""
        idx = self.combo_method.findData(p.method.upper())
        if idx >= 0:
            self.combo_method.setCurrentIndex(idx)
        self.stack_method.setCurrentIndex(self.combo_method.currentIndex())

        self.spin_pca_x.setValue(p.pca_component_x)
        self.spin_pca_y.setValue(p.pca_component_y)
        self.chk_pca_whiten.setChecked(p.pca_whiten)

        self.spin_umap_neighbors.setValue(p.umap_n_neighbors)
        self.spin_umap_min_dist.setValue(p.umap_min_dist)
        idx_m = self.combo_umap_metric.findText(p.umap_metric.lower())
        if idx_m >= 0:
            self.combo_umap_metric.setCurrentIndex(idx_m)

    def _reset_defaults(self) -> None:
        """Reset controls to default parameters."""
        defaults = EmbeddingParameters()
        self._populate_fields(defaults)

    def get_parameters(self) -> EmbeddingParameters:
        """Extract user-configured parameters into EmbeddingParameters dataclass."""
        method = self.combo_method.currentData() or "PCA"
        return EmbeddingParameters(
            method=str(method),
            pca_component_x=int(self.spin_pca_x.value()),
            pca_component_y=int(self.spin_pca_y.value()),
            pca_whiten=self.chk_pca_whiten.isChecked(),
            umap_n_neighbors=int(self.spin_umap_neighbors.value()),
            umap_min_dist=float(self.spin_umap_min_dist.value()),
            umap_metric=self.combo_umap_metric.currentText().lower(),
        )

    def get_feature_configuration(self) -> FeatureConfiguration:
        """Return active FeatureConfiguration."""
        return self.feature_config
