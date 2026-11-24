"""Main application window hosting QStackedWidget and navigation menubar."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PyQt6.QtCore import QEvent, QObject, QPoint, QRectF, QSettings, QSize, Qt
from PyQt6.QtGui import QAction, QColor, QFont, QFontMetrics, QIcon, QKeySequence, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from optocurator import __version__
from optocurator.models import StatsParameters
from optocurator.ui.dialog_stats import StatsParametersDialog
from optocurator.ui.theme import (
    CLUSTERS_PROGRESS_BAR_CONTAINER_STYLE,
    COLOR_ACCENT,
    COLOR_AUTO_ASSIGNED,
    COLOR_BORDER_STRONG,
    COLOR_CANVAS,
    COLOR_PANEL,
    COLOR_SEEDED,
    COLOR_SUCCESS,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    DETECTION_COMBO_STYLE,
    FONT_SIZE_CAPTION,
    HEADER_APP_BUTTON_STYLE,
    HEADER_BAR_STYLE,
    HEADER_EXPERIMENTS_BTN_STYLE,
    HEADER_HEIGHT,
    HEADER_MENUBAR_STYLE,
    HEADER_MODE_SEGMENT_STYLE,
    HEADER_MODE_SWITCH_CONTAINER_STYLE,
    HEADER_PROGRESS_BAR_WIDTH,
    HEADER_PROGRESS_TEXT_STYLE,
    HEADER_PROTOCOL_COMBO_STYLE,
    MAIN_STYLE,
    MENU_DANGER_ACTION_STYLE,
    STATUS_BAR_BTN_STYLE,
    STATUS_BAR_GEAR_BTN_STYLE,
    STATUS_BAR_HEIGHT,
    STATUS_BAR_SAVED_STYLE,
    STATUS_BAR_STYLE,
    STATUS_BAR_SWITCH_STYLE,
    STATUS_BAR_UNSAVED_STYLE,
    STATUS_BAR_VERSION_STYLE,
    TOOLBAR_LABEL_STYLE,
    build_progress_segment_style,
    make_app_mark_icon,
    make_font,
)
from optocurator.ui.view_clusters import ClusteringView
from optocurator.ui.view_detection import DetectionView
from optocurator.ui.view_launcher import LauncherDashboardWidget


_make_app_mark_icon = make_app_mark_icon


class StatusToggleSwitch(QAbstractButton):
    """Crisp 28x16 px pill toggle switch with text label for the status bar (§ 11.2)."""

    _TRACK_W: int = 28
    _TRACK_H: int = 16
    _THUMB_D: int = 12
    _GAP: int = 6

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(make_font(FONT_SIZE_CAPTION, weight=QFont.Weight.Medium))
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        fm = QFontMetrics(self.font())
        text_w = fm.horizontalAdvance(self.text()) if self.text() else 0
        w = self._TRACK_W + (self._GAP + text_w + 4 if text_w > 0 else 0)
        return QSize(w, max(20, fm.height() + 4))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        h = self.height()
        track_y = (h - self._TRACK_H) / 2.0
        track_rect = QRectF(0.0, track_y, float(self._TRACK_W), float(self._TRACK_H))
        radius = self._TRACK_H / 2.0

        track_color = QColor(COLOR_ACCENT if self.isChecked() else COLOR_BORDER_STRONG)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track_color)
        painter.drawRoundedRect(track_rect, radius, radius)

        thumb_margin = (self._TRACK_H - self._THUMB_D) / 2.0
        thumb_x = (
            float(self._TRACK_W - self._THUMB_D) - thumb_margin
            if self.isChecked()
            else thumb_margin
        )
        thumb_y = track_y + thumb_margin
        painter.setBrush(QColor(COLOR_PANEL))
        painter.drawEllipse(QRectF(thumb_x, thumb_y, float(self._THUMB_D), float(self._THUMB_D)))

        if self.text():
            painter.setFont(self.font())
            painter.setPen(QColor(COLOR_TEXT if self.isEnabled() else COLOR_TEXT_MUTED))
            text_x = self._TRACK_W + self._GAP
            text_rect = QRectF(float(text_x), 0.0, float(max(0, self.width() - text_x)), float(h))
            painter.drawText(
                text_rect,
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                self.text(),
            )


class HeaderProgressBar(QWidget):
    """Compact ~200 px header progress bar showing assigned / total and breakdown tooltip (§ 6)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.bar_container = QFrame(self)
        self.bar_container.setFixedSize(HEADER_PROGRESS_BAR_WIDTH, 8)
        self.bar_container.setStyleSheet(CLUSTERS_PROGRESS_BAR_CONTAINER_STYLE)
        self.bar_layout = QHBoxLayout(self.bar_container)
        self.bar_layout.setContentsMargins(0, 0, 0, 0)
        self.bar_layout.setSpacing(0)

        self.seg_curated = QFrame(self.bar_container)
        self.seg_curated.setStyleSheet(build_progress_segment_style(COLOR_SUCCESS))
        self.seg_seeded = QFrame(self.bar_container)
        self.seg_seeded.setStyleSheet(build_progress_segment_style(COLOR_SEEDED))
        self.seg_auto = QFrame(self.bar_container)
        self.seg_auto.setStyleSheet(build_progress_segment_style(COLOR_AUTO_ASSIGNED))
        self.seg_remaining = QFrame(self.bar_container)
        self.seg_remaining.setStyleSheet(build_progress_segment_style("transparent"))

        self.bar_layout.addWidget(self.seg_curated)
        self.bar_layout.addWidget(self.seg_seeded)
        self.bar_layout.addWidget(self.seg_auto)
        self.bar_layout.addWidget(self.seg_remaining)

        layout.addWidget(self.bar_container)

        self.lbl_count = QLabel("0 / 0", self)
        self.lbl_count.setStyleSheet(HEADER_PROGRESS_TEXT_STYLE)
        layout.addWidget(self.lbl_count)

        self.update_counts(0, 0, 0, 0, 0)

    def update_counts(
        self,
        curated: int,
        seeded: int,
        auto_assigned: int,
        suggested: int,
        total: int,
    ) -> None:
        """Update segmented bar proportions, label text, and breakdown tooltip."""
        assigned = max(0, int(curated)) + max(0, int(seeded)) + max(0, int(auto_assigned))
        total = max(0, int(total))
        remaining = max(0, total - assigned)

        self.seg_curated.setVisible(curated > 0)
        self.seg_seeded.setVisible(seeded > 0)
        self.seg_auto.setVisible(auto_assigned > 0)
        self.seg_remaining.setVisible(remaining > 0 or total == 0)

        self.bar_layout.setStretch(0, max(0, int(curated)))
        self.bar_layout.setStretch(1, max(0, int(seeded)))
        self.bar_layout.setStretch(2, max(0, int(auto_assigned)))
        self.bar_layout.setStretch(3, remaining if total > 0 else 1)

        self.lbl_count.setText(f"{assigned} / {total}")
        lines = [
            f"Curated: {curated}",
            f"Seeded: {seeded}",
            f"Auto-assigned: {auto_assigned}",
            f"Remaining: {remaining}",
        ]
        if suggested > 0:
            lines.append(f"Suggested (above threshold): {suggested}")
        tip = "\n".join(lines)
        self.setToolTip(tip)
        self.bar_container.setToolTip(tip)
        self.lbl_count.setToolTip(tip)


class MainWindow(QMainWindow):
    """OptoCurator primary desktop application window."""

    VIEW_LAUNCHER = 0
    VIEW_DETECTION = 1
    VIEW_CLUSTERING = 2

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OptoCurator — Optotagged Cell Detection & Response Clustering")
        self.resize(1340, 860)
        self.setMinimumSize(1024, 680)
        self._maximized_on_workspace_entry: bool = False
        self.setFont(make_font())

        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(pal.ColorRole.Window, QColor(COLOR_CANVAS))
        pal.setColor(pal.ColorRole.Base, QColor(COLOR_PANEL))
        self.setPalette(pal)

        app = QApplication.instance()
        if app is not None and not app.styleSheet():
            app.setStyleSheet(MAIN_STYLE)

        self._init_ui()
        self._init_menubar()
        self._init_statusbar()
        self.setStyleSheet(MAIN_STYLE)

    def menuBar(self) -> QMenuBar:  # noqa: N802
        """Return the inline header menubar so menus sit inside the 44 px header bar (§ 6)."""
        if hasattr(self, "header_menubar"):
            return self.header_menubar
        return super().menuBar()

    def _init_ui(self) -> None:
        """Initialize unified header bar, central stacked widget, and views (§ 6, § 13.1)."""
        super().menuBar().setVisible(False)

        central_container = QWidget(self)
        central_layout = QVBoxLayout(central_container)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        # --- Unified 44 px Header Bar (§ 6, § 13.1) ---
        self.header_bar = QFrame(central_container)
        self.header_bar.setObjectName("appHeaderBar")
        self.header_bar.setFixedHeight(HEADER_HEIGHT)
        self.header_bar.setStyleSheet(HEADER_BAR_STYLE)
        header_layout = QHBoxLayout(self.header_bar)
        header_layout.setContentsMargins(12, 0, 12, 0)
        header_layout.setSpacing(10)

        # 1. App mark + "OptoCurator" Home button
        self.btn_home = QPushButton("OptoCurator", self.header_bar)
        self.btn_home.setIcon(_make_app_mark_icon(18, y_offset=2))
        self.btn_home.setIconSize(QSize(18, 20))
        self.btn_home.setToolTip("Home")
        self.btn_home.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_home.setStyleSheet(HEADER_APP_BUTTON_STYLE)
        self.btn_home.clicked.connect(self._on_home_clicked)
        header_layout.addWidget(self.btn_home)

        # 2. Segmented Mode Switch [Detection | Clustering]
        self.mode_switch_frame = QFrame(self.header_bar)
        self.mode_switch_frame.setObjectName("modeSwitchContainer")
        self.mode_switch_frame.setFixedHeight(28)
        self.mode_switch_frame.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.mode_switch_frame.setStyleSheet(HEADER_MODE_SWITCH_CONTAINER_STYLE)
        mode_layout = QHBoxLayout(self.mode_switch_frame)
        mode_layout.setContentsMargins(2, 2, 2, 2)
        mode_layout.setSpacing(2)

        self.btn_mode_detection = QPushButton("Detection", self.mode_switch_frame)
        self.btn_mode_detection.setCheckable(True)
        self.btn_mode_detection.setFixedHeight(22)
        self.btn_mode_detection.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_mode_detection.setStyleSheet(HEADER_MODE_SEGMENT_STYLE)

        self.btn_mode_clustering = QPushButton("Clustering", self.mode_switch_frame)
        self.btn_mode_clustering.setCheckable(True)
        self.btn_mode_clustering.setFixedHeight(22)
        self.btn_mode_clustering.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_mode_clustering.setStyleSheet(HEADER_MODE_SEGMENT_STYLE)

        self.mode_switch_group = QButtonGroup(self)
        self.mode_switch_group.setExclusive(True)
        self.mode_switch_group.addButton(self.btn_mode_detection, self.VIEW_DETECTION)
        self.mode_switch_group.addButton(self.btn_mode_clustering, self.VIEW_CLUSTERING)
        self.btn_mode_detection.clicked.connect(self.switch_to_detection)
        self.btn_mode_clustering.clicked.connect(self.switch_to_clustering)

        mode_layout.addWidget(self.btn_mode_detection)
        mode_layout.addWidget(self.btn_mode_clustering)
        header_layout.addWidget(self.mode_switch_frame, alignment=Qt.AlignmentFlag.AlignVCenter)

        # 3. Inline Menus (File · View · [Stats | Clustering] · Help)
        self.header_menubar = QMenuBar(self.header_bar)
        self.header_menubar.setNativeMenuBar(False)
        self.header_menubar.setFixedHeight(28)
        self.header_menubar.setStyleSheet(HEADER_MENUBAR_STYLE)
        header_layout.addWidget(self.header_menubar, alignment=Qt.AlignmentFlag.AlignVCenter)

        # 4. Experiments dropdown (Clustering mode)
        self.btn_experiments = QPushButton("0 experiments · 0 units ▾", self.header_bar)
        self.btn_experiments.setToolTip("Manage experiments in this clustering project")
        self.btn_experiments.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_experiments.setStyleSheet(HEADER_EXPERIMENTS_BTN_STYLE)
        self.btn_experiments.clicked.connect(self._show_experiments_dropdown)
        header_layout.addWidget(self.btn_experiments)

        # 4b. Detection session & protocol controls (Detection mode)
        self.btn_detection_session = QPushButton("📁 Open experiment folder… ▾", self.header_bar)
        self.btn_detection_session.setToolTip("Load or manage the current Detection session")
        self.btn_detection_session.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_detection_session.setStyleSheet(HEADER_EXPERIMENTS_BTN_STYLE)
        self.btn_detection_session.clicked.connect(self._show_detection_session_dropdown)
        header_layout.addWidget(self.btn_detection_session)

        self.lbl_detection_protocol = QLabel("Protocol", self.header_bar)
        self.lbl_detection_protocol.setStyleSheet(TOOLBAR_LABEL_STYLE)
        header_layout.addWidget(self.lbl_detection_protocol)

        self.combo_detection_protocol = QComboBox(self.header_bar)
        self.combo_detection_protocol.setMinimumWidth(160)
        self.combo_detection_protocol.setFixedHeight(28)
        self.combo_detection_protocol.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.combo_detection_protocol.setToolTip("Select stimulus protocol for Detection metrics")
        self.combo_detection_protocol.setStyleSheet(DETECTION_COMBO_STYLE)
        from optocurator.ui.detection_widgets import install_combo_chevron

        install_combo_chevron(self.combo_detection_protocol)
        self.combo_detection_protocol.currentIndexChanged.connect(self._on_header_detection_protocol_changed)
        header_layout.addWidget(self.combo_detection_protocol)

        # 5. Header Progress Bar (~200 px, Clustering mode)
        self.header_progress_bar = HeaderProgressBar(self.header_bar)
        header_layout.addWidget(self.header_progress_bar)

        header_layout.addStretch(1)

        # Hidden on Launcher
        self.header_bar.setVisible(False)
        central_layout.addWidget(self.header_bar)

        # --- Stacked Workspace Views ---
        self.stack = QStackedWidget(central_container)
        self.stack.setObjectName("centralStack")
        self.stack.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.stack.setAutoFillBackground(True)

        # 1. Launcher Dashboard (Index 0)
        self.launcher_view = LauncherDashboardWidget(self)
        self.launcher_view.open_detection.connect(self.switch_to_detection)
        self.launcher_view.open_clustering.connect(self.switch_to_clustering)
        self.stack.addWidget(self.launcher_view)

        # 2. Phase 1: Detection Workspace (Index 1)
        self.detection_view = DetectionView(self)
        self.detection_view.request_home.connect(self.switch_to_launcher)
        self.detection_view.request_clustering_view.connect(self.switch_to_clustering)
        if hasattr(self.detection_view, "dirty_changed"):
            self.detection_view.dirty_changed.connect(self._on_workspace_dirty_changed)
        if hasattr(self.detection_view, "session_changed"):
            self.detection_view.session_changed.connect(self._sync_detection_header_session)
        if hasattr(self.detection_view, "protocols_changed"):
            self.detection_view.protocols_changed.connect(self._sync_detection_header_protocols)
        self.stack.addWidget(self.detection_view)

        # 3. Phase 2: Clustering Workspace (Index 2)
        self.clustering_view = ClusteringView(self)
        self.clustering_view.request_home.connect(self.switch_to_launcher)
        self.clustering_view.request_detection_view.connect(self.switch_to_detection)
        self.clustering_view.experiment_added.connect(self._on_clustering_experiment_added)
        self.clustering_view.experiment_removed.connect(self._on_clustering_experiment_removed)
        self.clustering_view.experiments_changed.connect(self._update_header_experiments_button)
        self.clustering_view.progress_updated.connect(self.header_progress_bar.update_counts)
        self.clustering_view.dirty_changed.connect(self._on_workspace_dirty_changed)
        self.clustering_view.confidence_threshold_changed.connect(self._on_confidence_threshold_changed)
        self.clustering_view.auto_assign_changed.connect(self._on_clustering_auto_assign_changed)
        self.clustering_view.active_learning_toggled.connect(self._on_clustering_active_learning_toggled)
        self.stack.addWidget(self.clustering_view)

        central_layout.addWidget(self.stack, stretch=1)
        self.setCentralWidget(central_container)
        self.stack.setCurrentIndex(self.VIEW_LAUNCHER)

    def _init_menubar(self) -> None:
        """Build mode-aware application menubar (§ 11.1, § 13.3)."""
        menubar = self.menuBar()

        # --- File Menu ---
        self.file_menu = menubar.addMenu("File")

        # Clustering mode File actions (§ 11.1)
        self.act_new_project = QAction("New project…", self)
        self.act_new_project.setShortcut(QKeySequence("Ctrl+N"))
        self.act_new_project.triggered.connect(self._on_new_clustering_project)
        self.file_menu.addAction(self.act_new_project)

        self.act_open_project = QAction("Open project…", self)
        self.act_open_project.setShortcut(QKeySequence("Ctrl+O"))
        self.act_open_project.triggered.connect(self._on_open_clustering_project)
        self.file_menu.addAction(self.act_open_project)

        self.menu_recent_projects = QMenu("Recent", self)
        self.menu_recent_projects.aboutToShow.connect(self._populate_recent_projects_menu)
        self.act_recent_projects = self.file_menu.addMenu(self.menu_recent_projects)

        self.act_save_project = QAction("Save", self)
        self.act_save_project.setShortcut(QKeySequence("Ctrl+S"))
        self.act_save_project.triggered.connect(self._on_save_clustering_project)
        self.file_menu.addAction(self.act_save_project)

        self.act_save_project_as = QAction("Save as…", self)
        self.act_save_project_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.act_save_project_as.triggered.connect(self._on_save_clustering_project_as)
        self.file_menu.addAction(self.act_save_project_as)

        self.act_file_sep_clustering = self.file_menu.addSeparator()

        self.act_add_experiment = QAction("Add experiment…", self)
        self.act_add_experiment.setShortcut(QKeySequence("Ctrl+E"))
        self.act_add_experiment.triggered.connect(self._on_add_clustering_experiment)
        self.file_menu.addAction(self.act_add_experiment)

        self.act_remove_experiment = QAction("Remove experiment…", self)
        self.act_remove_experiment.triggered.connect(self._on_remove_clustering_experiment)
        self.file_menu.addAction(self.act_remove_experiment)

        self.act_close_project = QAction("Close project", self)
        self.act_close_project.setShortcut(QKeySequence("Ctrl+W"))
        self.act_close_project.triggered.connect(self._on_close_clustering_project)
        self.file_menu.addAction(self.act_close_project)

        # Detection mode File actions (§ 13.3, § 13.8)
        self.act_load_experiment = QAction("Load experiment folder…", self)
        self.act_load_experiment.setShortcut(QKeySequence("Ctrl+O"))
        self.act_load_experiment.triggered.connect(self._on_load_experiment_folder)
        self.file_menu.addAction(self.act_load_experiment)

        self.act_load_nwb = QAction("Load NWB file…", self)
        self.act_load_nwb.triggered.connect(self._on_load_nwb_file)
        self.file_menu.addAction(self.act_load_nwb)

        self.menu_recent_detection = QMenu("Recent", self)
        self.menu_recent_detection.aboutToShow.connect(self._populate_recent_detection_menu)
        self.act_recent_detection = self.file_menu.addMenu(self.menu_recent_detection)

        self.act_save_detection = QAction("Save", self)
        self.act_save_detection.setShortcut(QKeySequence("Ctrl+S"))
        self.act_save_detection.triggered.connect(self._on_save_detection_session)
        self.file_menu.addAction(self.act_save_detection)

        self.act_close_detection = QAction("Close session", self)
        self.act_close_detection.setShortcut(QKeySequence("Ctrl+W"))
        self.act_close_detection.triggered.connect(self._on_close_detection_session)
        self.file_menu.addAction(self.act_close_detection)

        self.file_menu.addSeparator()

        act_exit = QAction("Exit", self)
        act_exit.setShortcut(QKeySequence("Ctrl+Q"))
        act_exit.triggered.connect(self.close)
        self.file_menu.addAction(act_exit)

        self._clustering_file_actions = [
            self.act_new_project,
            self.act_open_project,
            self.act_recent_projects,
            self.act_save_project,
            self.act_save_project_as,
            self.act_file_sep_clustering,
            self.act_add_experiment,
            self.act_remove_experiment,
            self.act_close_project,
        ]
        self._detection_file_actions = [
            self.act_load_experiment,
            self.act_load_nwb,
            self.act_recent_detection,
            self.act_save_detection,
            self.act_close_detection,
        ]

        # --- Internal Mode Actions (no "Modes" menu in visible menubar; § 6, § 13.1) ---
        self.act_mode_launcher = QAction("Home / Launcher", self)
        self.act_mode_launcher.setShortcut(QKeySequence("Ctrl+H"))
        self.act_mode_launcher.triggered.connect(self._on_home_clicked)
        self.addAction(self.act_mode_launcher)

        self.act_mode_detection = QAction("Optotagged Cell Detection", self)
        self.act_mode_detection.setShortcut(QKeySequence("Ctrl+1"))
        self.act_mode_detection.triggered.connect(self.switch_to_detection)
        self.addAction(self.act_mode_detection)

        self.act_mode_clustering = QAction("Response Clustering", self)
        self.act_mode_clustering.setShortcut(QKeySequence("Ctrl+2"))
        self.act_mode_clustering.triggered.connect(self.switch_to_clustering)
        self.addAction(self.act_mode_clustering)

        # --- View Menu (§ 11.1, § 13.3) ---
        self.view_menu = menubar.addMenu("View")

        self.act_view_overlay_focused = QAction("Overlay focused unit", self)
        self.act_view_overlay_focused.setCheckable(True)
        self.act_view_overlay_focused.setChecked(
            bool(getattr(self.clustering_view, "overlay_focused_unit_enabled", True))
        )
        self.act_view_overlay_focused.toggled.connect(self._on_view_overlay_focused_toggled)
        self.view_menu.addAction(self.act_view_overlay_focused)
        if hasattr(self.clustering_view, "act_overlay_focused_unit"):
            self.clustering_view.act_overlay_focused_unit.toggled.connect(
                self._sync_view_overlay_focused_from_cluster_view
            )

        self.act_view_shared_y = QAction("Shared y-axis", self)
        self.act_view_shared_y.setCheckable(True)
        self.act_view_shared_y.setChecked(
            bool(getattr(self.clustering_view, "shared_y_axis_enabled", False))
        )
        self.act_view_shared_y.toggled.connect(self._on_view_shared_y_toggled)
        self.view_menu.addAction(self.act_view_shared_y)
        if hasattr(self.clustering_view, "act_shared_y_axis"):
            self.clustering_view.act_shared_y_axis.toggled.connect(
                self._sync_view_shared_y_from_cluster_view
            )

        self.act_view_sep_clustering = self.view_menu.addSeparator()

        self.act_configure_visualizations = QAction("Configure visualizations…", self)
        self.act_configure_visualizations.triggered.connect(self._on_configure_visualizations)
        self.view_menu.addAction(self.act_configure_visualizations)

        self.act_reset_panel_layout = QAction("Reset panel layout", self)
        self.act_reset_panel_layout.triggered.connect(self._on_reset_phase2_layout)
        self.act_reset_phase2_layout = self.act_reset_panel_layout
        self.view_menu.addAction(self.act_reset_panel_layout)

        # --- Stats Menu (Detection mode, § 13.3, § 13.8) ---
        self.stats_menu = menubar.addMenu("Stats")
        self.act_stats_params = QAction("Metric parameters…", self)
        self.act_stats_params.triggered.connect(self._on_change_stats_parameters)
        self.stats_menu.addAction(self.act_stats_params)

        # --- Clustering Menu (§ 11.1 exact grouping) ---
        self.clustering_menu = menubar.addMenu("Clustering")

        # Group 1: Build clusters
        self.act_seed_clusters = QAction("Seed clusters…", self)
        self.act_seed_clusters.triggered.connect(self._on_seed_clusters)
        self.clustering_menu.addAction(self.act_seed_clusters)

        self.act_hierarchical = QAction("Run clustering…", self)
        self.act_hierarchical.triggered.connect(self._on_hierarchical_clustering)
        self.clustering_menu.addAction(self.act_hierarchical)

        self.clustering_menu.addSeparator()

        # Group 2: Settings
        self.act_feature_settings = QAction("Feature settings…", self)
        self.act_feature_settings.triggered.connect(self._on_feature_settings)
        self.clustering_menu.addAction(self.act_feature_settings)

        self.act_customize_clusters = QAction("Customize clusters…", self)
        self.act_customize_clusters.triggered.connect(self._on_customize_clusters)
        self.clustering_menu.addAction(self.act_customize_clusters)

        # Compatibility action (accessible via embedding gear button; kept as attribute)
        self.act_embedding_settings = QAction("2D Embedding Settings (PCA / UMAP)…", self)
        self.act_embedding_settings.triggered.connect(self._on_embedding_settings)

        self.clustering_menu.addSeparator()

        # Group 3: Active learning
        al_enabled = bool(getattr(self.clustering_view, "active_learning_enabled", True))
        self.act_active_learning = QAction("Active learning", self)
        self.act_active_learning.setCheckable(True)
        self.act_active_learning.setChecked(al_enabled)
        self.act_active_learning.toggled.connect(self._on_menu_active_learning_toggled)
        self.clustering_menu.addAction(self.act_active_learning)

        self.act_go_to_suggested_unit = QAction("Go to suggested unit", self)
        self.act_go_to_suggested_unit.setShortcut(QKeySequence("G"))
        self.act_go_to_suggested_unit.setEnabled(al_enabled)
        self.act_go_to_suggested_unit.triggered.connect(self._on_go_to_suggested_unit)
        self.clustering_menu.addAction(self.act_go_to_suggested_unit)

        self.act_active_learning_settings = QAction("Active learning settings…", self)
        self.act_active_learning_settings.triggered.connect(self._on_active_learning_settings)
        self.clustering_menu.addAction(self.act_active_learning_settings)

        self.clustering_menu.addSeparator()

        # Group 4: Auto-assignment
        aa_enabled = bool(getattr(self.clustering_view, "auto_assign_enabled", False))
        self.act_auto_assign = QAction("Auto-assign", self)
        self.act_auto_assign.setCheckable(True)
        self.act_auto_assign.setChecked(aa_enabled)
        self.act_auto_assign.toggled.connect(self._on_menu_auto_assign_toggled)
        self.clustering_menu.addAction(self.act_auto_assign)

        self.act_confidence_threshold = QAction("Confidence threshold…", self)
        self.act_confidence_threshold.triggered.connect(self._on_confidence_threshold)
        self.clustering_menu.addAction(self.act_confidence_threshold)

        self.act_revert_auto_assignments = QAction("Revert auto-assignments…", self)
        self.act_revert_auto_assignments.triggered.connect(self._on_revert_auto_assignments)
        self.clustering_menu.addAction(self.act_revert_auto_assignments)

        self.clustering_menu.addSeparator()

        # Danger action: Reset clustering…
        self.act_reset_clustering = QWidgetAction(self)
        self.act_reset_clustering.setText("Reset clustering…")
        self.act_reset_clustering.triggered.connect(self._on_reset_clustering)
        btn_reset_menu = QPushButton("Reset clustering…", self.clustering_menu)
        btn_reset_menu.setFlat(True)
        btn_reset_menu.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_reset_menu.setStyleSheet(MENU_DANGER_ACTION_STYLE)
        btn_reset_menu.clicked.connect(self.act_reset_clustering.trigger)
        btn_reset_menu.clicked.connect(self.clustering_menu.close)
        self.act_reset_clustering.setDefaultWidget(btn_reset_menu)
        self.clustering_menu.addAction(self.act_reset_clustering)

        # --- Help Menu (§ 11.1) ---
        self.help_menu = menubar.addMenu("Help")

        self.act_shortcuts = QAction("Keyboard shortcuts", self)
        self.act_shortcuts.setShortcut(QKeySequence("?"))
        self.act_shortcuts.triggered.connect(self._show_shortcuts_dialog)
        self.help_menu.addAction(self.act_shortcuts)

        self.act_documentation = QAction("Documentation", self)
        self.act_documentation.setShortcut(QKeySequence("F1"))
        self.act_documentation.triggered.connect(self._show_documentation_dialog)
        self.help_menu.addAction(self.act_documentation)

        self.act_cite = QAction("How to cite OptoCurator", self)
        self.act_cite.triggered.connect(self._show_citation_dialog)
        self.help_menu.addAction(self.act_cite)

        self.act_about = QAction("About OptoCurator", self)
        self.act_about.triggered.connect(self._show_about_dialog)
        self.help_menu.addAction(self.act_about)

    def _init_statusbar(self) -> None:
        """Initialize 24 px status bar (§ 11.2, § 13.7)."""
        self.status = QStatusBar(self)
        self.status.setFixedHeight(STATUS_BAR_HEIGHT)
        self.status.setStyleSheet(STATUS_BAR_STYLE)
        self.status.setSizeGripEnabled(False)
        self.setStatusBar(self.status)
        self.status.setVisible(False)

        # Left widgets (Clustering mode)
        self.status_clustering_left = QWidget(self.status)
        left_layout = QHBoxLayout(self.status_clustering_left)
        left_layout.setContentsMargins(6, 0, 0, 0)
        left_layout.setSpacing(6)

        pct = int(round(getattr(self.clustering_view, "auto_assign_threshold", 0.85) * 100))
        self.btn_status_threshold = QPushButton(
            f"Confidence threshold {pct}%",
            self.status_clustering_left,
        )
        self.btn_status_threshold.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_status_threshold.setToolTip("Set auto-assignment confidence threshold (50–99%)")
        self.btn_status_threshold.setStyleSheet(STATUS_BAR_BTN_STYLE)
        self.btn_status_threshold.clicked.connect(self._on_confidence_threshold)
        left_layout.addWidget(self.btn_status_threshold)

        aa_enabled = bool(getattr(self.clustering_view, "auto_assign_enabled", False))
        self.btn_status_auto_assign = StatusToggleSwitch(
            "Auto-assign",
            self.status_clustering_left,
        )
        self.btn_status_auto_assign.setChecked(aa_enabled)
        self.btn_status_auto_assign.setToolTip("Automatically assign units above confidence threshold")
        self.btn_status_auto_assign.toggled.connect(self._on_status_auto_assign_toggled)
        left_layout.addWidget(self.btn_status_auto_assign)

        al_enabled = bool(getattr(self.clustering_view, "active_learning_enabled", True))
        self.btn_status_active_learning = StatusToggleSwitch(
            "Active learning",
            self.status_clustering_left,
        )
        self.btn_status_active_learning.setChecked(al_enabled)
        self.btn_status_active_learning.setToolTip("Toggle uncertainty-guided active learning queue")
        self.btn_status_active_learning.toggled.connect(self._on_status_active_learning_toggled)
        left_layout.addWidget(self.btn_status_active_learning)

        self.btn_status_al_settings = QPushButton("⚙", self.status_clustering_left)
        self.btn_status_al_settings.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_status_al_settings.setToolTip("Active learning settings…")
        self.btn_status_al_settings.setStyleSheet(STATUS_BAR_GEAR_BTN_STYLE)
        self.btn_status_al_settings.clicked.connect(self._on_active_learning_settings)
        left_layout.addWidget(self.btn_status_al_settings)

        self.status.addWidget(self.status_clustering_left)

        # Right widgets (both Detection and Clustering modes)
        self.lbl_status_version = QLabel(f"v{__version__}", self.status)
        self.lbl_status_version.setStyleSheet(STATUS_BAR_VERSION_STYLE)
        self.status.addPermanentWidget(self.lbl_status_version)

        self.lbl_status_save_state = QLabel("● Saved", self.status)
        self.lbl_status_save_state.setStyleSheet(STATUS_BAR_SAVED_STYLE)
        self.status.addPermanentWidget(self.lbl_status_save_state)

    # --- View Switching Callbacks ---

    def _on_home_clicked(self) -> None:
        """Return to Launcher when App mark is clicked, asking to save if unsaved changes exist (§ 6)."""
        current_idx = self.stack.currentIndex()
        is_dirty = False
        if current_idx == self.VIEW_CLUSTERING:
            is_dirty = bool(getattr(self.clustering_view, "_is_dirty", False))
        elif current_idx == self.VIEW_DETECTION:
            is_dirty = bool(getattr(self.detection_view, "_is_dirty", False))

        if is_dirty:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Save before returning to Home?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.Save:
                self._on_save_session()
                if current_idx == self.VIEW_CLUSTERING and bool(
                    getattr(self.clustering_view, "_is_dirty", False)
                ):
                    return
                if current_idx == self.VIEW_DETECTION and bool(
                    getattr(self.detection_view, "_is_dirty", False)
                ):
                    return

        self.switch_to_launcher()

    def _update_header_experiments_button(self) -> None:
        """Refresh the Experiments dropdown button label with experiment and unit counts (§ 6)."""
        if not hasattr(self, "btn_experiments"):
            return
        n_exps = len(self.clustering_view.loaded_experiments)
        n_units = len(self.clustering_view._pooled_units_map)
        exp_word = "experiment" if n_exps == 1 else "experiments"
        unit_word = "unit" if n_units == 1 else "units"
        self.btn_experiments.setText(f"{n_exps} {exp_word} · {n_units} {unit_word} ▾")

    def _show_experiments_dropdown(self) -> None:
        """Show popup listing loaded experiments with right-click Remove and + Add experiment… (§ 6)."""
        menu = QMenu(self)
        exps = self.clustering_view.loaded_experiments
        if not exps:
            empty_act = menu.addAction("No experiments loaded")
            empty_act.setEnabled(False)
        else:
            for sid, meta in exps.items():
                label = f"{meta.session_id}  ·  {meta.protocol_name}  ·  {meta.tagged_units} tagged"
                act = menu.addAction(label)
                act.setData(sid)
                act.setToolTip("Right-click to remove experiment")

        menu.addSeparator()
        act_add = menu.addAction("+ Add experiment…")
        act_add.triggered.connect(self._on_add_clustering_experiment)

        # Support right-click "Remove experiment…" on experiment items
        class _ExpMenuFilter(QObject):
            def eventFilter(self_f, obj: Any, ev: Any) -> bool:  # noqa: N805
                if obj is menu and ev.type() == QEvent.Type.MouseButtonPress:
                    if ev.button() == Qt.MouseButton.RightButton:
                        act = menu.actionAt(ev.pos())
                        sid = act.data() if act is not None else None
                        if sid:
                            ctx = QMenu(menu)
                            act_rem = ctx.addAction("Remove experiment…")
                            chosen = ctx.exec(menu.mapToGlobal(ev.pos()))
                            if chosen is act_rem:
                                menu.close()
                                self.clustering_view.remove_selected_experiment(sid)
                            return True
                return False

        filt = _ExpMenuFilter(menu)
        menu.installEventFilter(filt)
        menu.exec(self.btn_experiments.mapToGlobal(QPoint(0, self.btn_experiments.height())))

    def _show_detection_session_dropdown(self) -> None:
        """Show Detection session menu with Load, Recent submenu, and Close session (§ 13.3)."""
        menu = QMenu(self)
        act_load = menu.addAction("Load experiment folder…")
        act_load.triggered.connect(self._on_load_experiment_folder)
        act_load_nwb = menu.addAction("Load NWB file…")
        act_load_nwb.triggered.connect(self._on_load_nwb_file)

        recent_menu = QMenu("Recent", menu)
        items = self._get_recent_list("recent_detection_folders")
        if not items:
            empty_act = recent_menu.addAction("No recent folders")
            empty_act.setEnabled(False)
        else:
            for p in items:
                act = recent_menu.addAction(p)
                act.triggered.connect(
                    lambda _checked=False, exp_dir=p: self._open_recent_detection_folder(exp_dir)
                )
        menu.addMenu(recent_menu)

        menu.addSeparator()
        act_close = menu.addAction("Close session")
        has_session = bool(getattr(self.detection_view, "current_folder", None) is not None or getattr(self.detection_view, "units", []))
        act_close.setEnabled(has_session)
        act_close.triggered.connect(self._on_close_detection_session)

        menu.exec(self.btn_detection_session.mapToGlobal(QPoint(0, self.btn_detection_session.height())))

    def _sync_detection_header_session(self) -> None:
        """Update Detection header session button label."""
        if not hasattr(self, "btn_detection_session"):
            return
        sid = getattr(self.detection_view, "session_id", "None")
        if sid and sid != "None":
            self.btn_detection_session.setText(f"📁 {sid} ▾")
        else:
            self.btn_detection_session.setText("📁 Open experiment folder… ▾")

    def _sync_detection_header_protocols(self) -> None:
        """Sync header protocol combo with detection_view.combo_proto."""
        if not hasattr(self, "combo_detection_protocol"):
            return
        src = getattr(self.detection_view, "combo_proto", None)
        if src is None:
            return
        self.combo_detection_protocol.blockSignals(True)
        self.combo_detection_protocol.clear()
        for idx in range(src.count()):
            self.combo_detection_protocol.addItem(src.itemText(idx), src.itemData(idx))
        if src.currentIndex() >= 0:
            self.combo_detection_protocol.setCurrentIndex(src.currentIndex())
        self.combo_detection_protocol.setEnabled(src.count() > 0)
        self.combo_detection_protocol.blockSignals(False)

    def _on_header_detection_protocol_changed(self, index: int) -> None:
        """Forward protocol selection from header combo to detection_view.combo_proto."""
        src = getattr(self.detection_view, "combo_proto", None)
        if src is not None and index >= 0 and index < src.count():
            if src.currentIndex() != index:
                src.setCurrentIndex(index)

    def _sync_header_for_mode(self, view_index: int) -> None:
        """Update header bar, mode switch, menu visibility, and status bar for active mode."""
        is_launcher = view_index == self.VIEW_LAUNCHER
        is_detection = view_index == self.VIEW_DETECTION
        is_clustering = view_index == self.VIEW_CLUSTERING

        self.header_bar.setVisible(not is_launcher)
        self.status.setVisible(not is_launcher)

        self.btn_mode_detection.blockSignals(True)
        self.btn_mode_clustering.blockSignals(True)
        self.btn_mode_detection.setChecked(is_detection)
        self.btn_mode_clustering.setChecked(is_clustering)
        self.btn_mode_detection.blockSignals(False)
        self.btn_mode_clustering.blockSignals(False)

        self.stats_menu.menuAction().setVisible(is_detection)
        self.clustering_menu.menuAction().setVisible(is_clustering)
        self.btn_experiments.setVisible(is_clustering)
        self.header_progress_bar.setVisible(is_clustering)
        if hasattr(self, "btn_detection_session"):
            self.btn_detection_session.setVisible(is_detection)
        if hasattr(self, "lbl_detection_protocol"):
            self.lbl_detection_protocol.setVisible(is_detection)
        if hasattr(self, "combo_detection_protocol"):
            self.combo_detection_protocol.setVisible(is_detection)

        for act in getattr(self, "_clustering_file_actions", ()):
            act.setVisible(is_clustering)
            act.setEnabled(is_clustering)
        if is_launcher:
            self.act_open_project.setEnabled(True)
            self.act_new_project.setEnabled(True)
        for act in getattr(self, "_detection_file_actions", ()):
            act.setVisible(is_detection)
            act.setEnabled(is_detection)

        if hasattr(self, "act_view_overlay_focused"):
            self.act_view_overlay_focused.setVisible(is_clustering)
        if hasattr(self, "act_view_shared_y"):
            self.act_view_shared_y.setVisible(is_clustering)
        if hasattr(self, "act_view_sep_clustering"):
            self.act_view_sep_clustering.setVisible(is_clustering)

        if hasattr(self, "status_clustering_left"):
            self.status_clustering_left.setVisible(is_clustering)

        if is_clustering:
            self._update_header_experiments_button()
            self._sync_status_and_clustering_controls()
        elif is_detection:
            self._sync_detection_header_session()
            self._sync_detection_header_protocols()
        self._update_status_save_indicator()

    def _ensure_workspace_maximized(self) -> None:
        """Maximize the window the first time a visible user enters Detection or Clustering mode."""
        if self.isVisible() and not self._maximized_on_workspace_entry:
            self._maximized_on_workspace_entry = True
            if not self.isMaximized():
                self.showMaximized()

    def switch_to_launcher(self) -> None:
        """Switch central view to Launcher Dashboard."""
        self.stack.setCurrentIndex(self.VIEW_LAUNCHER)
        self._sync_header_for_mode(self.VIEW_LAUNCHER)
        self.status.showMessage("Dashboard • Choose a workspace to begin")

    def switch_to_detection(self) -> None:
        """Switch central view to Optotagged Cell Detection."""
        self.stack.setCurrentIndex(self.VIEW_DETECTION)
        self._sync_header_for_mode(self.VIEW_DETECTION)
        self.status.clearMessage()
        self._ensure_workspace_maximized()

    def switch_to_clustering(self) -> None:
        """Switch central view to Response Clustering."""
        self.stack.setCurrentIndex(self.VIEW_CLUSTERING)
        self._sync_header_for_mode(self.VIEW_CLUSTERING)
        self.status.clearMessage()
        self._ensure_workspace_maximized()

    # --- Recent Projects / Sessions (QSettings) ---

    def _get_recent_list(self, settings_key: str) -> List[str]:
        """Return up to 5 recent paths stored under settings_key in QSettings."""
        settings = QSettings("OptoCurator", "OptoCurator")
        raw = settings.value(settings_key, [])
        if isinstance(raw, str):
            items = [raw] if raw.strip() else []
        elif isinstance(raw, (list, tuple)):
            items = [str(x) for x in raw if str(x).strip()]
        else:
            items = []
        return items[:5]

    def _record_recent_path(self, settings_key: str, path: Path) -> None:
        """Prepend path to recent list (max 5 unique resolved paths) in QSettings."""
        resolved = str(Path(path).resolve())
        items = [p for p in self._get_recent_list(settings_key) if p != resolved]
        items.insert(0, resolved)
        settings = QSettings("OptoCurator", "OptoCurator")
        settings.setValue(settings_key, items[:5])

    def _populate_recent_projects_menu(self) -> None:
        """Populate File -> Recent submenu for Clustering projects."""
        self.menu_recent_projects.clear()
        items = self._get_recent_list("recent_clustering_projects")
        if not items:
            empty_act = self.menu_recent_projects.addAction("No recent projects")
            empty_act.setEnabled(False)
            return
        for p in items:
            act = self.menu_recent_projects.addAction(p)
            act.triggered.connect(lambda _checked=False, proj_dir=p: self._open_recent_clustering_project(proj_dir))

    def _populate_recent_detection_menu(self) -> None:
        """Populate File -> Recent submenu for Detection experiment folders."""
        self.menu_recent_detection.clear()
        items = self._get_recent_list("recent_detection_folders")
        if not items:
            empty_act = self.menu_recent_detection.addAction("No recent folders")
            empty_act.setEnabled(False)
            return
        for p in items:
            act = self.menu_recent_detection.addAction(p)
            act.triggered.connect(lambda _checked=False, exp_dir=p: self._open_recent_detection_folder(exp_dir))

    def _open_recent_clustering_project(self, project_dir: str) -> None:
        """Open a recent clustering project folder."""
        if not self._confirm_discard_clustering_changes("opening a recent project"):
            return
        if self.stack.currentIndex() != self.VIEW_CLUSTERING:
            self.switch_to_clustering()
        ok = self.clustering_view.load_project(project_dir)
        if ok:
            self._record_recent_path("recent_clustering_projects", Path(project_dir))
            self._sync_status_and_clustering_controls()
            self._update_status_save_indicator()

    def _open_recent_detection_folder(self, folder_path: str) -> None:
        """Open a recent detection experiment folder or NWB file."""
        if not self._confirm_discard_detection_changes("opening a recent session"):
            return
        self.switch_to_detection()
        p = Path(folder_path)
        if p.suffix.lower() == ".nwb":
            ok = self.detection_view.load_nwb_file(p)
        else:
            ok = self.detection_view.load_experiment_folder(p)
        if ok:
            self._record_recent_path("recent_detection_folders", p)
            self._update_status_save_indicator()
            self.status.clearMessage()

    # --- Actions ---

    def _on_load_nwb_file(self, *args: Any, **kwargs: Any) -> None:
        """File selection dialog for loading an NWB session in Detection mode."""
        if not self._confirm_discard_detection_changes("loading an NWB file"):
            return
        self.switch_to_detection()
        ok = self.detection_view.load_nwb_file()
        if ok:
            rec_path = getattr(self.detection_view, "current_nwb_path", None) or getattr(
                self.detection_view, "current_folder", None
            )
            if rec_path is not None:
                self._record_recent_path("recent_detection_folders", rec_path)
            self._update_status_save_indicator()
            self.status.clearMessage()

    def _on_load_experiment_folder(self, *args: Any, **kwargs: Any) -> None:
        """Folder selection dialog for single-session detection."""
        if not self._confirm_discard_detection_changes("loading an experiment folder"):
            return
        self.switch_to_detection()
        ok = self.detection_view.load_experiment_folder()
        if ok:
            if getattr(self.detection_view, "current_folder", None) is not None:
                self.detection_view.current_nwb_path = None
                self._record_recent_path("recent_detection_folders", self.detection_view.current_folder)
            self._update_status_save_indicator()
            self.status.clearMessage()

    def _on_save_detection_session(self) -> None:
        """Save detection session CSV + tagged units .npz (§ 1.1, § 13.3)."""
        self.switch_to_detection()
        csv_path = self.detection_view.export_curated_session()
        if csv_path:
            if getattr(self.detection_view, "current_folder", None) is not None:
                self._record_recent_path("recent_detection_folders", self.detection_view.current_folder)
            self._update_status_save_indicator()
            self.status.clearMessage()

    def _confirm_discard_detection_changes(self, action_verb: str = "closing") -> bool:
        """Prompt to save unsaved detection changes before discarding; return True to proceed."""
        if not bool(getattr(self.detection_view, "_is_dirty", False)):
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            f"You have unsaved changes in this detection session. Save before {action_verb}?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return False
        if reply == QMessageBox.StandardButton.Save:
            self._on_save_detection_session()
            if bool(getattr(self.detection_view, "_is_dirty", False)):
                return False
        return True

    def _on_close_detection_session(self) -> None:
        """Close current detection session, prompting to save if dirty (§ 13.3)."""
        if not self._confirm_discard_detection_changes("closing"):
            return
        if hasattr(self.detection_view, "close_session"):
            self.detection_view.close_session()
        else:
            self.detection_view.current_folder = None
            self.detection_view.session_id = "None"
            if hasattr(self.detection_view, "lbl_session"):
                self.detection_view.lbl_session.setText("Session: None")
            if hasattr(self.detection_view, "table_units"):
                self.detection_view.table_units.setRowCount(0)
        self._update_status_save_indicator()

    def _on_add_clustering_experiment(self, *args: Any, **kwargs: Any) -> None:
        """Folder selection dialog for multi-experiment clustering."""
        if self.stack.currentIndex() != self.VIEW_CLUSTERING:
            self.switch_to_clustering()
        self.clustering_view.add_experiment_folder()

    def _on_remove_clustering_experiment(self) -> None:
        """Remove selected experiment from the clustering project pool."""
        if self.stack.currentIndex() != self.VIEW_CLUSTERING:
            self.switch_to_clustering()
        self.clustering_view.remove_clustering_experiment()

    def _on_clustering_experiment_added(self, session_id: str) -> None:
        """Handle new experiment registration in Phase 2."""
        n_exps = len(self.clustering_view.loaded_experiments)
        n_units = self.clustering_view.table_pooled_units.rowCount()
        self.status.showMessage(
            f"Ingested experiment '{session_id}'. Active experiments: {n_exps} | Pooled units: {n_units}"
        )

    def _on_clustering_experiment_removed(self, session_id: str) -> None:
        """Handle experiment removal in Phase 2."""
        n_exps = len(self.clustering_view.loaded_experiments)
        n_units = self.clustering_view.table_pooled_units.rowCount()
        self.status.showMessage(
            f"Removed experiment '{session_id}'. Remaining experiments: {n_exps} | Pooled units: {n_units}"
        )

    def _on_save_session(self, *args: Any, **kwargs: Any) -> None:
        """Context-dependent save action."""
        current_idx = self.stack.currentIndex()
        if current_idx == self.VIEW_DETECTION:
            self._on_save_detection_session()
        elif current_idx == self.VIEW_CLUSTERING:
            self._on_save_clustering_project()
        else:
            self.status.showMessage("Please open a workspace to save curation state.")

    def _confirm_discard_clustering_changes(self, action_verb: str = "closing") -> bool:
        """Prompt to save unsaved clustering changes before discarding; return True to proceed."""
        if not bool(getattr(self.clustering_view, "_is_dirty", False)):
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            f"You have unsaved changes in this clustering project. Save before {action_verb}?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return False
        if reply == QMessageBox.StandardButton.Save:
            saved = self.clustering_view.save_project()
            if saved is None:
                return False
        return True

    def _on_new_clustering_project(self) -> None:
        """Create a new clustering project in a user-chosen folder (§ 11.1)."""
        if self.stack.currentIndex() != self.VIEW_CLUSTERING:
            self.switch_to_clustering()
        if not self._confirm_discard_clustering_changes("starting a new project"):
            return
        proj_dir = self.clustering_view.new_clustering_project()
        if proj_dir is not None:
            self._record_recent_path("recent_clustering_projects", proj_dir)
            self._sync_status_and_clustering_controls()
            self._update_status_save_indicator()

    def _on_open_clustering_project(self) -> None:
        """Open a clustering project folder in Phase 2."""
        if self.stack.currentIndex() != self.VIEW_CLUSTERING:
            self.switch_to_clustering()
        if not self._confirm_discard_clustering_changes("opening another project"):
            return
        ok = self.clustering_view.prompt_load_project()
        if ok and self.clustering_view.current_project_dir is not None:
            self._record_recent_path("recent_clustering_projects", self.clustering_view.current_project_dir)
            self._sync_status_and_clustering_controls()
            self._update_status_save_indicator()

    def _on_save_clustering_project(self) -> None:
        """Save clustering project in current or prompted directory."""
        if self.stack.currentIndex() != self.VIEW_CLUSTERING:
            self.switch_to_clustering()
        res = self.clustering_view.save_project()
        if res:
            self._record_recent_path("recent_clustering_projects", res.parent)
            self._update_status_save_indicator()
            self.status.showMessage(f"Clustering project saved to {res.parent.name}")

    def _on_save_clustering_project_as(self) -> None:
        """Prompt user for directory to save clustering project."""
        if self.stack.currentIndex() != self.VIEW_CLUSTERING:
            self.switch_to_clustering()
        res = self.clustering_view.prompt_save_project(save_as=True)
        if res:
            self._record_recent_path("recent_clustering_projects", res.parent)
            self._update_status_save_indicator()
            self.status.showMessage(f"Clustering project saved to {res.parent.name}")

    def _on_close_clustering_project(self) -> None:
        """Close the current clustering project, prompting to save if dirty (§ 11.1)."""
        if self.stack.currentIndex() != self.VIEW_CLUSTERING:
            self.switch_to_clustering()
        if not self._confirm_discard_clustering_changes("closing"):
            return
        self.clustering_view.close_clustering_project()
        self._sync_status_and_clustering_controls()
        self._update_status_save_indicator()

    # --- View / Clustering / Status Bar Synchronization (§ 11.1, § 11.2) ---

    def _on_view_overlay_focused_toggled(self, checked: bool) -> None:
        if hasattr(self.clustering_view, "act_overlay_focused_unit"):
            self.clustering_view.act_overlay_focused_unit.setChecked(bool(checked))

    def _sync_view_overlay_focused_from_cluster_view(self, checked: bool) -> None:
        if hasattr(self, "act_view_overlay_focused"):
            self.act_view_overlay_focused.blockSignals(True)
            self.act_view_overlay_focused.setChecked(bool(checked))
            self.act_view_overlay_focused.blockSignals(False)

    def _on_view_shared_y_toggled(self, checked: bool) -> None:
        if hasattr(self.clustering_view, "act_shared_y_axis"):
            self.clustering_view.act_shared_y_axis.setChecked(bool(checked))

    def _sync_view_shared_y_from_cluster_view(self, checked: bool) -> None:
        if hasattr(self, "act_view_shared_y"):
            self.act_view_shared_y.blockSignals(True)
            self.act_view_shared_y.setChecked(bool(checked))
            self.act_view_shared_y.blockSignals(False)

    def _on_configure_visualizations(self) -> None:
        """Open VisualizationSettingsDialog for active workspace (§ 11.1, § 13.3)."""
        if self.stack.currentIndex() == self.VIEW_DETECTION:
            if hasattr(self.detection_view, "prompt_visualization_settings"):
                self.detection_view.prompt_visualization_settings()
        else:
            if self.stack.currentIndex() != self.VIEW_CLUSTERING:
                self.switch_to_clustering()
            self.clustering_view.prompt_visualization_settings()

    def _on_menu_active_learning_toggled(self, checked: bool) -> None:
        self.clustering_view.set_active_learning_mode(bool(checked))

    def _on_status_active_learning_toggled(self, checked: bool) -> None:
        self.clustering_view.set_active_learning_mode(bool(checked))

    def _on_clustering_active_learning_toggled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if hasattr(self, "act_active_learning"):
            self.act_active_learning.blockSignals(True)
            self.act_active_learning.setChecked(enabled)
            self.act_active_learning.blockSignals(False)
        if hasattr(self, "act_go_to_suggested_unit"):
            self.act_go_to_suggested_unit.setEnabled(enabled)
        if hasattr(self, "btn_status_active_learning"):
            self.btn_status_active_learning.blockSignals(True)
            self.btn_status_active_learning.setChecked(enabled)
            self.btn_status_active_learning.update()
            self.btn_status_active_learning.blockSignals(False)

    def _on_go_to_suggested_unit(self) -> None:
        if self.stack.currentIndex() != self.VIEW_CLUSTERING:
            self.switch_to_clustering()
        if getattr(self.clustering_view, "active_learning_enabled", True):
            self.clustering_view.return_to_active_candidate()

    def _on_active_learning_settings(self) -> None:
        if self.stack.currentIndex() != self.VIEW_CLUSTERING:
            self.switch_to_clustering()
        self.clustering_view.prompt_active_learning_settings()

    def _on_menu_auto_assign_toggled(self, checked: bool) -> None:
        if hasattr(self.clustering_view, "chk_auto_assign"):
            self.clustering_view.chk_auto_assign.setChecked(bool(checked))
        else:
            self.clustering_view._on_auto_assign_toggled(bool(checked))

    def _on_status_auto_assign_toggled(self, checked: bool) -> None:
        if hasattr(self.clustering_view, "chk_auto_assign"):
            self.clustering_view.chk_auto_assign.setChecked(bool(checked))
        else:
            self.clustering_view._on_auto_assign_toggled(bool(checked))

    def _on_clustering_auto_assign_changed(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if hasattr(self, "act_auto_assign"):
            self.act_auto_assign.blockSignals(True)
            self.act_auto_assign.setChecked(enabled)
            self.act_auto_assign.blockSignals(False)
        if hasattr(self, "btn_status_auto_assign"):
            self.btn_status_auto_assign.blockSignals(True)
            self.btn_status_auto_assign.setChecked(enabled)
            self.btn_status_auto_assign.update()
            self.btn_status_auto_assign.blockSignals(False)

    def _on_confidence_threshold(self) -> None:
        if self.stack.currentIndex() != self.VIEW_CLUSTERING:
            self.switch_to_clustering()
        self.clustering_view.prompt_confidence_threshold()

    def _on_confidence_threshold_changed(self, threshold: float) -> None:
        pct = int(round(float(threshold) * 100))
        if hasattr(self, "btn_status_threshold"):
            self.btn_status_threshold.setText(f"Confidence threshold {pct}%")

    def _on_revert_auto_assignments(self) -> None:
        if self.stack.currentIndex() != self.VIEW_CLUSTERING:
            self.switch_to_clustering()
        self.clustering_view.prompt_revert_auto_assignments()

    def _on_workspace_dirty_changed(self, _is_dirty: bool) -> None:
        self._update_status_save_indicator()

    def _update_status_save_indicator(self) -> None:
        """Update right-side status bar save state indicator for the active workspace (§ 11.2, § 13.7)."""
        if not hasattr(self, "lbl_status_save_state"):
            return
        idx = self.stack.currentIndex()
        is_dirty = False
        if idx == self.VIEW_CLUSTERING:
            is_dirty = bool(getattr(self.clustering_view, "_is_dirty", False))
        elif idx == self.VIEW_DETECTION:
            is_dirty = bool(getattr(self.detection_view, "_is_dirty", False))

        if is_dirty:
            self.lbl_status_save_state.setText("● Unsaved changes")
            self.lbl_status_save_state.setStyleSheet(STATUS_BAR_UNSAVED_STYLE)
        else:
            self.lbl_status_save_state.setText("● Saved")
            self.lbl_status_save_state.setStyleSheet(STATUS_BAR_SAVED_STYLE)

    def _sync_status_and_clustering_controls(self) -> None:
        """Synchronize menu checkmarks and status bar switches with ClusteringView state."""
        self._on_confidence_threshold_changed(getattr(self.clustering_view, "auto_assign_threshold", 0.85))
        self._on_clustering_auto_assign_changed(bool(getattr(self.clustering_view, "auto_assign_enabled", False)))
        self._on_clustering_active_learning_toggled(bool(getattr(self.clustering_view, "active_learning_enabled", True)))
        self._sync_view_overlay_focused_from_cluster_view(
            bool(getattr(self.clustering_view, "overlay_focused_unit_enabled", True))
        )
        self._sync_view_shared_y_from_cluster_view(
            bool(getattr(self.clustering_view, "shared_y_axis_enabled", False))
        )

    def _on_reset_phase2_layout(self) -> None:
        """Reset panel layout to default for the active workspace view (§ 11.1, § 13.3)."""
        if self.stack.currentIndex() == self.VIEW_DETECTION:
            if hasattr(self.detection_view, "reset_default_layout"):
                self.detection_view.reset_default_layout()
            self.status.clearMessage()
            return
        if self.stack.currentIndex() != self.VIEW_CLUSTERING:
            self.switch_to_clustering()
        self.clustering_view.reset_default_layout()
        self.status.showMessage("Clustering panel layout reset to default.")

    def _on_change_stats_parameters(self) -> None:
        """Open the Stats Parameters modal dialog to inspect calculations and configure hyperparameters."""
        current_params = getattr(self.detection_view, "stats_params", StatsParameters.default())
        dlg = StatsParametersDialog(
            current_params=current_params,
            default_params=self.detection_view.default_stats_parameters(),
            parent=self,
        )
        if dlg.exec():
            new_params = dlg.get_parameters()
            self.detection_view.update_stats_parameters(new_params)
            self.status.clearMessage()

    def _ensure_clustering_view(self):
        """Switch to the Phase 2 ClusteringView if needed and return it."""
        if self.stack.currentIndex() != self.VIEW_CLUSTERING:
            self.switch_to_clustering()
        return self.clustering_view

    def _on_feature_settings(self, *args: Any, **kwargs: Any) -> None:
        """Launch feature space and weights configuration dialog."""
        self._ensure_clustering_view().prompt_feature_settings()

    def _on_embedding_settings(self, *args: Any, **kwargs: Any) -> None:
        """Launch 2D embedding (PCA / UMAP) configuration dialog."""
        self._ensure_clustering_view().prompt_embedding_settings()

    def _on_seed_clusters(self, *args: Any, **kwargs: Any) -> None:
        """Launch Affinity Propagation cluster seeding dialog."""
        self._ensure_clustering_view().prompt_seed_clusters()

    def _on_customize_clusters(self, *args: Any, **kwargs: Any) -> None:
        """Launch customize clusters dialog."""
        self._ensure_clustering_view().prompt_customize_clusters()

    def _on_hierarchical_clustering(self, *args: Any, **kwargs: Any) -> None:
        """Launch response clustering configuration dialog via ClusteringView."""
        view = self._ensure_clustering_view()
        ok = view.prompt_hierarchical_clustering()
        if ok:
            n_units = len(view._pooled_rows)
            n_clusters = (
                len(view.current_clustering_result.cluster_summaries)
                if view.current_clustering_result
                else 0
            )
            alg = view.active_clustering_params.algorithm
            self.status.showMessage(
                f"Clustered {n_units} units into {n_clusters} response types ({alg})."
            )

    def _on_reset_clustering(self, *args: Any, **kwargs: Any) -> None:
        """Trigger Reset Clustering in Phase 2 view."""
        self._ensure_clustering_view().reset_clustering(confirm=True)

    def _show_about_dialog(self, *args: Any, **kwargs: Any) -> None:
        """Show about information."""
        QMessageBox.about(
            self,
            "About OptoCurator",
            f"<b>OptoCurator v{__version__}</b><br><br>"
            "Open-source desktop application for statistical identification, manual curation, "
            "and multi-session response-type clustering of optogenetically tagged electrophysiological units.<br><br>"
            "Author: Bram Nuttin<br>"
            "Licensed under the MIT License.",
        )

    def _show_shortcuts_dialog(self, *args: Any, **kwargs: Any) -> None:
        """Show keyboard shortcuts dialog."""
        shortcuts_text = (
            "<b>File &amp; Navigation:</b><br>"
            "• <b>Ctrl+N / Ctrl+O / Ctrl+S / Ctrl+Shift+S / Ctrl+W</b>: New / Open / Save / Save as / Close<br>"
            "• <b>Ctrl+E</b>: Add experiment… (Clustering mode)<br>"
            "• <b>Ctrl+Q</b>: Exit<br>"
            "• <b>Ctrl+H / Ctrl+1 / Ctrl+2</b>: Home / Detection / Clustering<br>"
            "• <b>? / F1</b>: Keyboard shortcuts / Documentation<br><br>"
            "<b>Optotagged Cell Detection:</b><br>"
            "• <b>T</b>: Mark as tagged / untagged<br>"
            "• <b>Ctrl+Z</b>: Undo criteria change<br><br>"
            "<b>Response Clustering:</b><br>"
            "• <b>1–9, 0, Q, W, E, R, T</b>: Assign focused unit to cluster<br>"
            "• <b>N</b>: Assign focused unit to a new cluster<br>"
            "• <b>Backspace</b>: Unassign focused unit<br>"
            "• <b>S / Space</b>: Skip focused unit<br>"
            "• <b>G</b>: Go to suggested unit<br>"
            "• <b>Ctrl+Z / Ctrl+Y / Ctrl+Shift+Z</b>: Undo / Redo curation action<br>"
            "• <b>Esc</b>: Close menu, dialog, or cluster inspector<br>"
        )
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts_text)

    def _show_documentation_dialog(self, *args: Any, **kwargs: Any) -> None:
        """Show OptoCurator workflow documentation dialog."""
        doc_text = (
            "<b>OptoCurator Documentation</b><br><br>"
            "<b>1. Detection mode:</b> Load a Kilosort/Phy experiment folder or an NWB (.nwb) file, "
            "inspect light-evoked rasters, PSTHs, and waveforms, tune statistical tagging criteria "
            "(latency, jitter, reliability, SALT p, waveform correlation), and save curated results to disk.<br><br>"
            "<b>2. Clustering mode:</b> Create or open a clustering project folder, pool tagged units "
            "across multiple experiments, seed or run unsupervised clustering, and refine response "
            "types interactively with active learning and confidence-thresholded auto-assignment."
        )
        QMessageBox.information(self, "OptoCurator Documentation", doc_text)

    def _show_citation_dialog(self, *args: Any, **kwargs: Any) -> None:
        """Show citation information and BibTeX entry."""
        cite_text = (
            "<b>How to cite OptoCurator</b><br><br>"
            "If you use OptoCurator in your research, please cite:<br><br>"
            "<pre>"
            "@software{optocurator,\n"
            "  author  = {Bram Nuttin},\n"
            "  title   = {OptoCurator: Statistical Identification and Response-Type Clustering "
            "of Optogenetically Tagged Units},\n"
            f"  version = {{{__version__}}},\n"
            "  year    = {2026},\n"
            "  license = {MIT},\n"
            "  url     = {https://github.com/bramn22/OptoCurator}\n"
            "}"
            "</pre>"
        )
        QMessageBox.information(self, "How to cite OptoCurator", cite_text)

    def closeEvent(self, event: Any) -> None:
        """Prompt for unsaved changes and shut down workspace background workers before destruction."""
        if not self._confirm_discard_detection_changes("exiting"):
            event.ignore()
            return
        if not self._confirm_discard_clustering_changes("exiting"):
            event.ignore()
            return
        if hasattr(self, "clustering_view") and self.clustering_view is not None:
            self.clustering_view.shutdown_background_workers()
        super().closeEvent(event)
