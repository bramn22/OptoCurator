"""Launcher dashboard widget for OptoCurator."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from optocurator.ui.theme import (
    COLOR_ACCENT,
    LAUNCHER_CARD_DESC_STYLE,
    LAUNCHER_CARD_TITLE_STYLE,
    LAUNCHER_SUBTITLE_STYLE,
    LAUNCHER_TITLE_STYLE,
    build_mode_card_style,
    make_app_mark_icon,
)


class ModeCardWidget(QFrame):
    """Minimalist clickable button card for selecting a workflow."""

    clicked = pyqtSignal()

    def __init__(
        self,
        title: str,
        description: str = "",
        accent_color: str = COLOR_ACCENT,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.accent_color = accent_color
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(296, 138)
        self.setObjectName("modeCard")
        self.setStyleSheet(build_mode_card_style(accent_color))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setWordWrap(True)
        title_label.setStyleSheet(LAUNCHER_CARD_TITLE_STYLE)
        layout.addWidget(title_label)

        if description:
            desc_label = QLabel(description)
            desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet(LAUNCHER_CARD_DESC_STYLE)
            layout.addWidget(desc_label)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


ModeCard = ModeCardWidget


class LauncherDashboardWidget(QWidget):
    """Home launcher dashboard allowing user to enter Phase 1 or Phase 2."""

    open_detection = pyqtSignal()
    open_clustering = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("launcherDashboard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(40)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Header Section
        header_layout = QVBoxLayout()
        header_layout.setSpacing(8)
        header_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        app_mark = QLabel()
        app_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        app_mark.setPixmap(make_app_mark_icon(40).pixmap(40, 40))
        header_layout.addWidget(app_mark)

        app_title = QLabel("OptoCurator")
        app_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        app_title.setStyleSheet(LAUNCHER_TITLE_STYLE)
        header_layout.addWidget(app_title)

        main_layout.addLayout(header_layout)

        # Two Main Workflow Buttons
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(24)
        cards_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Button 1: Optotagged Cell Detection
        self.detection_card = ModeCardWidget(
            title="Optotagged Cell Detection",
            description="Identify optotagged units",
            accent_color=COLOR_ACCENT,
        )
        self.detection_card.clicked.connect(self.open_detection.emit)
        cards_layout.addWidget(self.detection_card)
        self.card_detection = self.detection_card

        # Button 2: Response Clustering
        self.clustering_card = ModeCardWidget(
            title="Response Clustering",
            description="Interactively cluster optotagged units",
            accent_color=COLOR_ACCENT,
        )
        self.clustering_card.clicked.connect(self.open_clustering.emit)
        cards_layout.addWidget(self.clustering_card)
        self.card_clustering = self.clustering_card

        main_layout.addLayout(cards_layout)
