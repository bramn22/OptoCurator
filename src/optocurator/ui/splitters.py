"""Reusable QSplitter with 1 px divider styling, hover grip, double-click reset, and QSettings persistence."""

from typing import Any, List, Optional

from PyQt6.QtCore import QByteArray, QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QSplitter, QSplitterHandle, QWidget

from optocurator.ui.theme import (
    COLOR_BORDER_STRONG,
    COLOR_DIVIDER,
    SPLITTER_HANDLE_WIDTH,
    SPLITTER_MAIN_STYLE,
)


class ResettableSplitterHandle(QSplitterHandle):
    """Splitter handle rendering a 1 px divider with a ~6 px grab area, hover grip, and double-click reset."""

    def __init__(self, orientation: Qt.Orientation, parent: QSplitter) -> None:
        super().__init__(orientation, parent)
        self._hovered: bool = False
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def enterEvent(self, event: Any) -> None:  # noqa: N802
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:  # noqa: N802
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:  # noqa: N802
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            splitter = self.splitter()
            if isinstance(splitter, ResettableSplitter):
                splitter.reset_to_default()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        rect = self.rect()
        divider_pen = QPen(QColor(COLOR_DIVIDER), 1)
        painter.setPen(divider_pen)

        if self.orientation() == Qt.Orientation.Horizontal:
            cx = rect.width() // 2
            painter.drawLine(cx, 0, cx, rect.height())
            if self._hovered:
                grip_pen = QPen(QColor(COLOR_BORDER_STRONG), 2)
                grip_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(grip_pen)
                cy = rect.height() // 2
                painter.drawLine(cx, max(4, cy - 14), cx, min(rect.height() - 4, cy + 14))
        else:
            cy = rect.height() // 2
            painter.drawLine(0, cy, rect.width(), cy)
            if self._hovered:
                grip_pen = QPen(QColor(COLOR_BORDER_STRONG), 2)
                grip_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(grip_pen)
                cx = rect.width() // 2
                painter.drawLine(max(4, cx - 14), cy, min(rect.width() - 4, cx + 14), cy)

        painter.end()


class ResettableSplitter(QSplitter):
    """Non-collapsible QSplitter with default sizes, double-click reset, and QSettings persistence."""

    layout_reset = pyqtSignal()

    SETTINGS_ORG = "OptoCurator"
    SETTINGS_APP = "OptoCurator"

    def __init__(
        self,
        orientation: Qt.Orientation,
        parent: Optional[QWidget] = None,
        default_sizes: Optional[List[int]] = None,
        settings_key: Optional[str] = None,
    ) -> None:
        super().__init__(orientation, parent)
        self._default_sizes: List[int] = list(default_sizes) if default_sizes else []
        self._settings_key: Optional[str] = settings_key
        self.setChildrenCollapsible(False)
        self.setHandleWidth(SPLITTER_HANDLE_WIDTH)
        self.setOpaqueResize(True)
        self.setStyleSheet(SPLITTER_MAIN_STYLE)
        self.splitterMoved.connect(self._on_splitter_moved)

    def createHandle(self) -> QSplitterHandle:  # noqa: N802
        return ResettableSplitterHandle(self.orientation(), self)

    def set_default_sizes(self, sizes: List[int]) -> None:
        """Set default pane sizes used by double-click reset and Reset panel layout."""
        self._default_sizes = list(sizes)

    def reset_to_default(self) -> None:
        """Restore splitter panes to their default sizes and persist."""
        if self._default_sizes and len(self._default_sizes) == self.count():
            self.setSizes(list(self._default_sizes))
            self.save_to_settings()
            self.layout_reset.emit()

    def _on_splitter_moved(self, _pos: int, _index: int) -> None:
        self.save_to_settings()

    def save_to_settings(self) -> None:
        """Persist splitter state to QSettings."""
        if not self._settings_key:
            return
        settings = QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)
        settings.setValue(f"splitters/{self._settings_key}", self.saveState())

    def restore_from_settings(self) -> bool:
        """Restore splitter state from QSettings if available."""
        if not self._settings_key:
            return False
        settings = QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)
        raw = settings.value(f"splitters/{self._settings_key}")
        if isinstance(raw, QByteArray) and not raw.isEmpty():
            return bool(self.restoreState(raw))
        return False
