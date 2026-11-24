"""Reusable right-click SVG vector figure export utilities (Layer 2)."""

from pathlib import Path
import re
from typing import Any, Callable, List, Optional, Tuple, Union

import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtSvg import QSvgGenerator
from PyQt6.QtWidgets import QFileDialog, QGraphicsView, QMenu, QMessageBox, QWidget

from optocurator.ui.theme import CLUSTERS_MENU_STYLE, COLOR_PANEL

_LAST_EXPORT_DIR: Optional[Path] = None
_DRAG_THRESHOLD_PX: int = 4


def sanitize_filename(name: str) -> str:
    """Return a clean, filesystem-safe filename stem."""
    text = str(name or "").strip()
    if text.lower().endswith(".svg"):
        text = text[:-4]
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned or "figure"


def _resolve_default_name(
    default_name: Union[str, Callable[[], str]],
    fallback: str = "figure",
) -> str:
    """Resolve a static string or callable name provider to a sanitized stem."""
    raw: str
    if callable(default_name):
        try:
            raw = str(default_name() or "")
        except Exception:
            raw = fallback
    else:
        raw = str(default_name or "")
    return sanitize_filename(raw or fallback)


def _collect_graphics_views(widget: QWidget) -> List[QGraphicsView]:
    """Collect all unique QGraphicsView instances at or under `widget`."""
    views: List[QGraphicsView] = []
    seen_views: set[int] = set()
    if isinstance(widget, QGraphicsView):
        views.append(widget)
        seen_views.add(id(widget))
    for child in widget.findChildren(QGraphicsView):
        if id(child) not in seen_views:
            views.append(child)
            seen_views.add(id(child))
    return views


def _collect_scatter_items(views: List[QGraphicsView]) -> List[pg.ScatterPlotItem]:
    """Collect all unique pyqtgraph ScatterPlotItem instances in `views`."""
    scatters: List[pg.ScatterPlotItem] = []
    seen_items: set[int] = set()
    for view in views:
        scene = view.scene()
        if scene is None:
            continue
        for item in scene.items():
            if isinstance(item, pg.ScatterPlotItem) and id(item) not in seen_items:
                scatters.append(item)
                seen_items.add(id(item))
    return scatters


def export_widget_to_svg(widget: QWidget, path: Union[str, Path]) -> Path:
    """Render a Qt/pyqtgraph widget or composite panel into a true vector SVG file.

    Temporarily disables pixmap symbol caching on all descendant `pg.ScatterPlotItem`
    instances and background pixmap caching on `QGraphicsView` widgets so plots are
    emitted as pure vector primitives rather than embedded raster `<image>` blobs,
    then restores all original cache settings in a `finally:` block.
    """
    out_path = Path(path)
    if out_path.suffix.lower() != ".svg":
        out_path = Path(f"{out_path}.svg")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if widget.layout() is not None:
        widget.layout().activate()

    size = widget.size()
    if size.width() <= 0 or size.height() <= 0:
        size = widget.sizeHint()
    w = max(1, int(size.width()))
    h = max(1, int(size.height()))

    views = _collect_graphics_views(widget)
    saved_view_caches: List[Tuple[QGraphicsView, Any]] = []
    for view in views:
        orig_mode = view.cacheMode()
        saved_view_caches.append((view, orig_mode))
        if orig_mode != QGraphicsView.CacheModeFlag.CacheNone:
            view.setCacheMode(QGraphicsView.CacheModeFlag.CacheNone)

    scatter_items = _collect_scatter_items(views)
    saved_cache_flags: List[Tuple[pg.ScatterPlotItem, bool]] = []
    for item in scatter_items:
        orig_cache = bool(item.opts.get("useCache", True))
        saved_cache_flags.append((item, orig_cache))
        if orig_cache:
            item.opts["useCache"] = False
            try:
                item.updateSpots()
            except Exception:
                pass

    try:
        generator = QSvgGenerator()
        generator.setFileName(str(out_path))
        generator.setSize(QSize(w, h))
        generator.setViewBox(QRect(0, 0, w, h))
        generator.setResolution(96)
        generator.setTitle("OptoCurator Figure")

        painter = QPainter()
        if not painter.begin(generator):
            raise RuntimeError(f"Failed to begin SVG painting to {out_path}")
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            painter.fillRect(QRect(0, 0, w, h), QColor(COLOR_PANEL))
            widget.render(painter)
        finally:
            painter.end()
    finally:
        for item, orig_cache in saved_cache_flags:
            if item.opts.get("useCache", True) != orig_cache:
                item.opts["useCache"] = orig_cache
                try:
                    item.updateSpots()
                except Exception:
                    pass
        for view, orig_mode in saved_view_caches:
            if view.cacheMode() != orig_mode:
                view.setCacheMode(orig_mode)

    return out_path


def prompt_save_figure_svg(
    parent: QWidget,
    export_target: QWidget,
    default_name: Union[str, Callable[[], str]] = "figure",
) -> Optional[Path]:
    """Prompt the user for an SVG destination and export `export_target`."""
    global _LAST_EXPORT_DIR

    stem = _resolve_default_name(default_name, fallback="figure")
    base_dir = (
        _LAST_EXPORT_DIR
        if (_LAST_EXPORT_DIR is not None and _LAST_EXPORT_DIR.exists())
        else Path.home()
    )
    initial_path = str(base_dir / f"{stem}.svg")

    chosen_path, _ = QFileDialog.getSaveFileName(
        parent,
        "Save Figure as SVG",
        initial_path,
        "SVG Vector Graphics (*.svg)",
    )
    if not chosen_path:
        return None

    try:
        saved_path = export_widget_to_svg(export_target, chosen_path)
    except Exception as exc:
        QMessageBox.warning(
            parent,
            "Save Figure Failed",
            f"Could not export figure to SVG:\n{exc}",
        )
        return None

    _LAST_EXPORT_DIR = saved_path.parent
    return saved_path


def populate_figure_export_menu(
    menu: QMenu,
    export_target: QWidget,
    default_name: Union[str, Callable[[], str]] = "figure",
    label: str = "Save figure as SVG…",
    subplot_target: Optional[QWidget] = None,
    subplot_default_name: Optional[Union[str, Callable[[], str]]] = None,
    subplot_label: str = "Save this subplot as SVG…",
    add_separator: bool = False,
) -> None:
    """Append SVG export action(s) to an existing QMenu."""
    if add_separator:
        menu.addSeparator()

    dialog_parent = menu.parentWidget() if menu.parentWidget() is not None else export_target

    act_main = menu.addAction(label)
    act_main.triggered.connect(
        lambda _checked=False, p=dialog_parent, t=export_target, n=default_name: prompt_save_figure_svg(
            p, t, n
        )
    )

    if subplot_target is not None and subplot_target is not export_target:
        resolved_sub_name = (
            subplot_default_name if subplot_default_name is not None else default_name
        )
        act_sub = menu.addAction(subplot_label)
        act_sub.triggered.connect(
            lambda _checked=False, p=dialog_parent, t=subplot_target, n=resolved_sub_name: prompt_save_figure_svg(
                p, t, n
            )
        )


def _event_global_point(event: Any) -> Optional[QPoint]:
    """Extract global QPoint from a QMouseEvent or QContextMenuEvent."""
    if hasattr(event, "globalPosition"):
        try:
            return event.globalPosition().toPoint()
        except Exception:
            pass
    if hasattr(event, "globalPos"):
        try:
            return event.globalPos()
        except Exception:
            pass
    return None


class _FigureExportEventFilter(QObject):
    """Lightweight event filter opening an SVG export menu on stationary right-click."""

    def __init__(
        self,
        target_widget: QWidget,
        export_target: QWidget,
        default_name: Union[str, Callable[[], str]],
        label: str,
        subplot_target: Optional[QWidget],
        subplot_default_name: Optional[Union[str, Callable[[], str]]],
        subplot_label: str,
    ) -> None:
        super().__init__(target_widget)
        self._target_widget = target_widget
        self._export_target = export_target
        self._default_name = default_name
        self._label = label
        self._subplot_target = subplot_target
        self._subplot_default_name = subplot_default_name
        self._subplot_label = subplot_label
        self._rpress_global_pos: Optional[QPoint] = None
        self._rdragged: bool = False

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        etype = event.type()
        if etype == QEvent.Type.MouseButtonPress:
            if hasattr(event, "button") and event.button() == Qt.MouseButton.RightButton:
                self._rpress_global_pos = _event_global_point(event)
                self._rdragged = False
            return False

        if etype == QEvent.Type.MouseMove:
            if self._rpress_global_pos is not None:
                cur_pos = _event_global_point(event)
                if (
                    cur_pos is not None
                    and (cur_pos - self._rpress_global_pos).manhattanLength() > _DRAG_THRESHOLD_PX
                ):
                    self._rdragged = True
            return False

        if etype == QEvent.Type.MouseButtonRelease:
            if hasattr(event, "button") and event.button() == Qt.MouseButton.RightButton:
                if self._rpress_global_pos is not None:
                    cur_pos = _event_global_point(event)
                    if (
                        cur_pos is not None
                        and (cur_pos - self._rpress_global_pos).manhattanLength() > _DRAG_THRESHOLD_PX
                    ):
                        self._rdragged = True
                self._rpress_global_pos = None
            return False

        if etype != QEvent.Type.ContextMenu:
            return False

        was_dragged = self._rdragged
        self._rdragged = False
        self._rpress_global_pos = None
        if was_dragged:
            event.accept()
            return True

        menu = QMenu(self._target_widget)
        menu.setStyleSheet(CLUSTERS_MENU_STYLE)
        populate_figure_export_menu(
            menu,
            export_target=self._export_target,
            default_name=self._default_name,
            label=self._label,
            subplot_target=self._subplot_target,
            subplot_default_name=self._subplot_default_name,
            subplot_label=self._subplot_label,
            add_separator=False,
        )
        global_pos = _event_global_point(event)
        if global_pos is not None:
            menu.exec(global_pos)
        event.accept()
        return True


def install_figure_export_menu(
    target_widget: QWidget,
    export_target: Optional[QWidget] = None,
    default_name: Union[str, Callable[[], str]] = "figure",
    label: str = "Save figure as SVG…",
    subplot_target: Optional[QWidget] = None,
    subplot_default_name: Optional[Union[str, Callable[[], str]]] = None,
    subplot_label: str = "Save this subplot as SVG…",
) -> QObject:
    """Install a right-click SVG export menu on `target_widget` (and its viewport if present)."""
    resolved_export = export_target if export_target is not None else target_widget
    filt = _FigureExportEventFilter(
        target_widget=target_widget,
        export_target=resolved_export,
        default_name=default_name,
        label=label,
        subplot_target=subplot_target,
        subplot_default_name=subplot_default_name,
        subplot_label=subplot_label,
    )
    target_widget.installEventFilter(filt)
    if hasattr(target_widget, "viewport"):
        vp = target_widget.viewport()
        if vp is not None:
            vp.installEventFilter(filt)

    existing = getattr(target_widget, "_figure_export_filters", None)
    if not isinstance(existing, list):
        existing = []
        target_widget._figure_export_filters = existing
    existing.append(filt)
    return filt
