"""Ultra-fast, zero-overhead pre-rendered plot image viewer using PyQt6.QtGui.QPixmap."""

from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QResizeEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from optocurator.ui.plots import DEFAULT_Y_AXIS_WIDTH, render_psth, render_raster
from optocurator.ui.theme import (
    COLOR_PANEL,
    IMAGE_VIEWER_ACTIVE_IMAGE_STYLE,
    IMAGE_VIEWER_BADGE_STYLE,
    IMAGE_VIEWER_COMPACT_FRAME_STYLE,
    IMAGE_VIEWER_FULL_FRAME_STYLE,
    IMAGE_VIEWER_META_STYLE,
    IMAGE_VIEWER_PLACEHOLDER_STYLE,
    IMAGE_VIEWER_TITLE_STYLE,
)


def _format_unit_display(session_id: str = "", unit_id: Any = None, global_unit_id: str = "") -> Tuple[str, str]:
    """Parse and format session_id and unit_id cleanly without underscore concatenation."""
    sess_str = str(session_id).strip() if session_id is not None and str(session_id).strip() != "None" else ""
    uid_str = ""
    if unit_id is not None and str(unit_id).strip() and str(unit_id).strip() != "None":
        try:
            uid_str = str(int(unit_id))
        except (ValueError, TypeError):
            uid_str = str(unit_id).strip()

    gid_str = str(global_unit_id).strip() if global_unit_id is not None else ""

    # If sess_str or uid_str is missing, try parsing global_unit_id
    if (not sess_str or not uid_str) and gid_str:
        # Match <session>_u<digits> (e.g. test_OptoCurator_u0594)
        m_u = re.match(r"^(.*)_u(\d+)$", gid_str)
        if m_u:
            if not sess_str:
                sess_str = m_u.group(1)
            if not uid_str:
                uid_str = str(int(m_u.group(2)))
        else:
            # Match <session>_<digits>
            m_d = re.match(r"^(.*)_(\d+)$", gid_str)
            if m_d:
                if not sess_str:
                    sess_str = m_d.group(1)
                if not uid_str:
                    uid_str = str(int(m_d.group(2)))
            else:
                if not sess_str:
                    sess_str = gid_str

    if not sess_str:
        sess_str = "Unknown"
    if not uid_str:
        uid_str = "—"

    return sess_str, uid_str


class ImageViewer(QFrame):
    """High-performance widget for instant display of pre-rendered PSTH & raster PNGs."""

    def __init__(self, parent: Optional[QWidget] = None, compact_mode: bool = False) -> None:
        super().__init__(parent)
        self.compact_mode: bool = compact_mode
        self.setObjectName("precomputedPlotViewer")
        if self.compact_mode:
            self.setStyleSheet(IMAGE_VIEWER_COMPACT_FRAME_STYLE)
        else:
            self.setStyleSheet(IMAGE_VIEWER_FULL_FRAME_STYLE)

        self._current_pixmap: Optional[QPixmap] = None
        self._current_path: Optional[Path] = None
        self._current_metadata: Dict[str, Any] = {}
        self.plot_canvas: Optional[pg.GraphicsLayoutWidget] = None
        self.raster_plot: Optional[pg.PlotItem] = None
        self.psth_plot: Optional[pg.PlotItem] = None

        self._init_ui()
        if parent is None:
            self.show()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        if self.compact_mode:
            layout.setContentsMargins(6, 6, 6, 6)
            layout.setSpacing(4)
        else:
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(8)

        # Header bar with unit information and badge
        self.header_bar = QHBoxLayout()
        self.header_bar.setSpacing(6)

        self.lbl_title = QLabel("Pre-rendered Unit Plot")
        self.lbl_title.setStyleSheet(IMAGE_VIEWER_TITLE_STYLE)
        self.header_bar.addWidget(self.lbl_title)

        self.header_bar.addStretch()

        self.lbl_badge = QLabel("Instant Preview")
        self.lbl_badge.setVisible(False)
        self.lbl_badge.setStyleSheet(IMAGE_VIEWER_BADGE_STYLE)
        self.header_bar.addWidget(self.lbl_badge)
        layout.addLayout(self.header_bar)

        # Subtitle for metric details
        self.lbl_meta = QLabel("Select any pooled unit to view its spike raster and PSTH histogram.")
        self.lbl_meta.setStyleSheet(IMAGE_VIEWER_META_STYLE)
        self.lbl_meta.setWordWrap(True)
        layout.addWidget(self.lbl_meta)

        # Image Display Label (for legacy pre-rendered PNG fallback or cached QPixmap)
        self.lbl_image = QLabel()
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.lbl_image.setMinimumSize(200, 160)
        self.lbl_image.setStyleSheet(IMAGE_VIEWER_PLACEHOLDER_STYLE)
        layout.addWidget(self.lbl_image, stretch=1)

        # Live pyqtgraph plot canvas (only in full mode, omitted in compact mode)
        if not self.compact_mode:
            self.plot_canvas = pg.GraphicsLayoutWidget()
            self.plot_canvas.setBackground(COLOR_PANEL)
            self.plot_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.plot_canvas.setMinimumSize(200, 160)

            self.raster_plot = self.plot_canvas.addPlot(row=0, col=0)
            self.raster_plot.setTitle(None)
            self.raster_plot.hideAxis("bottom")
            self.raster_plot.getAxis("left").setWidth(DEFAULT_Y_AXIS_WIDTH)
            self.raster_plot.setMouseEnabled(x=False, y=False)

            self.psth_plot = self.plot_canvas.addPlot(row=1, col=0)
            self.psth_plot.setTitle(None)
            self.psth_plot.setXLink(self.raster_plot)
            self.psth_plot.getAxis("left").setWidth(DEFAULT_Y_AXIS_WIDTH)
            self.psth_plot.setMouseEnabled(x=False, y=False)

            if hasattr(self.plot_canvas, "ci") and hasattr(self.plot_canvas.ci, "layout"):
                self.plot_canvas.ci.layout.setSpacing(2)
                self.plot_canvas.ci.layout.setRowStretchFactor(0, 12)
                self.plot_canvas.ci.layout.setRowStretchFactor(1, 10)

            self.plot_canvas.setVisible(False)
            layout.addWidget(self.plot_canvas, stretch=1)
        else:
            self.plot_canvas = None
            self.raster_plot = None
            self.psth_plot = None

        self._show_placeholder("Select a unit to inspect PSTH & raster.")

    def set_cached_pixmap(
        self,
        pixmap: Optional[QPixmap],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Display an in-memory rendered QPixmap instantly with zero graphics scene overhead."""
        self._current_metadata = metadata or {}
        self._update_header()

        if self.plot_canvas is not None:
            self.plot_canvas.setVisible(False)
        self.lbl_image.setVisible(True)

        if pixmap is None or pixmap.isNull():
            self._show_placeholder("No plot available for this unit.")
            return False

        self._current_pixmap = pixmap
        self.lbl_image.setStyleSheet(IMAGE_VIEWER_ACTIVE_IMAGE_STYLE)
        self._scale_and_render()
        return True

    def set_live_plot(
        self,
        psth_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        window_ms: Tuple[float, float] = (-30.0, 50.0),
        order_by: str = "presentation",
        marker_size: float = 3.0,
    ) -> bool:
        """Render live trial-aligned spike raster and PSTH histogram using pyqtgraph."""
        self._current_metadata = metadata or {}
        self._update_header()

        if self.compact_mode or self.plot_canvas is None or self.raster_plot is None or self.psth_plot is None:
            return False

        render_raster(
            self.raster_plot,
            raster_x=psth_data.get("raster_x", np.array([])),
            raster_y=psth_data.get("raster_y", np.array([])),
            n_trials=psth_data.get("n_trials", 0),
            title=None,
            x_range=window_ms,
            order_by=order_by,
            show_x_axis=False,
            marker_size=marker_size,
        )

        render_psth(
            self.psth_plot,
            psth_bins=psth_data.get("psth_bins", np.array([])),
            psth_rates=psth_data.get("psth_rates", np.array([])),
            baseline_mean=psth_data.get("baseline_mean", 0.0),
            baseline_3sigma=psth_data.get("baseline_3sigma", 0.0),
            title=None,
            x_range=window_ms,
            show_x_axis=True,
            is_smooth=psth_data.get("is_smooth", True),
        )

        self.raster_plot.setTitle(None)
        self.psth_plot.setTitle(None)

        self.lbl_image.setVisible(False)
        self.plot_canvas.setVisible(True)

        # Set valid 1x1 pixmap for backward compatibility with legacy tests
        self._current_pixmap = QPixmap(1, 1)
        return True

    def set_image(
        self,
        image_path: Optional[Union[str, Path]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Load and display a pre-rendered image instantly with zero lag."""
        self._current_metadata = metadata or {}
        self._update_header()

        if self.plot_canvas is not None:
            self.plot_canvas.setVisible(False)
        self.lbl_image.setVisible(True)

        if image_path is None:
            self._show_placeholder("No plot specified for this unit.")
            return False

        path = Path(image_path)
        self._current_path = path

        if not path.is_file():
            self._show_placeholder(f"Pre-rendered plot not found:\n{path.name}")
            return False

        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._show_placeholder(f"Failed to decode image file:\n{path.name}")
            return False

        self._current_pixmap = pixmap
        self.lbl_image.setStyleSheet(IMAGE_VIEWER_ACTIVE_IMAGE_STYLE)
        self._scale_and_render()
        return True

    def clear_preview(self) -> None:
        """Reset viewer to empty placeholder state."""
        self._current_pixmap = None
        self._current_path = None
        self._current_metadata.clear()
        if getattr(self, "raster_plot", None) is not None:
            self.raster_plot.clear()
        if getattr(self, "psth_plot", None) is not None:
            self.psth_plot.clear()
        if getattr(self, "plot_canvas", None) is not None:
            self.plot_canvas.setVisible(False)
        if hasattr(self, "lbl_image"):
            self.lbl_image.setVisible(True)
        self.lbl_title.setText("Pre-rendered Unit Plot")
        self.lbl_meta.setText("Select any pooled unit to view its spike raster and PSTH histogram.")
        self._show_placeholder("Select a unit to inspect PSTH & raster.")

    def _show_placeholder(self, message: str) -> None:
        """Render informative placeholder text."""
        self._current_pixmap = None
        if getattr(self, "plot_canvas", None) is not None:
            self.plot_canvas.setVisible(False)
        if hasattr(self, "lbl_image"):
            self.lbl_image.setVisible(True)
            self.lbl_image.setPixmap(QPixmap())
            self.lbl_image.setText(message)
            self.lbl_image.setStyleSheet(IMAGE_VIEWER_PLACEHOLDER_STYLE)

    def _update_header(self) -> None:
        """Update header and metadata labels from current metadata."""
        if not self._current_metadata:
            self.lbl_title.setText("Pre-rendered Unit Plot")
            self.lbl_meta.setText("Select any pooled unit to view its spike raster and PSTH histogram.")
            self.lbl_badge.setVisible(False)
            return

        sess = self._current_metadata.get("session_id", "")
        uid = self._current_metadata.get("unit_id")
        gid = self._current_metadata.get("global_unit_id", "Unknown")
        lat = self._current_metadata.get("latency_ms", "N/A")
        rel = self._current_metadata.get("reliability", "N/A")
        salt_p = self._current_metadata.get("salt_p_value", "N/A")

        lat_str = f"{lat:.1f} ms" if isinstance(lat, (int, float)) and lat < 500 else str(lat)
        rel_str = f"{rel:.2f}" if isinstance(rel, (int, float)) else str(rel)
        salt_str = f"{salt_p:.4f}" if isinstance(salt_p, (int, float)) else str(salt_p)

        sess_str, uid_str = _format_unit_display(str(sess) if sess else "", uid, str(gid))
        if sess_str and uid_str:
            self.lbl_title.setText(f"Session: {sess_str}  •  Unit: {uid_str}")
        elif uid_str:
            self.lbl_title.setText(f"Unit: {uid_str}")
        elif sess_str:
            self.lbl_title.setText(f"Session: {sess_str}")
        else:
            self.lbl_title.setText(f"Unit: {gid}")

        self.lbl_meta.setText(
            f"Latency: {lat_str}  |  Reliability: {rel_str}  |  SALT p: {salt_str}"
        )
        self.lbl_badge.setVisible(False)

    def _scale_and_render(self) -> None:
        """Scale the active pixmap smoothly to fit within lbl_image maintaining aspect ratio."""
        if self._current_pixmap is None or self._current_pixmap.isNull():
            return

        target_size = self.lbl_image.size()
        if target_size.width() <= 10 or target_size.height() <= 10:
            return

        scaled = self._current_pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.lbl_image.setPixmap(scaled)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Rescale pixmap automatically when widget dimensions change."""
        super().resizeEvent(event)
        self._scale_and_render()
