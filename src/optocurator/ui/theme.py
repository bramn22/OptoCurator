"""Centralized Design System, Typography, Palette, and Global QSS Stylesheet.

Follows UI_OVERHAUL_SPEC.md § 4 and § 8.1:
- Calm light palette (#F6F7F9 canvas, #FFFFFF panels, #E3E5E8 dividers, #2563EB accent)
- Strict typographic scale (Inter / Segoe UI, 10px–32px)
- 15-colour colour-blind-safe cluster palette bound to cluster_id via ClusterColorService
- Fixed plot sizes, control heights, and component style builders
"""

import html
from pathlib import Path
import struct
import tempfile
from typing import Dict, Mapping, Optional, Tuple, Union
import zlib
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap

from optocurator.analysis.clustering import (
    CLUSTER_PALETTE,
    COLOR_UNASSIGNED,
    color_for_cluster,
)

# -----------------------------------------------------------------------------
# 1. TYPOGRAPHY (§ 4.1)
# -----------------------------------------------------------------------------
FONT_NAME: str = "Inter"
FONT_FALLBACK: str = "Segoe UI"
FONT_FAMILY: str = (
    '"Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, "Roboto", "Helvetica Neue", sans-serif'
)
FONT_FAMILY_MONO: str = '"Inter", "Segoe UI", "Consolas", "SF Mono", monospace'

FONT_SIZE_TICK: int = 10
FONT_SIZE_CAPTION: int = 11
FONT_SIZE_BODY: int = 12
FONT_SIZE_PANEL_TITLE: int = 13
FONT_SIZE_HEADER_TITLE: int = 16
FONT_SIZE_UNIT_TITLE: int = 16
FONT_SIZE_APP_TITLE: int = 32


def make_font(
    px_size: int = FONT_SIZE_BODY,
    weight: Optional[Union[int, QFont.Weight]] = None,
    bold: bool = False,
    italic: bool = False,
) -> QFont:
    """Create a standardized UI QFont using the design system font stack."""
    font = QFont(FONT_NAME)
    font.setFamilies([FONT_NAME, FONT_FALLBACK])
    font.setPixelSize(int(px_size))
    if bold:
        font.setBold(True)
    if weight is not None:
        if isinstance(weight, QFont.Weight):
            font.setWeight(weight)
        else:
            font.setWeight(QFont.Weight(int(weight)))
    if italic:
        font.setItalic(True)
    return font


def make_mono_font(
    px_size: int = FONT_SIZE_BODY,
    weight: Optional[Union[int, QFont.Weight]] = None,
    bold: bool = False,
) -> QFont:
    """Create a tabular/monospaced numeric QFont using the design system font stack."""
    font = make_font(px_size=px_size, weight=weight, bold=bold)
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setFixedPitch(True)
    return font


# -----------------------------------------------------------------------------
# 2. COLOUR TOKENS (§ 4.2)
# -----------------------------------------------------------------------------
COLOR_CANVAS: str = "#F6F7F9"          # Window background, cluster area
COLOR_PANEL: str = "#FFFFFF"           # Panels, cards, dialogs
COLOR_DIVIDER: str = "#E3E5E8"         # 1 px lines
COLOR_BORDER_STRONG: str = "#D0D5DB"   # Inputs, secondary buttons

COLOR_TEXT: str = "#1F2328"            # Primary text
COLOR_TEXT_SECONDARY: str = "#59636E"  # Secondary text
COLOR_TEXT_MUTED: str = "#6E7781"      # Tick labels, captions (passes 4.5:1)
COLOR_AXIS: str = "#8C959F"            # Axes, ticks

COLOR_ACCENT: str = "#2563EB"          # Primary buttons, selection, focus
COLOR_ACCENT_HOVER: str = "#1D4ED8"
COLOR_ACCENT_PRESS: str = "#1E40AF"
COLOR_ACCENT_BG: str = "#EEF3FE"       # Selected rows, focused items

COLOR_SUCCESS: str = "#1A7F37"         # Curated, saved
COLOR_SUCCESS_BG: str = "#E6F4EA"

COLOR_WARNING: str = "#9A6700"         # Threshold line, "!" badge
COLOR_WARNING_BG: str = "#FFF1C2"     # "!" badge bg

COLOR_DANGER: str = "#CF222E"          # Destructive actions only
COLOR_DANGER_BG: str = "#FFEBE9"

COLOR_LIGHT_PULSE_BAND: str = "#EAF2FD"  # 0–10 ms band in every PSTH/raster
COLOR_PASSING_FILL: str = "#BFD3F7"      # Passing area fill in criterion KDEs
COLOR_PASS_FILL: str = COLOR_PASSING_FILL
COLOR_SEEDED: str = "#8E7CC3"            # Seeded status

COLOR_POPULATION_CURVE: str = "#94A3B8"  # Population KDE curve / subtle overlay
COLOR_KDE_POP_FILL: str = "#E1E4E8"      # Population KDE area fill
COLOR_WAVEFORM_SPONT: str = "#D55E00"    # Spontaneous waveform (vermilion)
COLOR_WAVEFORM_EVOKED: str = "#0072B2"   # Light-evoked waveform (blue)
COLOR_CANDIDATE_MARKER: str = "#DC2626"  # Candidate line / marker

# Additional semantic surface / probe / status tokens
COLOR_SURFACE_SUBTLE: str = "#F8FAFC"
COLOR_SURFACE_MUTED: str = "#F1F5F9"
COLOR_PROBE_SITE: str = "#CBD5E1"
COLOR_PROBE_SHANK_BG: str = "#F8FAFC"
COLOR_PROBE_SHANK_BORDER: str = "#E2E8F0"
COLOR_AUTO_ASSIGNED: str = "#2563EB"

PLOT_HIST_BRUSH: Tuple[int, int, int, int] = (37, 99, 235, 45)


# -----------------------------------------------------------------------------
# 3. CLUSTER COLOUR SERVICE & LUMINANCE HELPERS (§ 4.3, § 8.1)
# -----------------------------------------------------------------------------
def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Parse a hex colour string (#RRGGBB or #RGB) into an (R, G, B) tuple."""
    h = str(hex_color).strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    if len(h) != 6:
        return (160, 164, 170)
    try:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except ValueError:
        return (160, 164, 170)


def rgba_from_hex(hex_color: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    """Convert a hex colour string and alpha (0–255) to an (R, G, B, A) tuple."""
    r, g, b = hex_to_rgb(hex_color)
    return (r, g, b, max(0, min(255, int(alpha))))


def perceived_luminance(hex_color: str) -> float:
    """Compute perceived luminance in [0.0, 1.0] for a hex colour."""
    r, g, b = hex_to_rgb(hex_color)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def text_color_for_cluster(
    cluster_id_or_hex: Union[int, str],
    custom_colors: Optional[Mapping[int, str]] = None,
) -> str:
    """Return a readable text colour on white for a cluster ID or hex colour.

    For light colours (perceived luminance > 0.6), darkens RGB channels to ~55%
    so text in a cluster colour stays readable on white, while swatches and plot
    lines keep the true colour (§ 4.3).
    """
    if isinstance(cluster_id_or_hex, int) or (
        isinstance(cluster_id_or_hex, str)
        and cluster_id_or_hex.lstrip("-").isdigit()
    ):
        hex_color = color_for_cluster(int(cluster_id_or_hex), custom_colors)
    else:
        hex_color = str(cluster_id_or_hex).strip()
    if perceived_luminance(hex_color) > 0.6:
        r, g, b = hex_to_rgb(hex_color)
        r_d = max(0, min(255, int(round(r * 0.55))))
        g_d = max(0, min(255, int(round(g * 0.55))))
        b_d = max(0, min(255, int(round(b * 0.55))))
        return f"#{r_d:02X}{g_d:02X}{b_d:02X}"
    return hex_color


class ClusterColorService:
    """Centralized service for cluster colours bound to cluster_id (§ 4.3, § 8.1)."""

    def __init__(self, custom_colors: Optional[Dict[int, str]] = None) -> None:
        if isinstance(custom_colors, dict):
            self._custom_colors: Dict[int, str] = custom_colors
        else:
            self._custom_colors = {}
            if custom_colors:
                self.set_colors(custom_colors)

    @property
    def custom_colors(self) -> Dict[int, str]:
        """Return underlying custom colour mapping dictionary."""
        return self._custom_colors

    def get_color(self, cluster_id: int) -> str:
        """Return true swatch/plot hex colour for cluster_id."""
        return color_for_cluster(int(cluster_id), self._custom_colors)

    def get_text_color(self, cluster_id: int) -> str:
        """Return readable text hex colour on white for cluster_id."""
        return text_color_for_cluster(int(cluster_id), self._custom_colors)

    @staticmethod
    def _is_valid_hex(val: str) -> bool:
        s = val.strip()
        if not s.startswith("#") or len(s) not in (4, 7):
            return False
        try:
            int(s[1:], 16)
            return True
        except ValueError:
            return False

    def set_color(self, cluster_id: int, hex_str: str) -> None:
        """Set or clear a custom colour override for cluster_id."""
        cid = int(cluster_id)
        if cid < 0:
            return
        if not hex_str or not str(hex_str).strip():
            self._custom_colors.pop(cid, None)
            return
        clean = str(hex_str).strip()
        if self._is_valid_hex(clean):
            self._custom_colors[cid] = clean

    def set_colors(self, mapping: Mapping[int, str]) -> None:
        """Replace all custom colour overrides in-place."""
        items = list(mapping.items())
        self._custom_colors.clear()
        for k, v in items:
            try:
                cid = int(k)
            except (ValueError, TypeError):
                continue
            if cid < 0 or not v or not str(v).strip():
                continue
            clean = str(v).strip()
            if self._is_valid_hex(clean):
                self._custom_colors[cid] = clean

    def reset(self, cluster_id: Optional[int] = None) -> None:
        """Reset one cluster's custom colour or all custom colours."""
        if cluster_id is None:
            self._custom_colors.clear()
        else:
            self._custom_colors.pop(int(cluster_id), None)

    def to_dict(self) -> Dict[int, str]:
        """Export custom colour overrides as a plain dict."""
        return dict(self._custom_colors)


# -----------------------------------------------------------------------------
# 4. FIXED PLOT SIZES & CONTROL CONSTANTS (§ 4.4, § 8.1, § 9.2, § 13.5, § 13.6)
# -----------------------------------------------------------------------------
PLOT_UNIT_RESPONSE_RASTER_SIZE: Tuple[int, int] = (232, 110)
PLOT_UNIT_RESPONSE_PSTH_SIZE: Tuple[int, int] = (232, 56)
PLOT_UNIT_RESPONSE_GAP: int = 6
PLOT_UNIT_RESPONSE_LEFT_MARGIN: int = 44

PLOT_CARD_PSTH_SIZE: Tuple[int, int] = (150, 84)
PLOT_CARD_PSTH_LEFT_MARGIN: int = 34

PLOT_CLUSTER_PROB_SIZE: Tuple[int, int] = (232, 90)
PLOT_METRIC_KDE_SIZE: Tuple[int, int] = (128, 44)
PLOT_EMBEDDING_SIZE: Tuple[int, int] = (268, 268)
PLOT_PROBE_SIZE: Tuple[int, int] = (76, 268)
CLUSTER_CARD_SIZE: Tuple[int, int] = (217, 148)

PLOT_DETECTION_RASTER_SIZE: Tuple[int, int] = (520, 240)
PLOT_DETECTION_PSTH_SIZE: Tuple[int, int] = (520, 100)
PLOT_DETECTION_LEFT_MARGIN: int = 52
PLOT_WAVEFORM_SIZE: Tuple[int, int] = (240, 134)
PLOT_CRITERION_KDE_SIZE: Tuple[int, int] = (240, 24)

DETECTION_RASTER_SIZE: Tuple[int, int] = PLOT_DETECTION_RASTER_SIZE
DETECTION_PSTH_SIZE: Tuple[int, int] = PLOT_DETECTION_PSTH_SIZE
DETECTION_PLOT_LEFT_MARGIN: int = PLOT_DETECTION_LEFT_MARGIN
DETECTION_WAVEFORM_PLOT_SIZE: Tuple[int, int] = PLOT_WAVEFORM_SIZE
DETECTION_CRITERION_PLOT_SIZE: Tuple[int, int] = PLOT_CRITERION_KDE_SIZE
DETECTION_PAIRPLOT_CELL_SIZE: Tuple[int, int] = (167, 167)

PSTH_LINE_WIDTH: float = 1.5

# Plot axis label dictionaries for pyqtgraph setLabel(...)
PLOT_AXIS_LABEL_STYLE: Dict[str, str] = {
    "color": COLOR_TEXT_SECONDARY,
    "font-size": f"{FONT_SIZE_CAPTION}px",
    "font-family": FONT_NAME,
}
PLOT_AXIS_LABEL_COMPACT_STYLE: Dict[str, str] = {
    "color": COLOR_TEXT_SECONDARY,
    "font-size": f"{FONT_SIZE_CAPTION}px",
    "font-family": FONT_NAME,
}


# -----------------------------------------------------------------------------
# 5. GLOBAL STYLESHEET (§ 4.4)
# -----------------------------------------------------------------------------
def _ensure_checkmark_icon_url() -> str:
    """Create a cached anti-aliased white checkmark PNG for QCheckBox::indicator:checked."""
    icon_path = Path(tempfile.gettempdir()) / "optocurator_checkmark_white_v1.png"
    if not icon_path.exists():
        w, h = 24, 24
        # Anti-aliased distance-to-segment rasterization for two tick segments:
        # (5.5, 12.5) -> (10.0, 17.0) and (10.0, 17.0) -> (18.5, 7.5)
        segs = ((5.5, 12.5, 10.0, 17.0), (10.0, 17.0, 18.5, 7.5))
        rows = bytearray()
        for y in range(h):
            rows.append(0)
            for x in range(w):
                min_d = 1e9
                for x1, y1, x2, y2 in segs:
                    dx, dy = x2 - x1, y2 - y1
                    t = max(0.0, min(1.0, ((x + 0.5 - x1) * dx + (y + 0.5 - y1) * dy) / (dx * dx + dy * dy)))
                    px, py = x1 + t * dx, y1 + t * dy
                    d = ((x + 0.5 - px) ** 2 + (y + 0.5 - py) ** 2) ** 0.5
                    if d < min_d:
                        min_d = d
                # Stroke half-width ~ 1.4 px with 1 px AA feather
                alpha = max(0.0, min(1.0, 1.9 - min_d))
                a_byte = int(round(alpha * 255))
                rows.extend((255, 255, 255, a_byte))

        def _chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

        ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
        png = (
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(bytes(rows)))
            + _chunk(b"IEND", b"")
        )
        icon_path.write_bytes(png)
    return icon_path.as_posix()


_CHECKMARK_ICON_URL: str = _ensure_checkmark_icon_url()


MAIN_STYLE: str = f"""
/* --- Global Reset & Typography --- */
* {{
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_BODY}px;
    color: {COLOR_TEXT};
    outline: none;
}}

QMainWindow, QWidget#central_workspace {{
    background-color: {COLOR_CANVAS};
}}

/* --- Menu Bar & Menus --- */
QMenuBar {{
    background-color: {COLOR_PANEL};
    border-bottom: 1px solid {COLOR_DIVIDER};
    padding: 2px 8px;
    font-weight: 500;
    color: {COLOR_TEXT};
}}

QMenuBar::item {{
    padding: 4px 10px;
    border-radius: 4px;
    background: transparent;
}}

QMenuBar::item:selected {{
    background-color: {COLOR_CANVAS};
    color: {COLOR_ACCENT};
}}

QMenu {{
    background-color: {COLOR_PANEL};
    border: 1px solid {COLOR_DIVIDER};
    border-radius: 6px;
    padding: 4px;
}}

QMenu::item {{
    padding: 6px 24px 6px 12px;
    border-radius: 4px;
    color: {COLOR_TEXT};
}}

QMenu::item:selected {{
    background-color: {COLOR_ACCENT_BG};
    color: {COLOR_ACCENT};
    font-weight: 500;
}}

QMenu::item:disabled {{
    color: {COLOR_TEXT_MUTED};
    font-weight: 600;
}}

QMenu::separator {{
    height: 1px;
    background: {COLOR_DIVIDER};
    margin: 4px 8px;
}}

/* --- Dialogs & Message Boxes --- */
QDialog, QMessageBox, QInputDialog, QProgressDialog {{
    background-color: {COLOR_PANEL};
    color: {COLOR_TEXT};
}}

QMessageBox QLabel, QInputDialog QLabel, QDialog QLabel {{
    color: {COLOR_TEXT};
    font-size: {FONT_SIZE_BODY}px;
}}

/* --- Cards & Panels --- */
QFrame#panel_card {{
    background-color: {COLOR_PANEL};
    border: 1px solid {COLOR_DIVIDER};
    border-radius: 6px;
}}

QFrame#sidebar_card {{
    background-color: {COLOR_PANEL};
    border: 1px solid {COLOR_DIVIDER};
    border-radius: 6px;
}}

QFrame#toolbar_strip {{
    background-color: {COLOR_PANEL};
    border-bottom: 1px solid {COLOR_DIVIDER};
}}

QFrame#footer_bar {{
    background-color: {COLOR_CANVAS};
    border-top: 1px solid {COLOR_DIVIDER};
}}

/* --- Launcher Mode Cards --- */
QFrame#mode_card {{
    background-color: {COLOR_PANEL};
    border: 1px solid {COLOR_DIVIDER};
    border-radius: 8px;
}}

QFrame#mode_card:hover {{
    border: 1px solid {COLOR_ACCENT};
    background-color: {COLOR_PANEL};
}}

/* --- Section Headers --- */
QLabel#section_header {{
    font-size: {FONT_SIZE_PANEL_TITLE}px;
    font-weight: 600;
    color: {COLOR_TEXT};
    padding: 2px 0px;
}}

QLabel#metric_badge {{
    background-color: {COLOR_CANVAS};
    border: 1px solid {COLOR_DIVIDER};
    border-radius: 4px;
    padding: 3px 8px;
    font-family: {FONT_FAMILY_MONO};
    font-weight: 600;
    color: {COLOR_TEXT};
}}

/* --- Buttons (§ 4.4) --- */
QPushButton {{
    background-color: {COLOR_PANEL};
    border: 1px solid {COLOR_BORDER_STRONG};
    border-radius: 5px;
    padding: 5px 12px;
    color: {COLOR_TEXT};
    font-weight: 500;
    min-height: 20px;
}}

QPushButton:hover {{
    background-color: {COLOR_CANVAS};
    border-color: {COLOR_AXIS};
}}

QPushButton:pressed {{
    background-color: {COLOR_DIVIDER};
}}

QPushButton:disabled {{
    background-color: {COLOR_CANVAS};
    border-color: {COLOR_DIVIDER};
    color: {COLOR_TEXT_MUTED};
}}

QPushButton#primary_btn, QPushButton[role="primary"], QPushButton#primaryButton {{
    background-color: {COLOR_ACCENT};
    border: 1px solid {COLOR_ACCENT};
    color: {COLOR_PANEL};
    font-weight: 600;
}}

QPushButton#primary_btn:hover, QPushButton[role="primary"]:hover, QPushButton#primaryButton:hover {{
    background-color: {COLOR_ACCENT_HOVER};
    border-color: {COLOR_ACCENT_HOVER};
}}

QPushButton#primary_btn:pressed, QPushButton[role="primary"]:pressed, QPushButton#primaryButton:pressed {{
    background-color: {COLOR_ACCENT_PRESS};
    border-color: {COLOR_ACCENT_PRESS};
}}

QPushButton#secondaryButton, QPushButton[role="secondary"] {{
    background-color: {COLOR_PANEL};
    border: 1px solid {COLOR_BORDER_STRONG};
    color: {COLOR_TEXT};
    font-weight: 500;
}}

QPushButton#secondaryButton:hover, QPushButton[role="secondary"]:hover {{
    background-color: {COLOR_CANVAS};
    border-color: {COLOR_AXIS};
}}

QPushButton#ghostButton, QPushButton[role="ghost"] {{
    background-color: transparent;
    border: 1px solid transparent;
    color: {COLOR_TEXT_SECONDARY};
    font-weight: 500;
}}

QPushButton#ghostButton:hover, QPushButton[role="ghost"]:hover {{
    background-color: {COLOR_CANVAS};
    color: {COLOR_TEXT};
}}

QPushButton#danger_btn, QPushButton#dangerButton, QPushButton[role="danger"] {{
    background-color: {COLOR_PANEL};
    border: 1px solid {COLOR_DANGER};
    color: {COLOR_DANGER};
    font-weight: 600;
}}

QPushButton#danger_btn:hover, QPushButton#dangerButton:hover, QPushButton[role="danger"]:hover {{
    background-color: {COLOR_DANGER_BG};
    border-color: {COLOR_DANGER};
}}

QPushButton#success_btn {{
    background-color: {COLOR_SUCCESS_BG};
    border: 1px solid {COLOR_SUCCESS};
    color: {COLOR_SUCCESS};
    font-weight: 600;
}}

QPushButton#success_btn:hover {{
    background-color: {COLOR_SUCCESS};
    color: {COLOR_PANEL};
}}

/* --- Form Inputs --- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {COLOR_PANEL};
    border: 1px solid {COLOR_BORDER_STRONG};
    border-radius: 5px;
    padding: 4px 8px;
    color: {COLOR_TEXT};
    selection-background-color: {COLOR_ACCENT};
    selection-color: {COLOR_PANEL};
    min-height: 20px;
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {COLOR_ACCENT};
    background-color: {COLOR_PANEL};
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLOR_PANEL};
    border: 1px solid {COLOR_DIVIDER};
    selection-background-color: {COLOR_ACCENT_BG};
    selection-color: {COLOR_ACCENT};
    outline: none;
}}

/* --- Tables & Data Grids --- */
QTableWidget, QTableView {{
    background-color: {COLOR_PANEL};
    alternate-background-color: {COLOR_CANVAS};
    border: 1px solid {COLOR_DIVIDER};
    border-radius: 6px;
    gridline-color: {COLOR_DIVIDER};
    font-family: {FONT_FAMILY_MONO};
    font-size: {FONT_SIZE_BODY}px;
    selection-background-color: {COLOR_ACCENT_BG};
    selection-color: {COLOR_TEXT};
}}

QTableWidget::item, QTableView::item {{
    padding: 4px 8px;
    border-bottom: 1px solid {COLOR_DIVIDER};
}}

QTableWidget::item:selected, QTableView::item:selected {{
    background-color: {COLOR_ACCENT_BG};
    color: {COLOR_ACCENT};
    font-weight: 600;
}}

QHeaderView {{
    background-color: {COLOR_CANVAS};
    border: none;
}}

QHeaderView::section {{
    background-color: {COLOR_CANVAS};
    color: {COLOR_TEXT_SECONDARY};
    font-size: {FONT_SIZE_CAPTION}px;
    font-weight: 600;
    padding: 6px 8px;
    border: none;
    border-bottom: 1px solid {COLOR_DIVIDER};
    border-right: 1px solid {COLOR_DIVIDER};
}}

/* --- Splitters --- */
QSplitter::handle {{
    background-color: {COLOR_DIVIDER};
}}

QSplitter::handle:horizontal {{
    width: 1px;
    margin: 0px 2px;
}}

QSplitter::handle:vertical {{
    height: 1px;
    margin: 2px 0px;
}}

QSplitter::handle:hover {{
    background-color: {COLOR_ACCENT};
}}

/* --- Scrollbars --- */
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 2px;
}}

QScrollBar::handle:vertical {{
    background: {COLOR_BORDER_STRONG};
    min-height: 24px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background: {COLOR_AXIS};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    margin: 2px;
}}

QScrollBar::handle:horizontal {{
    background: {COLOR_BORDER_STRONG};
    min-width: 24px;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {COLOR_AXIS};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* --- Checkboxes --- */
QCheckBox {{
    spacing: 6px;
    color: {COLOR_TEXT};
    font-weight: 500;
}}

QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border-radius: 3px;
    border: 1px solid {COLOR_BORDER_STRONG};
    background-color: {COLOR_PANEL};
}}

QCheckBox::indicator:hover {{
    border-color: {COLOR_ACCENT};
}}

QCheckBox::indicator:checked {{
    background-color: {COLOR_ACCENT};
    border-color: {COLOR_ACCENT};
    image: url({_CHECKMARK_ICON_URL});
}}

QCheckBox::indicator:disabled {{
    background-color: {COLOR_SURFACE_MUTED};
    border-color: {COLOR_DIVIDER};
}}

QCheckBox::indicator:checked:disabled {{
    background-color: {COLOR_TEXT_MUTED};
    border-color: {COLOR_TEXT_MUTED};
    image: url({_CHECKMARK_ICON_URL});
}}

/* --- GroupBox --- */
QGroupBox {{
    background-color: {COLOR_PANEL};
    border: 1px solid {COLOR_DIVIDER};
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 10px;
    font-weight: 600;
    color: {COLOR_TEXT};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {COLOR_TEXT_SECONDARY};
    font-size: {FONT_SIZE_CAPTION}px;
}}

/* --- ProgressBar --- */
QProgressBar {{
    border: 1px solid {COLOR_DIVIDER};
    border-radius: 4px;
    background-color: {COLOR_CANVAS};
    text-align: center;
    color: {COLOR_TEXT};
    font-weight: 600;
}}

QProgressBar::chunk {{
    background-color: {COLOR_ACCENT};
    border-radius: 3px;
}}

/* --- Status Bar --- */
QStatusBar {{
    background-color: {COLOR_PANEL};
    border-top: 1px solid {COLOR_DIVIDER};
    color: {COLOR_TEXT_SECONDARY};
    font-size: {FONT_SIZE_CAPTION}px;
}}
"""


# -----------------------------------------------------------------------------
# 6. COMPONENT STYLESHEET CONSTANTS & BUILDERS (NO INLINE CSS OUTSIDE THEME.PY)
# -----------------------------------------------------------------------------

# --- App & Dialogs ---
STATUS_BAR_STYLE: str = (
    f"background-color: {COLOR_PANEL}; color: {COLOR_TEXT_SECONDARY}; "
    f"border-top: 1px solid {COLOR_DIVIDER}; font-size: {FONT_SIZE_CAPTION}px;"
)

PROGRESS_DIALOG_STYLE: str = f"""
    QProgressDialog {{
        background-color: {COLOR_PANEL};
        color: {COLOR_TEXT};
    }}
    QLabel {{
        color: {COLOR_TEXT};
        font-weight: 600;
        font-size: {FONT_SIZE_BODY}px;
        padding: 8px;
    }}
    QProgressBar {{
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 4px;
        background-color: {COLOR_CANVAS};
        text-align: center;
        color: {COLOR_TEXT};
        min-height: 18px;
    }}
    QProgressBar::chunk {{
        background-color: {COLOR_ACCENT};
        border-radius: 3px;
    }}
    QPushButton {{
        background-color: {COLOR_CANVAS};
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 4px;
        padding: 4px 12px;
    }}
    QPushButton:hover {{
        background-color: {COLOR_DIVIDER};
    }}
"""


_ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def _asset_icon(filename: str) -> QIcon:
    """Load an icon asset from ``ui/assets/`` if present."""
    path = _ASSETS_DIR / filename
    if path.is_file():
        return QIcon(str(path))
    return QIcon()


# --- Launcher View ---
def make_app_mark_icon(size: int = 18, y_offset: int = 0) -> QIcon:
    """Load the OptoCurator mark icon from the bundled SVG asset (with PNG fallback)."""
    icon = _asset_icon("optocurator-icon.svg")
    if icon.isNull() or icon.pixmap(size, size).isNull():
        icon = _asset_icon("optocurator-icon.png")
    if y_offset == 0 or icon.isNull():
        return icon
    pm = QPixmap(size, size + y_offset)
    pm.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pm)
    painter.drawPixmap(0, y_offset, icon.pixmap(size, size))
    painter.end()
    return QIcon(pm)


def build_mode_card_style(accent_color: str = COLOR_ACCENT) -> str:
    """Build stylesheet for ModeCardWidget on launcher screen."""
    return f"""
        QFrame#modeCard {{
            background-color: {COLOR_PANEL};
            border: 1px solid {COLOR_BORDER_STRONG};
            border-radius: 8px;
        }}
        QFrame#modeCard:hover {{
            border: 1px solid {accent_color};
            background-color: {COLOR_ACCENT_BG};
        }}
    """


LAUNCHER_TITLE_STYLE: str = (
    f"font-size: {FONT_SIZE_APP_TITLE}px; font-weight: 700; color: {COLOR_TEXT}; letter-spacing: -0.5px;"
)
LAUNCHER_SUBTITLE_STYLE: str = (
    f"font-size: {FONT_SIZE_PANEL_TITLE}px; color: {COLOR_TEXT_SECONDARY}; font-weight: 400;"
)
LAUNCHER_CARD_TITLE_STYLE: str = (
    f"font-size: {FONT_SIZE_HEADER_TITLE}px; font-weight: 600; color: {COLOR_TEXT};"
)
LAUNCHER_CARD_DESC_STYLE: str = (
    f"font-size: {FONT_SIZE_BODY}px; color: {COLOR_TEXT_SECONDARY}; line-height: 1.4;"
)


# --- Shared Dialog Styles ---
DIALOG_TITLE_STYLE: str = (
    f"font-size: {FONT_SIZE_HEADER_TITLE}px; font-weight: 700; color: {COLOR_TEXT};"
)
DIALOG_SUBTITLE_STYLE: str = (
    f"color: {COLOR_TEXT_SECONDARY}; font-size: {FONT_SIZE_BODY}px; margin-bottom: 6px;"
)
DIALOG_NOTE_STYLE: str = (
    f"color: {COLOR_TEXT_MUTED}; font-size: {FONT_SIZE_CAPTION}px; font-style: italic;"
)
DIALOG_SECTION_TITLE_STYLE: str = (
    f"font-weight: 600; font-size: {FONT_SIZE_PANEL_TITLE}px; color: {COLOR_TEXT};"
)
DIALOG_SECTION_HEADER_STYLE: str = (
    f"color: {COLOR_TEXT}; font-weight: 600; font-size: {FONT_SIZE_BODY}px; margin-top: 4px;"
)
DIALOG_DESCRIPTION_STYLE: str = (
    f"color: {COLOR_TEXT_SECONDARY}; font-size: {FONT_SIZE_CAPTION}px; margin-bottom: 8px;"
)
DIALOG_MUTED_LABEL_STYLE: str = (
    f"color: {COLOR_TEXT_SECONDARY}; font-size: {FONT_SIZE_CAPTION}px;"
)
DIALOG_BOLD_LABEL_STYLE: str = (
    f"font-weight: 600; color: {COLOR_TEXT};"
)
DIALOG_INFO_BANNER_STYLE: str = (
    f"color: {COLOR_TEXT_SECONDARY}; font-size: {FONT_SIZE_CAPTION}px; "
    f"background-color: {COLOR_CANVAS}; border: 1px solid {COLOR_DIVIDER}; "
    f"border-radius: 5px; padding: 8px;"
)
DIALOG_GROUP_BOX_STYLE: str = f"""
    QGroupBox {{
        font-weight: 600;
        font-size: {FONT_SIZE_BODY}px;
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_DIVIDER};
        border-radius: 6px;
        margin-top: 8px;
        padding-top: 14px;
        background-color: {COLOR_PANEL};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
    }}
"""
DIALOG_SECONDARY_BUTTON_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_CANVAS};
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 5px;
        padding: 6px 14px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {COLOR_DIVIDER};
    }}
"""
DIALOG_PRIMARY_BUTTON_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_ACCENT};
        color: {COLOR_PANEL};
        border: 1px solid {COLOR_ACCENT};
        border-radius: 5px;
        padding: 6px 18px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {COLOR_ACCENT_HOVER};
    }}
"""
DIALOG_DANGER_BUTTON_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_DANGER};
        color: {COLOR_PANEL};
        border: 1px solid {COLOR_DANGER};
        border-radius: 5px;
        padding: 6px 18px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {COLOR_DANGER_BG};
        color: {COLOR_DANGER};
    }}
"""
DIALOG_RADIO_SECONDARY_STYLE: str = (
    f"font-weight: 600; color: {COLOR_TEXT}; font-size: {FONT_SIZE_BODY}px; padding: 2px 0;"
)
DIALOG_RADIO_DANGER_STYLE: str = (
    f"font-weight: 600; color: {COLOR_DANGER}; font-size: {FONT_SIZE_BODY}px; padding: 2px 0;"
)
DIALOG_VALUE_BADGE_STYLE: str = (
    f"font-family: {FONT_FAMILY_MONO}; font-size: {FONT_SIZE_PANEL_TITLE}px; "
    f"font-weight: 600; color: {COLOR_TEXT};"
)
DIALOG_MODAL_STYLE: str = f"""
    QDialog {{
        background-color: {COLOR_PANEL};
    }}
    QGroupBox {{
        font-weight: 600;
        font-size: {FONT_SIZE_BODY}px;
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_DIVIDER};
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 14px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
    }}
    QLabel {{
        font-size: {FONT_SIZE_CAPTION}px;
        color: {COLOR_TEXT_SECONDARY};
    }}
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
        background-color: {COLOR_PANEL};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 5px;
        padding: 4px 8px;
        font-size: {FONT_SIZE_CAPTION}px;
        color: {COLOR_TEXT};
        min-height: 22px;
    }}
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
        border: 1px solid {COLOR_ACCENT};
    }}
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
        background-color: {COLOR_CANVAS};
        color: {COLOR_TEXT_MUTED};
        border-color: {COLOR_DIVIDER};
    }}
    QRadioButton, QCheckBox {{
        font-size: {FONT_SIZE_CAPTION}px;
        color: {COLOR_TEXT};
        spacing: 6px;
    }}
    QPushButton {{
        font-size: {FONT_SIZE_CAPTION}px;
        font-weight: 600;
        border-radius: 5px;
        padding: 6px 14px;
    }}
"""
DIALOG_LINE_EDIT_STYLE: str = f"""
    QLineEdit {{
        background-color: {COLOR_PANEL};
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 5px;
        padding: 4px 8px;
        font-size: {FONT_SIZE_BODY}px;
        font-weight: 500;
    }}
    QLineEdit:focus {{
        border-color: {COLOR_ACCENT};
        background-color: {COLOR_PANEL};
    }}
"""
DIALOG_SCROLL_AREA_STYLE: str = (
    f"QScrollArea {{ border: 1px solid {COLOR_DIVIDER}; border-radius: 6px; background-color: {COLOR_PANEL}; }}"
)
DIALOG_REORDER_BTN_STYLE: str = f"""
    QPushButton {{
        font-size: {FONT_SIZE_TICK}px;
        font-weight: 600;
        background-color: {COLOR_CANVAS};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 4px;
        color: {COLOR_TEXT};
    }}
    QPushButton:hover {{
        background-color: {COLOR_DIVIDER};
    }}
    QPushButton:disabled {{
        color: {COLOR_TEXT_MUTED};
        background-color: {COLOR_CANVAS};
        border-color: {COLOR_DIVIDER};
    }}
"""
DIALOG_HOTKEY_EDIT_STYLE: str = f"""
    QLineEdit {{
        background-color: {COLOR_PANEL};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 4px;
        padding: 4px 6px;
        font-size: {FONT_SIZE_CAPTION}px;
        font-weight: 600;
        color: {COLOR_TEXT};
    }}
    QLineEdit:focus {{
        border: 1px solid {COLOR_ACCENT};
    }}
"""
DIALOG_COUNT_BADGE_STYLE: str = (
    f"color: {COLOR_TEXT_MUTED}; font-size: {FONT_SIZE_TICK}px; font-weight: 500;"
)
DIALOG_ACCENT_CHECKBOX_STYLE: str = (
    f"color: {COLOR_ACCENT}; font-weight: 600; font-size: {FONT_SIZE_TICK}px;"
)
DIALOG_GRID_HEADER_STYLE: str = (
    f"font-weight: 700; color: {COLOR_TEXT_SECONDARY}; font-size: {FONT_SIZE_TICK}px;"
)


def build_color_swatch_button_style(color_hex: str) -> str:
    """Build stylesheet for a colour swatch button in cluster customization dialogs."""
    return f"""
        QPushButton {{
            background-color: {color_hex};
            border: 1px solid {COLOR_BORDER_STRONG};
            border-radius: 4px;
        }}
        QPushButton:hover {{
            border: 2px solid {COLOR_TEXT};
        }}
    """


# --- Phase 1 Detection View Styles ---
TOOLBAR_LABEL_STYLE: str = (
    f"color: {COLOR_TEXT_SECONDARY}; font-size: {FONT_SIZE_CAPTION}px; font-weight: 600;"
)
TOOLBAR_SPIN_STYLE: str = (
    f"QDoubleSpinBox {{ padding: 2px 4px; font-size: {FONT_SIZE_CAPTION}px; min-height: 22px; }}"
)
DETECTION_UNIT_TITLE_STYLE: str = (
    f"font-size: {FONT_SIZE_HEADER_TITLE}px; font-weight: 700; color: {COLOR_TEXT};"
)
DETECTION_WAVEFORM_LEGEND_STYLE: str = (
    f"font-size: {FONT_SIZE_CAPTION}px; color: {COLOR_TEXT_SECONDARY}; font-weight: 500;"
)


# --- Phase 2 Clusters View & Cards Styles ---
CLUSTER_CARD_SUBTITLE_STYLE: str = (
    f"color: {COLOR_TEXT_MUTED}; font-size: {FONT_SIZE_TICK}px; "
    f"background: transparent; border: none;"
)


def build_active_learning_button_style(enabled: bool) -> str:
    """Build stylesheet for Active Learning toggle button."""
    if enabled:
        return f"""
            QPushButton {{
                background-color: {COLOR_ACCENT_BG};
                color: {COLOR_ACCENT};
                border: 1px solid {COLOR_ACCENT};
                border-radius: 5px;
                padding: 4px 10px;
                font-size: {FONT_SIZE_CAPTION}px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {COLOR_ACCENT};
                color: {COLOR_PANEL};
            }}
        """
    return f"""
        QPushButton {{
            background-color: {COLOR_CANVAS};
            color: {COLOR_TEXT_SECONDARY};
            border: 1px solid {COLOR_BORDER_STRONG};
            border-radius: 5px;
            padding: 4px 10px;
            font-size: {FONT_SIZE_CAPTION}px;
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: {COLOR_DIVIDER};
            color: {COLOR_TEXT};
        }}
    """


CLUSTERS_TOOLBAR_SUBLABEL_STYLE: str = (
    f"color: {COLOR_TEXT_SECONDARY}; font-size: {FONT_SIZE_CAPTION}px; font-weight: 600;"
)
CLUSTERS_TOOLBAR_ACTION_BTN_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_PANEL};
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 5px;
        padding: 4px 10px;
        font-size: {FONT_SIZE_CAPTION}px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {COLOR_CANVAS};
        border-color: {COLOR_ACCENT};
        color: {COLOR_ACCENT};
    }}
"""
CLUSTERS_TOOLBAR_SECONDARY_BTN_STYLE: str = CLUSTERS_TOOLBAR_ACTION_BTN_STYLE
CLUSTERS_TOOLBAR_PRIMARY_BTN_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_ACCENT};
        color: {COLOR_PANEL};
        border: 1px solid {COLOR_ACCENT};
        border-radius: 5px;
        padding: 4px 12px;
        font-size: {FONT_SIZE_CAPTION}px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {COLOR_ACCENT_HOVER};
        border-color: {COLOR_ACCENT_HOVER};
    }}
"""
PRIMARY_BTN_STYLE: str = CLUSTERS_TOOLBAR_PRIMARY_BTN_STYLE
CLUSTERS_ACCEPT_BTN_STYLE: str = PRIMARY_BTN_STYLE
SECONDARY_BTN_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_PANEL};
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 5px;
        padding: 4px 10px;
        font-weight: 500;
        font-size: {FONT_SIZE_CAPTION}px;
    }}
    QPushButton:hover {{
        background-color: {COLOR_CANVAS};
        border-color: {COLOR_AXIS};
    }}
"""
CLUSTERS_TOOLBAR_DANGER_BTN_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_PANEL};
        color: {COLOR_DANGER};
        border: 1px solid {COLOR_DANGER};
        border-radius: 5px;
        padding: 4px 10px;
        font-size: {FONT_SIZE_CAPTION}px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {COLOR_DANGER_BG};
    }}
"""
CLUSTERS_TOOLBAR_RESET_BTN_STYLE: str = f"""
    QPushButton {{
        background-color: transparent;
        color: {COLOR_DANGER};
        border: none;
        border-radius: 5px;
        padding: 4px 8px;
        font-size: {FONT_SIZE_CAPTION}px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {COLOR_DANGER_BG};
    }}
"""
CLUSTERS_TOOLBAR_COMBO_STYLE: str = (
    f"QComboBox {{ padding: 2px 6px; font-size: {FONT_SIZE_CAPTION}px; min-height: 22px; }}"
)
CLUSTERS_TOOLBAR_SPIN_STYLE: str = (
    f"QDoubleSpinBox, QSpinBox {{ padding: 2px 4px; font-size: {FONT_SIZE_CAPTION}px; min-height: 22px; }}"
)
CLUSTERS_TOOLBAR_SLIDER_STYLE: str = f"""
    QSlider::groove:horizontal {{
        height: 4px;
        background: {COLOR_DIVIDER};
        border-radius: 2px;
    }}
    QSlider::sub-page:horizontal {{
        background: {COLOR_ACCENT};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {COLOR_ACCENT};
        border: 2px solid {COLOR_PANEL};
        width: 12px;
        height: 12px;
        margin: -5px 0;
        border-radius: 6px;
    }}
"""

CLUSTERS_CANDIDATE_LABEL_STYLE: str = (
    f"color: {COLOR_TEXT}; font-weight: 600; font-size: {FONT_SIZE_BODY}px;"
)


def build_comparison_pane_style(color_hex: str) -> str:
    """Build stylesheet for cluster comparison drawer pane."""
    return (
        f"QWidget#comparisonPane, QFrame#comparison_pane {{ background-color: {COLOR_PANEL}; "
        f"border: 1px solid {COLOR_DIVIDER}; border-left: 3px solid {color_hex}; "
        f"border-top: 3px solid {color_hex}; border-radius: 6px; }}"
    )


CLUSTERS_COMPARISON_PANE_DEFAULT_STYLE: str = (
    f"QWidget#comparisonPane, QFrame#comparison_pane {{ background-color: {COLOR_PANEL}; "
    f"border: 1px solid {COLOR_DIVIDER}; border-left: 3px solid {COLOR_ACCENT}; border-radius: 6px; }}"
)

CLUSTERS_COMPARISON_PANE_NEUTRAL_STYLE: str = (
    f"QWidget#comparisonPane, QFrame#comparison_pane {{ background-color: {COLOR_PANEL}; "
    f"border: 1px solid {COLOR_DIVIDER}; border-radius: 6px; }}"
)


def build_comparison_title_style(color_hex: str) -> str:
    """Build stylesheet for comparison pane title using readable cluster text colour."""
    text_col = text_color_for_cluster(color_hex)
    return f"font-weight: 700; font-size: {FONT_SIZE_BODY}px; color: {text_col};"


CLUSTERS_COMPARISON_CLOSE_BTN_STYLE: str = f"""
    QPushButton {{
        background-color: transparent;
        color: {COLOR_TEXT_SECONDARY};
        border: none;
        font-weight: 700;
        font-size: {FONT_SIZE_BODY}px;
        padding: 2px 6px;
        min-height: 18px;
    }}
    QPushButton:hover {{
        color: {COLOR_DANGER};
        background-color: {COLOR_DANGER_BG};
        border-radius: 4px;
    }}
"""

CLUSTERS_COMPARISON_EMPTY_LABEL_STYLE: str = (
    f"color: {COLOR_TEXT_MUTED}; font-style: italic; padding: 12px;"
)

CLUSTERS_SUGGESTED_TOOLBAR_BTN_PRIMARY_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_SUCCESS_BG};
        color: {COLOR_SUCCESS};
        border: 1px solid {COLOR_SUCCESS};
        border-radius: 4px;
        padding: 3px 8px;
        font-size: {FONT_SIZE_CAPTION}px;
        font-weight: 600;
        min-height: 20px;
    }}
    QPushButton:hover {{
        background-color: {COLOR_SUCCESS};
        color: {COLOR_PANEL};
    }}
"""

CLUSTERS_SUGGESTED_TOOLBAR_BTN_SECONDARY_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_PANEL};
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 4px;
        padding: 3px 8px;
        font-size: {FONT_SIZE_CAPTION}px;
        font-weight: 500;
        min-height: 20px;
    }}
    QPushButton:hover {{
        background-color: {COLOR_CANVAS};
        border-color: {COLOR_ACCENT};
        color: {COLOR_ACCENT};
    }}
"""

CLUSTERS_CURATION_BTN_SKIP_STYLE: str = SECONDARY_BTN_STYLE
CLUSTERS_CURATION_BTN_UNDO_STYLE: str = SECONDARY_BTN_STYLE

CLUSTERS_PROGRESS_BAR_CONTAINER_STYLE: str = (
    f"background-color: {COLOR_CANVAS}; border: 1px solid {COLOR_DIVIDER}; border-radius: 4px;"
)


def build_progress_segment_style(color_hex: str, is_left: bool = False, is_right: bool = False) -> str:
    """Build stylesheet for a segmented progress bar chunk."""
    radius = ""
    if is_left and is_right:
        radius = "border-radius: 3px;"
    elif is_left:
        radius = "border-top-left-radius: 3px; border-bottom-left-radius: 3px;"
    elif is_right:
        radius = "border-top-right-radius: 3px; border-bottom-right-radius: 3px;"
    return f"background-color: {color_hex}; border: none; {radius}"


CLUSTERS_PROGRESS_LEGEND_STYLE: str = (
    f"color: {COLOR_TEXT_SECONDARY}; font-size: {FONT_SIZE_TICK}px; font-weight: 500;"
)

CLUSTERS_PROBE_EXP_LABEL_STYLE: str = (
    f"color: {COLOR_TEXT}; font-size: {FONT_SIZE_CAPTION}px; font-weight: 600;"
)
CLUSTERS_PROBE_INFO_LABEL_STYLE: str = (
    f"color: {COLOR_TEXT_SECONDARY}; font-size: {FONT_SIZE_TICK}px; font-family: {FONT_FAMILY_MONO};"
)
CLUSTERS_SECTION_SUBHEADER_STYLE: str = (
    f"color: {COLOR_TEXT_SECONDARY}; font-size: {FONT_SIZE_CAPTION}px; font-weight: 600;"
)
CLUSTERS_FILTER_COMBO_MAX_VISIBLE: int = 10
CLUSTERS_FILTER_COMBO_MAX_POPUP_HEIGHT: int = 260
CLUSTERS_FILTER_COMBO_STYLE: str = f"""
QComboBox {{
    padding: 3px 8px;
    font-size: {FONT_SIZE_BODY}px;
    min-height: 24px;
    background-color: {COLOR_PANEL};
    border: 1px solid {COLOR_BORDER_STRONG};
    border-radius: 6px;
    color: {COLOR_TEXT};
}}
QComboBox QAbstractItemView {{
    background-color: {COLOR_PANEL};
    border: 1px solid {COLOR_BORDER_STRONG};
    border-radius: 6px;
    selection-background-color: {COLOR_ACCENT_BG};
    selection-color: {COLOR_TEXT};
    max-height: {CLUSTERS_FILTER_COMBO_MAX_POPUP_HEIGHT}px;
}}
"""
CLUSTERS_UNIT_COUNT_LABEL_STYLE: str = (
    f"color: {COLOR_TEXT_MUTED}; font-size: {FONT_SIZE_CAPTION}px; font-weight: 500;"
)

SALT_P_RESOLUTION_FLOOR: float = 0.01
EMBEDDING_POINT_SIZE: int = 8
EMBEDDING_POINT_EMPHASIS_SIZE: int = 9
EMBEDDING_DIM_ALPHA: int = 45
EMBEDDING_FOCUSED_RING_SIZE: int = 15
EMBEDDING_FOCUSED_RING_WIDTH: float = 2.2
PROBE_UNIT_SIZE: int = 7
PROBE_UNIT_EMPHASIS_SIZE: int = 9
PROBE_UNIT_FOCUSED_SIZE: int = 11
PROBE_DIM_ALPHA: int = 45


def build_marginal_plot_title_html(label: str) -> str:
    """Build HTML title for MarginalDistributionsWidget subplots."""
    return (
        f"<span style='font-size:{FONT_SIZE_TICK}px; color:{COLOR_TEXT_SECONDARY}; font-weight:600;'>{label}</span>"
    )


def build_classification_progress_summary_html(
    classified: int,
    total: int,
    pct: int,
    curated_count: int,
    auto_str: str,
    seeded_count: int,
    remaining_unassigned: int,
) -> str:
    """Build rich HTML summary string for ClassificationProgressBar."""
    return (
        f"Classified: {classified} / {total} ({pct}%) • "
        f"<span style='color: {COLOR_SUCCESS}; font-weight:700;'>{curated_count} Curated</span> | "
        f"<span style='color: {COLOR_ACCENT}; font-weight:700;'>{auto_str}</span> | "
        f"<span style='color: {COLOR_SEEDED}; font-weight:700;'>{seeded_count} Seeded</span> | "
        f"<span style='color: {COLOR_TEXT_SECONDARY};'>{remaining_unassigned} Remaining</span>"
    )


def build_cluster_card_header_style(color_hex: str) -> str:
    """Build stylesheet for ClusterCardWidget title label using readable cluster text colour."""
    text_col = text_color_for_cluster(color_hex)
    return f"color: {text_col}; font-weight: 700; font-size: {FONT_SIZE_CAPTION}px;"


def build_cluster_card_hint_style(color_hex: str, active: bool = False) -> str:
    """Build stylesheet for ClusterCardWidget footer hint label."""
    if active:
        text_col = text_color_for_cluster(color_hex)
        return f"color: {text_col}; font-size: {FONT_SIZE_TICK}px; font-weight: 700;"
    return f"color: {COLOR_TEXT_MUTED}; font-size: {FONT_SIZE_TICK}px; font-weight: 500;"


def build_cluster_card_widget_style(
    color_hex: str,
    is_comparing: bool = False,
    is_top: bool = False,
) -> str:
    """Build stylesheet for ClusterCardWidget frame (§ 9.2)."""
    if is_comparing:
        return f"""
            ClusterCardWidget {{
                background-color: {COLOR_PANEL};
                border: 2px solid {color_hex};
                border-radius: 8px;
            }}
            ClusterCardWidget:hover {{
                border: 2px solid {color_hex};
                background-color: {COLOR_SURFACE_SUBTLE};
            }}
        """
    if is_top:
        return f"""
            ClusterCardWidget {{
                background-color: {COLOR_PANEL};
                border: 1.5px solid {color_hex};
                border-radius: 8px;
            }}
            ClusterCardWidget:hover {{
                border: 1.5px solid {color_hex};
                background-color: {COLOR_SURFACE_SUBTLE};
            }}
        """
    return f"""
        ClusterCardWidget {{
            background-color: {COLOR_PANEL};
            border: 1px solid {COLOR_DIVIDER};
            border-radius: 8px;
        }}
        ClusterCardWidget:hover {{
            border: 1.5px solid {color_hex};
            background-color: {COLOR_SURFACE_SUBTLE};
        }}
    """


CLUSTER_CARD_SUGGESTED_BADGE_STYLE: str = f"""
    QLabel {{
        font-size: {FONT_SIZE_TICK}px;
        font-weight: 700;
        color: {COLOR_WARNING};
        background-color: {COLOR_WARNING_BG};
        border: 1px solid {COLOR_WARNING};
        border-radius: 9px;
        padding: 1px 6px;
    }}
"""

CLUSTER_CARD_PROB_BADGE_DEFAULT_STYLE: str = f"""
    QLabel {{
        font-size: {FONT_SIZE_CAPTION}px;
        font-weight: 700;
        color: {COLOR_TEXT};
        background-color: {COLOR_CANVAS};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 4px;
        padding: 1px 5px;
    }}
"""


def build_cluster_card_prob_badge_style(color_hex: str, is_top: bool = False) -> str:
    """Build stylesheet for ClusterCardWidget candidate probability badge."""
    if is_top:
        return f"""
            QLabel {{
                font-size: {FONT_SIZE_CAPTION}px;
                font-weight: 700;
                color: {COLOR_PANEL};
                background-color: {color_hex};
                border-radius: 4px;
                padding: 1px 5px;
            }}
        """
    return CLUSTER_CARD_PROB_BADGE_DEFAULT_STYLE


CLUSTER_CARD_DRAG_MERGE_STYLE: str = f"""
    ClusterCardWidget {{
        background-color: {COLOR_ACCENT_BG};
        border: 2px dashed {COLOR_ACCENT};
        border-radius: 6px;
    }}
"""

CLUSTER_CARD_DRAG_MOVE_STYLE: str = f"""
    ClusterCardWidget {{
        background-color: {COLOR_SUCCESS_BG};
        border: 2px solid {COLOR_SUCCESS};
        border-radius: 6px;
    }}
"""

CLUSTERS_POOLED_BADGE_STYLE: str = (
    f"background-color: {COLOR_SUCCESS_BG}; color: {COLOR_SUCCESS}; font-size: {FONT_SIZE_TICK}px; "
    f"font-weight: 600; border: 1px solid {COLOR_SUCCESS}; border-radius: 4px; padding: 1px 6px;"
)

CLUSTERS_INSPECTOR_SPLITTER_STYLE: str = f"""
    QSplitter::handle:horizontal {{
        background-color: {COLOR_DIVIDER};
        width: 5px;
        margin: 4px 1px;
        border-radius: 2px;
    }}
    QSplitter::handle:horizontal:hover {{
        background-color: {COLOR_AXIS};
    }}
"""

CLUSTERS_HELP_BTN_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_CANVAS};
        color: {COLOR_TEXT_SECONDARY};
        font-weight: 700;
        font-size: {FONT_SIZE_PANEL_TITLE}px;
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 13px;
    }}
    QPushButton:hover {{
        background-color: {COLOR_DIVIDER};
        color: {COLOR_TEXT};
        border-color: {COLOR_AXIS};
    }}
"""

CLUSTERS_COMPARISON_SCROLL_STYLE: str = f"""
    QScrollArea {{
        background-color: transparent;
        border: 1px solid {COLOR_DIVIDER};
        border-radius: 4px;
    }}
"""

CLUSTERS_TRANSPARENT_BG_STYLE: str = "background-color: transparent;"

CLUSTERS_MEMBERS_SECTION_STYLE: str = (
    f"font-size: {FONT_SIZE_TICK}px; font-weight: 700; color: {COLOR_TEXT_SECONDARY}; letter-spacing: 0.5px;"
)

CLUSTERS_SHORTCUTS_HELP_TEXT_STYLE: str = (
    f"font-size: {FONT_SIZE_CAPTION}px; color: {COLOR_TEXT}; line-height: 1.5;"
)
CLUSTERS_WORKFLOW_HELP_TEXT_STYLE: str = (
    f"font-size: {FONT_SIZE_CAPTION}px; color: {COLOR_TEXT_SECONDARY}; line-height: 1.4;"
)

UNIT_RESPONSE_MEMBER_CARD_STYLE: str = f"""
    QFrame#unitResponseMemberCard {{
        background-color: {COLOR_PANEL};
        border: 1px solid {COLOR_DIVIDER};
        border-radius: 6px;
    }}
"""

UNIT_RESPONSE_MEMBER_CARD_ACTIVE_STYLE: str = f"""
    QFrame#unitResponseMemberCard {{
        background-color: {COLOR_ACCENT_BG};
        border: 1px solid {COLOR_ACCENT};
        border-radius: 6px;
    }}
"""

INSPECTOR_HEADER_HEIGHT: int = 46
INSPECTOR_SWATCH_SIZE: int = 10
INSPECTOR_HEADER_NAME_MAX_WIDTH: int = 170
INSPECTOR_CLOSE_BTN_SIZE: int = 22

INSPECTOR_HEADER_NAME_STYLE: str = (
    f"font-size: {FONT_SIZE_BODY}px; font-weight: 700; color: {COLOR_TEXT}; background-color: transparent;"
)

INSPECTOR_HEADER_CID_STYLE: str = (
    f"font-size: {FONT_SIZE_TICK}px; font-weight: 600; color: {COLOR_TEXT_SECONDARY}; "
    f"background-color: {COLOR_CANVAS}; border: 1px solid {COLOR_DIVIDER}; "
    f"border-radius: 4px; padding: 1px 5px;"
)

INSPECTOR_HEADER_SUMMARY_STYLE: str = (
    f"font-size: {FONT_SIZE_CAPTION}px; color: {COLOR_TEXT_SECONDARY}; background-color: transparent;"
)


def build_inspector_swatch_style(color_hex: str) -> str:
    """Build stylesheet for the 10x10 rounded cluster colour swatch in the inspector header."""
    return f"background-color: {color_hex}; border-radius: 2px; border: none;"

UNIT_RESPONSE_MEMBER_HEADER_STYLE: str = (
    f"font-size: {FONT_SIZE_CAPTION}px; font-weight: 600; color: {COLOR_TEXT};"
)

UNIT_RESPONSE_MEMBER_META_STYLE: str = (
    f"font-size: {FONT_SIZE_TICK}px; color: {COLOR_TEXT_MUTED};"
)


# -----------------------------------------------------------------------------
# 9. WP3 MAIN LAYOUT, HEADER, SPLITTERS & STATUS BAR (§ 5, § 6, § 11.2, § 13.1)
# -----------------------------------------------------------------------------
HEADER_HEIGHT: int = 44
STATUS_BAR_HEIGHT: int = 24
UNIT_LIST_DEFAULT_WIDTH: int = 380
UNIT_LIST_MIN_WIDTH: int = 300
UNIT_LIST_MAX_WIDTH: int = 600
FOCUSED_UNIT_WIDTH: int = 300
INSPECTOR_SLOT_WIDTH: int = 300
CENTRE_PANEL_WIDTH: int = FOCUSED_UNIT_WIDTH + INSPECTOR_SLOT_WIDTH
RIGHT_PANEL_MIN_WIDTH: int = 400
CLUSTER_GRID_MIN_HEIGHT: int = 164
LIST_PANEL_MIN_HEIGHT: int = 120
SPLITTER_HANDLE_WIDTH: int = 6
HEADER_PROGRESS_BAR_WIDTH: int = 200
CLUSTER_GRID_SPACING: int = 12

HEADER_BAR_STYLE: str = f"""
    QFrame#appHeaderBar {{
        background-color: {COLOR_PANEL};
        border-bottom: 1px solid {COLOR_DIVIDER};
    }}
"""

HEADER_APP_BUTTON_STYLE: str = f"""
    QPushButton {{
        background-color: transparent;
        border: none;
        border-radius: 6px;
        color: {COLOR_TEXT};
        font-size: {FONT_SIZE_HEADER_TITLE}px;
        font-weight: 700;
        padding: 4px 8px;
    }}
    QPushButton:hover {{
        background-color: {COLOR_SURFACE_MUTED};
    }}
"""

HEADER_MODE_SWITCH_CONTAINER_STYLE: str = f"""
    QFrame#modeSwitchContainer {{
        background-color: {COLOR_SURFACE_MUTED};
        border: 1px solid {COLOR_DIVIDER};
        border-radius: 6px;
    }}
"""

HEADER_MODE_SEGMENT_STYLE: str = f"""
    QPushButton {{
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 4px;
        color: {COLOR_TEXT_SECONDARY};
        font-size: {FONT_SIZE_BODY}px;
        font-weight: 500;
        min-height: 0px;
        padding: 2px 10px;
    }}
    QPushButton:hover:!checked {{
        color: {COLOR_TEXT};
    }}
    QPushButton:checked {{
        background-color: {COLOR_PANEL};
        border: 1px solid {COLOR_DIVIDER};
        color: {COLOR_TEXT};
        font-weight: 600;
    }}
"""

HEADER_MENUBAR_STYLE: str = f"""
    QMenuBar {{
        background-color: transparent;
        border: none;
        color: {COLOR_TEXT};
        font-size: {FONT_SIZE_BODY}px;
        spacing: 2px;
    }}
    QMenuBar::item {{
        background-color: transparent;
        padding: 5px 8px;
        border-radius: 4px;
    }}
    QMenuBar::item:selected {{
        background-color: {COLOR_SURFACE_MUTED};
    }}
    QMenuBar::item:pressed {{
        background-color: {COLOR_DIVIDER};
    }}
"""

HEADER_EXPERIMENTS_BTN_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_PANEL};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 6px;
        color: {COLOR_TEXT};
        font-size: {FONT_SIZE_BODY}px;
        font-weight: 500;
        padding: 4px 10px;
    }}
    QPushButton:hover {{
        background-color: {COLOR_SURFACE_SUBTLE};
        border-color: {COLOR_AXIS};
    }}
"""

HEADER_PROGRESS_TEXT_STYLE: str = (
    f"font-size: {FONT_SIZE_BODY}px; font-weight: 600; color: {COLOR_TEXT}; background-color: transparent;"
)

STATUS_BAR_STYLE: str = f"""
    QStatusBar {{
        background-color: {COLOR_PANEL};
        border-top: 1px solid {COLOR_DIVIDER};
        color: {COLOR_TEXT_SECONDARY};
        font-size: {FONT_SIZE_CAPTION}px;
    }}
    QStatusBar::item {{
        border: none;
    }}
"""

STATUS_BAR_BTN_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_SURFACE_SUBTLE};
        border: 1px solid {COLOR_DIVIDER};
        border-radius: 4px;
        color: {COLOR_TEXT};
        font-size: {FONT_SIZE_CAPTION}px;
        font-weight: 500;
        padding: 1px 7px;
        min-height: 16px;
        max-height: 18px;
    }}
    QPushButton:hover {{
        background-color: {COLOR_SURFACE_MUTED};
        border-color: {COLOR_BORDER_STRONG};
    }}
"""

STATUS_BAR_SWITCH_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_SURFACE_MUTED};
        border: 1px solid {COLOR_DIVIDER};
        border-radius: 9px;
        color: {COLOR_TEXT_SECONDARY};
        font-size: {FONT_SIZE_CAPTION}px;
        font-weight: 500;
        padding: 1px 8px;
        min-height: 16px;
        max-height: 18px;
    }}
    QPushButton:hover {{
        border-color: {COLOR_ACCENT};
        color: {COLOR_TEXT};
    }}
    QPushButton:checked {{
        background-color: {COLOR_ACCENT_BG};
        border: 1px solid {COLOR_ACCENT};
        color: {COLOR_ACCENT};
        font-weight: 600;
    }}
"""

STATUS_BAR_GEAR_BTN_STYLE: str = f"""
    QPushButton {{
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 4px;
        color: {COLOR_TEXT_SECONDARY};
        font-size: {FONT_SIZE_BODY}px;
        padding: 0px 4px;
        min-height: 16px;
        max-height: 18px;
        min-width: 18px;
    }}
    QPushButton:hover {{
        background-color: {COLOR_SURFACE_MUTED};
        color: {COLOR_TEXT};
    }}
"""

STATUS_BAR_VERSION_STYLE: str = (
    f"color: {COLOR_TEXT_MUTED}; font-size: {FONT_SIZE_CAPTION}px; "
    f"background-color: transparent; padding: 0px 4px;"
)

STATUS_BAR_SAVED_STYLE: str = (
    f"color: {COLOR_SUCCESS}; font-size: {FONT_SIZE_CAPTION}px; font-weight: 500; "
    f"background-color: transparent; padding: 0px 6px;"
)

STATUS_BAR_UNSAVED_STYLE: str = (
    f"color: {COLOR_WARNING}; font-size: {FONT_SIZE_CAPTION}px; font-weight: 600; "
    f"background-color: {COLOR_WARNING_BG}; border-radius: 4px; padding: 1px 6px;"
)

MENU_DANGER_ACTION_STYLE: str = f"""
    QPushButton {{
        color: {COLOR_DANGER};
        text-align: left;
        padding: 5px 14px 5px 10px;
        border: none;
        border-radius: 4px;
        background: transparent;
        font-size: {FONT_SIZE_BODY}px;
    }}
    QPushButton:hover {{
        background-color: {COLOR_DANGER_BG};
    }}
"""

SPLITTER_MAIN_STYLE: str = f"""
    QSplitter {{
        background-color: {COLOR_CANVAS};
    }}
    QSplitter::handle {{
        background-color: transparent;
    }}
"""

UNIT_LIST_PANEL_STYLE: str = f"""
    QFrame#unitListPanel {{
        background-color: {COLOR_PANEL};
    }}
"""

FOCUSED_UNIT_PANEL_STYLE: str = f"""
    QFrame#focusedUnitPanel {{
        background-color: {COLOR_PANEL};
        border-right: 1px solid {COLOR_DIVIDER};
    }}
"""

INSPECTOR_SLOT_PANEL_STYLE: str = f"""
    QFrame#inspectorSlotPanel {{
        background-color: {COLOR_PANEL};
    }}
"""

INSPECTOR_EMPTY_STATE_STYLE: str = f"""
    QFrame#inspectorEmptyState {{
        background-color: {COLOR_PANEL};
        border: 1px dashed {COLOR_BORDER_STRONG};
        border-radius: 8px;
    }}
"""

INSPECTOR_EMPTY_LABEL_STYLE: str = (
    f"font-size: {FONT_SIZE_BODY}px; color: {COLOR_TEXT_MUTED}; background-color: transparent;"
)

CLUSTERS_AREA_PANEL_STYLE: str = f"""
    QFrame#clustersAreaPanel {{
        background-color: {COLOR_CANVAS};
    }}
"""

CARD_GRID_SCROLL_STYLE: str = f"""
    QScrollArea#clustersScrollArea {{
        background-color: {COLOR_CANVAS};
        border: none;
    }}
    QWidget#clustersScrollContent {{
        background-color: {COLOR_CANVAS};
    }}
"""

LIST_PANEL_STYLE: str = f"""
    QFrame#listPanel {{
        background-color: {COLOR_PANEL};
    }}
"""

LIST_PANEL_HEADER_STYLE: str = (
    f"font-size: {FONT_SIZE_PANEL_TITLE}px; font-weight: 600; color: {COLOR_TEXT};"
)

LIST_PANEL_HINT_STYLE: str = (
    f"font-size: {FONT_SIZE_CAPTION}px; color: {COLOR_TEXT_MUTED};"
)


# -----------------------------------------------------------------------------
# 10. WP5 FOCUSED UNIT COLUMN (§ 8)
# -----------------------------------------------------------------------------
FOCUSED_UNIT_CONTENT_WIDTH: int = 268
FOCUSED_UNIT_HEADER_HEIGHT: int = 64

FOCUSED_UNIT_TITLE_STYLE: str = (
    f"font-size: {FONT_SIZE_HEADER_TITLE}px; font-weight: 600; "
    f"color: {COLOR_TEXT}; background-color: transparent;"
)

FOCUSED_UNIT_SESSION_STYLE: str = (
    f"font-size: {FONT_SIZE_BODY}px; color: {COLOR_TEXT_SECONDARY}; "
    f"background-color: transparent;"
)

FOCUSED_UNIT_SUGGESTED_BTN_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_ACCENT_BG};
        color: {COLOR_ACCENT};
        border: 1px solid {COLOR_ACCENT};
        border-radius: 11px;
        padding: 2px 8px;
        font-size: {FONT_SIZE_CAPTION}px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {COLOR_ACCENT};
        color: {COLOR_PANEL};
    }}
    QPushButton:disabled {{
        background-color: {COLOR_SURFACE_MUTED};
        color: {COLOR_TEXT_MUTED};
        border-color: {COLOR_DIVIDER};
    }}
"""

FOCUSED_UNIT_LEGEND_STYLE: str = (
    f"font-size: {FONT_SIZE_CAPTION}px; color: {COLOR_TEXT_SECONDARY}; "
    f"background-color: transparent;"
)

FOCUSED_UNIT_UNASSIGN_BTN_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_PANEL};
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 5px;
        padding: 4px 10px;
        font-weight: 500;
        font-size: {FONT_SIZE_CAPTION}px;
    }}
    QPushButton:hover {{
        background-color: {COLOR_CANVAS};
    }}
    QPushButton:disabled {{
        color: {COLOR_TEXT_MUTED};
        border-color: {COLOR_DIVIDER};
    }}
"""


def build_marginal_legend_html(cluster_name: str = "", color_hex: str = COLOR_ACCENT) -> str:
    """Build compact legend HTML for the Response metrics header (§ 8)."""
    all_swatch = f"<span style='color: {COLOR_POPULATION_CURVE};'>&#9632;</span> All"
    unit_marker = f"<span style='color: {COLOR_TEXT}; font-weight: 600;'>|</span> Unit"
    if cluster_name:
        cluster_swatch = f"<span style='color: {color_hex};'>&#9632;</span> {cluster_name}"
        return f"{all_swatch} &middot; {cluster_swatch} &middot; {unit_marker}"
    return f"{all_swatch} &middot; {unit_marker}"


# -----------------------------------------------------------------------------
# 11. WP6 RESPONSE CLUSTERS & LIST PANEL (§ 9)
# -----------------------------------------------------------------------------
CLUSTERS_COUNT_CHIP_STYLE: str = (
    f"background-color: {COLOR_SURFACE_MUTED}; color: {COLOR_TEXT_SECONDARY}; "
    f"font-size: {FONT_SIZE_TICK}px; font-weight: 600; "
    f"border: 1px solid {COLOR_DIVIDER}; border-radius: 9px; padding: 1px 7px;"
)

CLUSTERS_TOOLBAR_ICON_BTN_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_PANEL};
        color: {COLOR_TEXT_SECONDARY};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 5px;
        padding: 3px 8px;
        font-size: {FONT_SIZE_BODY}px;
        font-weight: 600;
        min-width: 26px;
        min-height: 22px;
    }}
    QPushButton:hover {{
        background-color: {COLOR_CANVAS};
        border-color: {COLOR_ACCENT};
        color: {COLOR_ACCENT};
    }}
    QPushButton::menu-indicator {{
        image: none;
        width: 0px;
    }}
"""

CLUSTERS_GHOST_ICON_BTN_STYLE: str = f"""
    QPushButton {{
        background-color: transparent;
        color: {COLOR_TEXT_SECONDARY};
        border: none;
        border-radius: 4px;
        padding: 3px 6px;
        font-size: {FONT_SIZE_BODY}px;
        font-weight: 600;
        min-width: 24px;
        min-height: 22px;
    }}
    QPushButton:hover {{
        background-color: {COLOR_SURFACE_MUTED};
        color: {COLOR_TEXT};
    }}
    QPushButton::menu-indicator {{
        image: none;
        width: 0px;
    }}
"""

CLUSTERS_TOOLBAR_DIVIDER_STYLE: str = (
    f"background-color: {COLOR_DIVIDER}; min-width: 1px; max-width: 1px; margin: 2px 2px;"
)
CLUSTERS_VERTICAL_DIVIDER_STYLE: str = CLUSTERS_TOOLBAR_DIVIDER_STYLE

CLUSTERS_MENU_STYLE: str = f"""
    QMenu {{
        background-color: {COLOR_PANEL};
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 6px;
        padding: 4px;
        font-size: {FONT_SIZE_BODY}px;
    }}
    QMenu::item {{
        padding: 5px 14px 5px 10px;
        border-radius: 4px;
    }}
    QMenu::item:selected {{
        background-color: {COLOR_ACCENT_BG};
        color: {COLOR_ACCENT};
    }}
    QMenu::separator {{
        height: 1px;
        background: {COLOR_DIVIDER};
        margin: 4px 6px;
    }}
"""

CLUSTER_CARD_COUNT_STYLE: str = (
    f"color: {COLOR_TEXT_MUTED}; font-size: {FONT_SIZE_CAPTION}px; font-weight: 500; "
    f"background-color: transparent;"
)

CLUSTER_CARD_KEYCAP_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_SURFACE_MUTED};
        color: {COLOR_TEXT_SECONDARY};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 4px;
        padding: 1px 5px;
        font-size: {FONT_SIZE_TICK}px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {COLOR_ACCENT};
        color: {COLOR_PANEL};
        border-color: {COLOR_ACCENT};
    }}
"""

CLUSTER_CARD_KEYCAP_HOVER_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_ACCENT};
        color: {COLOR_PANEL};
        border: 1px solid {COLOR_ACCENT};
        border-radius: 4px;
        padding: 1px 6px;
        font-size: {FONT_SIZE_TICK}px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {COLOR_ACCENT_HOVER};
        border-color: {COLOR_ACCENT_HOVER};
    }}
"""

NEW_CLUSTER_CARD_STYLE: str = f"""
    NewClusterCardWidget {{
        background-color: {COLOR_PANEL};
        border: 1.5px dashed {COLOR_BORDER_STRONG};
        border-radius: 8px;
    }}
    NewClusterCardWidget:hover {{
        background-color: {COLOR_ACCENT_BG};
        border: 1.5px dashed {COLOR_ACCENT};
    }}
"""

NEW_CLUSTER_CARD_LABEL_STYLE: str = (
    f"color: {COLOR_TEXT_SECONDARY}; font-size: {FONT_SIZE_BODY}px; font-weight: 600; "
    f"background-color: transparent;"
)

NEW_CLUSTER_CARD_KEYCAP_STYLE: str = (
    f"background-color: {COLOR_SURFACE_MUTED}; color: {COLOR_TEXT_SECONDARY}; "
    f"border: 1px solid {COLOR_BORDER_STRONG}; border-radius: 4px; "
    f"padding: 1px 6px; font-size: {FONT_SIZE_TICK}px; font-weight: 600;"
)

LIST_PANEL_COUNT_CHIP_STYLE: str = (
    f"background-color: {COLOR_SURFACE_MUTED}; color: {COLOR_TEXT_SECONDARY}; "
    f"font-size: {FONT_SIZE_TICK}px; font-weight: 600; "
    f"border: 1px solid {COLOR_DIVIDER}; border-radius: 9px; padding: 1px 7px;"
)

LIST_PANEL_CHEVRON_BTN_STYLE: str = f"""
    QPushButton {{
        background-color: transparent;
        color: {COLOR_TEXT_SECONDARY};
        border: none;
        border-radius: 4px;
        padding: 2px 4px;
        font-size: {FONT_SIZE_BODY}px;
        font-weight: 700;
        min-width: 20px;
        min-height: 20px;
    }}
    QPushButton:hover {{
        background-color: {COLOR_SURFACE_MUTED};
        color: {COLOR_TEXT};
    }}
"""

LIST_PANEL_TABLE_STYLE: str = f"""
    QTableWidget {{
        background-color: {COLOR_PANEL};
        border: 1px solid {COLOR_DIVIDER};
        border-radius: 6px;
        gridline-color: {COLOR_DIVIDER};
        font-family: {FONT_FAMILY_MONO};
        font-size: {FONT_SIZE_BODY}px;
        color: {COLOR_TEXT};
        selection-background-color: {COLOR_ACCENT_BG};
    }}
    QTableWidget::item {{
        padding: 3px 6px;
    }}
    QTableWidget::item:selected {{
        background-color: {COLOR_ACCENT_BG};
    }}
    QHeaderView::section {{
        background-color: {COLOR_CANVAS};
        color: {COLOR_TEXT_SECONDARY};
        font-weight: 600;
        font-size: {FONT_SIZE_TICK}px;
        border: none;
        border-bottom: 1px solid {COLOR_DIVIDER};
        padding: 4px 6px;
    }}
"""

CLUSTERS_POOLED_TABLE_STYLE: str = f"""
    QTableWidget {{
        background-color: {COLOR_PANEL};
        border: 1px solid {COLOR_DIVIDER};
        border-radius: 6px;
        gridline-color: transparent;
        font-family: {FONT_FAMILY_MONO};
        font-size: {FONT_SIZE_BODY}px;
        color: {COLOR_TEXT};
        selection-background-color: {COLOR_ACCENT_BG};
    }}
    QTableWidget::item {{
        padding: 1px 3px;
        border-bottom: 1px solid {COLOR_DIVIDER};
    }}
    QTableWidget::item:selected {{
        background-color: {COLOR_ACCENT_BG};
    }}
    QHeaderView::section {{
        background-color: {COLOR_CANVAS};
        color: {COLOR_TEXT_SECONDARY};
        font-weight: 600;
        font-size: {FONT_SIZE_TICK}px;
        border: none;
        border-bottom: 1px solid {COLOR_DIVIDER};
        padding: 3px 3px;
    }}
"""

CLUSTER_CARD_MENU_DANGER_BTN_STYLE: str = f"""
    QPushButton {{
        color: {COLOR_DANGER};
        text-align: left;
        padding: 5px 14px 5px 10px;
        border: none;
        border-radius: 4px;
        background: transparent;
        font-size: {FONT_SIZE_BODY}px;
    }}
    QPushButton:hover {{
        background-color: {COLOR_ACCENT_BG};
    }}
"""


def build_list_panel_candidates_title_html(cluster_name: str, color_hex: str) -> str:
    """Build rich-text title for the list panel when a cluster is open."""
    safe_name = html.escape(str(cluster_name))
    return f"Candidates for <span style='color:{color_hex}; font-weight:700;'>{safe_name}</span>"


# -----------------------------------------------------------------------------
# 12. WP9 DETECTION MODE STYLES (§ 13)
# -----------------------------------------------------------------------------
DETECTION_ROW_TAGGED_BG: str = "#F3FAF5"

DETECTION_STATUS_CHIP_TAGGED_STYLE: str = (
    f"background-color: {COLOR_SUCCESS_BG}; color: {COLOR_SUCCESS}; "
    f"border: 1px solid {COLOR_SUCCESS}; border-radius: 10px; "
    f"padding: 2px 10px; font-size: {FONT_SIZE_CAPTION}px; font-weight: 600;"
)

DETECTION_STATUS_CHIP_UNTAGGED_STYLE: str = (
    f"background-color: {COLOR_SURFACE_MUTED}; color: {COLOR_TEXT_SECONDARY}; "
    f"border: 1px solid {COLOR_BORDER_STRONG}; border-radius: 10px; "
    f"padding: 2px 10px; font-size: {FONT_SIZE_CAPTION}px; font-weight: 600;"
)

DETECTION_OVERRIDE_BTN_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_PANEL};
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 5px;
        padding: 4px 10px;
        font-size: {FONT_SIZE_CAPTION}px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {COLOR_CANVAS};
        border-color: {COLOR_ACCENT};
        color: {COLOR_ACCENT};
    }}
    QPushButton:disabled {{
        color: {COLOR_TEXT_MUTED};
        border-color: {COLOR_DIVIDER};
        background-color: {COLOR_CANVAS};
    }}
"""

DETECTION_CRITERION_ROW_STYLE: str = f"""
    QWidget#detectionCriterionRow {{
        background-color: {COLOR_PANEL};
        border: 1px solid {COLOR_DIVIDER};
        border-radius: 6px;
    }}
    QWidget#detectionCriterionRow[criterionDisabled="true"] {{
        background-color: {COLOR_SURFACE_SUBTLE};
        border-color: {COLOR_DIVIDER};
    }}
    QCheckBox {{
        color: {COLOR_TEXT};
        font-size: {FONT_SIZE_BODY}px;
        font-weight: 600;
        spacing: 6px;
    }}
    QCheckBox:unchecked {{
        color: {COLOR_TEXT_MUTED};
    }}
"""

DETECTION_CRITERION_INPUT_STYLE: str = f"""
    QLineEdit {{
        background-color: {COLOR_PANEL};
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 4px;
        padding: 2px 6px;
        font-family: {FONT_FAMILY_MONO};
        font-size: {FONT_SIZE_BODY}px;
        font-weight: 600;
        min-height: 22px;
    }}
    QLineEdit:focus {{
        border: 1px solid {COLOR_ACCENT};
    }}
    QLineEdit:disabled {{
        background-color: {COLOR_CANVAS};
        color: {COLOR_TEXT_MUTED};
        border-color: {COLOR_DIVIDER};
    }}
"""

DETECTION_CRITERION_META_LABEL_STYLE: str = (
    f"color: {COLOR_TEXT_SECONDARY}; font-size: {FONT_SIZE_BODY}px; "
    f"background-color: transparent;"
)

DETECTION_CRITERION_META_DIM_STYLE: str = (
    f"color: {COLOR_TEXT_MUTED}; font-size: {FONT_SIZE_BODY}px; "
    f"background-color: transparent;"
)

DETECTION_CRITERION_VALUE_PASS_STYLE: str = (
    f"color: {COLOR_TEXT}; font-family: {FONT_FAMILY_MONO}; "
    f"font-size: {FONT_SIZE_BODY}px; font-weight: 600; background-color: transparent;"
)

DETECTION_CRITERION_VALUE_FAIL_STYLE: str = (
    f"color: {COLOR_TEXT}; font-family: {FONT_FAMILY_MONO}; "
    f"font-size: {FONT_SIZE_BODY}px; font-weight: 600; background-color: transparent;"
)

DETECTION_CRITERION_ICON_PASS_STYLE: str = (
    f"color: {COLOR_SUCCESS}; font-size: {FONT_SIZE_BODY}px; "
    f"font-weight: 700; background-color: transparent;"
)

DETECTION_CRITERION_ICON_FAIL_STYLE: str = (
    f"color: {COLOR_DANGER}; font-size: {FONT_SIZE_BODY}px; "
    f"font-weight: 700; background-color: transparent;"
)

DETECTION_CRITERION_VALUE_NA_STYLE: str = (
    f"color: {COLOR_TEXT_MUTED}; font-family: {FONT_FAMILY_MONO}; "
    f"font-size: {FONT_SIZE_BODY}px; font-weight: 500; background-color: transparent;"
)

DETECTION_CRITERION_PASS_COUNT_STYLE: str = (
    f"color: {COLOR_TEXT_SECONDARY}; font-size: {FONT_SIZE_CAPTION}px; "
    f"background-color: transparent;"
)

DETECTION_CRITERION_PASS_COUNT_DIM_STYLE: str = (
    f"color: {COLOR_TEXT_MUTED}; font-size: {FONT_SIZE_CAPTION}px; "
    f"background-color: transparent;"
)

DETECTION_SUMMARY_BOX_STYLE: str = f"""
    QFrame#detectionSummaryBox {{
        background-color: {COLOR_SURFACE_SUBTLE};
        border: 1px solid {COLOR_DIVIDER};
        border-radius: 6px;
        padding: 8px 10px;
    }}
    QLabel {{
        color: {COLOR_TEXT_SECONDARY};
        font-size: {FONT_SIZE_CAPTION}px;
        background-color: transparent;
    }}
"""

DETECTION_SUMMARY_LABEL_STYLE: str = (
    f"color: {COLOR_TEXT_SECONDARY}; font-size: {FONT_SIZE_CAPTION}px; "
    f"font-weight: 500; background-color: transparent;"
)

DETECTION_SUMMARY_VALUE_STYLE: str = (
    f"color: {COLOR_TEXT}; font-family: {FONT_FAMILY_MONO}; "
    f"font-size: {FONT_SIZE_CAPTION}px; font-weight: 600; background-color: transparent;"
)

DETECTION_NEXT_LIKELY_BTN_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_ACCENT_BG};
        color: {COLOR_ACCENT};
        border: 1px solid {COLOR_ACCENT};
        border-radius: 5px;
        padding: 4px 10px;
        font-size: {FONT_SIZE_CAPTION}px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {COLOR_ACCENT};
        color: {COLOR_PANEL};
    }}
    QPushButton:disabled {{
        background-color: {COLOR_SURFACE_MUTED};
        color: {COLOR_TEXT_MUTED};
        border-color: {COLOR_DIVIDER};
    }}
"""

DETECTION_PROBE_LEGEND_STYLE: str = (
    f"color: {COLOR_TEXT_SECONDARY}; font-size: {FONT_SIZE_TICK}px; "
    f"background-color: transparent;"
)

DETECTION_SUBPANEL_TITLE_STYLE: str = (
    f"color: {COLOR_TEXT}; font-size: {FONT_SIZE_PANEL_TITLE}px; "
    f"font-weight: 600; background-color: transparent;"
)

DETECTION_SUBPANEL_META_STYLE: str = (
    f"color: {COLOR_TEXT_SECONDARY}; font-size: {FONT_SIZE_CAPTION}px; "
    f"background-color: transparent;"
)

DETECTION_WINDOW_INPUT_STYLE: str = f"""
    QLineEdit {{
        background-color: {COLOR_PANEL};
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 4px;
        padding: 2px 4px;
        font-family: {FONT_FAMILY_MONO};
        font-size: {FONT_SIZE_CAPTION}px;
        min-height: 20px;
    }}
    QLineEdit:focus {{
        border: 1px solid {COLOR_ACCENT};
    }}
"""

DETECTION_WAVEFORM_LEGEND_STYLE: str = (
    f"color: {COLOR_TEXT}; font-size: {FONT_SIZE_CAPTION}px; "
    f"font-weight: 500; background-color: transparent;"
)

DETECTION_WAVEFORM_NOTE_STYLE: str = (
    f"color: {COLOR_TEXT_MUTED}; font-size: {FONT_SIZE_TICK}px; "
    f"background-color: transparent;"
)

DETECTION_WAVEFORM_R_STYLE: str = (
    f"color: {COLOR_TEXT}; font-family: {FONT_FAMILY_MONO}; "
    f"font-size: {FONT_SIZE_BODY}px; font-weight: 600; background-color: transparent;"
)

DETECTION_PAIRPLOT_LEGEND_STYLE: str = (
    f"color: {COLOR_TEXT_SECONDARY}; font-size: {FONT_SIZE_TICK}px; "
    f"background-color: transparent;"
)

DETECTION_PAIRPLOT_AXIS_LABEL_STYLE: str = (
    f"color: {COLOR_TEXT_SECONDARY}; font-size: {FONT_SIZE_TICK}px; "
    f"font-weight: 500; background-color: transparent;"
)

HEADER_PROTOCOL_COMBO_STYLE: str = f"""
    QComboBox {{
        background-color: {COLOR_PANEL};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 6px;
        color: {COLOR_TEXT};
        font-size: {FONT_SIZE_BODY}px;
        font-weight: 500;
        padding: 4px 10px;
        min-height: 20px;
    }}
    QComboBox:hover {{
        background-color: {COLOR_SURFACE_SUBTLE};
        border-color: {COLOR_AXIS};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 18px;
    }}
"""

DETECTION_COLUMN_PANEL_STYLE: str = (
    f"QWidget#detectionColumnPanel {{ background-color: {COLOR_PANEL}; }}"
)

DETECTION_UNIT_META_STYLE: str = (
    f"color: {COLOR_TEXT_SECONDARY}; font-size: {FONT_SIZE_CAPTION}px; "
    f"background-color: transparent;"
)

DETECTION_CRITERIA_HELP_STYLE: str = (
    f"color: {COLOR_TEXT_SECONDARY}; font-size: {FONT_SIZE_CAPTION}px; "
    f"background-color: transparent;"
)

DETECTION_SAVE_BTN_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_ACCENT};
        color: {COLOR_PANEL};
        border: 1px solid {COLOR_ACCENT};
        border-radius: 6px;
        padding: 6px 16px;
        font-size: {FONT_SIZE_BODY}px;
        font-weight: 600;
        min-height: 28px;
    }}
    QPushButton:hover {{
        background-color: {COLOR_ACCENT_HOVER};
        border-color: {COLOR_ACCENT_HOVER};
    }}
    QPushButton:pressed {{
        background-color: {COLOR_ACCENT_PRESS};
        border-color: {COLOR_ACCENT_PRESS};
    }}
    QPushButton:disabled {{
        background-color: {COLOR_SURFACE_MUTED};
        color: {COLOR_TEXT_MUTED};
        border-color: {COLOR_DIVIDER};
    }}
"""

DETECTION_SUBTLE_BTN_STYLE: str = f"""
    QPushButton {{
        background-color: {COLOR_PANEL};
        color: {COLOR_TEXT_SECONDARY};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 5px;
        padding: 3px 10px;
        font-size: {FONT_SIZE_CAPTION}px;
        font-weight: 500;
        min-height: 22px;
    }}
    QPushButton:hover {{
        background-color: {COLOR_SURFACE_SUBTLE};
        color: {COLOR_TEXT};
        border-color: {COLOR_AXIS};
    }}
    QPushButton:disabled {{
        background-color: {COLOR_CANVAS};
        color: {COLOR_TEXT_MUTED};
        border-color: {COLOR_DIVIDER};
    }}
"""

DETECTION_PROBE_LEGEND_HTML: str = (
    f"<span style='color:{COLOR_SUCCESS};'>&#9679;</span> Tagged<br>"
    f"<span style='color:{COLOR_UNASSIGNED};'>&#9679;</span> Untagged<br>"
    f"<span style='color:{COLOR_ACCENT}; font-weight:700;'>&#9675;</span> Focused unit"
)

DETECTION_WAVEFORM_SPONT_LEGEND_HTML: str = (
    f"<span style='color:{COLOR_WAVEFORM_SPONT}; font-weight:700;'>&#9473;</span> Spontaneous"
)

DETECTION_WAVEFORM_EVOKED_LEGEND_HTML: str = (
    f"<span style='color:{COLOR_WAVEFORM_EVOKED}; font-weight:700;'>&#9473;</span> Light-evoked"
)

DETECTION_PAIRPLOT_LEGEND_HTML: str = (
    f"<span style='color:{COLOR_SUCCESS};'>&#9679;</span> Tagged<br>"
    f"<span style='color:{COLOR_TEXT_MUTED};'>&#9679;</span> Untagged<br>"
    f"<span style='color:{COLOR_ACCENT}; font-weight:700;'>&#9675;</span> Focused unit"
)

DETECTION_TABLE_STYLE: str = f"""
    QTableView, QTableWidget {{
        background-color: {COLOR_PANEL};
        border: 1px solid {COLOR_DIVIDER};
        border-radius: 6px;
        gridline-color: transparent;
        font-family: {FONT_FAMILY_MONO};
        font-size: {FONT_SIZE_BODY}px;
        color: {COLOR_TEXT};
        selection-background-color: {COLOR_ACCENT_BG};
        selection-color: {COLOR_TEXT};
    }}
    QTableView::item, QTableWidget::item {{
        border: none;
        border-bottom: 1px solid {COLOR_DIVIDER};
        padding: 0px 3px;
    }}
    QTableView::item:selected, QTableWidget::item:selected {{
        background-color: {COLOR_ACCENT_BG};
        color: {COLOR_TEXT};
    }}
    QHeaderView::section {{
        background-color: {COLOR_PANEL};
        color: {COLOR_TEXT_SECONDARY};
        font-family: {FONT_FAMILY};
        font-weight: 600;
        font-size: {FONT_SIZE_TICK}px;
        border: none;
        border-bottom: 1px solid {COLOR_DIVIDER};
        padding: 3px 3px;
    }}
"""

DETECTION_COMBO_STYLE: str = f"""
    QComboBox {{
        background-color: {COLOR_PANEL};
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_BORDER_STRONG};
        border-radius: 5px;
        padding: 3px 22px 3px 8px;
        font-size: {FONT_SIZE_CAPTION}px;
        font-weight: 500;
        min-height: 22px;
    }}
    QComboBox:hover {{
        background-color: {COLOR_SURFACE_SUBTLE};
        border-color: {COLOR_AXIS};
    }}
    QComboBox:focus {{
        border-color: {COLOR_ACCENT};
    }}
    QComboBox:disabled {{
        background-color: {COLOR_CANVAS};
        color: {COLOR_TEXT_MUTED};
        border-color: {COLOR_DIVIDER};
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border: none;
        background: transparent;
    }}
    QComboBox::down-arrow {{
        width: 0px;
        height: 0px;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {COLOR_TEXT_SECONDARY};
        margin-right: 4px;
    }}
    QComboBox::down-arrow:disabled {{
        border-top-color: {COLOR_TEXT_MUTED};
    }}
    QComboBox QAbstractItemView {{
        background-color: {COLOR_PANEL};
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_BORDER_STRONG};
        selection-background-color: {COLOR_ACCENT_BG};
        selection-color: {COLOR_TEXT};
        outline: none;
    }}
"""


