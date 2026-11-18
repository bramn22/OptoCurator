"""Modal configuration dialog for customizing cluster names, order, colors, and hotkeys."""

from typing import Any, Dict, List, Optional, Tuple
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QKeyEvent, QKeySequence
from PyQt6.QtWidgets import (
    QColorDialog,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from optocurator.analysis.clustering import CLUSTER_PALETTE, color_for_cluster
from optocurator.ui.theme import (
    DIALOG_COUNT_BADGE_STYLE,
    DIALOG_HOTKEY_EDIT_STYLE,
    DIALOG_LINE_EDIT_STYLE,
    DIALOG_MODAL_STYLE,
    DIALOG_MUTED_LABEL_STYLE,
    DIALOG_PRIMARY_BUTTON_STYLE,
    DIALOG_REORDER_BTN_STYLE,
    DIALOG_SECONDARY_BUTTON_STYLE,
    DIALOG_SUBTITLE_STYLE,
    build_color_swatch_button_style,
)


def normalize_hotkey(hotkey_str: str) -> str:
    """Normalize shortcut string like 'shift-A', 'ctrl-1', 'shift+a' to canonical 'Shift+A'."""
    if not hotkey_str:
        return ""
    s = str(hotkey_str).strip()
    if not s:
        return ""
    s = s.replace("-", "+")
    parts = [p.strip() for p in s.split("+") if p.strip()]
    if not parts:
        return ""
    norm_parts = []
    for p in parts[:-1]:
        low = p.lower()
        if low in ("shift", "shft"):
            norm_parts.append("Shift")
        elif low in ("ctrl", "control"):
            norm_parts.append("Ctrl")
        elif low in ("alt",):
            norm_parts.append("Alt")
        elif low in ("meta", "cmd", "command"):
            norm_parts.append("Meta")
        else:
            norm_parts.append(p.capitalize())

    last = parts[-1]
    if len(last) == 1:
        norm_parts.append(last.upper())
    else:
        norm_parts.append(last.capitalize())

    clean = "+".join(norm_parts)
    seq = QKeySequence.fromString(clean, QKeySequence.SequenceFormat.PortableText)
    res = seq.toString(QKeySequence.SequenceFormat.PortableText)
    return res if res else clean


class HotkeyLineEdit(QLineEdit):
    """Line edit specialized for capturing and normalizing hotkey shortcuts."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(normalize_hotkey(text), parent)

    def focusInEvent(self, event: Any) -> None:
        super().focusInEvent(event)
        self.selectAll()

    def focusOutEvent(self, event: Any) -> None:
        norm = normalize_hotkey(self.text())
        self.setText(norm)
        if event is not None:
            try:
                super().focusOutEvent(event)
            except TypeError:
                pass

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self.setText("")
            event.accept()
            return
        if key in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab, Qt.Key.Key_Escape, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            super().keyPressEvent(event)
            return

        modifiers = event.modifiers()
        if modifiers & (
            Qt.KeyboardModifier.ShiftModifier
            | Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.MetaModifier
        ):
            key_comb = event.keyCombination()
            seq = QKeySequence(key_comb)
            seq_str = seq.toString(QKeySequence.SequenceFormat.PortableText)
            if seq_str:
                self.setText(normalize_hotkey(seq_str))
                event.accept()
                return

        super().keyPressEvent(event)


class ClusterRowItem(QWidget):
    """Row widget representing a single cluster with reorder buttons, name input, color swatch, and hotkey."""

    def __init__(
        self,
        cluster_id: int,
        initial_name: str,
        initial_color: str,
        initial_hotkey: str = "",
        count: int = 0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.cluster_id = cluster_id
        self.color_hex = initial_color
        self.hotkey = normalize_hotkey(initial_hotkey)
        self.count = count

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Move Up and Move Down buttons
        self.btn_up = QPushButton("▲")
        self.btn_up.setToolTip("Move cluster up in order")
        self.btn_up.setFixedSize(26, 26)
        self.btn_up.setStyleSheet(DIALOG_REORDER_BTN_STYLE)

        self.btn_down = QPushButton("▼")
        self.btn_down.setToolTip("Move cluster down in order")
        self.btn_down.setFixedSize(26, 26)
        self.btn_down.setStyleSheet(DIALOG_REORDER_BTN_STYLE)

        layout.addWidget(self.btn_up)
        layout.addWidget(self.btn_down)

        # Color swatch button
        self.btn_color = QPushButton()
        self.btn_color.setToolTip("Click to choose cluster color")
        self.btn_color.setFixedSize(30, 26)
        self.btn_color.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_swatch()
        layout.addWidget(self.btn_color)

        # Cluster label indicator
        self.lbl_id = QLabel(f"ID {cluster_id}:")
        self.lbl_id.setStyleSheet(DIALOG_MUTED_LABEL_STYLE)
        layout.addWidget(self.lbl_id)

        # Name edit field
        self.edit_name = QLineEdit(initial_name)
        self.edit_name.setPlaceholderText(f"Cluster {cluster_id}")
        self.edit_name.setStyleSheet(DIALOG_LINE_EDIT_STYLE)
        layout.addWidget(self.edit_name, stretch=1)

        # Hotkey shortcut field
        self.lbl_key = QLabel("Key:")
        self.lbl_key.setStyleSheet(DIALOG_MUTED_LABEL_STYLE)
        layout.addWidget(self.lbl_key)

        self.edit_hotkey = HotkeyLineEdit(self.hotkey)
        self.edit_hotkey.setPlaceholderText("None")
        self.edit_hotkey.setFixedWidth(80)
        self.edit_hotkey.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.edit_hotkey.setToolTip("Hotkey shortcut to assign selected cells to this cluster (e.g. 0-9, Shift+A)")
        self.edit_hotkey.setStyleSheet(DIALOG_HOTKEY_EDIT_STYLE)
        layout.addWidget(self.edit_hotkey)

        # Count badge
        self.lbl_count = QLabel(f"(N={count})")
        self.lbl_count.setStyleSheet(DIALOG_COUNT_BADGE_STYLE)
        layout.addWidget(self.lbl_count)

    def _update_swatch(self) -> None:
        """Update color swatch background styling."""
        self.btn_color.setStyleSheet(build_color_swatch_button_style(self.color_hex))

    def set_color(self, hex_code: str) -> None:
        """Set new color hex and update swatch."""
        self.color_hex = hex_code
        self._update_swatch()

    def set_hotkey(self, hotkey: str) -> None:
        """Set new hotkey string and update input field."""
        norm = normalize_hotkey(hotkey)
        self.hotkey = norm
        self.edit_hotkey.setText(norm)


class CustomizeClustersDialog(QDialog):
    """Modal dialog allowing users to rename, reorder, and color-code response clusters."""

    def __init__(
        self,
        cluster_order: Optional[List[int]] = None,
        cluster_names: Optional[Dict[int, str]] = None,
        cluster_colors: Optional[Dict[int, str]] = None,
        cluster_counts: Optional[Dict[int, int]] = None,
        cluster_hotkeys: Optional[Dict[int, str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Customize Clusters")
        self.setMinimumSize(560, 360)
        self.setModal(True)
        self.setStyleSheet(DIALOG_MODAL_STYLE)

        self._order: List[int] = list(cluster_order or [])
        self._initial_order: List[int] = list(self._order)
        self._names: Dict[int, str] = dict(cluster_names or {})
        self._colors: Dict[int, str] = dict(cluster_colors or {})
        self._counts: Dict[int, int] = dict(cluster_counts or {})
        self._hotkeys: Dict[int, str] = dict(cluster_hotkeys or {})

        self._rows: Dict[int, ClusterRowItem] = {}

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(10)

        # Header description
        lbl_info = QLabel("Customize display names, grid ordering, color accents, and hotkeys for response clusters.")
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet(DIALOG_SUBTITLE_STYLE)
        main_layout.addWidget(lbl_info)

        # Scrollable rows area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.rows_container = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_container)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(6)

        # Build row items
        for cid in self._order:
            name = self._names.get(cid, f"Cluster {cid}")
            color = color_for_cluster(cid, self._colors)
            count = self._counts.get(cid, 0)
            hotkey = self._hotkeys.get(cid, str(cid) if 0 <= cid <= 9 else "")

            row_item = ClusterRowItem(
                cluster_id=cid,
                initial_name=name,
                initial_color=color,
                initial_hotkey=hotkey,
                count=count,
                parent=self.rows_container,
            )
            row_item.btn_up.clicked.connect(lambda _, c=cid: self.move_cluster_up(c))
            row_item.btn_down.clicked.connect(lambda _, c=cid: self.move_cluster_down(c))
            row_item.btn_color.clicked.connect(lambda _, c=cid: self._pick_color_for_cluster(c))

            self._rows[cid] = row_item
            self.rows_layout.addWidget(row_item)

        self.rows_layout.addStretch()
        scroll.setWidget(self.rows_container)
        main_layout.addWidget(scroll, stretch=1)

        self._refresh_reorder_buttons()

        # Dialog Action Buttons: Reset to Defaults, Cancel, OK / Apply
        btn_bar = QHBoxLayout()
        self.btn_reset = QPushButton("Reset to Defaults")
        self.btn_reset.setStyleSheet(DIALOG_SECONDARY_BUTTON_STYLE)
        self.btn_reset.setAutoDefault(False)
        self.btn_reset.clicked.connect(self.reset_to_defaults)
        btn_bar.addWidget(self.btn_reset)

        btn_bar.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(DIALOG_SECONDARY_BUTTON_STYLE)
        self.btn_cancel.clicked.connect(self.reject)
        btn_bar.addWidget(self.btn_cancel)

        self.btn_apply = QPushButton("Done")
        self.btn_apply.setStyleSheet(DIALOG_PRIMARY_BUTTON_STYLE)
        self.btn_apply.setDefault(True)
        self.btn_apply.clicked.connect(self.accept)
        btn_bar.addWidget(self.btn_apply)

        self.btn_ok = self.btn_apply  # Alias for OK / Apply

        main_layout.addLayout(btn_bar)

    def _refresh_reorder_buttons(self) -> None:
        """Update Up/Down button states and layout widget order."""
        # Re-insert widgets in order
        for card in self._rows.values():
            self.rows_layout.removeWidget(card)

        for idx, cid in enumerate(self._order):
            row = self._rows.get(cid)
            if row is not None:
                self.rows_layout.insertWidget(idx, row)
                row.btn_up.setEnabled(idx > 0)
                row.btn_down.setEnabled(idx < len(self._order) - 1)

    def move_cluster_up(self, cid: int) -> bool:
        """Move cluster up one position in order."""
        if cid not in self._order:
            return False
        idx = self._order.index(cid)
        if idx <= 0:
            return False
        self._order[idx], self._order[idx - 1] = self._order[idx - 1], self._order[idx]
        self._refresh_reorder_buttons()
        return True

    def move_cluster_down(self, cid: int) -> bool:
        """Move cluster down one position in order."""
        if cid not in self._order:
            return False
        idx = self._order.index(cid)
        if idx >= len(self._order) - 1:
            return False
        self._order[idx], self._order[idx + 1] = self._order[idx + 1], self._order[idx]
        self._refresh_reorder_buttons()
        return True

    def _pick_color_for_cluster(self, cid: int) -> None:
        """Open QColorDialog to choose a new color for the specified cluster."""
        row = self._rows.get(cid)
        if row is None:
            return
        initial = QColor(row.color_hex)
        chosen = QColorDialog.getColor(initial, self, f"Select Color for Cluster {cid}")
        if chosen.isValid():
            row.set_color(chosen.name())

    def set_cluster_color(self, cid: int, color_hex: str) -> None:
        """Programmatically set cluster color (useful for tests and automation)."""
        if cid in self._rows:
            self._rows[cid].set_color(color_hex)

    def set_cluster_name(self, cid: int, name: str) -> None:
        """Programmatically set cluster display name (useful for tests and automation)."""
        if cid in self._rows:
            self._rows[cid].edit_name.setText(name)

    def set_cluster_hotkey(self, cid: int, hotkey: str) -> None:
        """Programmatically set cluster hotkey (useful for tests and automation)."""
        if cid in self._rows:
            self._rows[cid].set_hotkey(hotkey)

    def reset_to_defaults(self) -> None:
        """Reset order, names, colors, and hotkeys to default clustering settings."""
        self._order = sorted(self._rows.keys())
        for cid in self._order:
            row = self._rows[cid]
            row.edit_name.setText(f"Cluster {cid}")
            default_color = color_for_cluster(cid)
            row.set_color(default_color)
            default_key = str(cid) if 0 <= cid <= 9 else ""
            row.set_hotkey(default_key)
        self._refresh_reorder_buttons()

    def get_customizations(self) -> Tuple[List[int], Dict[int, str], Dict[int, str], Dict[int, str]]:
        """Return the customized cluster order, names, colors, and hotkeys.

        Returns:
            Tuple of:
                - List[int]: Ordered cluster IDs
                - Dict[int, str]: Mapping from cluster_id to custom display name
                - Dict[int, str]: Mapping from cluster_id to color hex string
                - Dict[int, str]: Mapping from cluster_id to hotkey shortcut string
        """
        names: Dict[int, str] = {}
        colors: Dict[int, str] = {}
        hotkeys: Dict[int, str] = {}
        for cid in self._order:
            row = self._rows[cid]
            raw_text = row.edit_name.text().strip()
            names[cid] = raw_text if raw_text else f"Cluster {cid}"
            colors[cid] = row.color_hex
            hotkeys[cid] = normalize_hotkey(row.edit_hotkey.text().strip())
        return list(self._order), names, colors, hotkeys
