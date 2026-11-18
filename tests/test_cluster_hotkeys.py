"""Tests for cluster hotkeys customization and PSTH panel hotkey indicators."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QFocusEvent, QKeyEvent
from optocurator.ui.dialog_clusters_customize import (
    CustomizeClustersDialog,
    HotkeyLineEdit,
    normalize_hotkey,
)
from optocurator.ui.theme import CLUSTER_CARD_KEYCAP_STYLE
from optocurator.ui.view_clusters import ClusterCardWidget, ClusteringView


def test_normalize_hotkey_utility():
    """Verify normalize_hotkey normalizes key combinations (hyphens, case, single keys)."""
    assert normalize_hotkey("shift-A") == "Shift+A"
    assert normalize_hotkey("Shift-a") == "Shift+A"
    assert normalize_hotkey("shift+a") == "Shift+A"
    assert normalize_hotkey("ctrl-1") == "Ctrl+1"
    assert normalize_hotkey("alt-x") == "Alt+X"
    assert normalize_hotkey("4") == "4"
    assert normalize_hotkey("") == ""
    assert normalize_hotkey("  ") == ""


def test_hotkey_line_edit_normalization(qapp):
    """Verify HotkeyLineEdit normalizes text on focus out."""
    edit = HotkeyLineEdit("shift-a")
    assert edit.text() == "Shift+A"

    edit.setText("ctrl-1")
    # Simulate focus out event
    edit.focusOutEvent(QFocusEvent(QEvent.Type.FocusOut))
    assert edit.text() == "Ctrl+1"


def test_cluster_hotkeys_customization_and_card_badges(qapp):
    """Verify CustomizeClustersDialog and ClusterCardWidget hotkey configuration and gray label."""
    initial_order = [0, 1]
    initial_names = {0: "Cluster 0", 1: "Cluster 1"}
    initial_colors = {0: "#2563EB", 1: "#10B981"}
    initial_counts = {0: 5, 1: 8}
    initial_hotkeys = {0: "shift-A", 1: "2"}

    dlg = CustomizeClustersDialog(
        cluster_order=initial_order,
        cluster_names=initial_names,
        cluster_colors=initial_colors,
        cluster_counts=initial_counts,
        cluster_hotkeys=initial_hotkeys,
    )

    # 1. Dialog populates normalized hotkeys
    assert dlg._rows[0].edit_hotkey.text() == "Shift+A"
    assert dlg._rows[1].edit_hotkey.text() == "2"

    # 2. Modify hotkey on row 0
    dlg.set_cluster_hotkey(0, "shift-B")
    assert dlg._rows[0].edit_hotkey.text() == "Shift+B"

    # 3. get_customizations returns new hotkeys
    _, _, _, hotkeys_out = dlg.get_customizations()
    assert hotkeys_out[0] == "Shift+B"
    assert hotkeys_out[1] == "2"

    # 4. Reset to defaults resets hotkeys to numeric string
    dlg.reset_to_defaults()
    assert dlg._rows[0].edit_hotkey.text() == "0"
    assert dlg._rows[1].edit_hotkey.text() == "1"
    dlg.close()

    # 5. Verify ClusterCardWidget keycap badge
    card = ClusterCardWidget(cluster_id=0, count=5, color_hex="#2563EB", hotkey="Shift+A")
    assert not card.btn_keycap.isHidden()
    assert card.btn_keycap.text() == "Shift+A"
    assert card.btn_keycap.styleSheet() == CLUSTER_CARD_KEYCAP_STYLE

    # Empty hotkey hides keycap
    card.set_hotkey("")
    assert card.btn_keycap.isHidden()
    assert card.btn_keycap.text() == ""

    # Setting new hotkey updates keycap
    card.set_hotkey("ctrl-3")
    assert not card.btn_keycap.isHidden()
    assert card.btn_keycap.text() == "Ctrl+3"
    card.close()


def test_cluster_custom_hotkey_assignment_via_keypress(qapp):
    """Verify triggering custom cluster hotkey (e.g. Shift+A) via keyPressEvent assigns candidate / selected cell."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    assert view.current_clustering_result is not None

    # Customize cluster hotkeys: 0 -> Shift+A, 1 -> Shift+B, 2 -> Shift+S
    view.apply_cluster_customizations(
        new_order=[0, 1, 2, 3],
        new_names={0: "Cluster 0", 1: "Cluster 1", 2: "Cluster 2", 3: "Cluster 3"},
        new_colors={0: "#2563EB", 1: "#10B981", 2: "#F59E0B", 3: "#8B5CF6"},
        new_hotkeys={0: "Shift+A", 1: "Shift+B", 2: "Shift+S", 3: "3"},
    )

    assert view._get_cluster_hotkey(0) == "Shift+A"
    assert view._cluster_cards[0].btn_keycap.text() == "Shift+A"
    assert view._get_cluster_hotkey(1) == "Shift+B"
    assert view._cluster_cards[1].btn_keycap.text() == "Shift+B"

    # 1. Select row 0 and press Shift+A -> should assign to cluster 0
    gid0 = view._get_gid_for_row(0)
    view.select_pooled_unit(gid0)
    assert view._current_selected_gid == gid0

    event_shift_a = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_A,
        Qt.KeyboardModifier.ShiftModifier,
        "A",
    )
    view.keyPressEvent(event_shift_a)
    assert view._pooled_units_map[gid0]["cluster_id"] == 0
    assert view._unit_status_map[gid0] == "Curated"

    # 2. Select row 1 and press Shift+B -> should assign to cluster 1
    gid1 = view._get_gid_for_row(1)
    view.select_pooled_unit(gid1)
    assert view._current_selected_gid == gid1

    event_shift_b = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_B,
        Qt.KeyboardModifier.ShiftModifier,
        "B",
    )
    view.keyPressEvent(event_shift_b)
    assert view._pooled_units_map[gid1]["cluster_id"] == 1
    assert view._unit_status_map[gid1] == "Curated"

    # 3. Press Shift+S on selected row 2 -> should assign to cluster 2
    gid2 = view._get_gid_for_row(2)
    view.select_pooled_unit(gid2)
    assert view._current_selected_gid == gid2

    event_shift_s = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_S,
        Qt.KeyboardModifier.ShiftModifier,
        "S",
    )
    view.keyPressEvent(event_shift_s)
    assert view._pooled_units_map[gid2]["cluster_id"] == 2
    assert view._unit_status_map[gid2] == "Curated"

    # 4. Press 'S' without modifier when no cluster has hotkey 'S' -> skips active candidate
    view.table_pooled_units.clearSelection()
    view._on_pooled_unit_selected()
    cand_gid = view.active_candidate.global_unit_id if view.active_candidate else None
    if cand_gid:
        event_s = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_S,
            Qt.KeyboardModifier.NoModifier,
            "s",
        )
        view.keyPressEvent(event_s)
        assert cand_gid in view._skipped_gids

    view.close()


def test_clustering_project_save_load_hotkeys(qapp, tmp_path):
    """Verify custom cluster hotkeys persist in clustering_project.yaml and restore properly."""
    exp_dir = tmp_path / "test_session_hk"
    exp_dir.mkdir(parents=True, exist_ok=True)
    df_curated = pd.DataFrame([
        {
            "global_unit_id": "test_session_hk_u0001",
            "session_id": "test_session_hk",
            "unit_id": 1,
            "latency_ms": 3.4,
            "jitter_ms": 0.9,
            "reliability": 0.92,
            "waveform_correlation": 0.95,
            "salt_p_value": 0.0008,
            "is_tagged": True,
            "cluster_id": 0,
            "curated_label": "Tagged",
        },
        {
            "global_unit_id": "test_session_hk_u0002",
            "session_id": "test_session_hk",
            "unit_id": 2,
            "latency_ms": 4.1,
            "jitter_ms": 1.1,
            "reliability": 0.88,
            "waveform_correlation": 0.93,
            "salt_p_value": 0.0015,
            "is_tagged": True,
            "cluster_id": 1,
            "curated_label": "Tagged",
        },
    ])
    df_curated.to_csv(exp_dir / "test_session_hk_opto_curated.csv", index=False)
    np.save(exp_dir / "opto_idxs_OPTS03.npy", np.array([10, 20]))

    view = ClusteringView(start_with_clusters=True, load_mock=False)
    view.add_experiment_folder(exp_dir)
    view.assign_candidate_to_cluster(0, unit_id="test_session_hk_u0001")
    view.assign_candidate_to_cluster(1, unit_id="test_session_hk_u0002")

    view.apply_cluster_customizations(
        new_order=[0, 1],
        new_names={0: "Burst", 1: "Tonic"},
        new_colors={0: "#2563EB", 1: "#10B981"},
        new_hotkeys={0: "Shift+A", 1: "Shift+Z"},
    )

    proj_dir = tmp_path / "saved_clustering_proj"
    saved_path = view.save_project(proj_dir)
    assert saved_path is not None
    assert saved_path.is_file()

    # Load in new view
    view2 = ClusteringView(start_with_clusters=False, load_mock=False)
    ok = view2.load_project(saved_path)
    assert ok is True

    # Hotkeys restored
    assert view2._get_cluster_hotkey(0) == "Shift+A"
    assert view2._get_cluster_hotkey(1) == "Shift+Z"

    # Card badges reflect restored hotkeys
    assert view2._cluster_cards[0].btn_keycap.text() == "Shift+A"
    assert view2._cluster_cards[1].btn_keycap.text() == "Shift+Z"

    view.close()
    view2.close()
