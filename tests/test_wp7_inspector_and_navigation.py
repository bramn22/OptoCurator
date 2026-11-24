"""WP7 tests for Cluster Inspector, clickable members, focus/advance rules, keyboard map, and Undo/Redo."""

from __future__ import annotations

from typing import Optional

import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QColor, QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import QColorDialog, QInputDialog, QPushButton

from optocurator.ui.theme import INSPECTOR_HEADER_HEIGHT, UNIT_RESPONSE_MEMBER_CARD_ACTIVE_STYLE
from optocurator.ui.view_clusters import ClusteringView


def _send_key(
    view: ClusteringView,
    key: Qt.Key,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
    text: str = "",
) -> QKeyEvent:
    """Dispatch a synthetic QKeyEvent directly to ClusteringView.keyPressEvent."""
    ev = QKeyEvent(QEvent.Type.KeyPress, key, modifiers, text)
    view.keyPressEvent(ev)
    return ev


def _click_member_card(viewer) -> None:
    """Simulate a left-button mouse press on an inspector member card."""
    ev = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(8.0, 8.0),
        QPointF(8.0, 8.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    viewer.mousePressEvent(ev)


def _find_cluster_with_min_members(view: ClusteringView, min_members: int = 2) -> int:
    """Return the first cluster ID in card order with at least ``min_members`` units."""
    for cid in view._cluster_card_order:
        members = [
            gid for gid, rec in view._pooled_units_map.items()
            if rec.get("cluster_id") == cid
        ]
        if len(members) >= min_members:
            return cid
    raise AssertionError(f"Expected a cluster with >= {min_members} members in mock view")


def _select_table_row_as_user(view: ClusteringView, row: int) -> Optional[str]:
    """Select a row in table_pooled_units simulating user interaction (not programmatic)."""
    view._programmatic_unit_select = False
    view.table_pooled_units.selectRow(row)
    return view._get_gid_for_row(row)


def test_wp7_inspector_header_members_alignment_and_close(qapp):
    """Verify 46px inspector header, no overflow menu, hidden suggestions, raster alignment, and close/Esc."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    try:
        view.resize(1920, 1080)
        view.show()
        qapp.processEvents()

        cid = _find_cluster_with_min_members(view, min_members=2)
        long_name = "Very Long Optogenetic Inhibitory Interneuron Cluster Name That Should Elide"
        view._cluster_names[cid] = long_name

        view.set_cluster_comparison(cid)
        qapp.processEvents()

        # 1. Exact 46 px header height & metadata widgets
        assert INSPECTOR_HEADER_HEIGHT == 46
        assert view.inspector_header_widget.height() == 46

        expected_color = view._get_cluster_color(cid)
        assert expected_color.lower() in view.lbl_inspector_swatch.styleSheet().lower()

        assert view.lbl_inspector_name.toolTip() == long_name
        assert view.lbl_inspector_name.text() != ""
        assert "…" in view.lbl_inspector_name.text() or "..." in view.lbl_inspector_name.text()

        assert view.lbl_inspector_cid.text() == f"C{cid}"

        member_gids = [
            gid for gid, rec in view._pooled_units_map.items()
            if rec.get("cluster_id") == cid
        ]
        summary_text = view.lbl_inspector_summary.text()
        assert summary_text.startswith(f"{len(member_gids)} members · mean latency ")

        # 2. Close button present, NO '⋯' overflow menu
        assert not view.btn_comp_close.isHidden()
        assert not hasattr(view, "btn_comp_menu")
        header_buttons = view.inspector_header_widget.findChildren(QPushButton)
        assert [b.text() for b in header_buttons] == ["✕"]

        # 3. First member UnitResponsePlot aligns horizontally with focused unit_response_plot
        view._align_inspector_first_member_raster()
        qapp.processEvents()
        assert view.comparison_viewers, "Expected at least one member viewer in inspector"
        first_plot = view.comparison_viewers[0].unit_response_plot
        assert first_plot is not None
        y_focused = view.unit_response_plot.mapTo(view.centre_panel, QPoint(0, 0)).y()
        y_first_member = first_plot.mapTo(view.centre_panel, QPoint(0, 0)).y()
        assert abs(y_focused - y_first_member) <= 2

        # 4. Clicking close button closes inspector and returns focus to active-learning suggestion
        expected_cand_gid = view.active_candidate.global_unit_id if view.active_candidate is not None else None
        view.btn_comp_close.click()
        qapp.processEvents()
        assert view._comparison_cluster_id is None
        assert view.comparison_pane.isHidden()
        if expected_cand_gid is not None:
            assert view._current_selected_gid == expected_cand_gid
            assert view._focus_source == "queue"

        # 5. Re-open inspector and verify Esc closes it and returns focus to suggestion
        view.set_cluster_comparison(cid)
        qapp.processEvents()
        assert view._comparison_cluster_id == cid

        ev = _send_key(view, Qt.Key.Key_Escape)
        qapp.processEvents()
        assert ev.isAccepted()
        assert view._comparison_cluster_id is None
        assert view.comparison_pane.isHidden()
        if expected_cand_gid is not None:
            assert view._current_selected_gid == expected_cand_gid
            assert view._focus_source == "queue"
    finally:
        view.close()


def test_wp7_clickable_members_highlight_and_removal_on_reassign(qapp):
    """Clicking an inspector member highlights it, sets source='inspector', and removing it updates list & summary."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    try:
        cid = _find_cluster_with_min_members(view, min_members=2)
        view.set_cluster_comparison(cid)
        qapp.processEvents()

        member_gids = list(view._comparison_member_viewers_map.keys())
        assert len(member_gids) >= 2
        target_gid = member_gids[1]
        other_gid = member_gids[0]
        initial_count = len(member_gids)

        _click_member_card(view._comparison_member_viewers_map[target_gid])
        qapp.processEvents()

        assert view._current_selected_gid == target_gid
        assert view._focus_source == "inspector"
        assert view._comparison_member_viewers_map[target_gid].styleSheet() == UNIT_RESPONSE_MEMBER_CARD_ACTIVE_STYLE
        assert view._comparison_member_viewers_map[other_gid].styleSheet() != UNIT_RESPONSE_MEMBER_CARD_ACTIVE_STYLE

        # Reassign or unassign the focused member -> immediately removed from inspector & summary updated
        other_cids = [c for c in view._cluster_card_order if c != cid]
        if other_cids:
            ok = view.assign_candidate_to_cluster(other_cids[0], unit_id=target_gid)
        else:
            ok = view.unassign_focused_unit()
        qapp.processEvents()

        assert ok is True
        assert target_gid not in view._comparison_member_viewers_map
        assert all(getattr(v, "gid", None) != target_gid for v in view.comparison_viewers)
        assert view.lbl_inspector_summary.text().startswith(f"{initial_count - 1} members · mean latency ")
    finally:
        view.close()


def test_wp7_advance_rules_for_all_sources_and_suggested_unit_return(qapp, qtbot):
    """Verify §10.2 advance rules for queue, inspector, table_cluster, list_candidates, and G/return_to_active_candidate."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    try:
        qtbot.waitUntil(
            lambda: view.active_candidate is not None
            and (view._active_learning_worker is None or not view._active_learning_worker.isRunning()),
            timeout=10000,
        )
        # (a) "queue" with active_learning_enabled=True
        view.active_learning_enabled = True
        assert view.return_to_active_candidate() is True
        assert view._focus_source == "queue"
        first_cand = view.active_candidate.global_unit_id

        view.skip_current_candidate()
        qtbot.waitUntil(
            lambda: (view._active_learning_worker is None or not view._active_learning_worker.isRunning())
            and view.active_candidate is not None
            and view.active_candidate.global_unit_id != first_cand,
            timeout=10000,
        )
        assert view._focus_source == "queue"
        assert view.active_candidate is not None
        assert view._current_selected_gid == view.active_candidate.global_unit_id
        assert view._current_selected_gid != first_cand

        target_cid = view._cluster_card_order[0]
        second_cand = view.active_candidate.global_unit_id
        assert view.assign_candidate_to_cluster(target_cid) is True
        qtbot.waitUntil(
            lambda: (view._active_learning_worker is None or not view._active_learning_worker.isRunning())
            and view.active_candidate is not None
            and view.active_candidate.global_unit_id != second_cand,
            timeout=10000,
        )
        assert view._focus_source == "queue"
        if view.active_candidate is not None:
            assert view._current_selected_gid == view.active_candidate.global_unit_id
            assert view._current_selected_gid != second_cand

        # (a2) "queue" with active_learning_enabled=False -> advances to next unassigned unit by table order
        view.active_learning_enabled = False
        # Unassign 2 units from a cluster with >=3 members so we have >=2 unassigned units without collapsing clusters
        big_cid = _find_cluster_with_min_members(view, min_members=3)
        big_members = [g for g, r in view._pooled_units_map.items() if r.get("cluster_id") == big_cid]
        view.assign_candidate_to_cluster(-1, unit_id=big_members[:2])
        unassigned_rows = [
            r for r in range(view.table_pooled_units.rowCount())
            if not view.table_pooled_units.isRowHidden(r)
            and view._pooled_units_map.get(view._get_gid_for_row(r), {}).get("cluster_id", -1) < 0
        ]
        assert len(unassigned_rows) >= 2
        gid_r0 = _select_table_row_as_user(view, unassigned_rows[0])
        expected_next_gid = view._get_gid_for_row(unassigned_rows[1])
        assert view._focus_source == "queue"
        assert view._current_selected_gid == gid_r0

        view.assign_candidate_to_cluster(target_cid)
        qapp.processEvents()
        assert view._focus_source == "queue"
        assert view._current_selected_gid == expected_next_gid

        # Restore reassigned/unassigned units to big_cid so a cluster with >=3 members exists for (b)-(d)
        view.assign_candidate_to_cluster(big_cid, unit_id=big_members[:2])

        # Re-enable active learning for remaining checks
        view.active_learning_enabled = True
        view.refresh_active_learning()
        qtbot.waitUntil(
            lambda: view.active_candidate is not None
            and (view._active_learning_worker is None or not view._active_learning_worker.isRunning()),
            timeout=10000,
        )

        # (b) "inspector" -> advances to next member in open cluster
        insp_cid = _find_cluster_with_min_members(view, min_members=3)
        view.set_cluster_comparison(insp_cid)
        qapp.processEvents()
        insp_members = list(view._comparison_member_viewers_map.keys())
        assert len(insp_members) >= 3

        view._on_inspector_member_clicked(insp_members[0])
        assert view._focus_source == "inspector"
        assert view._current_selected_gid == insp_members[0]

        view.skip_current_candidate()
        assert view._focus_source == "inspector"
        assert view._current_selected_gid == insp_members[1]

        assert view.unassign_focused_unit() is True
        assert view._focus_source == "inspector"
        assert view._current_selected_gid == insp_members[2]

        # (c) "table_cluster" -> filtered table to a cluster advances to next visible row in that cluster
        view.clear_cluster_comparison(return_to_suggestion=False)
        flt_idx = -1
        for idx in range(view.combo_filter_cluster.count()):
            if view.combo_filter_cluster.itemData(idx) == insp_cid:
                flt_idx = idx
                break
        assert flt_idx >= 0
        view.combo_filter_cluster.setCurrentIndex(flt_idx)
        qapp.processEvents()

        visible_rows = [
            r for r in range(view.table_pooled_units.rowCount())
            if not view.table_pooled_units.isRowHidden(r)
        ]
        assert len(visible_rows) >= 2
        row0_gid = _select_table_row_as_user(view, visible_rows[0])
        row1_gid = view._get_gid_for_row(visible_rows[1])
        assert row0_gid is not None and row1_gid is not None
        assert view._focus_source == "table_cluster"

        view.skip_current_candidate()
        assert view._focus_source == "table_cluster"
        assert view._current_selected_gid == row1_gid

        # Reset filter to All units
        view.combo_filter_cluster.setCurrentIndex(0)
        qapp.processEvents()

        # (d) "list_candidates" -> List Panel 'Candidates for <cluster>' advances to next candidate
        other_gids = [
            g for g, r in view._pooled_units_map.items()
            if r.get("cluster_id", -1) >= 0 and r.get("cluster_id") != insp_cid
        ]
        if len(other_gids) >= 2:
            view.assign_candidate_to_cluster(-1, unit_id=other_gids[:2])
        view.set_cluster_comparison(insp_cid)
        qapp.processEvents()
        cand_count = view.table_list_panel.rowCount()
        assert cand_count >= 2
        cand0 = str(view.table_list_panel.item(0, 0).data(Qt.ItemDataRole.UserRole))
        cand1 = str(view.table_list_panel.item(1, 0).data(Qt.ItemDataRole.UserRole))

        view._on_list_panel_cell_clicked(0, 0)
        assert view._focus_source == "list_candidates"
        assert view._current_selected_gid == cand0

        view.skip_current_candidate()
        assert view._focus_source == "list_candidates"
        assert view._current_selected_gid == cand1

        assert view.assign_candidate_to_cluster(insp_cid) is True
        assert view._focus_source == "list_candidates"
        assert view._current_selected_gid != cand1

        # (e) G / return_to_active_candidate() always returns to active-learning suggestion
        expected_sug = view.active_candidate.global_unit_id if view.active_candidate is not None else None
        assert expected_sug is not None
        ev_g = _send_key(view, Qt.Key.Key_G, text="g")
        assert ev_g.isAccepted()
        assert view._current_selected_gid == expected_sug
        assert view._focus_source == "queue"

        # Also verify direct method call from another source
        view._on_inspector_member_clicked(insp_members[0])
        assert view._focus_source == "inspector"
        assert view.return_to_active_candidate() is True
        assert view._current_selected_gid == expected_sug
        assert view._focus_source == "queue"
    finally:
        view.close()


def test_wp7_keyboard_map_no_plus_and_undo_redo_restores_focus(qapp, qtbot, monkeypatch):
    """Verify WP7 keyboard map (1-9, 0, Q/W/E/R/T, N, Backspace, S/Space, G, Ctrl+Z/Y, Esc, no '+') and Undo/Redo."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    try:
        qtbot.waitUntil(
            lambda: view.active_candidate is not None
            and (view._active_learning_worker is None or not view._active_learning_worker.isRunning()),
            timeout=10000,
        )
        # Ensure we have clusters mapped to 1-9, 0, Q, W, E, R, T by creating any missing clusters via 'N'
        expected_hotkeys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "Q", "W", "E", "R", "T"]
        key_map = {
            "1": (Qt.Key.Key_1, "1"),
            "2": (Qt.Key.Key_2, "2"),
            "3": (Qt.Key.Key_3, "3"),
            "4": (Qt.Key.Key_4, "4"),
            "5": (Qt.Key.Key_5, "5"),
            "6": (Qt.Key.Key_6, "6"),
            "7": (Qt.Key.Key_7, "7"),
            "8": (Qt.Key.Key_8, "8"),
            "9": (Qt.Key.Key_9, "9"),
            "0": (Qt.Key.Key_0, "0"),
            "Q": (Qt.Key.Key_Q, "q"),
            "W": (Qt.Key.Key_W, "w"),
            "E": (Qt.Key.Key_E, "e"),
            "R": (Qt.Key.Key_R, "r"),
            "T": (Qt.Key.Key_T, "t"),
        }

        # '+' must NOT create a new cluster
        cluster_count_before_plus = len(view._cluster_card_order)
        ev_plus = _send_key(view, Qt.Key.Key_Plus, text="+")
        assert not ev_plus.isAccepted()
        assert len(view._cluster_card_order) == cluster_count_before_plus

        # 'N' creates a new cluster and assigns the focused candidate
        focused_before_n = view._current_selected_gid
        assert focused_before_n is not None
        ev_n = _send_key(view, Qt.Key.Key_N, text="n")
        assert ev_n.isAccepted()
        assert len(view._cluster_card_order) == cluster_count_before_plus + 1
        new_cid = view._pooled_units_map[focused_before_n]["cluster_id"]
        assert new_cid in view._cluster_card_order

        # Backspace unassigns the focused assigned unit
        view.select_pooled_unit(focused_before_n, source="queue")
        ev_bs = _send_key(view, Qt.Key.Key_Backspace)
        assert ev_bs.isAccepted()
        assert view._pooled_units_map[focused_before_n]["cluster_id"] == -1

        # Undo via Ctrl+Z restores assignment AND focus
        ev_undo = _send_key(view, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        assert ev_undo.isAccepted()
        assert view._pooled_units_map[focused_before_n]["cluster_id"] == new_cid
        assert view._current_selected_gid == focused_before_n
        assert view._focus_source == "queue"

        # Redo via Ctrl+Y re-applies unassignment
        ev_redo = _send_key(view, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
        assert ev_redo.isAccepted()
        assert view._pooled_units_map[focused_before_n]["cluster_id"] == -1

        # Verify all 15 WP7 hotkeys (1-9, 0, Q, W, E, R, T) work across active clusters
        assert len(view._cluster_card_order) >= 1
        target_cid = view._cluster_card_order[0]
        target_gid = view._current_selected_gid
        assert target_gid is not None
        prev_al = view.active_learning_enabled
        view.active_learning_enabled = False
        try:
            for hk in expected_hotkeys:
                view._cluster_hotkeys[target_cid] = hk
                qt_key, txt = key_map[hk]
                view.select_pooled_unit(target_gid, source="queue")
                ev = _send_key(view, qt_key, text=txt)
                assert ev.isAccepted(), f"Hotkey {hk} was not accepted"
                assert view._pooled_units_map[target_gid]["cluster_id"] == target_cid
        finally:
            view.active_learning_enabled = prev_al
            view.refresh_active_learning()
            qtbot.waitUntil(
                lambda: view.active_candidate is not None
                and (view._active_learning_worker is None or not view._active_learning_worker.isRunning())
                and view._current_selected_gid == view.active_candidate.global_unit_id,
                timeout=10000,
            )

        # S and Space skip current unit
        gid_before_s = view._current_selected_gid
        ev_s = _send_key(view, Qt.Key.Key_S, text="s")
        assert ev_s.isAccepted()
        qtbot.waitUntil(
            lambda: (view._active_learning_worker is None or not view._active_learning_worker.isRunning())
            and view._current_selected_gid != gid_before_s,
            timeout=10000,
        )
        assert gid_before_s in view._skipped_gids
        assert view._current_selected_gid != gid_before_s

        gid_before_space = view._current_selected_gid
        ev_sp = _send_key(view, Qt.Key.Key_Space, text=" ")
        assert ev_sp.isAccepted()
        qtbot.waitUntil(
            lambda: (view._active_learning_worker is None or not view._active_learning_worker.isRunning())
            and view._current_selected_gid != gid_before_space,
            timeout=10000,
        )
        assert gid_before_space in view._skipped_gids
        assert view._current_selected_gid != gid_before_space

        # G returns to active candidate; Esc closes open inspector
        cid_open = view._cluster_card_order[0]
        view.set_cluster_comparison(cid_open)
        members_open = list(view._comparison_member_viewers_map.keys())
        if members_open:
            view._on_inspector_member_clicked(members_open[0])
            assert view._focus_source == "inspector"
        ev_g = _send_key(view, Qt.Key.Key_G, text="g")
        assert ev_g.isAccepted()
        assert view._focus_source == "queue"

        view.set_cluster_comparison(cid_open)
        ev_esc = _send_key(view, Qt.Key.Key_Escape)
        assert ev_esc.isAccepted()
        assert view._comparison_cluster_id is None

        # Undo/Redo for merge_clusters restores assignments AND focused_gid / _focus_source
        cid_src = view._cluster_card_order[0]
        cid_tgt = view._cluster_card_order[1]
        src_members = [g for g, r in view._pooled_units_map.items() if r.get("cluster_id") == cid_src]
        assert src_members
        view.set_cluster_comparison(cid_src)
        view._on_inspector_member_clicked(src_members[0])
        assert view._current_selected_gid == src_members[0]
        assert view._focus_source == "inspector"

        assert view.merge_clusters(cid_src, cid_tgt, confirm=False) is True
        assert cid_src not in view._cluster_card_order
        assert all(view._pooled_units_map[g]["cluster_id"] == cid_tgt for g in src_members)

        assert view.undo() is True
        assert cid_src in view._cluster_card_order
        assert all(view._pooled_units_map[g]["cluster_id"] == cid_src for g in src_members)
        assert view._current_selected_gid == src_members[0]
        assert view._focus_source == "inspector"

        assert view.redo() is True
        assert cid_src not in view._cluster_card_order
        assert all(view._pooled_units_map[g]["cluster_id"] == cid_tgt for g in src_members)

        # Undo/Redo for revert_auto_assignments
        auto_gid = src_members[0]
        view._pooled_units_map[auto_gid]["cluster_id"] = cid_tgt
        view._unit_status_map[auto_gid] = "Auto-assigned"
        view.select_pooled_unit(auto_gid, source="queue")

        reverted = view.revert_auto_assignments()
        assert reverted >= 1
        assert view._pooled_units_map[auto_gid]["cluster_id"] == -1
        assert view._unit_status_map[auto_gid] == "Unassigned"

        assert view.undo() is True
        assert view._pooled_units_map[auto_gid]["cluster_id"] == cid_tgt
        assert view._unit_status_map[auto_gid] == "Auto-assigned"
        assert view._current_selected_gid == auto_gid
        assert view._focus_source == "queue"

        assert view.redo() is True
        assert view._pooled_units_map[auto_gid]["cluster_id"] == -1
        assert view._unit_status_map[auto_gid] == "Unassigned"

        # Undo/Redo for rename and recolour
        old_name = view._cluster_names.get(cid_tgt, f"Cluster {cid_tgt}")
        monkeypatch.setattr(QInputDialog, "getText", lambda *a, **kw: ("Renamed WP7", True))
        view._on_card_rename_requested(cid_tgt)
        assert view._cluster_names[cid_tgt] == "Renamed WP7"

        assert view.undo() is True
        assert view._cluster_names.get(cid_tgt, f"Cluster {cid_tgt}") == old_name
        assert view.redo() is True
        assert view._cluster_names[cid_tgt] == "Renamed WP7"

        old_color = view._get_cluster_color(cid_tgt)
        monkeypatch.setattr(QColorDialog, "getColor", lambda *a, **kw: QColor("#FF00AA"))
        view._on_card_recolor_requested(cid_tgt)
        assert view._get_cluster_color(cid_tgt).upper() == "#FF00AA"

        assert view.undo() is True
        assert view._get_cluster_color(cid_tgt).upper() == old_color.upper()
        assert view.redo() is True
        assert view._get_cluster_color(cid_tgt).upper() == "#FF00AA"
    finally:
        view.close()


def test_list_panel_shift_click_isolation_and_arrow_navigation(qapp):
    """Shift-clicking in table_list_panel must not range-select table_pooled_units, and arrow keys must update focus."""
    from PyQt6.QtCore import QItemSelectionModel
    from PyQt6.QtTest import QTest

    view = ClusteringView(load_mock=True, start_with_clusters=True)
    view.show()
    qapp.processEvents()
    try:
        cid = view._cluster_card_order[0]
        # Ensure at least 3 unassigned units exist so candidate list has >= 3 rows
        other_gids = [
            g for g, r in view._pooled_units_map.items()
            if r.get("cluster_id", -1) != cid
        ]
        if len(other_gids) >= 3:
            view.assign_candidate_to_cluster(-1, unit_id=other_gids[:3])
        view.set_cluster_comparison(cid)
        qapp.processEvents()

        assert view.table_list_panel.rowCount() >= 3
        cand0 = str(view.table_list_panel.item(0, 0).data(Qt.ItemDataRole.UserRole))
        cand1 = str(view.table_list_panel.item(1, 0).data(Qt.ItemDataRole.UserRole))
        cand2 = str(view.table_list_panel.item(2, 0).data(Qt.ItemDataRole.UserRole))

        # 1. Arrow-key navigation in table_list_panel updates _current_selected_gid
        view.table_list_panel.setFocus()
        view.table_list_panel.setCurrentCell(0, 0)
        qapp.processEvents()
        assert view._current_selected_gid == cand0

        QTest.keyClick(view.table_list_panel, Qt.Key.Key_Down)
        qapp.processEvents()
        assert view._current_selected_gid == cand1

        QTest.keyClick(view.table_list_panel, Qt.Key.Key_Up)
        qapp.processEvents()
        assert view._current_selected_gid == cand0

        # 2. Shift-selecting rows 0 and 1 in table_list_panel must NOT select a range in table_pooled_units
        # (Simulate Shift-click selection on row 1 after row 0)
        sel_list = view.table_list_panel.selectionModel()
        idx0 = view.table_list_panel.model().index(0, 0)
        idx1 = view.table_list_panel.model().index(1, 0)
        sel_list.select(idx0, QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows)
        sel_list.select(idx1, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        view._on_list_panel_cell_clicked(1, 0)
        qapp.processEvents()

        assert len(view.table_pooled_units.selectionModel().selectedRows()) == 1
        assert view._current_selected_gid == cand1

        # 3. Assigning via hotkey while focused on list_candidates assigns ONLY the list_panel selected rows (cand0, cand1),
        # NOT intermediate rows from table_pooled_units
        view._on_cluster_hotkey_assign(cid)
        qapp.processEvents()
        assert view._pooled_units_map[cand0]["cluster_id"] == cid
        assert view._pooled_units_map[cand1]["cluster_id"] == cid
    finally:
        view.close()


def test_clustering_undo_covers_clustering_runs_card_reorder_and_marks_dirty(qapp, qtbot):
    """run_clustering, seed_clusters, move/sort cards must push undo snapshots, undo/redo must mark dirty and sync color service (Bug UI5)."""
    from optocurator.analysis.clustering import ClusteringParameters

    view = ClusteringView(load_mock=True, start_with_clusters=True)
    try:
        qtbot.waitUntil(
            lambda: view._active_learning_worker is None or not view._active_learning_worker.isRunning(),
            timeout=10000,
        )
        orig_order = list(view._cluster_card_order)
        assert len(orig_order) >= 2
        orig_cids = {gid: rec["cluster_id"] for gid, rec in view._pooled_units_map.items()}
        custom_cid = orig_order[0]
        view._cluster_colors[custom_cid] = "#ABCDEF"
        view.cluster_color_service.set_colors(view._cluster_colors)
        view._set_dirty(False)

        # 1. move_cluster_card must push undo snapshot and be undoable (marking dirty)
        assert view.move_cluster_card(orig_order[0], orig_order[1]) is True
        assert list(view._cluster_card_order[:2]) == [orig_order[1], orig_order[0]]
        view._set_dirty(False)
        assert view.undo() is True
        assert list(view._cluster_card_order) == orig_order
        assert view._is_dirty is True

        # 2. _sort_cluster_cards must push undo snapshot
        view.move_cluster_card(orig_order[0], orig_order[1])
        swapped_order = list(view._cluster_card_order)
        view._sort_cluster_cards("id")
        assert view.undo() is True
        assert list(view._cluster_card_order) == swapped_order
        assert view.undo() is True
        assert list(view._cluster_card_order) == orig_order

        # 3. run_clustering(preserve_curated=False) must push undo snapshot and sync color service on undo
        first_gid = next(iter(view._pooled_units_map))
        view.assign_candidate_to_cluster(orig_order[0], unit_id=first_gid)
        before_run = {gid: rec["cluster_id"] for gid, rec in view._pooled_units_map.items()}
        assert view.run_clustering(ClusteringParameters(n_clusters=2, algorithm="hierarchical"), preserve_curated=False) is True
        view._cluster_colors[custom_cid] = "#112233"
        view._set_dirty(False)
        assert view.undo() is True
        assert {gid: rec["cluster_id"] for gid, rec in view._pooled_units_map.items()} == before_run
        assert view.cluster_color_service.get_color(custom_cid).upper() == "#ABCDEF"
        assert view._is_dirty is True

        # 4. seed_clusters_with_affinity_propagation must push undo snapshot
        before_seed = {gid: rec["cluster_id"] for gid, rec in view._pooled_units_map.items()}
        assert view.seed_clusters_with_affinity_propagation() is True
        assert view.undo() is True
        assert {gid: rec["cluster_id"] for gid, rec in view._pooled_units_map.items()} == before_seed
    finally:
        qtbot.waitUntil(
            lambda: view._active_learning_worker is None or not view._active_learning_worker.isRunning(),
            timeout=10000,
        )
        view.close()


def test_ui7_list_panel_new_cluster_multiselection_point_click_source_and_prewarm_resume(qapp, qtbot, monkeypatch):
    """create_new_cluster_for_candidate honors list_panel multiselection; point clicks reset _focus_source; prewarm resumes on gen mismatch (Bug UI7)."""
    from PyQt6.QtCore import QItemSelectionModel

    view = ClusteringView(load_mock=True, start_with_clusters=True)
    try:
        qtbot.waitUntil(
            lambda: view._active_learning_worker is None or not view._active_learning_worker.isRunning(),
            timeout=10000,
        )
        cid = view._cluster_card_order[0]
        other_gids = [
            g for g, r in view._pooled_units_map.items()
            if r.get("cluster_id", -1) != cid
        ]
        if len(other_gids) >= 3:
            view.assign_candidate_to_cluster(-1, unit_id=other_gids[:3])
        qtbot.waitUntil(
            lambda: view._active_learning_worker is None or not view._active_learning_worker.isRunning(),
            timeout=10000,
        )
        view.set_cluster_comparison(cid)
        qapp.processEvents()
        assert view.table_list_panel.rowCount() >= 2
        cand0 = str(view.table_list_panel.item(0, 0).data(Qt.ItemDataRole.UserRole))
        cand1 = str(view.table_list_panel.item(1, 0).data(Qt.ItemDataRole.UserRole))
        assert cand0 != cand1

        # 1. Multi-select two units in table_list_panel and create a new cluster -> both must move
        sel_list = view.table_list_panel.selectionModel()
        idx0 = view.table_list_panel.model().index(0, 0)
        idx1 = view.table_list_panel.model().index(1, 0)
        sel_list.select(idx0, QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows)
        sel_list.select(idx1, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        view._on_list_panel_cell_clicked(1, 0)
        qapp.processEvents()
        assert view._focus_source == "list_candidates"

        new_cid = view.create_new_cluster_for_candidate()
        assert new_cid >= 0
        assert view._pooled_units_map[cand0]["cluster_id"] == new_cid
        assert view._pooled_units_map[cand1]["cluster_id"] == new_cid

        # 2. Clicking scatter point or probe unit must update _focus_source to "queue"
        class _DummyPoint:
            def __init__(self, gid: str) -> None:
                self._gid = gid

            def data(self) -> str:
                return self._gid

        view._focus_source = "inspector"
        view._on_scatter_point_clicked(None, [_DummyPoint(cand0)])
        assert view._focus_source == "queue"

        view._focus_source = "list_candidates"
        view._on_probe_unit_clicked(None, [_DummyPoint(cand1)])
        assert view._focus_source == "queue"

        # 3. Stale-generation prewarm batch callback must still kick queued new-generation prewarm items
        kick_calls = 0

        def _counted_kick() -> None:
            nonlocal kick_calls
            kick_calls += 1

        monkeypatch.setattr(view, "_kick_prewarm_data_worker", _counted_kick)
        view._prewarm_worker_active = True
        view._prewarm_generation += 1
        view._prewarm_data_queue = [cand0]
        view._on_prewarm_batch_ready(view._prewarm_generation - 1, [])
        assert kick_calls == 1
    finally:
        qtbot.waitUntil(
            lambda: view._active_learning_worker is None or not view._active_learning_worker.isRunning(),
            timeout=10000,
        )
        view.close()
