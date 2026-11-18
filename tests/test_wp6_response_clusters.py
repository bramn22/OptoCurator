"""Tests for WP6 (§ 9) Response Clusters grid, toolbar, cards, and list panel."""

from PyQt6.QtCore import QEvent, QPoint, Qt
from PyQt6.QtGui import QColor, QContextMenuEvent, QEnterEvent
from PyQt6.QtWidgets import QColorDialog, QInputDialog, QMenu

from optocurator.ui.theme import (
    CLUSTER_CARD_KEYCAP_HOVER_STYLE,
    CLUSTER_CARD_KEYCAP_STYLE,
)
from optocurator.ui.view_clusters import (
    ClusterCardWidget,
    ClusteringView,
    NewClusterCardWidget,
)


def test_wp6_toolbar_controls_and_menus(qapp):
    """Verify § 9.1 toolbar widgets, count chip, View Options menu, More menu, and hidden btn_new_cluster."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    qapp.processEvents()

    # Count chip reflects active cluster count
    n_clusters = len(view._cluster_card_order)
    assert n_clusters > 0
    assert view.lbl_clusters_count_chip.text() == str(n_clusters)

    # Primary toolbar buttons
    assert view.btn_seed_clusters.text() == "Seed clusters"
    assert view.btn_cluster.text() == "Run clustering"
    assert view.btn_reset_clusters.text() == "↻ Reset clustering"
    assert view.btn_new_cluster.isHidden()

    # View Options menu (☰)
    view_menu = view.btn_view_options.menu()
    assert view_menu is not None
    top_texts = [a.text() for a in view_menu.actions() if not a.isSeparator()]
    assert "Overlay focused unit" in top_texts
    assert "Shared y-axis" in top_texts
    assert "Sort by" in top_texts

    sort_action = next(a for a in view_menu.actions() if a.text() == "Sort by")
    sort_menu = sort_action.menu()
    assert sort_menu is not None
    sort_labels = [a.text() for a in sort_menu.actions()]
    assert sort_labels == ["Cluster ID", "Size", "Latency", "Name"]

    # Toggle Overlay focused unit OFF -> candidate curve cleared; ON -> restored
    assert view.act_overlay_focused_unit.isChecked() is True
    view.act_overlay_focused_unit.setChecked(False)
    assert view.overlay_focused_unit_enabled is False
    for card in view._cluster_cards.values():
        x_data, _ = card.candidate_psth_curve.getData()
        assert x_data is None or len(x_data) == 0
    view.act_overlay_focused_unit.setChecked(True)
    assert view.overlay_focused_unit_enabled is True

    # Toggle Shared y-axis ON / OFF
    assert view.act_shared_y_axis.isChecked() is False
    view.act_shared_y_axis.setChecked(True)
    assert view.shared_y_axis_enabled is True
    for card in view._cluster_cards.values():
        assert bool(card.plot.getViewBox().autoRangeEnabled()[1]) is False
    view.act_shared_y_axis.setChecked(False)
    assert view.shared_y_axis_enabled is False
    for card in view._cluster_cards.values():
        assert bool(card.plot.getViewBox().autoRangeEnabled()[1]) is True

    # More menu (⋯)
    more_menu = view.btn_clusters_more.menu()
    assert more_menu is not None
    more_texts = [a.text() for a in more_menu.actions() if not a.isSeparator()]
    assert more_texts == [
        "Feature settings…",
        "Customize clusters…",
        "Confidence threshold…",
        "Revert auto-assignments…",
    ]

    view.close()


def test_wp6_cluster_card_header_elision_badge_and_keycap_hover(qapp):
    """Verify § 9.2 card count ('n = N'), elided name + tooltip, '!' badge, keycap hover/click, and 2 px outline."""
    long_name = "Very Long Fast-Spiking Opto-Tagged Cluster Name"
    card = ClusterCardWidget(
        cluster_id=0,
        count=14,
        color_hex="#2563EB",
        custom_name=long_name,
        hotkey="1",
    )

    # Count format
    assert card.lbl_count.text() == "n = 14"

    # Elided name with full name in tooltip
    assert card.header.toolTip() == long_name
    assert card.header.text().endswith("…")
    assert len(card.header.text()) < len(long_name)

    # Suggested badge ('!' glyph, tooltip, not mouse-transparent)
    assert card.lbl_suggested_badge.isHidden()
    assert not card.lbl_suggested_badge.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    card.set_suggested_count(3, threshold=0.85)
    assert not card.lbl_suggested_badge.isHidden()
    assert card.lbl_suggested_badge.text() == "! 3"
    assert card.lbl_suggested_badge.toolTip() == "Cells above the confidence threshold"

    # Keycap compact -> hover ("Assign K") -> leave ("K")
    assert card.btn_keycap.text() == "1"
    assert card.btn_keycap.styleSheet() == CLUSTER_CARD_KEYCAP_STYLE
    card.show()
    qapp.processEvents()
    card.clearFocus()
    card._animate_keycap(False)
    assert card.btn_keycap.text() == "1"
    assert card.btn_keycap.styleSheet() == CLUSTER_CARD_KEYCAP_STYLE
    card.enterEvent(QEnterEvent(QPoint(10, 10).toPointF(), QPoint(10, 10).toPointF(), QPoint(10, 10).toPointF()))
    assert card.btn_keycap.text() == "Assign 1"
    assert card.btn_keycap.styleSheet() == CLUSTER_CARD_KEYCAP_HOVER_STYLE
    card.leaveEvent(QEvent(QEvent.Type.Leave))
    assert card.btn_keycap.text() == "1"
    assert card.btn_keycap.styleSheet() == CLUSTER_CARD_KEYCAP_STYLE

    # Keycap click emits assign_requested(cluster_id)
    emitted = []
    card.assign_requested.connect(emitted.append)
    card.btn_keycap.click()
    assert emitted == [0]

    # 2 px outline when open for comparison
    card.set_comparison_active(True)
    assert "2px solid" in card.styleSheet()
    card.set_comparison_active(False)

    card.close()


def test_wp6_keycap_defaults_click_assign_and_new_cluster_card(qapp):
    """Verify § 9.2 default hotkeys (0-9, Q W E R T), keycap click assignment, and NewClusterCardWidget at grid end."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    qapp.processEvents()

    # Verify default hotkey mapping 0-9 and 10-14 -> Q W E R T
    for cid in range(10):
        assert view._get_cluster_hotkey(cid) == str(cid)
    assert view._get_cluster_hotkey(10) == "Q"
    assert view._get_cluster_hotkey(11) == "W"
    assert view._get_cluster_hotkey(12) == "E"
    assert view._get_cluster_hotkey(13) == "R"
    assert view._get_cluster_hotkey(14) == "T"

    # NewClusterCardWidget is placed at the end of the grid
    assert isinstance(view._new_cluster_card, NewClusterCardWidget)
    assert view._new_cluster_card.lbl_title.text() == "+ New cluster"
    assert view._new_cluster_card.lbl_keycap.text() == "N"
    n_cards = len(view._cluster_card_order)
    last_item = view.clusters_grid.itemAt(n_cards)
    assert last_item is not None and last_item.widget() is view._new_cluster_card

    # Clicking keycap assigns focused unit to that cluster
    target_gid = view._get_gid_for_row(0)
    view.select_pooled_unit(target_gid)
    target_cid = view._cluster_card_order[-1]
    view._cluster_cards[target_cid].btn_keycap.click()
    assert view._pooled_units_map[target_gid]["cluster_id"] == target_cid
    assert view._unit_status_map[target_gid] == "Curated"

    # Clicking NewClusterCardWidget creates a new cluster and assigns focused unit
    prev_count = len(view._cluster_card_order)
    gid2 = view._get_gid_for_row(1)
    view.select_pooled_unit(gid2)
    view._new_cluster_card.clicked.emit()
    assert len(view._cluster_card_order) == prev_count + 1
    new_cid = view._pooled_units_map[gid2]["cluster_id"]
    assert new_cid in view._cluster_cards
    assert view.lbl_clusters_count_chip.text() == str(len(view._cluster_card_order))

    view.close()


def test_wp6_sort_orders_preserve_keys_and_comparison_outline(qapp):
    """Verify § 9.1/9.2 Sort by (Cluster ID, Size, Latency, Name) keeps keys tied to cluster ID and 2 px open outline."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    qapp.processEvents()

    # Give clusters distinct names for Name sort
    for idx, cid in enumerate(list(view._cluster_card_order)):
        view._cluster_names[cid] = f"Zebra_{9 - idx}"

    # Record each cluster's hotkey before sorting
    initial_keys = {cid: view._cluster_cards[cid].btn_keycap.text() for cid in view._cluster_card_order}

    for mode in ("size", "latency", "name", "id"):
        view._sort_cluster_cards(mode)
        assert view._sort_actions[mode].isChecked()
        for cid in view._cluster_card_order:
            assert view._cluster_cards[cid].btn_keycap.text() == initial_keys[cid]

    # Verify Name sort actually ordered alphabetically
    view._sort_cluster_cards("name")
    ordered_names = [view._cluster_names[cid] for cid in view._cluster_card_order]
    assert ordered_names == sorted(ordered_names, key=str.lower)

    # Open comparison on first card -> 2 px outline on open card only
    open_cid = view._cluster_card_order[0]
    view.set_cluster_comparison(open_cid)
    assert "2px solid" in view._cluster_cards[open_cid].styleSheet()
    for cid, card in view._cluster_cards.items():
        if cid != open_cid:
            assert card._is_comparing is False

    view.close()


def test_wp6_card_context_menu_rename_recolor_merge_dissolve_undo(qapp, monkeypatch):
    """Verify § 9.2 right-click context menu actions and Rename, Change colour, and Dissolve + Undo."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    qapp.processEvents()

    cid = view._cluster_card_order[0]
    card = view._cluster_cards[cid]

    # Inspect context menu actions without blocking
    captured_actions = []

    def _fake_menu_exec(self, _pos):
        for act in self.actions():
            if not act.isSeparator():
                captured_actions.append(act.text())
        return None

    monkeypatch.setattr(QMenu, "exec", _fake_menu_exec)
    card.contextMenuEvent(QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(10, 10), QPoint(10, 10)))
    assert "Rename…" in captured_actions
    assert "Change colour…" in captured_actions
    assert "Merge into…" in captured_actions
    assert "Dissolve cluster…" in captured_actions

    # Rename via mocked QInputDialog
    monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: ("Fast Bursters", True))
    view._on_card_rename_requested(cid)
    assert view._cluster_names[cid] == "Fast Bursters"
    assert card.display_name == "Fast Bursters"

    # Change colour via mocked QColorDialog
    monkeypatch.setattr(QColorDialog, "getColor", lambda *args, **kwargs: QColor("#FF5500"))
    view._on_card_recolor_requested(cid)
    assert view._get_cluster_color(cid).lower() == "#ff5500"

    # Dissolve cluster + Undo
    prev_members = [gid for gid, rec in view._pooled_units_map.items() if rec.get("cluster_id") == cid]
    assert len(prev_members) > 0
    assert view.dissolve_cluster(cid, confirm=False) is True
    assert cid not in view._cluster_cards
    for gid in prev_members:
        assert view._pooled_units_map[gid]["cluster_id"] == -1

    # Undo restores the dissolved cluster and its members
    assert view.btn_undo.isEnabled()
    view.undo()
    assert cid in view._cluster_cards
    for gid in prev_members:
        assert view._pooled_units_map[gid]["cluster_id"] == cid

    view.close()


def test_wp6_list_panel_states_collapse_click_and_batch_accept(qapp):
    """Verify § 9.3 list panel collapse, 3 states, 9 columns, click focus, and 'Accept N selected'."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    view.refresh_active_learning(synchronous=True)
    qapp.processEvents()

    # Ensure some unassigned units exist so list panel is populated
    gids_to_unassign = [view._get_gid_for_row(i) for i in range(3)]
    view.assign_candidate_to_cluster(-1, unit_id=gids_to_unassign)
    view.refresh_active_learning(synchronous=True)
    qapp.processEvents()

    # Collapsible chevron
    view.resize(1400, 900)
    view.show()
    qapp.processEvents()
    expanded_h = view.list_panel.height()
    assert expanded_h >= 120
    assert view.btn_list_collapse.text() == "▾"
    assert not view.list_panel_body.isHidden()
    view.btn_list_collapse.click()
    qapp.processEvents()
    assert view.btn_list_collapse.text() == "▸"
    assert view.list_panel_body.isHidden()
    assert view.list_panel.height() <= 52
    assert view.lbl_list_panel_count.height() <= 24
    view.btn_list_collapse.click()
    qapp.processEvents()
    assert view.btn_list_collapse.text() == "▾"
    assert not view.list_panel_body.isHidden()
    assert view.list_panel.height() >= 120

    # 9 columns, no checkboxes
    assert view.table_list_panel.columnCount() == 9
    headers = [view.table_list_panel.horizontalHeaderItem(c).text() for c in range(9)]
    assert headers == [
        "Unit", "Session", "Top probability", "Uncertainty",
        "Best match", "Lat (ms)", "Jitter", "Rel.", "SALT p",
    ]
    assert "✓" not in headers
    assert view.table_list_panel.rowCount() >= 3
    for c in range(9):
        item = view.table_list_panel.item(0, c)
        assert item.data(Qt.ItemDataRole.CheckStateRole) is None

    # State 1: Suggested next (AL ON, no cluster open)
    assert view.lbl_list_panel_title.text() == "Suggested next"
    assert view.lbl_list_panel_subtitle.text() == "by classifier uncertainty"
    assert view.btn_list_accept_selected.isHidden()

    # State 2: Unassigned units (AL OFF, no cluster open)
    view.set_active_learning_mode(False)
    assert view.lbl_list_panel_title.text() == "Unassigned units"
    assert view.lbl_list_panel_subtitle.text() == "by unit ID"
    assert view.btn_list_accept_selected.isHidden()

    # Plain click focuses unit
    row1_gid = view.table_list_panel.item(1, 0).data(Qt.ItemDataRole.UserRole)
    view._on_list_panel_cell_clicked(1, 0)
    assert view._current_selected_gid == row1_gid

    # State 3: Candidates for <name> (cluster open)
    open_cid = view._cluster_card_order[0]
    view._cluster_names[open_cid] = "Burst Alpha"
    view.set_cluster_comparison(open_cid)
    assert "Burst Alpha" in view.lbl_list_panel_title.text()
    assert view.lbl_list_panel_subtitle.text() == "by confidence"
    assert view.table_list_panel.horizontalHeaderItem(2).text() == "Confidence"
    assert not view.btn_list_accept_selected.isHidden()

    # Multi-select updates "Accept N selected"
    view.table_list_panel.clearSelection()
    sel_gids = []
    for r in (0, 1):
        sel_gids.append(view.table_list_panel.item(r, 0).data(Qt.ItemDataRole.UserRole))
        view.table_list_panel.selectRow(r)
    qapp.processEvents()
    # Select 2 rows explicitly via selectionModel
    from PyQt6.QtCore import QItemSelectionModel
    sm = view.table_list_panel.selectionModel()
    sm.clearSelection()
    for r in (0, 1):
        sm.select(view.table_list_panel.model().index(r, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
    qapp.processEvents()

    assert view.btn_list_accept_selected.text() == "Accept 2 selected"
    assert view.btn_list_accept_selected.isEnabled()

    # Clicking "Accept N selected" assigns and removes units from list immediately
    before_rows = view.table_list_panel.rowCount()
    view.btn_list_accept_selected.click()
    qapp.processEvents()
    for gid in sel_gids:
        assert view._pooled_units_map[gid]["cluster_id"] == open_cid
        assert view._unit_status_map[gid] == "Curated"
    assert view.table_list_panel.rowCount() == before_rows - 2

    view.close()
