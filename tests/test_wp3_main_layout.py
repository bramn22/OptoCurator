"""Tests for Work Package 3 (WP3 — Main layout) from UI_OVERHAUL_SPEC.md (§ 5, § 6, § 11.2, § 13.1)."""

from PyQt6.QtCore import QPointF, QSettings, Qt
from PyQt6.QtGui import QMouseEvent

from optocurator.ui.app import MainWindow
from optocurator.ui.splitters import ResettableSplitter
from optocurator.ui.theme import (
    CENTRE_PANEL_WIDTH,
    CLUSTER_CARD_SIZE,
    CLUSTER_GRID_MIN_HEIGHT,
    FOCUSED_UNIT_WIDTH,
    HEADER_HEIGHT,
    HEADER_PROGRESS_BAR_WIDTH,
    INSPECTOR_SLOT_WIDTH,
    LIST_PANEL_MIN_HEIGHT,
    RIGHT_PANEL_MIN_WIDTH,
    STATUS_BAR_HEIGHT,
    UNIT_LIST_MAX_WIDTH,
    UNIT_LIST_MIN_WIDTH,
)


def test_wp3_header_and_status_bar_across_modes(qapp):
    """Verify 44 px header, mode switch, no Modes menu, experiments dropdown, progress bar, and 24 px status bar."""
    win = MainWindow()
    win.resize(1920, 1080)

    # Launcher starts clean (header and status bar hidden)
    assert win.stack.currentIndex() == MainWindow.VIEW_LAUNCHER
    assert win.header_bar.isHidden()
    assert win.status.isHidden()

    # Switch to Clustering via segmented mode switch / method
    win.switch_to_clustering()
    qapp.processEvents()

    assert not win.header_bar.isHidden()
    assert win.header_bar.height() == HEADER_HEIGHT == 44
    assert not win.status.isHidden()
    assert win.status.height() == STATUS_BAR_HEIGHT == 24

    # Mode switch state
    assert win.btn_mode_clustering.isChecked()
    assert not win.btn_mode_detection.isChecked()

    # Menus: File, View, Clustering, Help (no "Modes" menu; Stats hidden in Clustering)
    visible_menus = [
        a.text().replace("&", "")
        for a in win.menuBar().actions()
        if a.isVisible()
    ]
    assert visible_menus == ["File", "View", "Clustering", "Help"]
    assert "Modes" not in [a.text().replace("&", "") for a in win.menuBar().actions()]
    assert win.act_reset_panel_layout.text() == "Reset panel layout"

    # App mark tooltip
    assert win.btn_home.toolTip() == "Home"

    # Experiments dropdown & ~200 px header progress bar
    assert not win.btn_experiments.isHidden()
    assert "experiment" in win.btn_experiments.text()
    assert "unit" in win.btn_experiments.text()
    assert not win.header_progress_bar.isHidden()
    assert win.header_progress_bar.bar_container.width() == HEADER_PROGRESS_BAR_WIDTH == 200
    assert "/" in win.header_progress_bar.lbl_count.text()

    # Switch to Detection via segmented mode switch
    win.btn_mode_detection.click()
    qapp.processEvents()
    assert win.stack.currentIndex() == MainWindow.VIEW_DETECTION
    assert win.btn_mode_detection.isChecked()
    assert not win.btn_mode_clustering.isChecked()
    visible_det_menus = [
        a.text().replace("&", "")
        for a in win.menuBar().actions()
        if a.isVisible()
    ]
    assert visible_det_menus == ["File", "View", "Stats", "Help"]

    # Click App mark to return Home (clean state)
    win.btn_home.click()
    qapp.processEvents()
    assert win.stack.currentIndex() == MainWindow.VIEW_LAUNCHER
    assert win.header_bar.isHidden()
    assert win.status.isHidden()

    win.close()


def test_wp3_four_region_layout_and_reserved_inspector_slot(qapp):
    """Verify 4 regions, fixed 300+300 centre, empty state, and zero layout shift on cluster open/close."""
    win = MainWindow()
    win.resize(1920, 1080)
    win.clustering_view._start_with_clusters = True
    win.clustering_view._load_mock_data()
    win.switch_to_clustering()
    win.show()
    qapp.processEvents()

    cv = win.clustering_view
    assert isinstance(cv.main_splitter, ResettableSplitter)
    assert isinstance(cv.right_splitter, ResettableSplitter)
    assert cv.main_splitter.childrenCollapsible() is False
    assert cv.right_splitter.childrenCollapsible() is False

    # Region dimension constraints
    assert cv.unit_list_panel.minimumWidth() == UNIT_LIST_MIN_WIDTH == 300
    assert cv.unit_list_panel.maximumWidth() == UNIT_LIST_MAX_WIDTH == 600
    assert cv.centre_panel.width() == CENTRE_PANEL_WIDTH == 600
    assert cv.focused_unit_panel.width() == FOCUSED_UNIT_WIDTH == 300
    assert cv.inspector_slot.width() == INSPECTOR_SLOT_WIDTH == 300
    assert cv.right_panel.minimumWidth() == RIGHT_PANEL_MIN_WIDTH == 400
    assert cv.clusters_area_panel.minimumHeight() == CLUSTER_GRID_MIN_HEIGHT == 164
    assert cv.list_panel.minimumHeight() == LIST_PANEL_MIN_HEIGHT == 120

    # Empty state when no cluster is open
    assert cv._comparison_cluster_id is None
    assert not cv.inspector_empty_state.isHidden()
    assert cv.lbl_inspector_empty.text() == "Click a cluster to compare its members"
    assert cv.comparison_pane.isHidden()

    # Record geometries before opening a cluster
    g_left_before = cv.unit_list_panel.geometry()
    g_focused_before = cv.focused_unit_panel.geometry()
    g_slot_before = cv.inspector_slot.geometry()
    g_right_before = cv.right_panel.geometry()

    # Open first cluster
    cids = sorted(cv._cluster_cards.keys())
    assert len(cids) >= 1
    cv.set_cluster_comparison(cids[0])
    qapp.processEvents()

    assert cv.inspector_empty_state.isHidden()
    assert not cv.comparison_pane.isHidden()
    assert cv.unit_list_panel.geometry() == g_left_before
    assert cv.focused_unit_panel.geometry() == g_focused_before
    assert cv.inspector_slot.geometry() == g_slot_before
    assert cv.right_panel.geometry() == g_right_before

    # Close cluster
    cv.clear_cluster_comparison()
    qapp.processEvents()

    assert not cv.inspector_empty_state.isHidden()
    assert cv.comparison_pane.isHidden()
    assert cv.unit_list_panel.geometry() == g_left_before
    assert cv.focused_unit_panel.geometry() == g_focused_before
    assert cv.inspector_slot.geometry() == g_slot_before
    assert cv.right_panel.geometry() == g_right_before

    win.close()


def test_wp3_splitter_double_click_reset_and_qsettings_persistence(qapp):
    """Verify splitter double-click reset, View -> Reset panel layout, and QSettings save/restore."""
    settings = QSettings("OptoCurator", "OptoCurator")
    settings.remove("splitters/clustering_main")
    settings.remove("splitters/clustering_right")

    win = MainWindow()
    win.resize(1920, 1080)
    win.switch_to_clustering()
    win.show()
    qapp.processEvents()

    cv = win.clustering_view
    # Modify splitter sizes and save
    cv.main_splitter.setSizes([480, 600, 820])
    cv.main_splitter.save_to_settings()
    total_h = sum(cv.right_splitter.sizes())
    cv.right_splitter.setSizes([total_h - 350, 350])
    cv.right_splitter.save_to_settings()
    qapp.processEvents()

    # New window restores persisted sizes
    win2 = MainWindow()
    win2.resize(1920, 1080)
    win2.switch_to_clustering()
    win2.show()
    qapp.processEvents()

    assert abs(win2.clustering_view.main_splitter.sizes()[0] - 480) <= 20
    assert abs(win2.clustering_view.right_splitter.sizes()[1] - 350) <= 20

    # Double-click on main_splitter handle resets to default sizes
    handle = win2.clustering_view.main_splitter.handle(1)
    dbl_ev = QMouseEvent(
        QMouseEvent.Type.MouseButtonDblClick,
        QPointF(2.0, 2.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    handle.mouseDoubleClickEvent(dbl_ev)
    qapp.processEvents()
    assert abs(win2.clustering_view.main_splitter.sizes()[0] - 380) <= 20

    # View -> Reset panel layout resets both splitters
    win2.clustering_view.right_splitter.setSizes([450, 400])
    win2.act_reset_panel_layout.trigger()
    qapp.processEvents()
    expected_bottom = round(260 * sum(win2.clustering_view.right_splitter.sizes()) / 880)
    assert abs(win2.clustering_view.right_splitter.sizes()[1] - expected_bottom) <= 20

    win.close()
    win2.close()
    settings.remove("splitters/clustering_main")
    settings.remove("splitters/clustering_right")


def test_wp3_card_grid_scrolling_and_dynamic_reflow(qapp):
    """Verify card grid scrolls vertically only and reflows columns dynamically without clipping."""
    win = MainWindow()
    win.resize(1920, 1080)
    win.clustering_view._start_with_clusters = True
    win.clustering_view._load_mock_data()
    win.switch_to_clustering()
    win.clustering_view.reset_default_layout()
    win.show()
    qapp.processEvents()

    cv = win.clustering_view
    assert cv.clusters_scroll_area.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert cv.clusters_scroll_area.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded

    # At 1920 px, right panel is ~928 px -> 4 columns of 217 px cards
    assert cv._compute_card_grid_columns() == 4
    for card in cv._cluster_cards.values():
        assert (card.width(), card.height()) == CLUSTER_CARD_SIZE

    # Shrink window to 1366x768 -> fewer columns without shrinking cards
    win.resize(1366, 768)
    qapp.processEvents()
    cols_narrow = cv._compute_card_grid_columns()
    assert 1 <= cols_narrow <= 2
    for card in cv._cluster_cards.values():
        assert (card.width(), card.height()) == CLUSTER_CARD_SIZE

    win.close()
