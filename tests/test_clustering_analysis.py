"""Unit and integration tests for Iteration 4: Phase 2 Clustering and Scatter Exploration."""

from pathlib import Path
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from PyQt6.QtCore import QEvent, QMimeData, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QHeaderView, QLabel, QMessageBox, QSplitter

from optocurator.analysis.clustering import (
    AffinityPropagationParameters,
    CLUSTER_PALETTE,
    ClusteringParameters,
    ClusteringResult,
    ClusterSummary,
    EmbeddingParameters,
    FeatureConfiguration,
    compute_2d_embedding,
    compute_cluster_summaries,
    compute_pca_embedding,
    extract_feature_matrix,
    normalize_psth_waveform,
    perform_clustering,
    run_clustering_algorithm,
    run_hierarchical_clustering,
    seed_clusters_affinity_propagation,
)
from optocurator.analysis.metrics import compute_psth_time_ticks
from optocurator.io.phy import PhyKilosortReader, load_probe_and_unit_positions
from optocurator.io.validation import validate_phase1_export
from optocurator.ui.app import MainWindow
from optocurator.ui.dialog_clustering import (
    EmbeddingSettingsDialog,
    HierarchicalClusteringDialog,
)
from optocurator.ui.dialog_clusters_customize import CustomizeClustersDialog, HotkeyLineEdit, normalize_hotkey
from optocurator.ui.dialog_feature_settings import FeatureSettingsDialog, format_feature_summary
from optocurator.ui.dialog_seed_clusters import SeedClustersDialog
from optocurator.ui.plots import UnitResponsePlot, format_unit_display as _format_unit_display
from optocurator.ui.splitters import ResettableSplitter
from optocurator.ui.theme import (
    CLUSTER_CARD_SIZE,
    CLUSTER_GRID_MIN_HEIGHT,
    COLOR_POPULATION_CURVE,
    COLOR_PROBE_SHANK_BG,
    COLOR_TEXT_MUTED,
    FOCUSED_UNIT_WIDTH,
    INSPECTOR_SLOT_WIDTH,
    PLOT_CLUSTER_PROB_SIZE,
    text_color_for_cluster,
)
from optocurator.ui.view_clusters import ClusterCardWidget, ClusteringView
from optocurator.ui.view_detection import DetectionView


def _make_dummy_pooled_df(n_units: int = 12) -> pd.DataFrame:
    """Create a mock DataFrame representing pooled optotagged units."""
    rows = []
    np.random.seed(42)
    for i in range(n_units):
        group = i % 3
        if group == 0:
            lat, jit, rel = 3.2, 0.8, 0.95
        elif group == 1:
            lat, jit, rel = 8.5, 1.8, 0.80
        else:
            lat, jit, rel = 14.0, 2.5, 0.55

        rows.append({
            "global_unit_id": f"sessA_u{i:04d}",
            "session_id": "sessA",
            "unit_id": i + 1,
            "latency_ms": lat + float(np.random.normal(0, 0.3)),
            "jitter_ms": jit + float(np.random.normal(0, 0.1)),
            "reliability": max(0.1, min(1.0, rel + float(np.random.normal(0, 0.05)))),
            "waveform_correlation": 0.94 + float(np.random.normal(0, 0.02)),
            "salt_p_value": 0.001 * (group + 1),
            "is_tagged": True,
            "cluster_id": -1,
        })
    return pd.DataFrame(rows)



def test_extract_feature_matrix():
    """Verify feature matrix extraction across scalar metrics, PSTH traces, and PCA compression."""
    df = _make_dummy_pooled_df(10)

    # 1. Analytical & Summary Metrics only (default 5 scalar metrics)
    cfg_metrics = FeatureConfiguration(use_psth_profile=False)
    feat_m, psth_m, t_bins = extract_feature_matrix(df, cfg_metrics)
    assert feat_m.shape == (10, 5)
    assert not np.isnan(feat_m).any()
    assert psth_m.shape[0] == 10
    assert len(t_bins) == 50  # [0, 25] with default 0.5 ms step -> 50 bin centers

    # 2. PSTH Traces only (without PCA)
    cfg_psth = FeatureConfiguration(
        use_latency=False,
        use_jitter=False,
        use_reliability=False,
        use_salt=False,
        use_peak_response_rate=False,
        use_time_to_response=False,
        use_psth_profile=True,
        psth_use_pca=False,
    )
    feat_p, _, _ = extract_feature_matrix(df, cfg_psth)
    assert feat_p.shape == (10, 50)
    assert not np.isnan(feat_p).any()

    # 3. PSTH Traces with PCA compression (variance explained)
    cfg_pca = FeatureConfiguration(
        use_latency=False,
        use_jitter=False,
        use_reliability=False,
        use_salt=False,
        use_peak_response_rate=False,
        use_time_to_response=False,
        use_psth_profile=True,
        psth_use_pca=True,
        psth_pca_variance_explained=0.80,
    )
    feat_pca, _, _ = extract_feature_matrix(df, cfg_pca)
    assert feat_pca.shape[0] == 10
    assert 1 <= feat_pca.shape[1] < 50
    assert not np.isnan(feat_pca).any()


def test_compute_pca_embedding():
    """Verify 2D PCA projection coordinates and variance calculation."""
    df = _make_dummy_pooled_df(15)
    feat_m, _, _ = extract_feature_matrix(df, FeatureConfiguration(use_psth_profile=False))

    coords, exp_var = compute_pca_embedding(feat_m, n_components=2)
    assert coords.shape == (15, 2)
    assert len(exp_var) == 2
    assert 0.0 <= exp_var[0] <= 1.0
    assert 0.0 <= exp_var[1] <= 1.0
    assert sum(exp_var) <= 1.0


def test_hierarchical_clustering_k_clusters():
    """Verify agglomerative clustering partitions into exact requested k clusters."""
    df = _make_dummy_pooled_df(18)
    feat_m, _, _ = extract_feature_matrix(df, FeatureConfiguration(use_psth_profile=False))

    params = ClusteringParameters(n_clusters=3, metric="euclidean", linkage="ward")
    labels = run_hierarchical_clustering(feat_m, params)

    assert len(labels) == 18
    unique_clusters = np.unique(labels)
    assert len(unique_clusters) == 3
    assert set(unique_clusters) == {0, 1, 2}


def test_hierarchical_clustering_distance_threshold():
    """Verify agglomerative clustering partitions using distance threshold."""
    df = _make_dummy_pooled_df(15)
    feat_m, _, _ = extract_feature_matrix(df, FeatureConfiguration(use_psth_profile=False))

    params = ClusteringParameters(
        cutoff_mode="distance",
        distance_threshold=2.0,
        metric="euclidean",
        linkage="average",
    )
    labels = run_hierarchical_clustering(feat_m, params)
    assert len(labels) == 15
    assert len(np.unique(labels)) >= 1


def test_hierarchical_clustering_ward_enforces_euclidean():
    """Verify ward linkage with non-euclidean parameter gracefully runs without crash."""
    df = _make_dummy_pooled_df(10)
    feat_m, _, _ = extract_feature_matrix(df, FeatureConfiguration(use_psth_profile=False))

    params = ClusteringParameters(n_clusters=2, metric="cosine", linkage="ward")
    labels = run_hierarchical_clustering(feat_m, params)
    assert len(np.unique(labels)) == 2


def test_perform_clustering_end_to_end():
    """Verify complete perform_clustering pipeline returning ClusteringResult."""
    df = _make_dummy_pooled_df(12)
    params = ClusteringParameters(n_clusters=4)

    res = perform_clustering(df, params)
    assert isinstance(res, ClusteringResult)
    assert len(res.cluster_labels) == 12
    assert res.pca_coordinates.shape == (12, 2)
    assert len(res.cluster_summaries) == 4

    for cid, summary in res.cluster_summaries.items():
        assert isinstance(summary, ClusterSummary)
        assert summary.cluster_id == cid
        assert summary.count > 0
        assert len(summary.mean_psth) == len(summary.time_bins_ms)
        assert len(summary.sem_psth) == len(summary.time_bins_ms)
        assert summary.color_hex.startswith("#")


def test_hierarchical_clustering_dialog(qapp):
    """Verify HierarchicalClusteringDialog initialization, linkage constraints, and parameter extraction."""
    init_params = ClusteringParameters(n_clusters=5)
    dlg = HierarchicalClusteringDialog(current_params=init_params)

    assert dlg.spin_k.value() == 5
    assert dlg.radio_k.isChecked()

    # Changing linkage to ward should disable and set metric to euclidean
    dlg.combo_linkage.setCurrentText("complete")
    dlg.combo_metric.setCurrentText("cosine")
    assert dlg.combo_metric.currentText() == "cosine"

    dlg.combo_linkage.setCurrentText("ward")
    assert dlg.combo_metric.currentText() == "euclidean"
    assert not dlg.combo_metric.isEnabled()

    # Switch cutoff to distance threshold
    dlg.radio_dist.setChecked(True)
    dlg.spin_dist.setValue(2.25)
    assert not dlg.spin_k.isEnabled()
    assert dlg.spin_dist.isEnabled()

    extracted = dlg.get_parameters()
    assert extracted.cutoff_mode == "distance"
    assert extracted.distance_threshold == 2.25
    assert extracted.linkage == "ward"
    assert extracted.metric == "euclidean"


def test_clustering_view_cluster_card_widget(qapp):
    """Verify ClusterCardWidget renders title, confidence envelope, axes, and stimulus markers."""
    t_bins = np.linspace(-50, 50, 40)
    mean_y = np.exp(-((t_bins - 5.0) ** 2) / 20.0) * 50.0
    sem_y = mean_y * 0.15

    summary = ClusterSummary(
        cluster_id=0,
        count=14,
        color_hex="#2563EB",
        unit_ids=["u001", "u002"],
        time_bins_ms=t_bins,
        mean_psth=mean_y,
        sem_psth=sem_y,
    )
    card = ClusterCardWidget(
        cluster_id=0,
        count=14,
        color_hex="#2563EB",
        summary=summary,
        pulse_duration_ms=25.0,
    )
    assert card.header.text() == "Cluster 0"
    assert card.lbl_count.text() == "n = 14"

    # Visual sizing checks (§ 5.1 fixed card size)
    assert (card.width(), card.height()) == CLUSTER_CARD_SIZE

    # Axes visibility checks
    assert card.plot is not None
    assert card.plot.getAxis("bottom").isVisible()
    assert card.plot.getAxis("left").isVisible()
    assert "Time" in card.plot.getAxis("bottom").labelText
    assert card.plot.getAxis("left").labelText in ("Norm. rate", "Firing Rate (Hz)")

    # Also verify raw mode sets Firing Rate (Hz)
    card_raw = ClusterCardWidget(
        cluster_id=0,
        count=14,
        color_hex="#2563EB",
        summary=summary,
        pulse_duration_ms=25.0,
        norm_mode="raw",
    )
    assert "Firing Rate" in card_raw.plot.getAxis("left").labelText

    # Vertical onset and offset line checks
    inf_lines = [item for item in card.plot.plotItem.items if isinstance(item, pg.InfiniteLine)]
    assert len(inf_lines) >= 2
    line_vals = [line.value() for line in inf_lines]
    assert 0.0 in line_vals
    assert 25.0 in line_vals

    # Shaded stimulus region check
    regions = [item for item in card.plot.plotItem.items if isinstance(item, pg.LinearRegionItem)]
    assert len(regions) >= 1
    assert regions[0].getRegion() == (0.0, 25.0)


def test_cluster_card_widget_fallback_mock(qapp):
    """Verify fallback mock ClusterCardWidget without summary also displays axes and stimulus lines."""
    card = ClusterCardWidget(
        cluster_id=1,
        count=8,
        color_hex="#059669",
        summary=None,
        pulse_duration_ms=20.0,
    )
    assert (card.width(), card.height()) == CLUSTER_CARD_SIZE
    assert card.plot.getAxis("bottom").isVisible()
    assert card.plot.getAxis("left").isVisible()

    inf_lines = [item for item in card.plot.plotItem.items if isinstance(item, pg.InfiniteLine)]
    line_vals = [line.value() for line in inf_lines]
    assert 0.0 in line_vals
    assert 20.0 in line_vals


def test_clustering_view_four_column_card_grid(qapp):
    """Verify cluster cards in ClusteringView are arranged in a 4-column grid."""
    view = ClusteringView(load_mock=True)
    view.resize(1920, 1080)
    view.show()
    qapp.processEvents()
    params = ClusteringParameters(n_clusters=4)
    ok = view.run_clustering(params)
    assert ok is True
    assert len(view._cluster_cards) == 4
    assert view.clusters_grid.count() == 5  # 4 cluster cards + '+ New cluster' card

    # 4 cards arranged in 4 columns: (0,0), (0,1), (0,2), (0,3); '+ New cluster' at (1,0)
    card_0_0 = view.clusters_grid.itemAtPosition(0, 0).widget()
    card_0_1 = view.clusters_grid.itemAtPosition(0, 1).widget()
    card_0_2 = view.clusters_grid.itemAtPosition(0, 2).widget()
    card_0_3 = view.clusters_grid.itemAtPosition(0, 3).widget()
    assert isinstance(card_0_0, ClusterCardWidget)
    assert isinstance(card_0_1, ClusterCardWidget)
    assert isinstance(card_0_2, ClusterCardWidget)
    assert isinstance(card_0_3, ClusterCardWidget)
    assert view.clusters_grid.itemAtPosition(1, 0).widget() is view._new_cluster_card


def test_clustering_view_run_clustering_acceptance(qapp):
    """Acceptance Test: run k=4 clustering -> verify scatter colors and cluster cards."""
    view = ClusteringView(load_mock=True)
    assert view.table_pooled_units.rowCount() > 0

    params = ClusteringParameters(n_clusters=4)
    ok = view.run_clustering(params)

    assert ok is True
    assert view.current_clustering_result is not None
    assert len(view.current_clustering_result.cluster_summaries) == 4

    # Top grid has 4 cluster cards + '+ New cluster' card
    assert len(view._cluster_cards) == 4
    assert view.clusters_grid.count() == 5

    # 2D PCA scatter canvas has spots with cluster colors
    spots = view.scatter_item.points()
    assert len(spots) == view.table_pooled_units.rowCount()

    # Verify scatter coordinates map is populated
    assert len(view._scatter_points_map) == len(spots)


def test_clustering_view_point_and_card_selection_callbacks(qapp):
    """Verify clicking 2D scatter point or cluster card selects unit and updates preview."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    view.run_clustering(ClusteringParameters(n_clusters=3))

    first_gid = view._get_gid_for_row(0)

    # Simulate clicking on scatter point
    spots = view.scatter_item.points()
    assert len(spots) > 0
    target_spot = spots[0]
    view._on_scatter_point_clicked(view.scatter_item, [target_spot])

    # Verifies unit was selected and highlighted with red ring
    assert view.table_pooled_units.selectedItems() is not None
    assert view._current_selected_gid == str(target_spot.data())
    assert view.cand_scatter.data is not None

    # Simulate clicking on cluster card
    card_widget = view.clusters_grid.itemAt(0).widget()
    assert isinstance(card_widget, ClusterCardWidget)
    card_widget.card_clicked.emit(card_widget.cluster_id)

    # Unit from that cluster is selected and active in viewer
    assert view.table_pooled_units.currentRow() >= 0


def test_main_window_hierarchical_clustering_trigger(qapp):
    """Verify MainWindow modal hierarchical clustering action."""
    win = MainWindow()
    win.clustering_view._load_mock_data()
    win.switch_to_clustering()

    # Trigger run_clustering programmatically on view
    ok = win.clustering_view.run_clustering(ClusteringParameters(n_clusters=3))
    assert ok is True
    assert len(win.clustering_view._cluster_cards) == 3
    assert win.clustering_view.clusters_grid.count() == 4  # 3 cluster cards + '+ New cluster' card


def test_iteration5_acceptance_hotkey_assignment_and_advance(qapp, qtbot):
    """Iteration 5 Acceptance Test: 1-9 hotkey assignment, active learning advance, and drag-and-drop enhancements."""
    view = ClusteringView(start_with_clusters=True)
    assert view.current_clustering_result is not None
    assert len(view.current_clustering_result.cluster_summaries) == 4

    # Wait for initial active learning worker to surface active candidate
    qtbot.waitUntil(
        lambda: (view._active_learning_worker is None or not view._active_learning_worker.isRunning())
        and view.active_candidate is not None,
        timeout=10000,
    )
    assert view.active_candidate is not None
    initial_cand_gid = view.active_candidate.global_unit_id
    active_cids = sorted(view.current_clustering_result.cluster_summaries.keys())
    target_cluster = 1 if 1 in view.current_clustering_result.cluster_summaries else active_cids[0]

    # Send '1' KeyPress event
    key_event = QKeyEvent(
        QKeyEvent.Type.KeyPress,
        Qt.Key.Key_1,
        Qt.KeyboardModifier.NoModifier,
        "1",
    )
    view.keyPressEvent(key_event)

    # Unit should be assigned to first sorted active cluster
    assert view._pooled_units_map[initial_cand_gid]["cluster_id"] == target_cluster
    # Wait for active learning worker to finish and advance candidate
    qtbot.waitUntil(
        lambda: (view._active_learning_worker is None or not view._active_learning_worker.isRunning())
        and (view.active_candidate is None or view.active_candidate.global_unit_id != initial_cand_gid),
        timeout=10000,
    )
    if view.active_candidate is not None:
        assert view.active_candidate.global_unit_id != initial_cand_gid

    # Test QShortcut registration (Space, S, N + 1 shortcut per active cluster card)
    assert hasattr(view, "_shortcuts")
    assert len(view._shortcuts) == 3
    assert len(getattr(view, "_cluster_shortcuts", {})) == len(view._cluster_cards)
    assert len(view._shortcuts) + len(getattr(view, "_cluster_shortcuts", {})) >= 6

    # Test ClusterCardWidget dragMoveEvent and child mouse transparency
    card = ClusterCardWidget(cluster_id=target_cluster, count=5, color_hex="#2563EB")
    assert card.plot.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    class MockDragMoveEvent:
        def __init__(self, mime):
            self._mime = mime
            self.accepted = False

        def mimeData(self):
            return self._mime

        def acceptProposedAction(self):
            self.accepted = True

    mime = QMimeData()
    other_cid = 2 if target_cluster != 2 else 0
    mime.setData("application/x-cluster-id", str(other_cid).encode("utf-8"))
    move_event = MockDragMoveEvent(mime)
    card.dragMoveEvent(move_event)
    assert move_event.accepted is True
    view.close()


def test_clustering_view_dock_area_layout(qapp):
    """Verify WP3 splitter/panel layout initialization and layout reset."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    assert hasattr(view, "main_splitter")
    assert hasattr(view, "right_splitter")
    assert isinstance(view.main_splitter, ResettableSplitter)
    assert isinstance(view.right_splitter, ResettableSplitter)

    # Verify WP3 3-region main_splitter hierarchy
    assert hasattr(view, "unit_list_panel")
    assert hasattr(view, "centre_panel")
    assert hasattr(view, "focused_unit_panel")
    assert hasattr(view, "inspector_slot")
    assert hasattr(view, "right_panel")
    assert hasattr(view, "clusters_area_panel")
    assert hasattr(view, "list_panel")

    assert view.main_splitter.count() == 3
    assert view.main_splitter.widget(0) is view.unit_list_panel
    assert view.main_splitter.widget(1) is view.centre_panel
    assert view.main_splitter.widget(2) is view.right_panel
    assert view.focused_unit_panel.parent() is view.centre_panel
    assert view.inspector_slot.parent() is view.centre_panel

    assert view.right_splitter.count() == 2
    assert view.right_splitter.widget(0) is view.clusters_area_panel
    assert view.right_splitter.widget(1) is view.list_panel

    # Verify public widgets exist
    assert view.table_exps is not None
    assert view.table_pooled_units is not None
    assert view.pca_plot is not None
    assert view.scatter_item is not None
    assert view.cand_scatter is not None
    assert view.lbl_candidate is not None
    assert view.image_viewer is not None
    assert view.probe_plot is not None
    assert view.marginal_widget is not None
    assert view.btn_save_pool is not None
    assert view.btn_cluster is not None
    assert view.clusters_grid is not None

    # Verify reset_default_layout restores splitter defaults cleanly
    view.reset_default_layout()
    assert view.main_splitter.count() == 3
    assert view.right_splitter.count() == 2


def test_clustering_view_cluster_filtering(qapp):
    """Validate that combo_filter_cluster filters pooled units table by cluster_id in ClusteringView."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    assert view.current_clustering_result is not None
    assert len(view.current_clustering_result.cluster_summaries) == 4
    assert hasattr(view, "combo_filter_cluster")

    # In default mock state, clustering has run with 4 clusters
    assert view.combo_filter_cluster.count() >= 2  # "All units" + statuses + active clusters
    assert view.combo_filter_cluster.itemData(0) == -2
    assert "All units" in view.combo_filter_cluster.itemText(0)

    # All active clusters should be listed in combo_filter_cluster
    assert view.current_clustering_result is not None
    active_cids = sorted(view.current_clustering_result.cluster_summaries.keys())
    for cid in active_cids:
        found = False
        for idx in range(view.combo_filter_cluster.count()):
            if view.combo_filter_cluster.itemData(idx) == cid:
                found = True
                assert f"Cluster {cid}" in view.combo_filter_cluster.itemText(idx)
                break
        assert found, f"Cluster {cid} not found in combo_filter_cluster"

    total_rows = view.table_pooled_units.rowCount()
    assert total_rows > 0

    # Initially "All units" is selected -> all rows visible
    assert view.combo_filter_cluster.currentData() == -2
    for r in range(total_rows):
        assert not view.table_pooled_units.isRowHidden(r)
    assert f"{total_rows} units" in view.lbl_pooled_units_badge.text()

    # Select the first active cluster
    target_cid = active_cids[0]
    target_idx = view.combo_filter_cluster.findData(target_cid)
    assert target_idx >= 0
    view.combo_filter_cluster.setCurrentIndex(target_idx)

    visible_count = 0
    for r in range(total_rows):
        gid = view._get_gid_for_row(r)
        rec = view._pooled_units_map[gid]
        cid = rec.get("cluster_id", -1)
        if cid == target_cid:
            assert not view.table_pooled_units.isRowHidden(r)
            visible_count += 1
        else:
            assert view.table_pooled_units.isRowHidden(r)

    assert visible_count > 0
    if visible_count < total_rows:
        assert f"{visible_count} / {total_rows} units" in view.lbl_pooled_units_badge.text()

    # Switch back to "All units" -> unhides all rows
    view.combo_filter_cluster.setCurrentIndex(0)
    for r in range(total_rows):
        assert not view.table_pooled_units.isRowHidden(r)
    assert f"{total_rows} units" in view.lbl_pooled_units_badge.text()

    # Verify sorting preserves row hiddenness while cluster filter is active
    view.combo_filter_cluster.setCurrentIndex(target_idx)
    # Sort by column 3 (latency)
    view._on_units_header_clicked(3)
    for r in range(total_rows):
        gid = view._get_gid_for_row(r)
        rec = view._pooled_units_map[gid]
        cid = rec.get("cluster_id", -1)
        if cid == target_cid:
            assert not view.table_pooled_units.isRowHidden(r)
        else:
            assert view.table_pooled_units.isRowHidden(r)

    # Sort by column 0 (gid)
    view._on_units_header_clicked(0)
    for r in range(total_rows):
        gid = view._get_gid_for_row(r)
        rec = view._pooled_units_map[gid]
        cid = rec.get("cluster_id", -1)
        if cid == target_cid:
            assert not view.table_pooled_units.isRowHidden(r)
        else:
            assert view.table_pooled_units.isRowHidden(r)

    # Verify selection desync fix: if selected row becomes hidden, selection shifts to visible row
    other_cid = [c for c in active_cids if c != target_cid][0]
    other_idx = [i for i in range(view.combo_filter_cluster.count()) if view.combo_filter_cluster.itemData(i) == other_cid][0]
    view.combo_filter_cluster.setCurrentIndex(other_idx)

    curr_selected = view.table_pooled_units.currentRow()
    assert curr_selected >= 0
    assert not view.table_pooled_units.isRowHidden(curr_selected)
    gid = view._get_gid_for_row(curr_selected)
    assert view._pooled_units_map[gid]["cluster_id"] == other_cid

    view.close()


def test_clustering_view_cluster_column_header_and_colors(qapp):
    """Verify column 6 header is 'Cluster', cells display cluster text & color, and unassigned is gray."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    assert view.current_clustering_result is not None
    assert len(view.current_clustering_result.cluster_summaries) == 4

    # 1. Header label for column 6 is "Cluster"
    header_item = view.table_pooled_units.horizontalHeaderItem(2)
    assert header_item is not None
    assert header_item.text() == "Cluster"

    # 2. Units in the table display their assigned "Cluster {cid}" with the appropriate cluster color
    assert view.current_clustering_result is not None
    total_rows = view.table_pooled_units.rowCount()
    assert total_rows > 0

    for r in range(total_rows):
        gid = view._get_gid_for_row(r)
        rec = view._pooled_units_map[gid]
        cid = rec.get("cluster_id", -1)
        cluster_item = view.table_pooled_units.item(r, 2)
        assert cluster_item is not None
        assert cluster_item.text() == f"Cluster {cid}"
        assert cluster_item.sort_key == cid
        expected_color = view.current_clustering_result.cluster_summaries[cid].color_hex
        assert cluster_item.foreground().color().name().upper() == text_color_for_cluster(expected_color).upper()

    # 3. Test Unassigned state: gray (#64748B), text "—", tooltip "Unassigned", sort_key -1
    first_gid = view._get_gid_for_row(0)
    view._pooled_units_map[first_gid]["cluster_id"] = -1
    view._update_cluster_column_for_row(0, -1)
    unassigned_item = view.table_pooled_units.item(0, 2)
    assert unassigned_item.text() == "—"
    assert unassigned_item.toolTip() == "Unassigned"
    assert unassigned_item.sort_key == -1
    assert unassigned_item.foreground().color().name().upper() == COLOR_TEXT_MUTED.upper()

    # 4. Test numeric sorting on column 6
    view._update_all_pooled_cluster_items()
    view.table_pooled_units.sortItems(2, Qt.SortOrder.AscendingOrder)
    sort_keys_asc = [view.table_pooled_units.item(r, 2).sort_key for r in range(total_rows)]
    assert sort_keys_asc == sorted(sort_keys_asc)
    assert sort_keys_asc[0] == -1

    view.table_pooled_units.sortItems(2, Qt.SortOrder.DescendingOrder)
    sort_keys_desc = [view.table_pooled_units.item(r, 2).sort_key for r in range(total_rows)]
    assert sort_keys_desc == sorted(sort_keys_desc, reverse=True)
    assert sort_keys_desc[-1] == -1

    # Also verify _on_units_header_clicked toggles and sorts column 6 numerically
    view._on_units_header_clicked(2)
    order = view.table_pooled_units.horizontalHeader().sortIndicatorOrder()
    current_keys = [view.table_pooled_units.item(r, 2).sort_key for r in range(total_rows)]
    expected = sorted(current_keys, reverse=(order == Qt.SortOrder.DescendingOrder))
    assert current_keys == expected

    view.close()


def test_clustering_view_card_click_autoselects_filter_and_representative(qapp):
    """Verify clicking a cluster card toggles side-by-side comparison pane, and combo filters table."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    assert view.current_clustering_result is not None
    assert len(view.current_clustering_result.cluster_summaries) == 4

    active_cids = sorted(view.current_clustering_result.cluster_summaries.keys())
    assert len(active_cids) >= 2
    target_cid = active_cids[1]

    # Find the card widget for target_cid
    target_card = None
    for i in range(view.clusters_grid.count()):
        w = view.clusters_grid.itemAt(i).widget()
        if isinstance(w, ClusterCardWidget) and w.cluster_id == target_cid:
            target_card = w
            break
    assert target_card is not None

    # Simulate card click -> opens side-by-side comparison
    target_card.card_clicked.emit(target_cid)
    assert view.comparison_pane.isVisible() is True
    assert view._comparison_cluster_id == target_cid
    assert target_card._is_comparing is True
    assert len(view.comparison_viewers) > 0

    # Second click toggles comparison off
    target_card.card_clicked.emit(target_cid)
    assert view.comparison_pane.isVisible() is False
    assert view._comparison_cluster_id is None
    assert target_card._is_comparing is False

    # Verify combo_filter_cluster direct selection filters the table
    target_idx = -1
    for idx in range(view.combo_filter_cluster.count()):
        if view.combo_filter_cluster.itemData(idx) == target_cid:
            target_idx = idx
            break
    assert target_idx >= 0
    view.combo_filter_cluster.setCurrentIndex(target_idx)

    # Verify table_pooled_units only displays cells from target_cid
    total_rows = view.table_pooled_units.rowCount()
    visible_count = 0
    for r in range(total_rows):
        gid = view._get_gid_for_row(r)
        rec = view._pooled_units_map[gid]
        cid = rec.get("cluster_id", -1)
        if cid == target_cid:
            assert not view.table_pooled_units.isRowHidden(r)
            visible_count += 1
        else:
            assert view.table_pooled_units.isRowHidden(r)
    assert visible_count > 0

    view.close()


def test_clustering_view_cluster_column_updates_on_merge_and_assignment(qapp):
    """Verify column 6 updates dynamically when merging clusters and assigning candidates."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    assert view.current_clustering_result is not None
    assert len(view.current_clustering_result.cluster_summaries) == 4
    assert view.current_clustering_result is not None

    active_cids = sorted(view.current_clustering_result.cluster_summaries.keys())
    src_id, tgt_id = active_cids[0], active_cids[1]

    # Get GIDs in src_id before merge
    src_gids = set(view.current_clustering_result.cluster_summaries[src_id].unit_ids)
    assert len(src_gids) > 0

    # Merge clusters without modal dialog
    ok = view.merge_clusters(src_id, tgt_id, confirm=False)
    assert ok is True

    # Verify all units that belonged to src_id now have column 6 as f"Cluster {tgt_id}"
    tgt_color = view.current_clustering_result.cluster_summaries[tgt_id].color_hex
    for r in range(view.table_pooled_units.rowCount()):
        gid = view._get_gid_for_row(r)
        if gid in src_gids:
            item = view.table_pooled_units.item(r, 2)
            assert item.text() == f"Cluster {tgt_id}"
            assert item.sort_key == tgt_id
            assert item.foreground().color().name().upper() == text_color_for_cluster(tgt_color).upper()

    # Test candidate assignment
    cand = view.active_candidate
    if cand is not None:
        cand_gid = cand.global_unit_id
        assign_ok = view.assign_candidate_to_cluster(tgt_id)
        assert assign_ok is True

        for r in range(view.table_pooled_units.rowCount()):
            gid = view._get_gid_for_row(r)
            if gid == cand_gid:
                item = view.table_pooled_units.item(r, 2)
                assert item.text() == f"Cluster {tgt_id}"
                assert item.sort_key == tgt_id
                assert item.foreground().color().name().upper() == text_color_for_cluster(tgt_color).upper()
                break

    view.close()


def test_clustering_view_move_cluster_cards(qapp, monkeypatch):
    """Verify cluster card swapping via move_cluster_card and 4-column layout."""
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)

    view = ClusteringView(start_with_clusters=True, load_mock=True)
    view.resize(1920, 1080)
    qapp.processEvents()
    assert view.current_clustering_result is not None
    active_cids = sorted(view.current_clustering_result.cluster_summaries.keys())
    assert len(active_cids) >= 3

    # 1. Initial card order matches active clusters in 4-column grid
    assert view._cluster_card_order == active_cids
    for idx, cid in enumerate(view._cluster_card_order):
        row = idx // 4
        col = idx % 4
        item = view.clusters_grid.itemAtPosition(row, col)
        assert item is not None
        card = item.widget()
        assert isinstance(card, ClusterCardWidget)
        assert card.cluster_id == cid
        assert view._cluster_cards[cid] is card

    # 2. Swap cards by moving src_id to tgt_id
    src_id = active_cids[-1]
    tgt_id = active_cids[0]

    moves_received = []
    view.cluster_moved.connect(lambda s, t: moves_received.append((s, t)))

    ok = view.move_cluster_card(src_id, tgt_id)
    assert ok is True
    assert moves_received == [(src_id, tgt_id)]

    # view._cluster_card_order swapped directly
    assert view._cluster_card_order[0] == src_id
    assert view._cluster_card_order[-1] == tgt_id

    # Card at position (0, 0) is src_id
    top_left_item = view.clusters_grid.itemAtPosition(0, 0)
    assert top_left_item is not None
    assert top_left_item.widget().cluster_id == src_id

    # Card at last position is tgt_id
    last_idx = len(view._cluster_card_order) - 1
    last_item = view.clusters_grid.itemAtPosition(last_idx // 4, last_idx % 4)
    assert last_item is not None
    assert last_item.widget().cluster_id == tgt_id

    # Verify no-op on invalid moves
    assert view.move_cluster_card(src_id, src_id) is False
    assert view.move_cluster_card(9999, tgt_id) is False

    # 3. Merging clusters preserves order of remaining clusters and removes merged cluster
    order_before_merge = list(view._cluster_card_order)
    merge_src = order_before_merge[0]  # src_id
    merge_tgt = order_before_merge[1]

    ok_merge = view.merge_clusters(merge_src, merge_tgt, confirm=False)
    assert ok_merge is True

    # Merged cluster is removed from order
    assert merge_src not in view._cluster_card_order
    assert merge_tgt in view._cluster_card_order

    # Remaining clusters maintain their relative order
    expected_remaining = [c for c in order_before_merge if c != merge_src]
    assert view._cluster_card_order == expected_remaining

    # Grid contains updated cards matching new order in 4 columns
    for idx, cid in enumerate(view._cluster_card_order):
        row = idx // 4
        col = idx % 4
        item = view.clusters_grid.itemAtPosition(row, col)
        assert item is not None
        assert item.widget().cluster_id == cid

    view.close()


def test_customize_clusters_dialog(qapp, monkeypatch):
    """Verify CustomizeClustersDialog renaming, color changing, reordering, and reset to defaults."""
    initial_order = [0, 1, 2]
    initial_names = {0: "Cluster 0", 1: "Cluster 1", 2: "Cluster 2"}
    initial_colors = {0: "#2563EB", 1: "#10B981", 2: "#F59E0B"}
    initial_counts = {0: 10, 1: 15, 2: 8}

    dlg = CustomizeClustersDialog(
        cluster_order=initial_order,
        cluster_names=initial_names,
        cluster_colors=initial_colors,
        cluster_counts=initial_counts,
    )

    # 1. Verify rows created and initial values populated
    assert dlg._order == [0, 1, 2]
    assert len(dlg._rows) == 3
    assert dlg._rows[0].edit_name.text() == "Cluster 0"
    assert dlg._rows[0].color_hex == "#2563EB"
    assert "(N=10)" in dlg._rows[0].lbl_count.text()

    # 2. Rename cluster 0
    dlg.set_cluster_name(0, "Fast Onset")
    assert dlg._rows[0].edit_name.text() == "Fast Onset"

    # 3. Change color of cluster 1
    dlg.set_cluster_color(1, "#8B5CF6")
    assert dlg._rows[1].color_hex == "#8B5CF6"

    # 4. Reorder: move cluster 2 up
    # Initial: [0, 1, 2] -> move up 2 -> [0, 2, 1]
    ok_up = dlg.move_cluster_up(2)
    assert ok_up is True
    assert dlg._order == [0, 2, 1]

    # Move down cluster 0 -> [2, 0, 1]
    ok_down = dlg.move_cluster_down(0)
    assert ok_down is True
    assert dlg._order == [2, 0, 1]

    # Boundary moves return False
    assert dlg.move_cluster_up(2) is False   # already at top (index 0)
    assert dlg.move_cluster_down(1) is False # already at bottom (index 2)

    # 5. Extract customizations
    ordered_cids, names_dict, colors_dict, hotkeys_dict = dlg.get_customizations()
    assert ordered_cids == [2, 0, 1]
    assert names_dict[0] == "Fast Onset"
    assert names_dict[2] == "Cluster 2"
    assert colors_dict[1] == "#8B5CF6"
    assert colors_dict[0] == "#2563EB"
    assert hotkeys_dict[0] == "0"
    assert hotkeys_dict[1] == "1"
    assert hotkeys_dict[2] == "2"

    # 6. Reset to Defaults
    dlg.reset_to_defaults()
    assert dlg._order == [0, 1, 2]
    assert dlg._rows[0].edit_name.text() == "Cluster 0"
    assert dlg._rows[1].edit_name.text() == "Cluster 1"
    assert dlg._rows[2].edit_name.text() == "Cluster 2"
    assert dlg._rows[0].color_hex == CLUSTER_PALETTE[0]

    dlg.close()


def test_clustering_view_apply_cluster_customizations(qapp):
    """Verify apply_cluster_customizations updates cards, table column 6, PCA scatter spots, and combobox."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    view.resize(1920, 1080)
    qapp.processEvents()
    assert view.current_clustering_result is not None
    assert len(view.current_clustering_result.cluster_summaries) == 4
    assert view.current_clustering_result is not None
    assert hasattr(view, "btn_customize_clusters")
    assert view.btn_customize_clusters.text() == "Customize clusters"

    active_cids = sorted(view.current_clustering_result.cluster_summaries.keys())
    assert len(active_cids) >= 2
    cid_a, cid_b = active_cids[0], active_cids[1]

    # Custom configuration
    new_order = list(reversed(active_cids))
    new_names = {cid_a: "Inhibitory Fast", cid_b: "Excitatory Delayed"}
    new_colors = {cid_a: "#EC4899", cid_b: "#14B8A6"}

    # Apply customizations
    view.apply_cluster_customizations(new_order, new_names, new_colors)

    # 1. Verify card order and grid arrangement
    assert view._cluster_card_order == new_order
    for idx, cid in enumerate(new_order):
        row = idx // 4
        col = idx % 4
        item = view.clusters_grid.itemAtPosition(row, col)
        assert item is not None
        card = item.widget()
        assert isinstance(card, ClusterCardWidget)
        assert card.cluster_id == cid

    # 2. Verify card display name and color updates (header may be elided; tooltip holds full name)
    card_a = view._cluster_cards[cid_a]
    assert card_a.display_name == "Inhibitory Fast"
    assert card_a.header.toolTip() == "Inhibitory Fast"
    assert card_a.color_hex == "#EC4899"

    card_b = view._cluster_cards[cid_b]
    assert card_b.display_name == "Excitatory Delayed"
    assert card_b.header.toolTip() == "Excitatory Delayed"
    assert card_b.color_hex == "#14B8A6"

    # 3. Verify table column 6 updates with custom name and color
    for r in range(view.table_pooled_units.rowCount()):
        gid = view._get_gid_for_row(r)
        rec = view._pooled_units_map[gid]
        cid = rec.get("cluster_id", -1)
        item_col6 = view.table_pooled_units.item(r, 2)
        if cid == cid_a:
            assert item_col6.text() == "Inhibitory Fast"
            assert item_col6.foreground().color().name().upper() == text_color_for_cluster("#EC4899").upper()
        elif cid == cid_b:
            assert item_col6.text() == "Excitatory Delayed"
            assert item_col6.foreground().color().name().upper() == text_color_for_cluster("#14B8A6").upper()

    # 4. Verify combo_filter_cluster has custom names
    combo_texts = [view.combo_filter_cluster.itemText(i) for i in range(view.combo_filter_cluster.count())]
    assert any("Inhibitory Fast" in t for t in combo_texts)
    assert any("Excitatory Delayed" in t for t in combo_texts)

    # 5. Verify PCA scatter plot spots are colored with the new colors
    spots = view.scatter_item.points()
    assert len(spots) > 0
    for spot in spots:
        gid = spot.data()
        if gid in view._pooled_units_map:
            cid = view._pooled_units_map[gid].get("cluster_id")
            spot_color = spot.brush().color().name().upper()
            if cid == cid_a:
                assert spot_color == "#EC4899"
            elif cid == cid_b:
                assert spot_color == "#14B8A6"

    # 6. Verify cluster_summaries color_hex is updated
    assert view.current_clustering_result.cluster_summaries[cid_a].color_hex == "#EC4899"
    assert view.current_clustering_result.cluster_summaries[cid_b].color_hex == "#14B8A6"

    view.close()


def test_reassign_selected_cell_with_filter_and_hotkey(qapp):
    """Verify that selecting a filtered cell and triggering a cluster hotkey reassigns that cell.

    1. Have ClusteringView in mock state with 4 clusters.
    2. Filter combo_filter_cluster to Cluster 1.
    3. Select the first visible row in the table (which belongs to Cluster 1).
    4. Note initial counts of Cluster 1 and Cluster 3.
    5. Trigger hotkey for Cluster 3 (or call assign_candidate_to_cluster(3) with row selected).
    6. Assert:
       - Cluster 1 count decreased by 1.
       - Cluster 3 count increased by 1.
       - That moved unit is now hidden in the table (filtered out of Cluster 1 list).
       - Selection shifted to the next visible row in Cluster 1.
       - The moved unit's cluster in _pooled_units_map is 3.
    """
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    assert view.current_clustering_result is not None
    assert len(view.current_clustering_result.cluster_summaries) == 4

    # 2. Filter combo_filter_cluster to Cluster 1
    idx_c1 = -1
    for i in range(view.combo_filter_cluster.count()):
        if view.combo_filter_cluster.itemData(i) == 1:
            idx_c1 = i
            break
    assert idx_c1 >= 0
    view.combo_filter_cluster.setCurrentIndex(idx_c1)
    assert view.combo_filter_cluster.currentData() == 1

    # 3. Select the first visible row in the table (which belongs to Cluster 1)
    first_visible_row = -1
    for r in range(view.table_pooled_units.rowCount()):
        if not view.table_pooled_units.isRowHidden(r):
            first_visible_row = r
            break
    assert first_visible_row >= 0
    view.table_pooled_units.selectRow(first_visible_row)
    view._on_pooled_unit_selected()

    selected_gid = view._get_gid_for_row(first_visible_row)
    assert view._current_selected_gid == selected_gid
    assert view._pooled_units_map[selected_gid]["cluster_id"] == 1

    # 4. Note initial counts of Cluster 1 and Cluster 3
    init_count_1 = view.current_clustering_result.cluster_summaries[1].count
    init_count_3 = view.current_clustering_result.cluster_summaries[3].count
    assert init_count_1 >= 2, "Cluster 1 must have at least 2 units for next visible shift verification"

    # 5. Trigger hotkey for Cluster 3
    key_event = QKeyEvent(
        QKeyEvent.Type.KeyPress,
        Qt.Key.Key_3,
        Qt.KeyboardModifier.NoModifier,
        "3",
    )
    view.keyPressEvent(key_event)

    # 6. Assertions
    # - Cluster 1 count decreased by 1
    assert view.current_clustering_result.cluster_summaries[1].count == init_count_1 - 1
    # - Cluster 3 count increased by 1
    assert view.current_clustering_result.cluster_summaries[3].count == init_count_3 + 1
    # - That moved unit is now hidden in the table (filtered out of Cluster 1 list)
    assert view.table_pooled_units.isRowHidden(first_visible_row)
    # - Selection shifted to the next visible row in Cluster 1
    new_selected_row = view.table_pooled_units.currentRow()
    assert new_selected_row >= 0
    assert new_selected_row != first_visible_row
    assert not view.table_pooled_units.isRowHidden(new_selected_row)
    new_gid = view._get_gid_for_row(new_selected_row)
    assert view._current_selected_gid == new_gid
    assert view._pooled_units_map[new_gid]["cluster_id"] == 1
    # - The moved unit's cluster in _pooled_units_map is 3
    assert view._pooled_units_map[selected_gid]["cluster_id"] == 3

    view.close()


def test_cluster_hotkey_0_assigns_to_cluster_0(qapp):
    """Verify that pressing hotkey '0' (Qt.Key.Key_0) assigns selected cell to Cluster 0."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    assert view.current_clustering_result is not None
    assert 0 in view.current_clustering_result.cluster_summaries

    # Find a unit not initially in Cluster 0
    target_row = -1
    for r in range(view.table_pooled_units.rowCount()):
        gid = view._get_gid_for_row(r)
        if view._pooled_units_map[gid]["cluster_id"] != 0:
            target_row = r
            break
    assert target_row >= 0

    view.table_pooled_units.selectRow(target_row)
    view._on_pooled_unit_selected()
    selected_gid = view._get_gid_for_row(target_row)
    assert view._current_selected_gid == selected_gid
    assert view._pooled_units_map[selected_gid]["cluster_id"] != 0

    init_c0_count = view.current_clustering_result.cluster_summaries[0].count

    # Press hotkey '0'
    key_event = QKeyEvent(
        QKeyEvent.Type.KeyPress,
        Qt.Key.Key_0,
        Qt.KeyboardModifier.NoModifier,
        "0",
    )
    view.keyPressEvent(key_event)

    # Unit reassigned to Cluster 0
    assert view._pooled_units_map[selected_gid]["cluster_id"] == 0
    assert view.current_clustering_result.cluster_summaries[0].count == init_c0_count + 1

    view.close()


def test_reassign_last_cell_in_filtered_cluster_clears_selection(qapp):
    """Verify that moving the last remaining cell in a filtered cluster cleanly clears table selection."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    assert view.current_clustering_result is not None

    summaries = view.current_clustering_result.cluster_summaries
    single_unit_cid = None
    for cid, summ in summaries.items():
        if summ.count == 1:
            single_unit_cid = cid
            break

    # If no cluster has count 1, reassign units until one cluster has exactly 1 unit
    if single_unit_cid is None:
        single_unit_cid = list(summaries.keys())[0]
        units = list(summaries[single_unit_cid].unit_ids)
        tgt_cid = list(summaries.keys())[1]
        for u in units[1:]:
            view.assign_candidate_to_cluster(tgt_cid, unit_id=u)

    assert view.current_clustering_result.cluster_summaries[single_unit_cid].count == 1

    # Filter combo_filter_cluster to single_unit_cid
    idx = view.combo_filter_cluster.findData(single_unit_cid)
    assert idx >= 0
    view.combo_filter_cluster.setCurrentIndex(idx)
    assert view.combo_filter_cluster.currentData() == single_unit_cid

    # Select the only visible row
    visible_row = -1
    for r in range(view.table_pooled_units.rowCount()):
        if not view.table_pooled_units.isRowHidden(r):
            visible_row = r
            break
    assert visible_row >= 0
    view.table_pooled_units.selectRow(visible_row)
    view._on_pooled_unit_selected()

    last_gid = view._get_gid_for_row(visible_row)
    assert view._current_selected_gid == last_gid

    # Move this last cell to another cluster
    other_cid = [c for c in summaries.keys() if c != single_unit_cid][0]
    view.assign_candidate_to_cluster(other_cid)

    # Selection should be cleanly cleared:
    assert view.table_pooled_units.selectionModel().selectedRows() == []
    assert view._current_selected_gid is None

    view.close()


def test_hotkey_fallback_to_active_candidate_when_selection_cleared(qapp, qtbot):
    """Verify that table_pooled_units.clearSelection() causes hotkey assignment to fall back to active_candidate."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    assert view.current_clustering_result is not None
    qtbot.waitUntil(
        lambda: (view._active_learning_worker is None or not view._active_learning_worker.isRunning())
        and view.active_candidate is not None,
        timeout=10000,
    )
    assert view.active_candidate is not None

    cand_gid = view.active_candidate.global_unit_id

    # Select a different unit in table first
    table_gids = [view._get_gid_for_row(r) for r in range(view.table_pooled_units.rowCount())]
    other_gids = [g for g in table_gids if g != cand_gid]
    assert len(other_gids) > 0
    other_gid = other_gids[0]

    view.select_pooled_unit(other_gid)
    assert view._current_selected_gid == other_gid

    # Explicitly clear selection on table
    view.table_pooled_units.clearSelection()
    view._on_pooled_unit_selected()
    assert view._current_selected_gid is None
    assert len(view.table_pooled_units.selectionModel().selectedRows()) == 0

    # Ensure active candidate is still available
    assert view.active_candidate is not None
    active_cand_gid = view.active_candidate.global_unit_id

    # Find a target cluster different from active candidate's current cluster
    curr_cand_cid = view._pooled_units_map[active_cand_gid].get("cluster_id", -1)
    summaries = view.current_clustering_result.cluster_summaries
    target_cid = [c for c in summaries.keys() if c != curr_cand_cid][0]

    # Trigger hotkey assignment
    view.assign_candidate_to_cluster(target_cid)

    # Active candidate should be updated to target_cid
    assert view._pooled_units_map[active_cand_gid]["cluster_id"] == target_cid

    qtbot.waitUntil(
        lambda: (view._active_learning_worker is None or not view._active_learning_worker.isRunning())
        and (view.active_candidate is None or view.active_candidate.global_unit_id != active_cand_gid),
        timeout=10000,
    )
    view.close()


def test_reset_clustering_button_and_reset_functionality(qapp, monkeypatch):
    """Verify Reset Clustering button attributes and complete reset of units, statuses, and cards."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    assert view.current_clustering_result is not None
    assert len(view.current_clustering_result.cluster_summaries) == 4

    # 1. Verify button existence, text, and tooltip
    assert hasattr(view, "btn_reset_clusters")
    assert "reset clustering" in view.btn_reset_clusters.text().lower()
    assert "Reset" in view.btn_reset_clusters.toolTip()
    assert view.btn_reset_clusters.objectName() == "resetClustersButton"

    # 2. Verify initial clustered state
    assert view.current_clustering_result is not None
    assert len(view._cluster_cards) > 0
    assigned_count = sum(
        1 for rec in view._pooled_units_map.values() if rec.get("cluster_id", -1) >= 0
    )
    assert assigned_count > 0

    # Populate skipped units to verify they get cleared on reset
    sample_gid = list(view._pooled_units_map.keys())[0]
    view._skipped_gids.add(sample_gid)
    assert len(view._skipped_gids) > 0

    # 3. Test cancellation via confirmation dialog
    from optocurator.ui.dialog_clustering_settings import ResetClusteringDialog
    monkeypatch.setattr(ResetClusteringDialog, "exec", lambda self: ResetClusteringDialog.DialogCode.Rejected)
    cancelled = view.reset_clustering(confirm=True)
    assert cancelled is False
    assert len(view._skipped_gids) > 0
    assert len(view._cluster_cards) > 0

    # 4. Test calling reset_clustering(confirm=False)
    res = view.reset_clustering(confirm=False)
    assert res is True

    # Verify all units in _pooled_units_map are reset to cluster_id = -1
    for gid, rec in view._pooled_units_map.items():
        assert rec["cluster_id"] == -1
        assert view._unit_status_map[gid] == "Unassigned"

    # Verify all pooled rows are reset
    for r in view._pooled_rows:
        assert r["cluster_id"] == -1

    # Verify cluster cards and card order are cleared
    assert len(view._cluster_cards) == 0
    assert len(view._cluster_card_order) == 0
    assert view.clusters_grid.count() == 0

    # Verify skipped units are cleared
    assert len(view._skipped_gids) == 0

    # 5. Test clicking btn_reset_clusters directly
    # Assign candidate to cluster 0 first
    view.assign_candidate_to_cluster(0, unit_id=sample_gid)
    assert view._pooled_units_map[sample_gid]["cluster_id"] == 0
    assert view._unit_status_map[sample_gid] == "Curated"
    view._skipped_gids.add(sample_gid)
    assert len(view._skipped_gids) > 0

    monkeypatch.setattr(ResetClusteringDialog, "exec", lambda self: ResetClusteringDialog.DialogCode.Accepted)
    monkeypatch.setattr(ResetClusteringDialog, "reset_mode", lambda self: "everything")
    view.btn_reset_clusters.click()

    # Verify units, statuses, cards, and skipped units are reset
    for gid, rec in view._pooled_units_map.items():
        assert rec["cluster_id"] == -1
        assert view._unit_status_map[gid] == "Unassigned"
    assert len(view._cluster_cards) == 0
    assert len(view._cluster_card_order) == 0
    assert len(view._skipped_gids) == 0

    view.close()


def test_clustering_view_default_clean_unassigned_state(qapp):
    """Verify that ClusteringView starts in 0-cluster unassigned active learning state by default."""
    view = ClusteringView(load_mock=True)

    assert view._start_with_clusters is False
    assert len(view._cluster_cards) == 0
    assert len(view._cluster_card_order) == 0
    assert view.clusters_grid.count() == 0

    total_rows = view.table_pooled_units.rowCount()
    assert total_rows > 0

    # All units should be unassigned (-1)
    for gid, rec in view._pooled_units_map.items():
        assert rec.get("cluster_id") == -1
        assert view._unit_status_map.get(gid) == "Unassigned"

    for r in range(total_rows):
        cluster_item = view.table_pooled_units.item(r, 2)
        assert cluster_item.text() == "—"
        assert cluster_item.toolTip() == "Unassigned"
        status_item = view.table_pooled_units.item(r, 6)
        assert status_item.text() == ""
        assert status_item.toolTip() == "Unassigned"

    # Filter combo contains "All units" and the 4 status items
    assert view.combo_filter_cluster.count() == 5
    assert view.combo_filter_cluster.itemData(0) == -2
    assert view.combo_filter_cluster.itemData(1) == "status:Unassigned"

    view.close()


def test_add_experiment_folder_progress_dialog_and_zero_cluster_loading(qapp, tmp_path):
    """Verify add_experiment_folder progress dialog and zero-cluster loading when start_with_clusters=False."""
    exp_dir = tmp_path / "01942_74R_20261019"
    exp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = exp_dir / "01942_74R_20261019_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_data = pd.DataFrame([
        {
            "global_unit_id": "01942_74R_20261019_u0001",
            "session_id": "01942_74R_20261019",
            "unit_id": 1,
            "latency_ms": 3.4,
            "jitter_ms": 1.2,
            "reliability": 0.92,
            "waveform_correlation": 0.95,
            "salt_p_value": 0.0008,
            "is_tagged": True,
            "depth": 850.0,
        },
        {
            "global_unit_id": "01942_74R_20261019_u0002",
            "session_id": "01942_74R_20261019",
            "unit_id": 2,
            "latency_ms": 7.5,
            "jitter_ms": 2.1,
            "reliability": 0.81,
            "waveform_correlation": 0.91,
            "salt_p_value": 0.0025,
            "is_tagged": True,
            "depth": 1150.0,
        },
    ])
    csv_file = exp_dir / "01942_74R_20261019_opto_curated.csv"
    csv_data.to_csv(csv_file, index=False)

    view = ClusteringView(start_with_clusters=False)
    assert view._start_with_clusters is False

    ok = view.add_experiment_folder(exp_dir)
    assert ok is True

    # Pooled units should be populated
    assert view.table_pooled_units.rowCount() == 2
    for gid, rec in view._pooled_units_map.items():
        assert rec["cluster_id"] == -1
        assert view._unit_status_map[gid] == "Unassigned"

    # Zero clusters initially
    assert len(view._cluster_cards) == 0
    assert len(view._cluster_card_order) == 0

    view.close()


def test_generate_psth_matrix_with_real_spikes():
    """Verify generate_psth_matrix computes real PSTH when spike_data_map is provided."""
    from optocurator.analysis.clustering import generate_psth_matrix
    from optocurator.analysis.metrics import compute_psth_and_raster

    df = pd.DataFrame([{
        "global_unit_id": "sess1_u0027",
        "session_id": "sess1",
        "unit_id": 27,
        "latency_ms": 10.0,
        "jitter_ms": 2.0,
        "reliability": 0.8,
        "is_tagged": True,
    }])

    # Synthetic pulses at 10.0s, 20.0s, 30.0s
    pulses = np.array([10.0, 20.0, 30.0])
    # Spikes around pulses
    spikes = np.array([10.005, 10.010, 20.006, 30.007, 30.012])
    spike_map = {"sess1_u0027": (spikes, pulses)}
    cache = {}

    psth_mat, t_bins = generate_psth_matrix(
        df,
        time_window_ms=(-30.0, 50.0),
        bin_size_ms=2.0,
        spike_data_map=spike_map,
        psth_cache=cache,
    )
    expected_res = compute_psth_and_raster(spikes, pulses, window_ms=(-30.0, 50.0), bin_size_ms=2.0)
    np.testing.assert_allclose(psth_mat[0, :], expected_res["psth_rates"])

    # Verify result was stored in cache
    cache_key = ("sess1_u0027", (-30.0, 50.0), 2.0, True, 2.0)
    assert cache_key in cache
    np.testing.assert_allclose(cache[cache_key], expected_res["psth_rates"])


def test_feature_settings_change_does_not_alter_presented_psth_window(qapp):
    """Verify that changing feature configuration does NOT change the presented cluster card time window."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    assert view.current_window_ms == (-30.0, 50.0)
    assert view.edit_win_start.text() == "-30"
    assert view.edit_win_end.text() == "+50"

    # Verify initial cluster cards have time bins matching (-30.0, 50.0) at default 0.5 ms bin width
    c0_summary = view.current_clustering_result.cluster_summaries[0]
    assert c0_summary.time_bins_ms[0] == pytest.approx(-29.75)
    assert c0_summary.time_bins_ms[-1] == pytest.approx(49.75)

    # Change feature configuration time window (e.g. to [-10, +25])
    new_cfg = FeatureConfiguration(
        use_latency=True,
        use_jitter=False,
        use_psth_profile=True,
        time_window_ms=(-10.0, 25.0),
        bin_size_ms=1.0,
    )
    view.feature_config = new_cfg
    view.active_clustering_params.feature_config = new_cfg
    view.refresh_active_learning()
    view._recompute_and_render_clusters()

    # The presented window must remain unchanged!
    assert view.current_window_ms == (-30.0, 50.0)
    assert view.edit_win_start.text() == "-30"
    assert view.edit_win_end.text() == "+50"
    c0_updated = view.current_clustering_result.cluster_summaries[0]
    assert c0_updated.time_bins_ms[0] == pytest.approx(-29.75)
    assert c0_updated.time_bins_ms[-1] == pytest.approx(49.75)

    view.close()


def test_set_visualized_window_updates_cluster_card_psths_and_text_boxes(qapp):
    """Verify set_visualized_window updates text boxes, cluster card PSTH waveforms, and emits signal."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)

    emitted_windows = []
    view.view_settings_changed.connect(lambda w: emitted_windows.append(w))

    # Set new presented window (-50, +100)
    view.set_visualized_window((-50.0, 100.0))

    assert view.current_window_ms == (-50.0, 100.0)
    assert view.edit_win_start.text() == "-50"
    assert view.edit_win_end.text() == "+100"
    assert len(emitted_windows) == 1
    assert emitted_windows[0] == (-50.0, 100.0)

    # Cluster cards must reflect the new time bins (default 0.5 ms bin width)
    c0 = view.current_clustering_result.cluster_summaries[0]
    assert c0.time_bins_ms[0] == pytest.approx(-49.75)
    assert c0.time_bins_ms[-1] == pytest.approx(99.75)

    # Test editing text boxes directly
    view.edit_win_start.setText("-20")
    view.edit_win_end.setText("+40")
    view._on_window_text_edited()

    assert view.current_window_ms == (-20.0, 40.0)
    c0_edited = view.current_clustering_result.cluster_summaries[0]
    assert c0_edited.time_bins_ms[0] == pytest.approx(-19.75)
    assert c0_edited.time_bins_ms[-1] == pytest.approx(39.75)

    # Test invalid text reverts to current_window_ms
    view.edit_win_start.setText("invalid")
    view._on_window_text_edited()
    assert view.edit_win_start.text() == "-20"
    assert view.current_window_ms == (-20.0, 40.0)

    # Test start >= end reverts
    view.edit_win_start.setText("+60")
    view.edit_win_end.setText("+40")
    view._on_window_text_edited()
    assert view.current_window_ms == (-20.0, 40.0)

    view.close()


def test_set_cluster_bin_size_updates_cluster_card_psths(qapp):
    """Verify default bin size is 0.5 ms, updating to 5.0 ms updates time_bins_ms resolution and cluster cards."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    assert view.current_bin_size_ms == pytest.approx(0.5)
    assert view.spin_bin_size.value() == pytest.approx(0.5)

    # With default window (-30, 50) and bin size 0.5 ms: 80 ms / 0.5 ms = 160 bins
    c0 = view.current_clustering_result.cluster_summaries[0]
    assert len(c0.time_bins_ms) == 160
    assert np.isclose(c0.time_bins_ms[1] - c0.time_bins_ms[0], 0.5)

    # Change bin size to 5.0 ms
    view.set_cluster_bin_size(5.0)
    assert view.current_bin_size_ms == 5.0
    assert view.spin_bin_size.value() == 5.0

    # 80 ms / 5 ms = 16 bins
    c0_new = view.current_clustering_result.cluster_summaries[0]
    assert len(c0_new.time_bins_ms) == 16
    assert np.isclose(c0_new.time_bins_ms[1] - c0_new.time_bins_ms[0], 5.0)

    # Verify cluster card plot has items rendered using stepMode
    card_0 = view._cluster_cards[0]
    assert card_0.plot is not None

    view.close()


def test_clustering_view_add_experiment_folder_loads_spikes_npz(tmp_path, qapp):
    """Verify add_experiment_folder loads <session_id>_spikes.npz into _experiment_spikes and stimulus."""
    session_id = "test_sess_01"
    session_dir = tmp_path / session_id
    session_dir.mkdir()

    # Create valid Phase 1 export CSV (<session_id>_opto_curated.csv)
    df_curated = pd.DataFrame([
        {
            "global_unit_id": f"{session_id}_u10",
            "session_id": session_id,
            "unit_id": 10,
            "latency_ms": 6.5,
            "jitter_ms": 1.2,
            "reliability": 0.95,
            "waveform_correlation": 0.98,
            "salt_p_value": 0.001,
            "is_tagged": True,
            "cluster_id": -1,
            "curated_label": "unassigned",
            "plot_path": "",
        },
        {
            "global_unit_id": f"{session_id}_u20",
            "session_id": session_id,
            "unit_id": 20,
            "latency_ms": 12.0,
            "jitter_ms": 2.1,
            "reliability": 0.85,
            "waveform_correlation": 0.92,
            "salt_p_value": 0.002,
            "is_tagged": True,
            "cluster_id": -1,
            "curated_label": "unassigned",
            "plot_path": "",
        },
    ])
    df_curated.to_csv(session_dir / f"{session_id}_opto_curated.csv", index=False)

    # Save real spike arrays and stimulus in <session_id>_spikes.npz
    spikes_u10 = np.array([1.005, 2.005, 3.005], dtype=np.float64)
    spikes_u20 = np.array([1.010, 2.010, 3.010], dtype=np.float64)
    pulse_onsets = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    np.savez_compressed(
        session_dir / f"{session_id}_spikes.npz",
        unit_10=spikes_u10,
        unit_20=spikes_u20,
        pulse_onsets=pulse_onsets,
    )

    view = ClusteringView(start_with_clusters=False)
    # Clear mock rows
    view.clear_all_experiments()

    view.add_experiment_folder(str(session_dir))
    assert session_id in view.loaded_experiments
    assert session_id in view._experiment_spikes
    assert session_id in view._experiment_stimulus

    # Check loaded spikes
    assert 10 in view._experiment_spikes[session_id]
    assert 20 in view._experiment_spikes[session_id]
    np.testing.assert_array_equal(view._experiment_spikes[session_id][10], spikes_u10)
    np.testing.assert_array_equal(view._experiment_spikes[session_id][20], spikes_u20)
    np.testing.assert_array_equal(view._experiment_stimulus[session_id], pulse_onsets)

    # Spike data map should contain both units
    df_active = view._get_active_pooled_dataframe()
    assert len(df_active) == 2
    spike_map = view._get_spike_data_map(df_active)
    gid_10 = f"{session_id}_u10"
    gid_20 = f"{session_id}_u20"
    assert gid_10 in spike_map
    assert gid_20 in spike_map

    # Run clustering and verify real PSTH computation
    view.run_clustering()
    assert view.current_clustering_result is not None
    # Verify cached PSTHs exist
    cached_keys = [k for k in view._unit_psth_cache.keys() if k[0] in (gid_10, gid_20)]
    assert len(cached_keys) >= 2
    assert len({k[0] for k in cached_keys}) == 2

    # Remove experiment and verify cleanup
    view.remove_selected_experiment(session_id)
    assert session_id not in view.loaded_experiments
    assert session_id not in view._experiment_spikes
    assert session_id not in view._experiment_stimulus

    view.close()


def test_cell_list_columns_exact_order_and_count(qapp):
    """Verify Phase 2 cell list has exactly 7 columns in precise order and excludes 'Global ID'."""
    view = ClusteringView(load_mock=True)

    # 1. Column count is exactly 7
    assert view.table_pooled_units.columnCount() == 7

    # 2. Expected column headers in exact order
    expected_headers = ["Unit", "Session", "Cluster", "Lat", "Rel.", "SALT p", "Status"]
    actual_headers = [
        view.table_pooled_units.horizontalHeaderItem(c).text()
        for c in range(view.table_pooled_units.columnCount())
    ]
    assert actual_headers == expected_headers

    # 3. Explicit check for each column index
    assert view.table_pooled_units.horizontalHeaderItem(0).text() == "Unit"
    assert view.table_pooled_units.horizontalHeaderItem(1).text() == "Session"
    assert view.table_pooled_units.horizontalHeaderItem(2).text() == "Cluster"
    assert view.table_pooled_units.horizontalHeaderItem(3).text() == "Lat"
    assert view.table_pooled_units.horizontalHeaderItem(4).text() == "Rel."
    assert view.table_pooled_units.horizontalHeaderItem(5).text() == "SALT p"
    assert view.table_pooled_units.horizontalHeaderItem(6).text() == "Status"

    # 4. "Global ID" is removed from visible columns
    assert "Global ID" not in actual_headers
    assert "GID" not in actual_headers

    view.close()


def test_cell_list_gid_retrieved_via_user_role(qapp):
    """Verify 'Global ID' is stored in UserRole and retrieved correctly via _get_gid_for_row()."""
    view = ClusteringView(load_mock=True)
    assert view.table_pooled_units.rowCount() > 0

    # Verify for each row that GID is stored in column 0 UserRole
    for row in range(view.table_pooled_units.rowCount()):
        sess_item = view.table_pooled_units.item(row, 0)
        assert sess_item is not None
        gid_from_role = sess_item.data(Qt.ItemDataRole.UserRole)
        assert gid_from_role is not None
        assert str(gid_from_role).strip() != ""

        # view._get_gid_for_row(row) must retrieve this exact GID
        retrieved_gid = view._get_gid_for_row(row)
        assert retrieved_gid == str(gid_from_role)

        # GID must match the record in _pooled_units_map
        assert retrieved_gid in view._pooled_units_map
        rec = view._pooled_units_map[retrieved_gid]
        assert rec["global_unit_id"] == retrieved_gid
        assert rec["session_id"] == view.table_pooled_units.item(row, 1).toolTip()
        assert str(rec["unit_id"]) == sess_item.text()

        # None of the visible cell texts should equal the full GID (e.g. '01942_74R_20261019_u0001')
        for col in range(view.table_pooled_units.columnCount()):
            cell_item = view.table_pooled_units.item(row, col)
            if cell_item:
                assert cell_item.text() != retrieved_gid

    # Test sorting maintains correct GID lookup via UserRole
    # Sort by Latency column (index 3) descending
    view.table_pooled_units.sortItems(3, Qt.SortOrder.DescendingOrder)
    for row in range(view.table_pooled_units.rowCount()):
        gid = view._get_gid_for_row(row)
        sess_item = view.table_pooled_units.item(row, 0)
        assert sess_item.data(Qt.ItemDataRole.UserRole) == gid
        assert gid in view._pooled_units_map

    # Test out-of-range row handling
    assert view._get_gid_for_row(-1) == ""
    assert view._get_gid_for_row(9999) == ""

    view.close()


def test_cell_list_horizontal_resizing_and_splitter_freedom(qapp):
    """Verify all 7 columns have ResizeMode.Interactive and clusters_container minimum size permits free splitter movement."""
    view = ClusteringView(load_mock=True)

    # 1. All 7 columns have ResizeMode.Interactive
    header = view.table_pooled_units.horizontalHeader()
    for col in range(7):
        mode = header.sectionResizeMode(col)
        assert mode == QHeaderView.ResizeMode.Interactive, f"Column {col} does not have Interactive resize mode"

    # 2. Horizontal scrollbar is disabled (compact columns + text elision fit cleanly)
    assert view.table_pooled_units.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff

    # 3. Verify clusters_container minimum size matches § 5.1 one full card row (164 px)
    assert hasattr(view, "clusters_container")
    min_size = view.clusters_container.minimumSize()
    assert min_size.width() <= 300, f"clusters_container minimum width {min_size.width()} is too large"
    assert min_size.height() == CLUSTER_GRID_MIN_HEIGHT

    # Also verify via clusters_area_panel
    assert view.clusters_area_panel.minimumHeight() == CLUSTER_GRID_MIN_HEIGHT

    view.close()


def test_active_learning_toggle_on_off(qapp):
    """Verify Active Learning toggle button updates state, UI styling, and sequential curation behavior."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)

    # 1. Initial state is ON
    assert view.active_learning_enabled is True
    assert "Active Learning: ON" in view.btn_toggle_active_learning.text()

    # 2. Toggle button clicked to turn OFF
    view.btn_toggle_active_learning.click()
    assert view.active_learning_enabled is False
    assert "Active Learning: OFF" in view.btn_toggle_active_learning.text()

    # 3. In OFF mode, hotkey assignment assigns selected row and sequentially advances to next row
    view.table_pooled_units.clearSelection()
    view.table_pooled_units.selectRow(0)
    gid_row_0 = view._get_gid_for_row(0)
    gid_row_1 = view._get_gid_for_row(1)

    view._on_cluster_hotkey(0)

    # Row 0 unit is now Curated into cluster 0
    assert view._unit_status_map[gid_row_0] == "Curated"
    assert view._pooled_units_map[gid_row_0]["cluster_id"] == 0

    # Sequential advance selected row 1
    curr_selected = view.table_pooled_units.currentRow()
    assert curr_selected == 1
    assert view._get_gid_for_row(curr_selected) == gid_row_1

    # 4. Programmatic toggle back ON
    view.set_active_learning_mode(True)
    assert view.active_learning_enabled is True
    assert "Active Learning: ON" in view.btn_toggle_active_learning.text()

    view.close()


def test_return_to_suggested_candidate_button(qapp, qtbot):
    """Verify '🎯 Suggested Cell' button is enabled when looking at another unit and returns selection."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    qtbot.waitUntil(
        lambda: (view._active_learning_worker is None or not view._active_learning_worker.isRunning())
        and view.active_candidate is not None,
        timeout=10000,
    )
    assert view.active_candidate is not None
    top_cand_gid = view.active_candidate.global_unit_id

    # Find a row different from top candidate
    other_row = -1
    for r in range(view.table_pooled_units.rowCount()):
        if view._get_gid_for_row(r) != top_cand_gid:
            other_row = r
            break
    assert other_row >= 0
    other_gid = view._get_gid_for_row(other_row)

    # Manually select other row
    view.table_pooled_units.clearSelection()
    view.table_pooled_units.selectRow(other_row)
    assert view._current_selected_gid == other_gid
    assert view._current_selected_gid != top_cand_gid

    # Button is enabled
    assert view.btn_return_to_candidate.isEnabled() is True

    # Click Suggested Cell button
    view.btn_return_to_candidate.click()

    # Verify selection returned to candidate
    assert view._current_selected_gid == top_cand_gid
    assert view.table_pooled_units.currentRow() >= 0
    assert view._get_gid_for_row(view.table_pooled_units.currentRow()) == top_cand_gid

    view.close()


def test_cluster_card_click_scrollable_comparison(qapp):
    """Verify clicking cluster card opens side-by-side comparison pane, displays members, and assign button."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    assert view.comparison_pane.isHidden()

    target_cid = 0
    card = view._cluster_cards[target_cid]

    # Click card to open comparison pane
    card.card_clicked.emit(target_cid)
    assert not view.comparison_pane.isHidden()
    assert view._comparison_cluster_id == target_cid
    assert card._is_comparing is True
    assert "Comparing: Cluster 0" in view.lbl_comparison_title.text()

    # Member viewers populated
    member_count = sum(1 for rec in view._pooled_units_map.values() if rec.get("cluster_id") == target_cid)
    assert len(view.comparison_viewers) == member_count
    for viewer in view.comparison_viewers:
        assert hasattr(viewer, "unit_response_plot") and isinstance(viewer.unit_response_plot, UnitResponsePlot) and viewer.unit_response_plot.size() == view.unit_response_plot.size()

    # Test "✓ Assign Here" button assigns active unit
    cand_gid = view.active_candidate.global_unit_id if view.active_candidate else view._get_gid_for_row(0)
    view.select_pooled_unit(cand_gid)
    view.btn_comp_assign.click()
    assert view._pooled_units_map[cand_gid]["cluster_id"] == target_cid

    # Test close button
    view.btn_comp_close.click()
    assert view.comparison_pane.isHidden()
    assert view._comparison_cluster_id is None
    assert view._cluster_cards[target_cid]._is_comparing is False

    view.close()


def test_marginal_distributions_gray_population_and_cluster_color_overlay(qapp):
    """Verify MarginalDistributionsWidget has gray population curve and cluster color comparison overlay."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    marginal = view.marginal_widget

    # Population curve styling
    for metric_name, curve in marginal.pop_curves.items():
        assert curve is not None
        # Pen color is light gray (#94A3B8)
        color = curve.opts["pen"].color()
        assert color.name().upper() == COLOR_POPULATION_CURVE.upper()

    # Y-range normalized to (0.0, 1.2)
    for plot in marginal.plots.values():
        yrange = plot.viewRange()[1]
        assert yrange[0] == pytest.approx(0.0, abs=0.05)
        assert yrange[1] == pytest.approx(1.2, abs=0.05)

    # Overlay cluster distribution
    target_cid = 1
    cluster_records = [rec for rec in view._pooled_units_map.values() if rec.get("cluster_id") == target_cid]
    cluster_df = pd.DataFrame(cluster_records)
    cluster_color = "#059669"

    marginal.update_cluster_distribution(cluster_df, cluster_color)
    assert len(marginal.cluster_curves) == 4
    for curve in marginal.cluster_curves.values():
        c_color = curve.opts["pen"].color()
        assert c_color.red() == 0x05 and c_color.green() == 0x96 and c_color.blue() == 0x69
        assert curve.xData is not None and len(curve.xData) > 0

    # Clear cluster distribution
    marginal.clear_cluster_distribution()
    for curve in marginal.cluster_curves.values():
        assert curve.xData is None or len(curve.xData) == 0

    view.close()


def test_cluster_cards_candidate_psth_light_gray_overlay(qapp):
    """Verify ClusterCardWidget contains candidate_psth_curve in thin light-gray with stepMode='center'."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)

    t_bins = np.linspace(-30, 50, 41)
    psth_data = np.random.uniform(5.0, 30.0, len(t_bins))

    for card in view._cluster_cards.values():
        assert hasattr(card, "candidate_psth_curve")
        pen_col = card.candidate_psth_curve.opts["pen"].color()
        assert pen_col.name().upper() == "#1F2328"
        assert card.candidate_psth_curve.opts["pen"].widthF() == pytest.approx(1.2, abs=0.1)

        # Set candidate PSTH
        card.set_candidate_psth(t_bins, psth_data)
        x_data, y_data = card.candidate_psth_curve.getData()
        if getattr(card, "is_smooth", False):
            assert len(x_data) == len(t_bins)
        else:
            assert len(x_data) == len(t_bins) + 1  # stepMode="center" requires N+1 edges
        assert len(y_data) == len(psth_data)

    # Test selecting pooled unit updates candidate curve across cards
    first_gid = view._get_gid_for_row(0)
    view.select_pooled_unit(first_gid)

    for card in view._cluster_cards.values():
        x_data, y_data = card.candidate_psth_curve.getData()
        assert x_data is not None
        assert len(y_data) > 0

    view.close()


def test_compute_psth_time_ticks():
    """Verify compute_psth_time_ticks guarantees 5.0 and 10.0 ms ticks within range."""
    # Standard window [-30, 50]
    ticks = compute_psth_time_ticks(-30.0, 50.0)
    assert 0.0 in ticks
    assert 5.0 in ticks
    assert 10.0 in ticks
    assert 20.0 in ticks
    assert -20.0 in ticks
    assert ticks == sorted(ticks)

    # Window [-10, 20]
    ticks_narrow = compute_psth_time_ticks(-10.0, 20.0)
    assert 5.0 in ticks_narrow
    assert 10.0 in ticks_narrow
    assert 0.0 in ticks_narrow

    # Window [15, 60] (5 and 10 are outside range)
    ticks_post = compute_psth_time_ticks(15.0, 60.0)
    assert 5.0 not in ticks_post
    assert 10.0 not in ticks_post
    assert 20.0 in ticks_post


def test_format_unit_display_helper():
    """Verify _format_unit_display properly formats session and unit strings."""
    # Normal explicit session and unit
    s, u = _format_unit_display("session_123", 45, "session_123_u0045")
    assert s == "session_123"
    assert u == "45"

    # Inferred from global_unit_id with _u
    s, u = _format_unit_display(None, None, "test_OptoCurator_u0594")
    assert s == "test_OptoCurator"
    assert u == "594"

    # Inferred from global_unit_id with underscore number
    s, u = _format_unit_display(None, None, "mouseA_sess1_0042")
    assert s == "mouseA_sess1"
    assert u == "42"

    # Empty inputs fallback gracefully
    s, u = _format_unit_display(None, None, None)
    assert s == "Unknown"
    assert u == "—"


def test_phase2_ui_cleanups_clustering_view(qapp, qtbot):
    """Integration test verifying all 5 UI cleanup deliverables in ClusteringView."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    qtbot.waitUntil(
        lambda: (view._active_learning_worker is None or not view._active_learning_worker.isRunning())
        and view.active_candidate is not None
        and "Entropy: H=" in view.lbl_candidate.text(),
        timeout=10000,
    )

    # Deliverable 1: Unit/Session Info Formatting & Deduplication
    # Left image viewer title is hidden so it doesn't duplicate candidate header
    assert isinstance(view.unit_response_plot, UnitResponsePlot)
    assert not view.unit_response_plot.raster_plot.getPlotItem().titleLabel.text
    assert not view.unit_response_plot.psth_plot.getPlotItem().titleLabel.text

    # Candidate label format contains separated Session and Unit info, not raw concatenated string
    cand_text = view.lbl_candidate.text()
    assert "Session:" in cand_text
    assert "Unit:" in cand_text
    assert "Cluster:" in cand_text
    assert "Entropy: H=" in cand_text

    # Deliverable 2: "Tagged Candidate" banner is permanently removed
    assert not hasattr(view.unit_response_plot, "lbl_badge")
    assert not hasattr(view.unit_response_plot, "lbl_meta")
    assert "Status:" not in view.lbl_candidate.text()

    # Deliverable 3: ClusterProbabilityBarWidget layout & labels (compact bottom axis without redundant title)
    assert view.cluster_prob_bar.plot_widget.height() == PLOT_CLUSTER_PROB_SIZE[1]
    bottom_axis = view.cluster_prob_bar.plot_widget.getAxis("bottom")
    assert bottom_axis.labelText == ""

    # Deliverable 4: PSTH time ticks guarantee 5 and 10 ms
    for cid, card in view._cluster_cards.items():
        card_axis = card.plot.getAxis("bottom")
        ticks_spec = card_axis._tickLevels
        if ticks_spec and len(ticks_spec) > 0 and len(ticks_spec[0]) > 0:
            tick_vals = [val for val, label in ticks_spec[0]]
            assert 5.0 in tick_vals
            assert 10.0 in tick_vals

    # Deliverable 5: Curation Shortcuts Help Button and Dialog
    assert hasattr(view, "btn_shortcuts_help")
    assert view.btn_shortcuts_help.text() == "?"
    assert "Curation Keyboard Shortcuts & Help" in view.btn_shortcuts_help.toolTip()

    # Help dialog is properly constructed and styled
    help_dialog = view.show_curation_shortcuts_help()
    assert help_dialog is not None
    assert help_dialog.windowTitle() == "Curation Shortcuts & Workflow Help"
    help_dialog.close()

    # Member comparison viewers have visible titles
    card_0 = view._cluster_cards[0]
    card_0.card_clicked.emit(0)
    for comp_viewer in view.comparison_viewers:
        assert isinstance(comp_viewer.unit_response_plot, UnitResponsePlot)
        assert comp_viewer.unit_response_plot.size() == view.unit_response_plot.size()
        assert any("Unit " in lbl.text() for lbl in comp_viewer.findChildren(QLabel) if not lbl.isHidden())
        assert any("ms" in lbl.text() or "—" in lbl.text() for lbl in comp_viewer.findChildren(QLabel) if not lbl.isHidden())

    view.btn_comp_close.click()
    view.close()


def test_dynamic_array_export_and_validation(tmp_path, qapp):
    """Verify DetectionView.export_curated_session creates arrays_dir, saves spikes npz, and skips PNGs by default."""
    view = DetectionView(load_mock=True)
    export_dest = tmp_path / "sess_export"
    csv_path = view.export_curated_session(
        export_dir=export_dest,
        show_dialog=False,
        only_tagged_plots=True,
        precompute_pngs=False,
    )
    assert csv_path is not None
    assert csv_path.is_file()

    # arrays_dir created and contains spikes npz
    arrays_dir = export_dest / f"{view.session_id}_arrays"
    assert arrays_dir.is_dir()
    spikes_npz = arrays_dir / f"{view.session_id}_spikes.npz"
    assert spikes_npz.is_file()

    # Also saved at root for backward compatibility
    root_npz = export_dest / f"{view.session_id}_spikes.npz"
    assert root_npz.is_file()

    # Verify content of spikes npz
    with np.load(spikes_npz) as data:
        assert "pulse_onsets" in data
        unit_keys = [k for k in data.files if k.startswith("unit_")]
        assert len(unit_keys) > 0

    # plots_dir exists but no PNGs because precompute_pngs was False
    plots_dir = export_dest / f"{view.session_id}_plots"
    assert plots_dir.is_dir()
    assert len(list(plots_dir.glob("*.png"))) == 0

    # validate_phase1_export parses arrays_dir
    meta = validate_phase1_export(export_dest)
    assert meta.arrays_dir == arrays_dir

    view.close()


def test_clustering_view_view_configuration_and_dynamic_inspection(qapp):
    """Verify ClusteringView dynamic array live rendering, pulse ordering, and marker size view config."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)

    # Initial view configuration defaults
    assert view.raster_order_by == "presentation"
    assert view.raster_marker_size == 3.0
    assert hasattr(view, "combo_order")
    assert view.combo_order.currentData() == "presentation"

    # Main inspection viewer has live plot active
    assert not view.unit_response_plot.raster_plot.isHidden() and not view.unit_response_plot.psth_plot.isHidden()
    assert view.unit_response_plot.raster_plot.getPlotItem()._raster_cache["scatter"].isVisible() is True
    assert view.unit_response_plot.psth_plot.getPlotItem()._psth_cache["smooth_curve"].isVisible() is True
    assert isinstance(view.unit_response_plot, UnitResponsePlot)

    # Test changing pulse ordering
    view.set_pulse_ordering("latency")
    assert view.raster_order_by == "latency"
    assert view.combo_order.currentData() == "latency"
    assert not view.unit_response_plot.raster_plot.isHidden() and not view.unit_response_plot.psth_plot.isHidden()
    assert view.unit_response_plot.raster_plot.getPlotItem()._raster_cache["scatter"].isVisible() is True
    assert view.unit_response_plot.psth_plot.getPlotItem()._psth_cache["smooth_curve"].isVisible() is True

    # Test changing raster marker size
    view.set_raster_marker_size(5.0)
    assert view.raster_marker_size == 5.0

    # Test set_visualized_window re-renders inspected unit
    view.set_visualized_window((-40.0, 60.0))
    assert view.current_window_ms == (-40.0, 60.0)
    assert not view.unit_response_plot.raster_plot.isHidden() and not view.unit_response_plot.psth_plot.isHidden()
    assert view.unit_response_plot.raster_plot.getPlotItem()._raster_cache["scatter"].isVisible() is True
    assert view.unit_response_plot.psth_plot.getPlotItem()._psth_cache["smooth_curve"].isVisible() is True

    # Test set_cluster_bin_size re-renders inspected unit
    view.set_cluster_bin_size(4.0)
    assert view.current_bin_size_ms == 4.0
    assert not view.unit_response_plot.raster_plot.isHidden() and not view.unit_response_plot.psth_plot.isHidden()
    assert view.unit_response_plot.raster_plot.getPlotItem()._raster_cache["scatter"].isVisible() is True
    assert view.unit_response_plot.psth_plot.getPlotItem()._psth_cache["smooth_curve"].isVisible() is True

    # Test cluster comparison pane live viewers
    view.set_cluster_comparison(0)
    assert len(view.comparison_viewers) > 0
    for cv in view.comparison_viewers:
        assert isinstance(cv.unit_response_plot, UnitResponsePlot)
        assert cv.unit_response_plot.size() == view.unit_response_plot.size()
        assert not cv.unit_response_plot.raster_plot.isHidden()
        assert not cv.unit_response_plot.psth_plot.isHidden()
        assert cv.unit_response_plot.raster_plot.getPlotItem()._raster_cache["scatter"].isVisible() is True

    view.clear_cluster_comparison()
    view.close()


def test_background_prewarm_and_compact_comparison_viewers(qapp):
    """Verify background idle pre-warming and compact comparison cards."""
    import time
    from PyQt6.QtCore import QThreadPool

    view = ClusteringView(start_with_clusters=True, load_mock=True)

    # 1. Background pre-warming was triggered
    # Allow background QThreadPool worker and 10 ms QTimer prewarm slice to populate _member_pixmap_cache
    QThreadPool.globalInstance().waitForDone(2000)
    for _ in range(15):
        qapp.processEvents()
        if len(view._member_pixmap_cache) > 0:
            break
        time.sleep(0.015)

    # Prewarm queue should have rendered member pixmap items
    assert len(view._member_pixmap_cache) > 0

    # 2. Open cluster comparison pane -> uses _MemberRowFrame with lazy UnitResponsePlot promotion
    view.set_cluster_comparison(0)
    assert len(view.comparison_viewers) > 0
    for comp_viewer in view.comparison_viewers:
        assert isinstance(comp_viewer.unit_response_plot, UnitResponsePlot)
        assert comp_viewer.unit_response_plot.size() == view.unit_response_plot.size()
        assert not comp_viewer.unit_response_plot.raster_plot.isHidden() and not comp_viewer.unit_response_plot.psth_plot.isHidden()
        assert comp_viewer.unit_response_plot.raster_plot.getPlotItem()._raster_cache["scatter"].isVisible() is True
        assert comp_viewer.unit_response_plot.psth_plot.getPlotItem()._psth_cache["smooth_curve"].isVisible() is True

    # 3. Changing view configuration invalidates / updates member pixmap cache
    view.set_visualized_window((-50.0, 70.0))
    qapp.processEvents()

    # New cache keys created for new window in _member_pixmap_cache
    new_keys = [k for k in view._member_pixmap_cache.keys() if k[1] == (-50.0, 70.0)]
    assert len(new_keys) > 0

    view.clear_cluster_comparison()
    view.close()


def test_inspection_horizontal_splitter_resizing_and_toggling(qapp):
    """Verify WP3 inspector_slot toggles between empty state and comparison_pane with fixed 300+300 px widths."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    view.resize(1400, 900)
    view.show()
    qapp.processEvents()

    # 1. Fixed 300 px + 300 px centre panels exist
    assert hasattr(view, "focused_unit_panel")
    assert hasattr(view, "inspector_slot")
    assert hasattr(view, "inspector_empty_state")
    assert hasattr(view, "comparison_pane")
    assert view.focused_unit_panel.width() == FOCUSED_UNIT_WIDTH
    assert view.inspector_slot.width() == INSPECTOR_SLOT_WIDTH

    # 2. Initially inspector_empty_state is visible and comparison_pane is hidden
    assert not view.inspector_empty_state.isHidden()
    assert view.comparison_pane.isHidden()

    # 3. Opening cluster comparison hides empty state, shows comparison_pane, and preserves widths
    view.set_cluster_comparison(0)
    qapp.processEvents()
    assert view.inspector_empty_state.isHidden()
    assert not view.comparison_pane.isHidden()
    assert view.focused_unit_panel.width() == FOCUSED_UNIT_WIDTH
    assert view.inspector_slot.width() == INSPECTOR_SLOT_WIDTH

    # 4. Closing comparison pane restores empty state and preserves widths
    view.clear_cluster_comparison()
    qapp.processEvents()
    assert view.comparison_pane.isHidden()
    assert not view.inspector_empty_state.isHidden()
    assert view.focused_unit_panel.width() == FOCUSED_UNIT_WIDTH
    assert view.inspector_slot.width() == INSPECTOR_SLOT_WIDTH

    view.close()


def test_feature_configuration_weights_and_normalization():
    """Verify FeatureConfiguration scalar metric toggles, per-feature weights, and PSTH normalization modes."""
    df = _make_dummy_pooled_df(12)

    # 1. Default FeatureConfiguration
    cfg_default = FeatureConfiguration()
    assert cfg_default.use_latency is True
    assert cfg_default.weight_latency == 5.0
    assert cfg_default.use_jitter is True
    assert cfg_default.use_reliability is True
    assert cfg_default.use_salt is False
    assert cfg_default.salt_log10_transform is True
    assert cfg_default.use_baseline_rate is False
    assert cfg_default.use_peak_rate is False
    assert cfg_default.use_peak_response_rate is True
    assert cfg_default.use_time_to_response is True
    assert cfg_default.response_std_threshold == 2.0
    assert cfg_default.use_psth_profile is True
    assert cfg_default.psth_normalization == "area_normalized"
    assert cfg_default.time_window_ms == (0.0, 25.0)
    assert cfg_default.bin_size_ms == pytest.approx(0.5)
    assert cfg_default.psth_use_pca is True
    assert cfg_default.psth_pca_variance_explained == pytest.approx(0.80)

    # Default extraction: 5 scalar metrics + PCA-compressed PSTH components (< 12 bins)
    feat_default, psth_default, _ = extract_feature_matrix(df, feature_config=cfg_default)
    assert feat_default.shape[0] == 12
    assert 5 < feat_default.shape[1] < 5 + psth_default.shape[1]

    # Without PSTH PCA: 5 scalar metrics + 50 PSTH bins (0.5 ms default) = 55 features
    cfg_no_pca = FeatureConfiguration(psth_use_pca=False)
    feat_no_pca, _, _ = extract_feature_matrix(df, feature_config=cfg_no_pca)
    assert feat_no_pca.shape == (12, 55)

    # 2. Enable all 8 scalar features (without PSTH profile)
    cfg_all_scalars = FeatureConfiguration(
        use_latency=True,
        use_jitter=True,
        use_reliability=True,
        use_salt=True,
        use_baseline_rate=True,
        use_peak_rate=True,
        use_peak_response_rate=True,
        use_time_to_response=True,
        use_psth_profile=False,
    )
    feat_all_scalars, _, _ = extract_feature_matrix(df, feature_config=cfg_all_scalars)
    assert feat_all_scalars.shape == (12, 8)
    assert not np.isnan(feat_all_scalars).any()

    # 3. Weights effect
    cfg_weighted = FeatureConfiguration(
        use_latency=True,
        use_jitter=False,
        use_reliability=False,
        use_salt=False,
        use_peak_response_rate=False,
        use_time_to_response=False,
        use_psth_profile=False,
        weight_latency=3.5,
    )
    feat_weighted, _, _ = extract_feature_matrix(df, feature_config=cfg_weighted)
    assert feat_weighted.shape == (12, 1)
    # Scaled feature with weight 3.5 has standard deviation close to 3.5
    assert np.isclose(float(np.std(feat_weighted[:, 0])), 3.5, atol=0.2)

    # 4. -log10(SALT) transform vs raw p-value
    cfg_raw_salt = FeatureConfiguration(
        use_latency=False,
        use_jitter=False,
        use_reliability=False,
        use_salt=True,
        salt_log10_transform=False,
        use_peak_response_rate=False,
        use_time_to_response=False,
        use_psth_profile=False,
    )
    cfg_log_salt = FeatureConfiguration(
        use_latency=False,
        use_jitter=False,
        use_reliability=False,
        use_salt=True,
        salt_log10_transform=True,
        use_peak_response_rate=False,
        use_time_to_response=False,
        use_psth_profile=False,
    )
    feat_raw, _, _ = extract_feature_matrix(df, feature_config=cfg_raw_salt)
    feat_log, _, _ = extract_feature_matrix(df, feature_config=cfg_log_salt)
    assert feat_raw.shape == (12, 1)
    assert feat_log.shape == (12, 1)
    assert not np.allclose(feat_raw, feat_log)

    # 5. Supported PSTH normalization modes
    norm_modes = ["area_normalized", "peak_normalized", "raw"]
    for mode in norm_modes:
        cfg_norm = FeatureConfiguration(
            use_latency=False,
            use_jitter=False,
            use_reliability=False,
            use_salt=False,
            use_peak_response_rate=False,
            use_time_to_response=False,
            use_psth_profile=True,
            psth_normalization=mode,
            psth_use_pca=False,
        )
        feat_norm, _, _ = extract_feature_matrix(df, feature_config=cfg_norm)
        assert feat_norm.shape == (12, 50)
        assert not np.isnan(feat_norm).any()


def test_feature_settings_dialog(qapp):
    """Verify FeatureSettingsDialog layout, toggles, weights, and summary formatting."""
    cfg = FeatureConfiguration()
    dlg = FeatureSettingsDialog(cfg)

    # 1. Summary string
    summary_text = format_feature_summary(cfg)
    assert "Latency" in summary_text
    assert "PSTH" in summary_text

    # 2. Checkboxes and widgets reflect new defaults
    assert dlg.chk_latency.isChecked() is True
    assert dlg.spin_w_latency.value() == 5.0
    assert dlg.chk_jitter.isChecked() is True
    assert dlg.chk_reliability.isChecked() is True
    assert dlg.chk_salt.isChecked() is False
    assert dlg.chk_salt_log10.isChecked() is True
    assert dlg.chk_salt_log10.isEnabled() is False
    assert dlg.chk_baseline_rate.isChecked() is False
    assert dlg.chk_peak_rate.isChecked() is False
    assert dlg.chk_peak_response_rate.isChecked() is True
    assert dlg.chk_time_to_response.isChecked() is True
    assert dlg.spin_response_std_threshold.value() == 2.0
    assert dlg.chk_psth_use_pca.isChecked() is True
    assert dlg.spin_psth_pca_var.value() == pytest.approx(0.80)

    # 3. Modifying toggles and weights
    dlg.chk_baseline_rate.setChecked(True)
    dlg.spin_w_baseline.setValue(2.5)
    dlg.chk_latency.setChecked(False)
    assert dlg.spin_w_latency.isEnabled() is False

    compiled = dlg.get_feature_configuration()
    assert compiled.use_baseline_rate is True
    assert compiled.weight_baseline_rate == 2.5
    assert compiled.use_latency is False

    # 4. Reset to defaults
    dlg.reset_to_defaults()
    reset_cfg = dlg.get_feature_configuration()
    assert reset_cfg.use_latency is True
    assert reset_cfg.weight_latency == 5.0
    assert reset_cfg.use_baseline_rate is False
    assert reset_cfg.use_salt is False
    assert reset_cfg.use_peak_response_rate is True
    assert reset_cfg.use_time_to_response is True
    assert reset_cfg.psth_normalization == "area_normalized"
    assert reset_cfg.psth_use_pca is True

    dlg.close()


def test_embedding_parameters_and_computation():
    """Verify EmbeddingParameters and compute_2d_embedding with PCA and UMAP."""
    df = _make_dummy_pooled_df(15)
    feat_m, _, _ = extract_feature_matrix(df, FeatureConfiguration(use_psth_profile=False))

    # 1. Default PCA PC1 vs PC2
    p_default = EmbeddingParameters(method="PCA", pca_component_x=1, pca_component_y=2)
    coords_pca, exp_var = compute_2d_embedding(feat_m, embedding_params=p_default)
    assert coords_pca.shape == (15, 2)
    assert len(exp_var) == 2
    assert exp_var[0] >= exp_var[1]

    # 2. Alternative PCA components (e.g. PC2 vs PC3)
    p_pc23 = EmbeddingParameters(method="PCA", pca_component_x=2, pca_component_y=3)
    coords_pc23, exp_var23 = compute_2d_embedding(feat_m, embedding_params=p_pc23)
    assert coords_pc23.shape == (15, 2)

    # 3. UMAP (or fallback)
    p_umap = EmbeddingParameters(method="UMAP", umap_n_neighbors=5, umap_min_dist=0.1)
    coords_umap, exp_umap = compute_2d_embedding(feat_m, embedding_params=p_umap)
    assert coords_umap.shape == (15, 2)
    assert not np.isnan(coords_umap).any()


def test_embedding_settings_dialog(qapp):
    """Verify EmbeddingSettingsDialog parameter configuration and UI toggles."""
    params = EmbeddingParameters(method="PCA", pca_component_x=1, pca_component_y=2)
    cfg = FeatureConfiguration()
    dlg = EmbeddingSettingsDialog(current_params=params, feature_config=cfg)

    assert dlg.combo_method.currentText().startswith("Principal Component")
    assert dlg.spin_pca_x.value() == 1
    assert dlg.spin_pca_y.value() == 2

    # Switch to UMAP
    idx_umap = dlg.combo_method.findData("UMAP")
    dlg.combo_method.setCurrentIndex(idx_umap)
    dlg.spin_umap_neighbors.setValue(25)
    dlg.spin_umap_min_dist.setValue(0.20)

    out_params = dlg.get_parameters()
    assert out_params.method == "UMAP"
    assert out_params.umap_n_neighbors == 25
    assert np.isclose(out_params.umap_min_dist, 0.20)

    dlg.close()


def test_seed_clusters_dialog_and_affinity_propagation(qapp):
    """Verify SeedClustersDialog controls and execution of seed_clusters_affinity_propagation."""
    df = _make_dummy_pooled_df(16)
    ap_params = AffinityPropagationParameters(
        damping=0.85,
        similarity_percentile=60.0,
        min_cluster_size=2,
    )
    cfg = FeatureConfiguration()

    # 1. Dialog instantiation
    dlg = SeedClustersDialog(current_params=ap_params, feature_config=cfg)
    assert np.isclose(dlg.spin_damping.value(), 0.85)
    assert np.isclose(dlg.spin_percentile.value(), 60.0)
    assert dlg.spin_min_size.value() == 2
    assert dlg.chk_pref_auto.isChecked() is True
    assert dlg.spin_preference.isEnabled() is False
    assert dlg.chk_preserve_curated.isChecked() is True

    # 2. Parameters extraction (auto preference -> None, manual preference -> float)
    dlg.spin_percentile.setValue(75.0)
    extracted_auto = dlg.get_parameters()
    assert extracted_auto.preference is None

    dlg.chk_pref_auto.setChecked(False)
    assert dlg.spin_preference.isEnabled() is True
    dlg.spin_preference.setValue(-25.0)
    extracted = dlg.get_parameters()
    assert np.isclose(extracted.similarity_percentile, 75.0)
    assert extracted.damping == 0.85
    assert np.isclose(extracted.preference, -25.0)

    dlg._reset_defaults()
    assert dlg.chk_pref_auto.isChecked() is True
    assert dlg.get_parameters().preference is None
    dlg.close()

    # 3. Seed clusters execution with custom ap_params
    res = seed_clusters_affinity_propagation(
        df=df,
        feature_config=cfg,
        ap_params=extracted,
    )
    assert len(res.cluster_labels) == 16
    assert res.pca_coordinates.shape == (16, 2)
    assert isinstance(res.cluster_summaries, dict)


def test_multi_algorithm_clustering_dialog_and_execution(qapp):
    """Verify HierarchicalClusteringDialog / ClusteringDialog algorithm switching and execution."""
    df = _make_dummy_pooled_df(18)

    # 1. Dialog supporting Hierarchical, K-Means, and GMM
    c_params = ClusteringParameters(
        algorithm="hierarchical",
        n_clusters=3,
    )
    dlg = HierarchicalClusteringDialog(current_params=c_params)
    assert dlg.combo_algorithm.currentData() == "hierarchical"
    assert dlg.stack_algo.currentIndex() == 0

    # Switch to K-Means
    idx_km = dlg.combo_algorithm.findData("kmeans")
    dlg.combo_algorithm.setCurrentIndex(idx_km)
    assert dlg.stack_algo.currentIndex() == 1
    dlg.spin_k_kmeans.setValue(3)
    p_kmeans = dlg.get_parameters()
    assert p_kmeans.algorithm == "kmeans"
    assert p_kmeans.n_clusters == 3

    # Switch to GMM
    idx_gmm = dlg.combo_algorithm.findData("gmm")
    dlg.combo_algorithm.setCurrentIndex(idx_gmm)
    assert dlg.stack_algo.currentIndex() == 2
    dlg.spin_k_gmm.setValue(3)
    p_gmm = dlg.get_parameters()
    assert p_gmm.algorithm == "gmm"
    assert p_gmm.n_clusters == 3

    dlg.close()

    # 2. Run algorithms directly
    feat_m, _, _ = extract_feature_matrix(df, FeatureConfiguration(use_psth_profile=False))

    # Hierarchical
    labels_hier = run_clustering_algorithm(feat_m, c_params)
    assert len(labels_hier) == 18
    assert len(np.unique(labels_hier)) == 3

    # K-Means
    labels_km = run_clustering_algorithm(feat_m, p_kmeans)
    assert len(labels_km) == 18
    assert len(np.unique(labels_km)) == 3

    # GMM
    labels_gmm = run_clustering_algorithm(feat_m, p_gmm)
    assert len(labels_gmm) == 18
    assert len(np.unique(labels_gmm)) == 3

    # 3. Perform end-to-end clustering with K-Means
    res_km = perform_clustering(df, params=p_kmeans)
    assert len(res_km.cluster_labels) == 18
    assert len(res_km.cluster_summaries) == 3


def test_clustering_view_embedding_controls_and_seeding_dialog(qapp):
    """Verify ClusteringView 2D embedding method selector, configure button, and seeding dialog."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)

    # 1. 2D Embedding controls in dock_pca
    assert hasattr(view, "combo_embedding_method")
    assert hasattr(view, "btn_embedding_settings")
    assert view.combo_embedding_method.currentText() == "PCA"

    # Switch embedding method to UMAP
    idx_umap = view.combo_embedding_method.findData("UMAP")
    view.combo_embedding_method.setCurrentIndex(idx_umap)
    qapp.processEvents()

    assert view.embedding_params.method == "UMAP"
    assert view.combo_embedding_method.currentText() == "UMAP"
    assert "UMAP" in view.pca_plot.getAxis("bottom").labelText

    # Switch back to PCA
    idx_pca = view.combo_embedding_method.findData("PCA")
    view.combo_embedding_method.setCurrentIndex(idx_pca)
    qapp.processEvents()

    assert view.embedding_params.method == "PCA"
    assert view.combo_embedding_method.currentText() == "PCA"

    # 2. Seed Clusters button
    assert hasattr(view, "btn_seed_clusters")
    assert hasattr(view, "prompt_seed_clusters")

    # 3. Clustering button
    assert hasattr(view, "btn_cluster")
    assert "clustering" in view.btn_cluster.text().lower()

    view.close()


def test_mainwindow_clustering_menu_actions(qapp, monkeypatch):
    """Verify MainWindow Clustering top menubar actions and connections."""
    win = MainWindow()

    # 1. Actions in clustering_menu
    assert hasattr(win, "clustering_menu")
    assert hasattr(win, "act_feature_settings")
    assert hasattr(win, "act_embedding_settings")
    assert hasattr(win, "act_seed_clusters")
    assert hasattr(win, "act_hierarchical")
    assert hasattr(win, "act_customize_clusters")
    assert hasattr(win, "act_reset_clustering")

    called = []
    monkeypatch.setattr(win.clustering_view, "prompt_feature_settings", lambda: called.append("features"))
    monkeypatch.setattr(win.clustering_view, "prompt_embedding_settings", lambda: called.append("embedding"))
    monkeypatch.setattr(win.clustering_view, "prompt_seed_clusters", lambda: called.append("seed"))

    # 2. Triggering actions switches to clustering view and invokes prompt
    win.switch_to_launcher()
    assert win.stack.currentIndex() == win.VIEW_LAUNCHER

    win._on_feature_settings()
    assert win.stack.currentIndex() == win.VIEW_CLUSTERING
    assert "features" in called

    win.switch_to_launcher()
    win._on_embedding_settings()
    assert win.stack.currentIndex() == win.VIEW_CLUSTERING
    assert "embedding" in called

    win.switch_to_launcher()
    win._on_seed_clusters()
    assert win.stack.currentIndex() == win.VIEW_CLUSTERING
    assert "seed" in called

    win.close()


def test_phy_reader_probe_and_unit_positions(tmp_path):
    """Verify load_probe_and_unit_positions and PhyKilosortReader load probe geometry and coordinates."""
    from optocurator.io.phy import PhyKilosortReader, load_probe_and_unit_positions

    # 1. Setup channel_positions.npy (4 channels)
    positions = np.array([
        [16.0, 100.0],
        [48.0, 150.0],
        [16.0, 200.0],
        [48.0, 250.0],
    ], dtype=np.float64)
    np.save(tmp_path / "channel_positions.npy", positions)

    # 2. Setup spike_times.npy, spike_clusters.npy, params.py
    spk_times = np.array([300, 600, 900, 1200], dtype=np.int64)
    spk_clusters = np.array([0, 0, 1, 1], dtype=np.int64)
    np.save(tmp_path / "spike_times.npy", spk_times)
    np.save(tmp_path / "spike_clusters.npy", spk_clusters)
    (tmp_path / "params.py").write_text("sample_rate = 30000.0\n", encoding="utf-8")

    # 3. Setup cluster_info.tsv with ch, depth, x
    tsv_content = "id\tch\tdepth\tx\n0\t1\t150.0\t48.0\n1\t2\t200.0\t16.0\n"
    (tmp_path / "cluster_info.tsv").write_text(tsv_content, encoding="utf-8")

    ch_map, depth_map, probe_x_map, probe_pos = load_probe_and_unit_positions(tmp_path)
    assert probe_pos is not None
    assert np.allclose(probe_pos, positions)
    assert ch_map[0] == 1
    assert ch_map[1] == 2
    assert depth_map[0] == 150.0
    assert depth_map[1] == 200.0
    assert probe_x_map[0] == 48.0
    assert probe_x_map[1] == 16.0

    # Load with reader
    reader = PhyKilosortReader()
    units = reader.load(tmp_path, session_id="test_sess")
    assert len(units) == 2
    u0 = units[0]
    assert u0.unit_id == 0
    assert u0.channel == 1
    assert u0.depth == 150.0
    assert u0.probe_x == 48.0
    u1 = units[1]
    assert u1.unit_id == 1
    assert u1.channel == 2
    assert u1.depth == 200.0
    assert u1.probe_x == 16.0


def test_phy_reader_template_peak_channel_fallback(tmp_path):
    """Verify load_probe_and_unit_positions falls back to templates.npy when cluster_info is absent."""
    from optocurator.io.phy import PhyKilosortReader, load_probe_and_unit_positions

    # Probe geometry (4 channels)
    positions = np.array([
        [0.0, 50.0],
        [20.0, 100.0],
        [0.0, 150.0],
        [20.0, 200.0],
    ], dtype=np.float64)
    np.save(tmp_path / "channel_positions.npy", positions)

    # templates.npy: (2 templates, 10 timepoints, 4 channels)
    # Template 0 peak on channel 3; Template 1 peak on channel 1
    templates = np.zeros((2, 10, 4), dtype=np.float32)
    templates[0, 5, 3] = 100.0
    templates[1, 5, 1] = 80.0
    np.save(tmp_path / "templates.npy", templates)

    spk_times = np.array([100, 200, 300, 400], dtype=np.int64)
    spk_clusters = np.array([0, 0, 1, 1], dtype=np.int64)
    spk_templates = np.array([0, 0, 1, 1], dtype=np.int64)
    np.save(tmp_path / "spike_times.npy", spk_times)
    np.save(tmp_path / "spike_clusters.npy", spk_clusters)
    np.save(tmp_path / "spike_templates.npy", spk_templates)
    (tmp_path / "params.py").write_text("sample_rate = 30000.0\n", encoding="utf-8")

    ch_map, depth_map, probe_x_map, probe_pos = load_probe_and_unit_positions(tmp_path)
    assert ch_map[0] == 3
    assert depth_map[0] == 200.0
    assert probe_x_map[0] == 20.0

    assert ch_map[1] == 1
    assert depth_map[1] == 100.0
    assert probe_x_map[1] == 20.0

    reader = PhyKilosortReader()
    units = reader.load(tmp_path, session_id="test_sess_tmpl")
    assert units[0].channel == 3
    assert units[0].depth == 200.0
    assert units[0].probe_x == 20.0
    assert units[1].channel == 1
    assert units[1].depth == 100.0
    assert units[1].probe_x == 20.0


def test_detection_view_export_probe_coordinates(tmp_path, qapp):
    """Verify DetectionView exports channel, depth, probe_x into CSV and channel_positions into npz."""
    det_view = DetectionView(load_mock=True)
    export_dir = tmp_path / "export_output"
    export_dir.mkdir()

    # Trigger export
    csv_path = det_view.export_curated_session(
        export_dir=export_dir,
        precompute_pngs=False,
        show_dialog=False,
    )
    assert csv_path.is_file()

    df = pd.read_csv(csv_path)
    assert "channel" in df.columns
    assert "depth" in df.columns
    assert "probe_x" in df.columns
    assert len(df) > 0
    assert not df["depth"].isna().all()

    # Check npz export
    npz_file = export_dir / f"{det_view.session_id}_spikes.npz"
    assert npz_file.is_file()
    with np.load(npz_file) as data:
        assert "channel_positions" in data
        assert data["channel_positions"].shape[1] >= 2

    det_view.close()


def test_clustering_view_dynamic_probe_switching(qapp, qtbot):
    """Verify dynamic Silicon Probe view re-frames geometry and dots when switching sessions."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    qtbot.waitUntil(
        lambda: (view._active_learning_worker is None or not view._active_learning_worker.isRunning())
        and view.active_candidate is not None,
        timeout=10000,
    )

    # 1. Probe geometries registered
    assert "01942_74R_20261019" in view._experiment_probes
    assert "02034_83L_20261020" in view._experiment_probes
    assert "02035_83R_20261021" in view._experiment_probes

    p1 = view._experiment_probes["01942_74R_20261019"]
    p2 = view._experiment_probes["02034_83L_20261020"]
    p3 = view._experiment_probes["02035_83R_20261021"]

    # Session 1 is 2-column staggered (1 shank)
    assert np.any(p1[:, 0] == 16.0) and np.any(p1[:, 0] == 48.0)
    # Session 2 is dual shank (x=0 and x=250)
    assert np.any(p2[:, 0] == 0.0) and np.any(p2[:, 0] == 250.0)
    # Session 3 is 4-shank probe (x=0, 250, 500, 750)
    assert np.any(p3[:, 0] == 0.0) and np.any(p3[:, 0] == 250.0) and np.any(p3[:, 0] == 500.0) and np.any(p3[:, 0] == 750.0)

    # Test _detect_shanks for 1, 2, and 4 shanks
    shanks1 = view._detect_shanks(p1)
    shanks2 = view._detect_shanks(p2)
    shanks3 = view._detect_shanks(p3)
    assert len(shanks1) == 1
    assert len(shanks2) == 2
    assert len(shanks3) == 4

    # 2. Select unit from Session 1
    u1_gid = "01942_74R_20261019_u0001"
    view.select_pooled_unit(u1_gid)
    qapp.processEvents()

    assert view._current_probe_session_id == "01942_74R_20261019"
    assert "01942_74R_20261019" in view.lbl_probe_experiment.text()
    assert "Shank 1 / 1" in view.lbl_probe_unit_info.text()
    assert "Ch 8" in view.lbl_probe_unit_info.text()
    assert "240" in view.lbl_probe_unit_info.text()
    assert "16" in view.lbl_probe_unit_info.text()
    assert len(view._probe_shank_items) == 1

    # Check selected unit is highlighted by cand_marker box and present in units_scatter
    unit_points1 = view.probe_units_scatter.points()
    u1_pts = [pt for pt in unit_points1 if pt.data() == u1_gid]
    assert len(u1_pts) == 1
    assert view.probe_cand_marker.isVisible() is True

    # Probe sites scatter matches Session 1 probe
    sites_data = view.probe_sites_scatter.getData()
    assert np.allclose(sites_data[0], p1[:, 0])
    assert np.allclose(sites_data[1], p1[:, 1])

    # Probe marker is positioned at the focused unit's normalized sub-column x and depth=240.0
    marker_data = view.probe_cand_marker.getData()
    assert marker_data[0][0] == pytest.approx(float(u1_pts[0].pos().x()))
    assert marker_data[1][0] == pytest.approx(240.0)
    assert view.probe_cand_line.value() == pytest.approx(240.0)

    # 3. Select unit from Session 2 on shank 2 (x=250.0)
    u2_gid = "02034_83L_20261020_u0008"
    view.select_pooled_unit(u2_gid)
    qapp.processEvents()

    assert view._current_probe_session_id == "02034_83L_20261020"
    assert "02034_83L_20261020" in view.lbl_probe_experiment.text()
    assert "Shank 2 / 2" in view.lbl_probe_unit_info.text()
    assert "Ch 42" in view.lbl_probe_unit_info.text()
    assert "500" in view.lbl_probe_unit_info.text()
    assert "250" in view.lbl_probe_unit_info.text()
    assert len(view._probe_shank_items) == 2

    # 4. Select unit from Session 3 on Shank 1 (x=0) and Shank 4 (x=750)
    u3_gid = "02035_83R_20261021_u0003"
    view.select_pooled_unit(u3_gid)
    qapp.processEvents()

    assert view._current_probe_session_id == "02035_83R_20261021"
    assert "02035_83R_20261021" in view.lbl_probe_experiment.text()
    assert "Shank 1 / 4" in view.lbl_probe_unit_info.text()
    assert "Ch 4" in view.lbl_probe_unit_info.text()
    assert len(view._probe_shank_items) == 4

    sites_data3 = view.probe_sites_scatter.getData()
    assert np.allclose(sites_data3[0], p3[:, 0])

    # Select unit on Shank 4 (x=750)
    u3_s4_gid = "02035_83R_20261021_u0013"
    view.select_pooled_unit(u3_s4_gid)
    qapp.processEvents()
    assert "Shank 4 / 4" in view.lbl_probe_unit_info.text()
    assert "Ch 52" in view.lbl_probe_unit_info.text()

    view.close()


def test_clustering_view_add_experiment_folder_position_resolution(qapp, tmp_path):
    """Verify add_experiment_folder updates meta.df and meta.tagged_df without AttributeError."""
    import pandas as pd
    session_id = "test_exp_session"
    exp_dir = tmp_path / session_id
    exp_dir.mkdir()

    # Create <session_id>_opto_curated.csv with missing depth and probe_x
    df_curated = pd.DataFrame([
        {
            "global_unit_id": f"{session_id}_u1",
            "session_id": session_id,
            "unit_id": 1,
            "channel": 5,
            "latency_ms": 3.2,
            "jitter_ms": 0.8,
            "reliability": 0.95,
            "waveform_correlation": 0.98,
            "salt_p_value": 0.001,
            "is_tagged": True,
            "cluster_id": -1,
            "curated_label": "Opto-tagged",
            "plot_path": "",
        },
        {
            "global_unit_id": f"{session_id}_u2",
            "session_id": session_id,
            "unit_id": 2,
            "channel": 10,
            "latency_ms": 4.5,
            "jitter_ms": 1.1,
            "reliability": 0.85,
            "waveform_correlation": 0.96,
            "salt_p_value": 0.002,
            "is_tagged": True,
            "cluster_id": -1,
            "curated_label": "Opto-tagged",
            "plot_path": "",
        },
    ])
    df_curated.to_csv(exp_dir / f"{session_id}_opto_curated.csv", index=False)

    # Create <session_id>_spikes.npz with channel_positions
    channel_pos = np.zeros((32, 2), dtype=np.float64)
    channel_pos[:, 0] = 0.0
    channel_pos[:, 1] = np.arange(32) * 50.0
    np.savez(
        exp_dir / f"{session_id}_spikes.npz",
        channel_positions=channel_pos,
        pulse_onsets=np.array([10.0, 20.0, 30.0]),
        unit_1=np.array([10.003, 20.003]),
        unit_2=np.array([10.004, 20.004]),
    )

    view = ClusteringView(start_with_clusters=False, load_mock=False)
    # Should complete without raising AttributeError on meta.tagged_df
    view.add_experiment_folder(str(exp_dir))

    assert session_id in view.loaded_experiments
    meta = view.loaded_experiments[session_id]
    assert "depth" in meta.df.columns
    assert "probe_x" in meta.df.columns
    assert meta.df.loc[meta.df["unit_id"] == 1, "depth"].iloc[0] == 250.0
    assert meta.df.loc[meta.df["unit_id"] == 2, "depth"].iloc[0] == 500.0

    # meta.tagged_df property reflects the updated meta.df
    assert "depth" in meta.tagged_df.columns
    assert meta.tagged_df.loc[meta.tagged_df["unit_id"] == 1, "depth"].iloc[0] == 250.0

    view.close()


def test_clustering_view_probe_interactive_click_and_colors(qapp, qtbot):
    """Verify clicking unit dots in Silicon Probe View selects unit and colors reflect clusters."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    qtbot.waitUntil(
        lambda: (view._active_learning_worker is None or not view._active_learning_worker.isRunning())
        and view.active_candidate is not None,
        timeout=10000,
    )

    # Unit dots are rendered on probe_units_scatter
    unit_spots = view.probe_units_scatter.points()
    assert len(unit_spots) > 0

    # Test clicking a probe unit dot
    target_point = unit_spots[0]
    target_gid = target_point.data()
    assert target_gid is not None

    view._on_probe_unit_clicked(view.probe_units_scatter, [target_point])
    qapp.processEvents()

    assert view._current_selected_gid == target_gid

    # Test color updates when cluster colors customize
    new_colors = {0: "#FF0000", 1: "#00FF00", 2: "#0000FF", 3: "#FFFF00"}
    view.apply_cluster_customizations(
        new_order=list(view._cluster_card_order),
        new_names=dict(view._cluster_names),
        new_colors=new_colors,
    )
    qapp.processEvents()

    # Selected unit's cluster color on probe
    rec = view._pooled_units_map[target_gid]
    cid = rec.get("cluster_id", -1)
    if cid in new_colors:
        points = view.probe_units_scatter.points()
        for p in points:
            if p.data() == target_gid:
                brush = p.brush()
                assert brush.color().name().upper() == new_colors[cid].upper()
                break

    view.close()


def test_clustering_view_probe_marker_jitter_and_shank_shading(qapp):
    """Verify that multiple units sharing the same site receive slight marker jitter so they don't occlude."""
    view = ClusteringView(start_with_clusters=False, load_mock=False)

    sess_id = "test_jitter_session"
    view._experiment_probes[sess_id] = np.array([
        [16.0, 100.0],
        [48.0, 100.0],
        [16.0, 120.0],
        [48.0, 120.0],
    ], dtype=np.float64)

    # 3 units sharing site at x=16.0, depth=100.0
    u1 = {"global_unit_id": f"{sess_id}_u001", "session_id": sess_id, "unit_id": 1, "depth": 100.0, "probe_x": 16.0, "cluster_id": 0}
    u2 = {"global_unit_id": f"{sess_id}_u002", "session_id": sess_id, "unit_id": 2, "depth": 100.0, "probe_x": 16.0, "cluster_id": 0}
    u3 = {"global_unit_id": f"{sess_id}_u003", "session_id": sess_id, "unit_id": 3, "depth": 100.0, "probe_x": 16.0, "cluster_id": 1}

    view._pooled_units_map = {
        u1["global_unit_id"]: u1,
        u2["global_unit_id"]: u2,
        u3["global_unit_id"]: u3,
    }

    view._update_probe_view_for_unit(u1["global_unit_id"])
    qapp.processEvents()

    # Shank shading items: only individual shanks are shaded with COLOR_PROBE_SHANK_FILL
    from optocurator.ui.theme import COLOR_PROBE_SHANK_FILL

    assert len(view._probe_shank_items) == 1
    poly_item = view._probe_shank_items[0]
    assert poly_item.brush().color().name().upper() == COLOR_PROBE_SHANK_FILL.upper()

    # Verify that the 3 units are placed inside shank 0's dual sub-columns (±0.17)
    points = view.probe_units_scatter.points()
    assert len(points) == 3

    x_positions = [float(pt.pos().x()) for pt in points]
    for x_val in x_positions:
        assert abs(x_val) == pytest.approx(0.17)

    # Verify clicking overlapping points array does not raise ValueError
    pts_array = np.array(points[:2])
    view._on_probe_unit_clicked(view.probe_units_scatter, pts_array)
    qapp.processEvents()

    view._on_scatter_point_clicked(view.scatter_item, pts_array)
    qapp.processEvents()

    # Verify adding units with slightly different depths renders all 5 horizontal bars at their exact depths
    u4 = {"global_unit_id": f"{sess_id}_u004", "session_id": sess_id, "unit_id": 4, "depth": 104.2, "probe_x": 16.0, "cluster_id": 0}
    u5 = {"global_unit_id": f"{sess_id}_u005", "session_id": sess_id, "unit_id": 5, "depth": 108.5, "probe_x": 16.0, "cluster_id": 1}
    view._pooled_units_map[u4["global_unit_id"]] = u4
    view._pooled_units_map[u5["global_unit_id"]] = u5

    view._update_probe_view_for_unit(u1["global_unit_id"])
    qapp.processEvents()

    pts_5 = view.probe_units_scatter.points()
    assert len(pts_5) == 5
    y_by_gid = {pt.data(): float(pt.pos().y()) for pt in pts_5}
    assert y_by_gid[u4["global_unit_id"]] == pytest.approx(104.2)
    assert y_by_gid[u5["global_unit_id"]] == pytest.approx(108.5)

    view.close()


def test_feature_configuration_defaults_and_clustering_norm():
    """Verify new FeatureConfiguration defaults and ClusteringParameters.cluster_psth_norm default."""
    fc = FeatureConfiguration()
    assert fc.weight_latency == 5.0
    assert fc.use_salt is False
    assert fc.use_peak_response_rate is True
    assert fc.use_time_to_response is True
    assert fc.use_baseline_rate is False
    assert fc.use_peak_rate is False
    assert fc.psth_normalization == "area_normalized"
    assert fc.psth_use_pca is True
    assert fc.psth_pca_variance_explained == pytest.approx(0.80)
    assert fc.time_window_ms == (0.0, 25.0)
    assert fc.response_std_threshold == 2.0

    cp = ClusteringParameters()
    assert cp.cluster_psth_norm == "probability"


def test_peak_rate_and_signed_peak_response_rate_pulse_window():
    """Verify peak_rate and signed peak_response_rate are computed strictly in [0, pulse_duration_ms]."""
    pulse_dur = 10.0
    win = (0.0, pulse_dur + 5.0)
    bin_ms = 2.0
    # t_bins for (0, 15) step 2 -> [1, 3, 5, 7, 9, 11, 13] (indices 0..4 in [0, 10], 5..6 outside)
    df = pd.DataFrame([
        {"global_unit_id": "u_exc", "session_id": "s", "unit_id": 0},
        {"global_unit_id": "u_rebound", "session_id": "s", "unit_id": 1},
        {"global_unit_id": "u_inh", "session_id": "s", "unit_id": 2},
    ])
    base_10 = np.full(25, 10.0)
    base_40 = np.full(25, 40.0)
    psth_cache = {
        ("u_exc", (-50.0, 0.0), bin_ms, False, 2.0): base_10,
        ("u_exc", win, bin_ms, False, 2.0): np.array([10.0, 50.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
        ("u_rebound", (-50.0, 0.0), bin_ms, False, 2.0): base_10,
        ("u_rebound", win, bin_ms, False, 2.0): np.array([10.0, 50.0, 10.0, 10.0, 10.0, 200.0, 300.0]),
        ("u_inh", (-50.0, 0.0), bin_ms, False, 2.0): base_40,
        ("u_inh", win, bin_ms, False, 2.0): np.array([40.0, 5.0, 40.0, 40.0, 40.0, 250.0, 250.0]),
    }

    cfg = FeatureConfiguration(
        use_latency=False,
        use_jitter=False,
        use_reliability=False,
        use_salt=False,
        use_baseline_rate=False,
        use_peak_rate=True,
        use_peak_response_rate=True,
        use_time_to_response=False,
        use_psth_profile=False,
        time_window_ms=win,
        bin_size_ms=bin_ms,
        psth_smooth=False,
    )
    feat, _, _ = extract_feature_matrix(
        df,
        feature_config=cfg,
        psth_cache=psth_cache,
        pulse_duration_ms=pulse_dur,
    )
    assert feat.shape == (3, 2)

    # Column 0: peak_rate in [0, 10] -> u_exc == u_rebound (50 Hz) > u_inh (40 Hz), ignoring 200-300 Hz outside [0, 10]
    assert feat[0, 0] == pytest.approx(feat[1, 0], abs=1e-9)
    assert feat[0, 0] > feat[2, 0]

    # Column 1: signed peak_response_rate -> u_exc == u_rebound (+40 Hz) > 0 > u_inh (-35 Hz)
    assert feat[0, 1] == pytest.approx(feat[1, 1], abs=1e-9)
    assert feat[0, 1] > 0.0
    assert feat[2, 1] < 0.0


def test_time_to_response_threshold_and_fallback():
    """Verify time_to_response respects response_std_threshold and falls back to window end."""
    win = (-20.0, 20.0)
    bin_ms = 2.0
    # 10 baseline bins (t < 0) alternating 8/12 -> mean=10, std=2; 10 evoked bins (t = [1, 3, ..., 19])
    base = np.array([8.0, 12.0] * 5)
    df = pd.DataFrame([
        {"global_unit_id": "u_two_step", "session_id": "s", "unit_id": 0},
        {"global_unit_id": "u_early", "session_id": "s", "unit_id": 1},
        {"global_unit_id": "u_flat", "session_id": "s", "unit_id": 2},
        {"global_unit_id": "u_end", "session_id": "s", "unit_id": 3},
    ])
    psth_cache = {
        ("u_two_step", win, bin_ms, False, 2.0): np.concatenate([
            base, np.array([10.0, 13.0, 10.0, 10.0, 20.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        ]),
        ("u_early", win, bin_ms, False, 2.0): np.concatenate([
            base, np.array([10.0, 20.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        ]),
        ("u_flat", win, bin_ms, False, 2.0): np.concatenate([base, np.full(10, 10.0)]),
        ("u_end", win, bin_ms, False, 2.0): np.concatenate([base, np.array([10.0] * 9 + [25.0])]),
    }

    def _run(std_thresh: float) -> np.ndarray:
        cfg = FeatureConfiguration(
            use_latency=False,
            use_jitter=False,
            use_reliability=False,
            use_salt=False,
            use_peak_response_rate=False,
            use_time_to_response=True,
            response_std_threshold=std_thresh,
            use_psth_profile=False,
            time_window_ms=win,
            bin_size_ms=bin_ms,
            psth_smooth=False,
        )
        feat, _, _ = extract_feature_matrix(df, feature_config=cfg, psth_cache=psth_cache)
        return feat[:, 0]

    # At 1.0 SD (2 Hz): u_two_step crosses at t=3ms (13 Hz) -> matches u_early; u_flat falls back to t=19ms (u_end)
    ttr_1sd = _run(1.0)
    assert ttr_1sd[0] == pytest.approx(ttr_1sd[1], abs=1e-9)
    assert ttr_1sd[2] == pytest.approx(ttr_1sd[3], abs=1e-9)
    assert ttr_1sd[0] < ttr_1sd[2]

    # At 2.0 SD (4 Hz): u_two_step skips t=3ms and crosses at t=9ms -> strictly between u_early and u_flat
    ttr_2sd = _run(2.0)
    assert ttr_2sd[1] < ttr_2sd[0] < ttr_2sd[2]
    assert ttr_2sd[2] == pytest.approx(ttr_2sd[3], abs=1e-9)


def test_area_normalized_psth_l1_and_pca_compression():
    """Verify L1 area_normalized PSTH makes scaled copies identical and psth_use_pca compresses dimensions."""
    win = (0.0, 25.0)
    bin_ms = 2.0
    shape_a = np.array([0.0, 5.0, 20.0, 10.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0])
    shape_b = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 10.0, 20.0, 10.0, 2.0, 1.0, 0.0, 0.0])
    df = pd.DataFrame([
        {"global_unit_id": "u_a1", "session_id": "s", "unit_id": 0},
        {"global_unit_id": "u_a10", "session_id": "s", "unit_id": 1},
        {"global_unit_id": "u_b", "session_id": "s", "unit_id": 2},
    ])
    psth_cache = {
        ("u_a1", win, bin_ms, False, 2.0): shape_a,
        ("u_a10", win, bin_ms, False, 2.0): shape_a * 10.0,  # 10x firing rate, identical L1 shape
        ("u_b", win, bin_ms, False, 2.0): shape_b,
        ("u_a1", (-50.0, 0.0), bin_ms, False, 2.0): np.ones(25),
        ("u_a10", (-50.0, 0.0), bin_ms, False, 2.0): np.ones(25) * 10.0,
        ("u_b", (-50.0, 0.0), bin_ms, False, 2.0): np.ones(25),
    }

    # Verify normalize_psth_waveform L1 sum == 1.0
    l1_a = normalize_psth_waveform(shape_a, norm_mode="probability")
    assert np.sum(l1_a) == pytest.approx(1.0)
    np.testing.assert_allclose(l1_a, normalize_psth_waveform(shape_a * 10.0, norm_mode="probability"))

    # Without PCA: u_a1 and u_a10 have identical area_normalized feature vectors
    cfg_no_pca = FeatureConfiguration(
        use_latency=False,
        use_jitter=False,
        use_reliability=False,
        use_salt=False,
        use_peak_response_rate=False,
        use_time_to_response=False,
        use_psth_profile=True,
        psth_normalization="area_normalized",
        psth_use_pca=False,
        time_window_ms=win,
        bin_size_ms=bin_ms,
        psth_smooth=False,
    )
    feat_no_pca, _, t_bins = extract_feature_matrix(df, feature_config=cfg_no_pca, psth_cache=psth_cache)
    assert feat_no_pca.shape == (3, len(t_bins))
    np.testing.assert_allclose(feat_no_pca[0], feat_no_pca[1], atol=1e-7)
    assert not np.allclose(feat_no_pca[0], feat_no_pca[2])

    # With PCA enabled on a larger dataset, dimensionality is reduced relative to psth_use_pca=False
    df_many = _make_dummy_pooled_df(16)
    cfg_full_many = FeatureConfiguration(
        use_latency=False,
        use_jitter=False,
        use_reliability=False,
        use_salt=False,
        use_peak_response_rate=False,
        use_time_to_response=False,
        use_psth_profile=True,
        psth_use_pca=False,
    )
    cfg_pca_80 = FeatureConfiguration(
        use_latency=False,
        use_jitter=False,
        use_reliability=False,
        use_salt=False,
        use_peak_response_rate=False,
        use_time_to_response=False,
        use_psth_profile=True,
        psth_use_pca=True,
        psth_pca_variance_explained=0.80,
    )
    feat_full, _, _ = extract_feature_matrix(df_many, feature_config=cfg_full_many)
    feat_pca, _, _ = extract_feature_matrix(df_many, feature_config=cfg_pca_80)
    assert feat_full.shape == (16, 50)
    assert 1 <= feat_pca.shape[1] < feat_full.shape[1]


def test_preserve_curated_keeps_cluster_cards_and_summaries_in_sync(qapp):
    """preserve_curated=True must keep curated cluster cards and ClusteringResult summaries in sync (Bug CL3)."""
    view = ClusteringView(load_mock=True, start_with_clusters=False)
    try:
        all_gids = list(view._pooled_units_map.keys())
        curated_gids = all_gids[:2]
        for gid in curated_gids:
            view._pooled_units_map[gid]["cluster_id"] = 5
            view._unit_status_map[gid] = "Curated"
        view._cluster_card_order = [5]

        ok = view.run_clustering(ClusteringParameters(n_clusters=2), preserve_curated=True)
        assert ok is True
        qapp.processEvents()

        for gid in curated_gids:
            assert view._pooled_units_map[gid]["cluster_id"] == 5
            assert view._unit_status_map[gid] == "Curated"

        assert 5 in view._cluster_cards
        assert 5 in view.current_clustering_result.cluster_summaries
        assert view._cluster_cards[5].count == 2
        assert view.current_clustering_result.cluster_summaries[5].count == 2
        total_in_summaries = sum(s.count for s in view.current_clustering_result.cluster_summaries.values())
        assert total_in_summaries == len(view._pooled_units_map)

        # Seeding with preserve_curated=True also keeps curated Cluster 5 card & forwards view smoothing/norm params
        view.psth_smooth = False
        view.cluster_psth_norm = "peak"
        ok_seed = view.seed_clusters_with_affinity_propagation(AffinityPropagationParameters(preserve_curated=True))
        assert ok_seed is True
        qapp.processEvents()
        assert 5 in view._cluster_cards
        assert 5 in view.current_clustering_result.cluster_summaries
        assert view.current_clustering_result.cluster_summaries[5].count == 2
        assert view.current_clustering_result.parameters.psth_smooth is False
        assert view.current_clustering_result.parameters.cluster_psth_norm == "peak"
    finally:
        view.close()

