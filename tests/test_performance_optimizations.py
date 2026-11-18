"""Unit and integration tests for Phase 2 Performance Optimizations.

Validates:
1. In-place item reuse in plots.py (render_raster and render_psth).
2. Selection signal deduplication and single execution on unit selection.
3. Persistent ClusterCardWidget reuse across assignment cycles.
4. Unified PSTH caching between raster/PSTH extraction and cluster PSTH computation.
5. Incremental member ImageViewer updates in cluster comparison drawer.
6. Probe shank/layout caching and invalidation lifecycle.
7. Feature matrix and spike data caching.
8. Suggested candidates selection deduplication and sequential advance.
9. ClusterCardWidget.update_card across mode switches, empty clusters, and normalization.
10. render_raster and render_psth empty data handling and theme styling.
11. Incremental comparison drawer member viewers across cluster switches.
12. Incremental pooled table cluster items updates.
"""

from pathlib import Path
from typing import Sequence
import numpy as np
import pytest
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem

from optocurator.analysis.clustering import ClusterSummary
from optocurator.ui.plots import render_raster, render_psth
from optocurator.ui.view_clusters import ClusteringView, ClusterCardWidget
from tests.test_detection_workflow import create_synthetic_experiment_folder


def test_render_raster_in_place_item_reuse(qapp):
    """Verify render_raster reuses ScatterPlotItem and InfiniteLine without calling clear()."""
    plot_widget = pg.PlotWidget()
    plot_item = plot_widget.getPlotItem()

    # First call: initializes cache
    x1 = np.array([1.0, 5.0, 10.0])
    y1 = np.array([0, 1, 2])
    render_raster(plot_widget, raster_x=x1, raster_y=y1, n_trials=5)

    assert hasattr(plot_item, "_raster_cache")
    cache = plot_item._raster_cache
    scatter_id = id(cache["scatter"])
    onset_id = id(cache["onset_line"])

    # Second call: updates data in-place
    x2 = np.array([2.0, 4.0, 8.0, 15.0])
    y2 = np.array([0, 1, 1, 3])
    render_raster(plot_widget, raster_x=x2, raster_y=y2, n_trials=5)

    assert id(plot_item._raster_cache["scatter"]) == scatter_id
    assert id(plot_item._raster_cache["onset_line"]) == onset_id
    assert len(plot_item.items) == 3

    # Third call with empty data: ensures no crash and sets visibility
    render_raster(plot_widget, raster_x=np.array([]), raster_y=np.array([]), n_trials=0)
    assert id(plot_item._raster_cache["scatter"]) == scatter_id
    assert not cache["scatter"].isVisible()


def test_render_psth_in_place_item_reuse(qapp):
    """Verify render_psth reuses PlotCurveItem, BarGraphItem, and InfiniteLines without calling clear()."""
    plot_widget = pg.PlotWidget()
    plot_item = plot_widget.getPlotItem()

    bins1 = np.linspace(-30.0, 50.0, 41)
    rates1 = np.abs(np.sin(bins1[:-1])) * 20.0

    # First call: initializes cache
    render_psth(
        plot_widget,
        psth_bins=bins1,
        psth_rates=rates1,
        baseline_mean=5.0,
        baseline_3sigma=12.0,
        is_smooth=True,
    )

    assert hasattr(plot_item, "_psth_cache")
    cache = plot_item._psth_cache
    onset_id = id(cache["onset_line"])
    baseline_id = id(cache["baseline_line"])
    smooth_id = id(cache["smooth_curve"])
    bar_id = id(cache["bar_item"])

    # Second call (smooth=True): mutates in-place
    rates2 = rates1 * 1.5
    render_psth(
        plot_widget,
        psth_bins=bins1,
        psth_rates=rates2,
        baseline_mean=6.0,
        baseline_3sigma=14.0,
        is_smooth=True,
    )

    assert id(plot_item._psth_cache["onset_line"]) == onset_id
    assert id(plot_item._psth_cache["baseline_line"]) == baseline_id
    assert id(plot_item._psth_cache["smooth_curve"]) == smooth_id
    assert id(plot_item._psth_cache["bar_item"]) == bar_id
    assert cache["smooth_curve"].isVisible()
    assert not cache["bar_item"].isVisible()

    # Third call (smooth=False): switches visibility without rebuilding
    render_psth(
        plot_widget,
        psth_bins=bins1,
        psth_rates=rates2,
        baseline_mean=6.0,
        baseline_3sigma=14.0,
        is_smooth=False,
    )

    assert id(plot_item._psth_cache["smooth_curve"]) == smooth_id
    assert id(plot_item._psth_cache["bar_item"]) == bar_id
    assert not cache["smooth_curve"].isVisible()
    assert cache["bar_item"].isVisible()


def test_select_pooled_unit_signal_deduplication(qapp, tmp_path):
    """Verify selecting a unit in table_pooled_units fires handler exactly once."""
    exp_dir = create_synthetic_experiment_folder(tmp_path / "exp_sig")
    view = ClusteringView(start_with_clusters=True)
    view.show()
    view.add_experiment_folder(exp_dir)

    all_gids = [str(r["global_unit_id"]) for r in view._pooled_rows]
    assert len(all_gids) >= 2
    target_gid = all_gids[1]

    call_count = 0
    orig_handler = view._on_pooled_unit_selected

    def wrapped_handler():
        nonlocal call_count
        call_count += 1
        return orig_handler()

    view._on_pooled_unit_selected = wrapped_handler

    try:
        call_count = 0
        success = view.select_pooled_unit(target_gid)
        assert success is True
        # Assert called exactly once (signal deduplication prevented redundant fires)
        assert call_count == 1
    finally:
        view._on_pooled_unit_selected = orig_handler
        view.close()


def test_cluster_card_widget_in_place_reuse(qapp, tmp_path):
    """Verify ClusterCardWidget instances are preserved and updated in-place during assignment."""
    exp_dir = create_synthetic_experiment_folder(tmp_path / "exp_cards")
    view = ClusteringView(start_with_clusters=True)
    view.show()
    view.add_experiment_folder(exp_dir)

    all_gids = [str(r["global_unit_id"]) for r in view._pooled_rows]
    assert len(all_gids) >= 4

    # Seed initial clusters
    view.assign_candidate_to_cluster(0, unit_id=all_gids[0])
    view.assign_candidate_to_cluster(1, unit_id=all_gids[1])

    assert 0 in view._cluster_cards
    assert 1 in view._cluster_cards
    card_0 = view._cluster_cards[0]
    card_0_id = id(card_0)

    # Assign another unit to Cluster 0
    view.assign_candidate_to_cluster(0, unit_id=all_gids[2])

    # Verify the card object identity is retained (not recreated)
    assert 0 in view._cluster_cards
    assert view._cluster_cards[0] is card_0
    assert id(view._cluster_cards[0]) == card_0_id
    assert view._cluster_cards[0].count >= 2

    view.close()


def test_psth_cache_unification(qapp):
    """Verify _get_raster_and_psth_for_unit populates _unit_psth_cache for _get_psth_for_unit."""
    view = ClusteringView(load_mock=True, start_with_clusters=False)
    view.show()

    all_gids = [str(r["global_unit_id"]) for r in view._pooled_rows]
    assert len(all_gids) > 0
    gid = all_gids[0]

    # Precondition: cache may be empty or clear it
    view._unit_psth_cache.clear()

    # Call _get_raster_and_psth_for_unit
    psth_data = view._get_raster_and_psth_for_unit(gid)
    assert psth_data is not None

    psth_key = (
        gid,
        (float(view.current_window_ms[0]), float(view.current_window_ms[1])),
        float(view.current_bin_size_ms),
        bool(view.psth_smooth),
        float(view.psth_gaussian_sigma_ms),
    )
    # Check that _unit_psth_cache was directly populated
    assert psth_key in view._unit_psth_cache
    np.testing.assert_array_equal(view._unit_psth_cache[psth_key], psth_data["psth_rates"])

    # Calling _get_psth_for_unit should now return immediately with cache hit
    t_bins, rates = view._get_psth_for_unit(gid)
    np.testing.assert_array_equal(rates, psth_data["psth_rates"])

    view.close()


def test_incremental_member_viewers_in_cluster_comparison(qapp, tmp_path):
    """Verify cluster comparison pane incrementally adds member ImageViewers without full rebuild."""
    exp_dir = create_synthetic_experiment_folder(tmp_path / "exp_comp")
    view = ClusteringView(start_with_clusters=True)
    view.show()
    view.add_experiment_folder(exp_dir)

    all_gids = [str(r["global_unit_id"]) for r in view._pooled_rows]
    assert len(all_gids) >= 3

    # Assign first unit to Cluster 0
    view.assign_candidate_to_cluster(0, unit_id=all_gids[0])
    view.set_cluster_comparison(0)

    assert all_gids[0] in view._comparison_member_viewers_map
    first_viewer = view._comparison_member_viewers_map[all_gids[0]]

    # Assign second unit to Cluster 0
    view.assign_candidate_to_cluster(0, unit_id=all_gids[1])
    view.set_cluster_comparison(0)

    # First viewer must be preserved (same object reference)
    assert all_gids[0] in view._comparison_member_viewers_map
    assert view._comparison_member_viewers_map[all_gids[0]] is first_viewer
    # Second viewer must now also exist
    assert all_gids[1] in view._comparison_member_viewers_map

    view.close()


def test_probe_and_clustering_caches_lifecycle(qapp, tmp_path):
    """Verify probe shanks/layout caching and clustering matrix caching lifecycle."""
    exp_dir = create_synthetic_experiment_folder(tmp_path / "exp_cache")
    view = ClusteringView(start_with_clusters=True)
    view.show()
    view.add_experiment_folder(exp_dir)

    all_gids = [str(r["global_unit_id"]) for r in view._pooled_rows]
    assert len(all_gids) > 0
    gid = all_gids[0]
    sess = view._pooled_units_map[gid]["session_id"]

    # Trigger probe render
    view._update_probe_view_for_unit(gid)
    assert sess in view._probe_shanks_cache

    # Invalidate probe caches
    view._invalidate_probe_caches()
    assert len(view._probe_shanks_cache) == 0
    assert len(view._probe_layout_cache) == 0

    # Test clustering cache invalidation
    df_active = view._get_active_pooled_dataframe()
    spk_map = view._get_cached_spike_data_map(df_active)
    psth_mat, t_bins = view._get_cached_psth_matrix(df_active, spk_map)

    assert view._cached_spike_data_map is not None
    assert view._cached_psth_matrix_tuple is not None

    # Invalidate clustering caches
    view._invalidate_clustering_caches()
    assert view._cached_spike_data_map is None
    assert view._cached_psth_matrix_tuple is None

    view.close()


def test_selection_deduplication_suggested_candidates_and_advance(qapp):
    """Verify suggested candidates cellClicked and _advance_to_next_table_row prevent duplicate selections."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    all_gids = list(view._pooled_units_map.keys())
    assert len(all_gids) >= 2

    # Set up suggested candidates table
    view.table_suggested_candidates.setRowCount(2)
    item0 = QTableWidgetItem("Candidate 0")
    item0.setData(Qt.ItemDataRole.UserRole, all_gids[0])
    view.table_suggested_candidates.setItem(0, 0, item0)

    item1 = QTableWidgetItem("Candidate 1")
    item1.setData(Qt.ItemDataRole.UserRole, all_gids[1])
    view.table_suggested_candidates.setItem(1, 0, item1)

    # Currently selected GID is already all_gids[0]
    view._current_selected_gid = all_gids[0]

    call_count = 0
    orig_select = view.select_pooled_unit

    def spy_select(gid):
        nonlocal call_count
        call_count += 1
        return orig_select(gid)

    view.select_pooled_unit = spy_select

    # Clicking on the already selected unit does not re-invoke select_pooled_unit
    view._on_suggested_candidate_cell_clicked(0, 0)
    assert call_count == 0

    # Clicking on a different candidate invokes select_pooled_unit once
    view._on_suggested_candidate_cell_clicked(1, 0)
    assert call_count == 1
    assert view._current_selected_gid == all_gids[1]

    # Test sequential advance in non-active learning mode
    view.active_learning_enabled = False
    view._advance_to_next_table_row(assigned_gid=all_gids[0])
    assert view.table_pooled_units.currentRow() >= 0

    view.close()


def test_cluster_card_update_card_stress_and_edge_cases(qapp):
    """Stress-test ClusterCardWidget.update_card across mode switches, empty clusters, and normalization."""
    card = ClusterCardWidget(cluster_id=0, count=5, color_hex="#2563EB", is_smooth=True, norm_mode="peak")
    curve_id = id(card.mean_curve)
    plot_id = id(card.plot)
    header_id = id(card.header)

    # 1. Update to empty cluster (count=0, summary=None)
    card.update_card(
        summary=None,
        count=0,
        color_hex="#EF4444",
        custom_name="Empty Cluster",
        pulse_duration_ms=25.0,
        is_smooth=False,
        norm_mode="raw",
        hotkey="Shift+X",
    )

    assert id(card.mean_curve) == curve_id
    assert id(card.plot) == plot_id
    assert id(card.header) == header_id
    assert card.count == 0
    assert card.display_name == "Empty Cluster"
    assert card.header.toolTip() == "Empty Cluster"
    assert card.lbl_count.text() == "n = 0"
    assert not card.btn_keycap.isHidden()
    assert card.btn_keycap.text() == "Shift+X"
    assert "Firing Rate" in card.plot.getAxis("left").labelText
    assert not card.mean_curve.isVisible()

    # 2. Update to populated cluster with summary in probability mode
    t_bins = np.linspace(-30.0, 50.0, 40)
    mean_psth = np.maximum(0.0, np.sin(np.linspace(0, np.pi, 40)))
    sem_psth = mean_psth * 0.1
    summ = ClusterSummary(
        cluster_id=0,
        count=8,
        color_hex="#10B981",
        unit_ids=["u1", "u2"],
        time_bins_ms=t_bins,
        mean_psth=mean_psth,
        sem_psth=sem_psth,
        norm_mode="probability",
    )

    card.update_card(
        summary=summ,
        count=8,
        color_hex="#10B981",
        custom_name="Burst",
        pulse_duration_ms=10.0,
        is_smooth=True,
        norm_mode="probability",
        hotkey="Ctrl+1",
    )

    assert id(card.mean_curve) == curve_id
    assert card.count == 8
    assert card.display_name == "Burst"
    assert card.header.toolTip() == "Burst"
    assert card.lbl_count.text() == "n = 8"
    assert not card.btn_keycap.isHidden()
    assert card.btn_keycap.text() == "Ctrl+1"
    assert "Probability" in card.plot.getAxis("left").labelText
    assert card.mean_curve.isVisible()

    # 3. Mode switch back to peak mode with smooth=False
    card.update_card(
        summary=summ,
        count=8,
        color_hex="#10B981",
        custom_name="Burst",
        pulse_duration_ms=10.0,
        is_smooth=False,
        norm_mode="peak",
        hotkey="",
    )

    assert id(card.mean_curve) == curve_id
    assert card.btn_keycap.isHidden()
    assert card.btn_keycap.text() == ""
    assert "Norm. rate" in card.plot.getAxis("left").labelText
    assert card.mean_curve.isVisible()

    card.close()


def test_render_raster_and_psth_empty_data_and_theme(qapp):
    """Verify render_raster and render_psth gracefully handle empty arrays and configure_plot_theme."""
    plot_widget = pg.PlotWidget()

    # Empty render_raster
    render_raster(plot_widget, raster_x=np.array([]), raster_y=np.array([]), n_trials=0)
    assert hasattr(plot_widget.getPlotItem(), "_raster_cache")
    assert not plot_widget.getPlotItem()._raster_cache["scatter"].isVisible()

    # Empty render_psth smooth
    render_psth(
        plot_widget,
        psth_bins=np.array([]),
        psth_rates=np.array([]),
        baseline_mean=0.0,
        baseline_3sigma=0.0,
        is_smooth=True,
    )
    assert hasattr(plot_widget.getPlotItem(), "_psth_cache")
    assert not plot_widget.getPlotItem()._psth_cache["smooth_curve"].isVisible()
    assert not plot_widget.getPlotItem()._psth_cache["bar_item"].isVisible()

    # Empty render_psth non-smooth
    render_psth(
        plot_widget,
        psth_bins=np.array([]),
        psth_rates=np.array([]),
        baseline_mean=0.0,
        baseline_3sigma=0.0,
        is_smooth=False,
    )
    assert not plot_widget.getPlotItem()._psth_cache["smooth_curve"].isVisible()
    assert not plot_widget.getPlotItem()._psth_cache["bar_item"].isVisible()

    plot_widget.close()


def test_incremental_comparison_drawer_switch_clusters(qapp, tmp_path):
    """Verify comparison drawer performs full rebuild when switching cluster_id but incremental update for same cluster."""
    exp_dir = create_synthetic_experiment_folder(tmp_path / "exp_switch")
    view = ClusteringView(start_with_clusters=True)
    view.show()
    view.add_experiment_folder(exp_dir)

    all_gids = [str(r["global_unit_id"]) for r in view._pooled_rows]
    assert len(all_gids) >= 3

    # Assign unit 0 to cluster 0, unit 1 to cluster 1
    view.assign_candidate_to_cluster(0, unit_id=all_gids[0])
    view.assign_candidate_to_cluster(1, unit_id=all_gids[1])

    # Open cluster 0 comparison
    view.set_cluster_comparison(0)
    assert view._comparison_rendered_cid == 0
    assert all_gids[0] in view._comparison_member_viewers_map
    assert all_gids[1] not in view._comparison_member_viewers_map

    # Switch to cluster 1 comparison (full rebuild across cluster switch)
    view.set_cluster_comparison(1)
    assert view._comparison_rendered_cid == 1
    assert all_gids[1] in view._comparison_member_viewers_map
    assert all_gids[0] not in view._comparison_member_viewers_map

    view.close()


def test_incremental_pooled_table_cluster_items_update(qapp):
    """Verify _update_all_pooled_cluster_items with target_gids updates rows in-place without touching other rows."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    all_gids = list(view._pooled_units_map.keys())
    assert len(all_gids) >= 3

    target_gid = all_gids[0]
    untouched_gid = all_gids[1]

    # Find row index for untouched_gid and get its QTableWidgetItem reference
    untouched_row = -1
    for r in range(view.table_pooled_units.rowCount()):
        if view._get_gid_for_row(r) == untouched_gid:
            untouched_row = r
            break
    assert untouched_row >= 0
    orig_untouched_item = view.table_pooled_units.item(untouched_row, 2)
    orig_untouched_item_id = id(orig_untouched_item)

    # Change cluster assignment for target_gid
    view._pooled_units_map[target_gid]["cluster_id"] = 3
    view._unit_status_map[target_gid] = "Curated"

    # Call incremental update targeting ONLY target_gid
    view._update_all_pooled_cluster_items(target_gids={target_gid})

    # Untouched row item must be the exact same QTableWidgetItem in memory
    assert id(view.table_pooled_units.item(untouched_row, 2)) == orig_untouched_item_id

    # Target row item was updated in-place
    target_row = -1
    for r in range(view.table_pooled_units.rowCount()):
        if view._get_gid_for_row(r) == target_gid:
            target_row = r
            break
    assert target_row >= 0
    assert "Cluster 3" in view.table_pooled_units.item(target_row, 2).text()
    assert view.table_pooled_units.item(target_row, 6).toolTip() == "Curated"

    view.close()


def test_inspector_viewport_lazy_hydration_and_scroll(qapp):
    """Verify Inspector uses viewport-lazy HiDPI pixmap hydration, 0 scroll-area QGraphicsViews, and lazy promotion."""
    from optocurator.ui.plots import UnitResponsePlot

    view = ClusteringView(load_mock=True, start_with_clusters=True)
    rng = np.random.default_rng(7)
    onsets = np.arange(1.0, 21.0, 0.5)
    sess = "01942_74R_20261019"
    view._experiment_stimulus[sess] = onsets
    view._experiment_spikes.setdefault(sess, {})
    base_rec = dict(next(iter(view._pooled_units_map.values())))
    for uid in range(200, 232):
        gid = f"{sess}_u{uid:04d}"
        view._experiment_spikes[sess][uid] = np.sort(
            np.concatenate([onsets + rng.uniform(-0.02, 0.04, len(onsets)), onsets + 0.005])
        )
        rec = dict(base_rec)
        rec["global_unit_id"] = gid
        rec["unit_id"] = uid
        rec["session_id"] = sess
        rec["cluster_id"] = 0
        rec["status"] = "Curated"
        view._pooled_units_map[gid] = rec
        view._unit_status_map[gid] = "Curated"

    view.resize(1920, 1080)
    view.show()
    qapp.processEvents()

    view.set_cluster_comparison(0)
    # Pause background idle pixmap prewarm to assert initial viewport culling strictly
    view._prewarm_timer.stop()
    view._prewarm_queue.clear()

    total_viewers = len(view.comparison_viewers)
    assert total_viewers >= 32
    # 0 live UnitResponsePlot widgets mounted inside scroll area
    assert view.comparison_list_widget.findChildren(UnitResponsePlot) == []

    hydrated_initial = sum(1 for v in view.comparison_viewers if not v._pixmap_label.pixmap().isNull())
    assert 1 <= hydrated_initial <= 8
    assert view.comparison_viewers[-1]._pixmap_label.pixmap().isNull()

    # Let layout settle, scroll to bottom, and flush deferred scroll hydration
    qapp.processEvents()
    view._prewarm_timer.stop()
    sb = view.comparison_scroll_area.verticalScrollBar()
    sb.setValue(sb.maximum())
    view._on_inspector_scrolled(sb.value())
    view._on_scroll_hydrate_timeout()
    assert not view.comparison_viewers[-1]._pixmap_label.pixmap().isNull()

    # Changing PSTH smoothing invalidates cached member pixmaps and re-hydrates visible cards
    old_key = view.comparison_viewers[-1]._rendered_key
    view.set_psth_smooth(False)
    assert len(view._member_pixmap_cache) < total_viewers
    assert view.comparison_viewers[-1]._rendered_key != old_key

    # Explicit .unit_response_plot access lazily promotes only that card to a real UnitResponsePlot
    promoted = view.comparison_viewers[0].unit_response_plot
    assert isinstance(promoted, UnitResponsePlot)
    assert promoted.size() == view.unit_response_plot.size()
    assert promoted.raster_plot.getPlotItem()._raster_cache["scatter"].isVisible() is True

    view.close()
