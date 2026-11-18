"""Tests verifying that OptoCurator starts up completely clean with no mock data loaded."""

import numpy as np
from optocurator.ui.app import MainWindow
from optocurator.ui.view_clusters import ClusteringView
from optocurator.ui.view_detection import DetectionView


def test_detection_view_starts_empty(qapp):
    """Verify DetectionView initializes completely clean with 0 units and no mock data."""
    view = DetectionView()

    assert len(view.units) == 0
    assert view.session_id == "None"
    assert view.lbl_session.text() == "Session: None"
    assert view.lbl_unit_summary.text() == "0 units"
    assert view.table_units.rowCount() == 0
    assert not view.raster_plot.getPlotItem().titleLabel.text

    # X-axis range matches current_window_ms
    x_range = view.raster_plot.getViewBox().viewRange()[0]
    assert np.isclose(x_range[0], view.current_window_ms[0])
    assert np.isclose(x_range[1], view.current_window_ms[1])

    # Edge cases: operations on empty state should not crash
    view._apply_thresholds()
    assert view.lbl_unit_summary.text() == "0 units"

    view._toggle_selected_unit_tag()
    view._apply_status_filter()
    assert view.lbl_unit_summary.text() == "0 units"

    view.close()


def test_clustering_view_starts_empty(qapp):
    """Verify ClusteringView initializes completely clean with 0 experiments and no mock data."""
    view = ClusteringView()

    assert view._is_mock is False
    assert len(view.loaded_experiments) == 0
    assert len(view._pooled_rows) == 0
    assert len(view._pooled_units_map) == 0
    assert view.table_exps.rowCount() == 0
    assert view.table_pooled_units.rowCount() == 0
    assert view.lbl_pool_count.text() == "0 units"
    assert view.lbl_candidate.text() == "No Experiment Selected"
    assert view.combo_filter_cluster.itemText(0) == "All units"
    assert len(view.scatter_item.points()) == 0
    raster_scatter = view.unit_response_plot.raster_plot.getPlotItem()._raster_cache["scatter"]
    psth_curve = view.unit_response_plot.psth_plot.getPlotItem()._psth_cache["smooth_curve"]
    rx, _ = raster_scatter.getData()
    px, _ = psth_curve.getData()
    assert len(rx) == 0
    assert len(px) == 0
    assert raster_scatter.isVisible() is False
    assert psth_curve.isVisible() is False

    # Edge cases: operations on empty state should not crash
    res = view.create_new_cluster_for_candidate()
    assert res == -1

    cand = view.refresh_active_learning()
    assert cand is None
    assert view.lbl_candidate.text() == "No Experiment Selected"

    view._apply_cluster_filter()
    assert view.lbl_pool_count.text() == "0 units"

    view.close()


def test_main_window_starts_empty(qapp):
    """Verify MainWindow initializes with no data in either detection or clustering views."""
    win = MainWindow()

    # Detection view is clean
    assert win.detection_view.table_units.rowCount() == 0
    assert win.detection_view.session_id == "None"
    assert win.detection_view.lbl_session.text() == "Session: None"
    assert win.detection_view.lbl_unit_summary.text() == "0 units"
    assert not win.detection_view.raster_plot.getPlotItem().titleLabel.text

    # Clustering view is clean
    assert win.clustering_view.table_exps.rowCount() == 0
    assert win.clustering_view.table_pooled_units.rowCount() == 0
    assert win.clustering_view.lbl_pool_count.text() == "0 units"
    assert win.clustering_view.lbl_candidate.text() == "No Experiment Selected"

    win.close()


def test_detection_view_no_title_and_x_range_sync(qapp):
    """Verify raster_plot has no heading title and x-range syncs with current_window_ms."""
    view = DetectionView(load_mock=True)

    # Initial selected unit: no title and accurate xRange
    assert not view.raster_plot.getPlotItem().titleLabel.text
    x_range_init = view.raster_plot.getViewBox().viewRange()[0]
    assert np.isclose(x_range_init[0], -30.0)
    assert np.isclose(x_range_init[1], 50.0)

    # Change visualized window to (-10.0, 30.0)
    view.set_visualized_window((-10.0, 30.0))
    assert not view.raster_plot.getPlotItem().titleLabel.text
    x_range_new = view.raster_plot.getViewBox().viewRange()[0]
    assert np.isclose(x_range_new[0], -10.0)
    assert np.isclose(x_range_new[1], 30.0)

    # Select another unit: still no title and window preserved
    view.table_units.selectRow(1)
    assert not view.raster_plot.getPlotItem().titleLabel.text
    x_range_row1 = view.raster_plot.getViewBox().viewRange()[0]
    assert np.isclose(x_range_row1[0], -10.0)
    assert np.isclose(x_range_row1[1], 30.0)

    view.close()


def test_dock_theme_and_orientations(qapp):
    """Verify WP9 3-region splitter layout in DetectionView (no DockArea, § 13.2)."""
    from optocurator.ui.splitters import ResettableSplitter

    det_view = DetectionView()
    assert isinstance(det_view.main_splitter, ResettableSplitter)
    assert det_view.main_splitter.count() == 3
    det_view.close()

    # Check ClusteringView splitters (WP3 replaced DockArea with ResettableSplitter)
    from optocurator.ui.splitters import ResettableSplitter

    clust_view = ClusteringView()
    assert isinstance(clust_view.main_splitter, ResettableSplitter)
    assert isinstance(clust_view.right_splitter, ResettableSplitter)

    # Verify no redundant internal plot titles in PCA
    assert not clust_view.pca_plot.getPlotItem().titleLabel.text

    clust_view.close()
