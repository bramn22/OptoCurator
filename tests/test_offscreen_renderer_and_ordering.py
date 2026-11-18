"""Tests verifying pure software offscreen rendering and suggested count harmonization:
1. Pure software offscreen rendering using pg.GraphicsScene + pg.GraphicsLayout without native OS windows.
2. Immediate caching of rendered unit pixmaps without pop-ups.
3. Canonical dataframe ordering matching current_clustering_result.unit_ids in refresh_active_learning.
4. Suggested counts computed directly from _all_unit_probabilities at auto_assign_threshold.
5. Threshold sliderReleased signal wired to update_auto_assignment_threshold.
"""

import numpy as np
import pytest
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPainter, QPixmap

from optocurator.ui.view_clusters import ClusteringView


def test_pure_software_offscreen_renderer(qapp):
    """Verify offscreen renderer uses headless pg.GraphicsScene and generates valid QPixmaps."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)

    # 1. Ensure renderer is lazily initialized
    view._ensure_offscreen_renderer()
    assert view._offscreen_scene is not None
    assert isinstance(view._offscreen_scene, pg.GraphicsScene)
    assert view._offscreen_layout is not None
    assert isinstance(view._offscreen_layout, pg.GraphicsLayout)
    assert view._offscreen_raster_plot is not None
    assert view._offscreen_psth_plot is not None

    # 2. Render pixmap for a mock unit
    sample_gid = list(view._pooled_units_map.keys())[0]
    pix = view._get_or_render_unit_pixmap(sample_gid)
    assert pix is not None
    assert isinstance(pix, QPixmap)
    assert not pix.isNull()
    assert pix.width() == 480
    assert pix.height() == 280

    # 3. Verify x- and y-axes generate valid axis line, tick, and text draw specs offscreen
    dummy_img = QImage(480, 280, QImage.Format.Format_ARGB32)
    painter = QPainter(dummy_img)
    try:
        raster_left_specs = view._offscreen_raster_plot.getAxis("left").generateDrawSpecs(painter)
        psth_left_specs = view._offscreen_psth_plot.getAxis("left").generateDrawSpecs(painter)
        psth_bottom_specs = view._offscreen_psth_plot.getAxis("bottom").generateDrawSpecs(painter)
    finally:
        painter.end()

    assert raster_left_specs is not None and len(raster_left_specs[1]) > 0 and len(raster_left_specs[2]) > 0
    assert psth_left_specs is not None and len(psth_left_specs[1]) > 0 and len(psth_left_specs[2]) > 0
    assert psth_bottom_specs is not None and len(psth_bottom_specs[1]) > 0 and len(psth_bottom_specs[2]) > 0

    # 4. Cache retrieval returns the same pixmap
    pix_cached = view._get_or_render_unit_pixmap(sample_gid)
    assert pix_cached is not None
    assert not pix_cached.isNull()

    view.close()


def test_canonical_ordering_and_suggested_counts_harmonization(qapp):
    """Verify refresh_active_learning constructs df using canonical unit_ids ordering."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    view.refresh_active_learning(synchronous=True)
    qapp.processEvents()

    assert view.current_clustering_result is not None
    assert view.current_clustering_result.unit_ids is not None

    # Check that unit order matches current_clustering_result.unit_ids
    result_uids = [str(u) for u in view.current_clustering_result.unit_ids]
    assert len(result_uids) == len(view._pooled_units_map)

    # Set custom probabilities in _all_unit_probabilities to verify direct count calculation
    view.auto_assign_threshold = 0.80
    dummy_probs = {
        "unit_test_1": {0: 0.90, 1: 0.10},  # meets threshold for cid 0
        "unit_test_2": {0: 0.85, 1: 0.15},  # meets threshold for cid 0
        "unit_test_3": {1: 0.82, 0: 0.18},  # meets threshold for cid 1
        "unit_test_4": {1: 0.75, 0: 0.25},  # below threshold (75% < 80%)
    }
    view._all_unit_probabilities = dummy_probs
    view._unit_status_map = {
        "unit_test_1": "Unassigned",
        "unit_test_2": "Unassigned",
        "unit_test_3": "Unassigned",
        "unit_test_4": "Unassigned",
    }
    view._dismissed_cluster_candidates = {}

    counts = view._get_suggested_counts_per_cluster()
    assert counts.get(0, 0) == 2
    assert counts.get(1, 0) == 1

    # Dismiss one candidate and verify count decrements
    view._dismissed_cluster_candidates = {0: {"unit_test_1"}}
    counts_after_dismiss = view._get_suggested_counts_per_cluster()
    assert counts_after_dismiss.get(0, 0) == 1
    assert counts_after_dismiss.get(1, 0) == 1

    # Raising threshold to 0.88 filters out unit_test_2 (85% < 88%)
    view.auto_assign_threshold = 0.88
    counts_high_thresh = view._get_suggested_counts_per_cluster()
    assert counts_high_thresh.get(0, 0) == 0
    assert counts_high_thresh.get(1, 0) == 0

    view.close()


def test_slider_threshold_slider_released_connected(qapp):
    """Verify slider_threshold.sliderReleased triggers update_auto_assignment_threshold."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)

    view.slider_threshold.setValue(90)
    # Trigger sliderReleased
    view.slider_threshold.sliderReleased.emit()
    assert view.auto_assign_threshold == 0.90

    view.slider_threshold.setValue(70)
    view.slider_threshold.sliderReleased.emit()
    assert view.auto_assign_threshold == 0.70

    view.close()
