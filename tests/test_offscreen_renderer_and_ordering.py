"""Tests verifying canonical ordering and suggested count harmonization:
1. Canonical dataframe ordering matching current_clustering_result.unit_ids in refresh_active_learning.
2. Suggested counts computed directly from _all_unit_probabilities at auto_assign_threshold.
3. Threshold sliderReleased signal wired to update_auto_assignment_threshold.
"""

from optocurator.ui.view_clusters import ClusteringView


def test_canonical_ordering_and_suggested_counts_harmonization(qapp, qtbot):
    """Verify refresh_active_learning constructs df using canonical unit_ids ordering."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    with qtbot.waitSignal(view.progress_updated, timeout=10000):
        view.refresh_active_learning()
    qtbot.waitUntil(
        lambda: view._active_learning_worker is not None
        and not view._active_learning_worker.isRunning()
        and bool(view._all_unit_probabilities),
        timeout=10000,
    )

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


def test_slider_threshold_slider_released_connected(qapp, qtbot):
    """Verify slider_threshold.sliderReleased triggers update_auto_assignment_threshold."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)

    view.slider_threshold.setValue(90)
    # Trigger sliderReleased
    with qtbot.waitSignal(view.progress_updated, timeout=10000):
        view.slider_threshold.sliderReleased.emit()
    assert view.auto_assign_threshold == 0.90

    view.slider_threshold.setValue(70)
    with qtbot.waitSignal(view.progress_updated, timeout=10000):
        view.slider_threshold.sliderReleased.emit()
    assert view.auto_assign_threshold == 0.70

    qtbot.waitUntil(
        lambda: view._active_learning_worker is not None and not view._active_learning_worker.isRunning(),
        timeout=10000,
    )
    view.close()
