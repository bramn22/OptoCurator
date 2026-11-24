"""Tests verifying the Active Learning UX enhancements:
- ClusterCardWidget candidate probability badges and top probability highlighting
- ClusterProbabilityBarWidget bar distributions and threshold line
- ClassificationProgressBar multi-segment breakdown and empty state handling
- End-to-end active learning UX loop triggering card and progress bar updates
"""

import numpy as np
import pandas as pd
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from optocurator.analysis.clustering import ActiveLearningCandidate
from optocurator.ui.theme import build_cluster_card_widget_style
from optocurator.ui.view_clusters import (
    ClassificationProgressBar,
    ClusterCardWidget,
    ClusterProbabilityBarWidget,
    ClusteringView,
)

def test_cluster_card_candidate_probability(qapp):
    """Verify ClusterCardWidget keeps lbl_prob hidden while storing probability text and applying is_top border highlighting."""
    card = ClusterCardWidget(cluster_id=0, count=15, color_hex="#2563EB")
    assert card.lbl_prob.isHidden()
    assert card.lbl_prob.text() == ""

    # Set probability 0.74 without top flag
    card.set_candidate_probability(0.74, is_top=False)
    assert card.lbl_prob.isHidden()
    assert card.lbl_prob.text() == "74%"
    assert card.styleSheet() == build_cluster_card_widget_style("#2563EB", is_top=False)

    # Set probability 0.92 with top flag (glow highlight)
    card.set_candidate_probability(0.92, is_top=True)
    assert card.lbl_prob.isHidden()
    assert card.lbl_prob.text() == "92%"
    assert card.styleSheet() == build_cluster_card_widget_style("#2563EB", is_top=True)

    # Clear probability
    card.set_candidate_probability(None)
    assert card.lbl_prob.isHidden()
    assert card.lbl_prob.text() == ""
    assert card.styleSheet() == build_cluster_card_widget_style("#2563EB", is_top=False)
    card.close()

def test_cluster_probability_bar_widget(qapp):
    """Verify ClusterProbabilityBarWidget bars rendering and threshold update."""
    bar_widget = ClusterProbabilityBarWidget()
    assert bar_widget.threshold_line.value() == 0.85

    probs = {0: 0.82, 1: 0.15, 2: 0.03}
    colors = {0: "#2563EB", 1: "#10B981", 2: "#F59E0B"}
    names = {0: "Cluster 0", 1: "Cluster 1", 2: "Cluster 2"}

    bar_widget.set_probabilities(probs, colors, names, threshold=0.80)
    assert bar_widget.threshold_line.value() == 0.80
    assert len(bar_widget.bar_items) == 3

    # Clear
    bar_widget.clear()
    assert len(bar_widget.bar_items) == 0
    bar_widget.close()

def test_classification_progress_bar_widget(qapp):
    """Verify ClassificationProgressBar segment stretch factors and summary formatting."""
    prog = ClassificationProgressBar()

    # Empty state
    prog.update_counts(0, 0, 0, 0)
    assert "0 / 0 (0%)" in prog.lbl_summary.text()
    assert prog.seg_curated.isHidden()
    assert prog.seg_auto.isHidden()
    assert prog.seg_seeded.isHidden()
    assert not prog.seg_unassigned.isHidden()

    # Populated state
    prog.update_counts(curated=10, auto_assigned=40, seeded=20, unassigned=30)
    assert "70 / 100 (70%)" in prog.lbl_summary.text()
    assert "10 Curated" in prog.lbl_summary.text()
    assert "40 Auto" in prog.lbl_summary.text()
    assert "20 Seeded" in prog.lbl_summary.text()
    assert "30 Remaining" in prog.lbl_summary.text()

    assert not prog.seg_curated.isHidden()
    assert not prog.seg_auto.isHidden()
    assert not prog.seg_seeded.isHidden()
    assert not prog.seg_unassigned.isHidden()
    prog.close()

def test_active_learning_ux_integration(qapp, qtbot, monkeypatch):
    """Verify ClusteringView updates progress bar and probability displays when running active learning."""
    # Start clustering view with mock data
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    qtbot.waitUntil(
        lambda: (view._active_learning_worker is None or not view._active_learning_worker.isRunning())
        and view.active_candidate is not None,
        timeout=10000,
    )
    assert hasattr(view, "progress_bar")
    assert hasattr(view, "cluster_prob_bar")

    # Pick an existing unit from the pooled mock units
    target_gid = list(view._pooled_units_map.keys())[0]
    rec = view._pooled_units_map[target_gid]

    # Mock candidate with probabilities
    candidate = ActiveLearningCandidate(
        global_unit_id=target_gid,
        session_id=str(rec.get("session_id", "mock_sess")),
        unit_id=int(rec.get("unit_id", 1)),
        entropy=0.98,
        probabilities={0: 0.88, 1: 0.12},
        feature_values={"latency_ms": 3.8, "reliability": 0.9},
    )

    view._apply_active_learning_results(
        candidates=[candidate],
        auto_assignments={},
        confidences={},
    )

    # Check candidate display updated
    assert view.active_candidate is not None
    assert view.active_candidate.global_unit_id == target_gid
    assert len(view.cluster_prob_bar.bar_items) >= 2

    # Check that cluster cards received top-card highlight (lbl_prob stays hidden on card header)
    card_0 = view._cluster_cards.get(0)
    if card_0:
        assert card_0.lbl_prob.isHidden()
        assert card_0.lbl_prob.text() == "88%"
        assert card_0.styleSheet() == build_cluster_card_widget_style(card_0.color_hex, is_top=True)

    # Progress bar should reflect active population counts
    assert view.progress_bar.lbl_summary.text() != ""
    assert "Classified:" in view.progress_bar.lbl_summary.text()

    # Assign candidate to cluster 0 and check loop update
    view.assign_candidate_to_cluster(0, target_gid)
    assert view._unit_status_map.get(target_gid) == "Curated"
    qtbot.waitUntil(
        lambda: (view._active_learning_worker is None or not view._active_learning_worker.isRunning())
        and view.progress_bar.curated_count >= 1,
        timeout=10000,
    )
    assert view.progress_bar.curated_count >= 1

    # Reset clustering and verify progress bar resets
    view.reset_to_unassigned()
    qtbot.waitUntil(
        lambda: (view._active_learning_worker is None or not view._active_learning_worker.isRunning())
        and view.progress_bar.curated_count == 0
        and len(view.cluster_prob_bar.bar_items) == 0,
        timeout=10000,
    )
    assert view.progress_bar.curated_count == 0
    assert view.progress_bar.auto_count == 0
    assert len(view.cluster_prob_bar.bar_items) == 0

    view.close()

def test_cycling_through_cells_updates_probabilities(qapp):
    """Verify that selecting or cycling through different cells in the table dynamically updates probabilities and badges."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    assert view.table_pooled_units.rowCount() >= 2

    gid0 = view._get_gid_for_row(0)
    gid1 = view._get_gid_for_row(1)

    # Pre-populate probability distribution cache across units
    view._all_unit_probabilities = {
        gid0: {0: 0.92, 1: 0.08},
        gid1: {0: 0.14, 1: 0.86},
    }

    # 1. Select first cell
    view.table_pooled_units.selectRow(0)
    view._on_pooled_unit_selected()

    assert view._current_selected_gid == gid0
    rec0 = view._pooled_units_map[gid0]
    assert f"Session: {rec0['session_id']}" in view.lbl_candidate.text()
    assert f"Unit: {rec0['unit_id']}" in view.lbl_candidate.text()

    card_0 = view._cluster_cards[0]
    card_1 = view._cluster_cards[1]

    assert card_0.lbl_prob.isHidden()
    assert card_0.lbl_prob.text() == "92%"
    assert card_0.styleSheet() == build_cluster_card_widget_style(card_0.color_hex, is_top=True)
    assert card_1.lbl_prob.isHidden()
    assert card_1.lbl_prob.text() == "8%"
    assert card_1.styleSheet() == build_cluster_card_widget_style(card_1.color_hex, is_top=False)
    assert len(view.cluster_prob_bar.bar_items) >= 2

    # 2. Select second cell (probabilities must switch dynamically)
    view.table_pooled_units.selectRow(1)
    view._on_pooled_unit_selected()

    assert view._current_selected_gid == gid1
    rec1 = view._pooled_units_map[gid1]
    assert f"Session: {rec1['session_id']}" in view.lbl_candidate.text()
    assert f"Unit: {rec1['unit_id']}" in view.lbl_candidate.text()

    assert card_0.lbl_prob.isHidden()
    assert card_0.lbl_prob.text() == "14%"
    assert card_0.styleSheet() == build_cluster_card_widget_style(card_0.color_hex, is_top=False)
    assert card_1.lbl_prob.isHidden()
    assert card_1.lbl_prob.text() == "86%"
    assert card_1.styleSheet() == build_cluster_card_widget_style(card_1.color_hex, is_top=True)

    # 3. Clear probabilities and select a unit with no predictions
    view._all_unit_probabilities.clear()
    view._unit_candidates_map.clear()
    view.table_pooled_units.selectRow(0)
    view._on_pooled_unit_selected()

    assert card_0.lbl_prob.isHidden()
    assert card_1.lbl_prob.isHidden()
    assert card_0.styleSheet() == build_cluster_card_widget_style(card_0.color_hex, is_top=False)
    assert card_1.styleSheet() == build_cluster_card_widget_style(card_1.color_hex, is_top=False)
    assert len(view.cluster_prob_bar.bar_items) == 0

    view.close()

def test_marginal_distributions_populate_on_load(qapp):
    """Verify that marginal distributions populate density curves upon loading without needing cluster seeding."""
    view = ClusteringView(load_mock=True, start_with_clusters=False)

    # 1. Assert all 4 marginal distribution curves have rendered density data
    for key, curve in view.marginal_widget.curves.items():
        assert curve.xData is not None and len(curve.xData) > 0, f"Curve for {key} has no xData"
        assert curve.yData is not None and len(curve.yData) > 0, f"Curve for {key} has no yData"

    # 2. Verify candidate value tracking positions indicator line within bounds
    candidate_vals = {
        "latency_ms": 3.4,
        "jitter_ms": 1.2,
        "reliability": 0.92,
        "salt_p_value": 0.0008,
    }
    view.marginal_widget.set_candidate_values(candidate_vals)

    for key, val in candidate_vals.items():
        line = view.marginal_widget.lines[key]
        assert line.value() == pytest.approx(val, rel=1e-3)
        plot = view.marginal_widget.plots[key]
        view_range = plot.viewRange()
        # Ensure the indicator is within the plot X-range
        x_min, x_max = view_range[0]
        assert x_min <= val <= x_max

    view.close()

def test_toggle_active_learning_mode_and_sequential_advancement(qapp):
    """Verify enabling/disabling Active Learning mode, preserving Auto-assigned status, and sequential table advance."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)

    # 1. Active Learning is ON by default
    assert view.active_learning_enabled is True
    assert view.btn_toggle_active_learning.isChecked() is True
    assert "Active Learning: ON" in view.btn_toggle_active_learning.text()

    # Mark a unit (not in rows 0 or 1) as Auto-assigned
    all_gids = list(view._pooled_units_map.keys())
    auto_gid = all_gids[-1]
    view._unit_status_map[auto_gid] = "Auto-assigned"

    # 2. Toggle Active Learning OFF
    view.btn_toggle_active_learning.click()
    assert view.active_learning_enabled is False
    assert view.btn_toggle_active_learning.isChecked() is False
    assert "Active Learning: OFF" in view.btn_toggle_active_learning.text()

    # Auto-assigned cells must remain intact
    assert view._unit_status_map[auto_gid] == "Auto-assigned"

    # 3. When OFF, hotkey assignment advances sequentially down the table
    view.table_pooled_units.selectRow(0)
    view._on_pooled_unit_selected()
    row0_gid = view._get_gid_for_row(0)

    # Hotkey '0'
    view._on_cluster_hotkey(0)
    assert view._pooled_units_map[row0_gid]["cluster_id"] == 0
    assert view.table_pooled_units.currentRow() == 1
    row1_gid = view._get_gid_for_row(1)
    assert view._current_selected_gid == row1_gid

    # Hotkey '1' advances to row 2
    view._on_cluster_hotkey(1)
    assert view._pooled_units_map[row1_gid]["cluster_id"] == 1
    assert view.table_pooled_units.currentRow() == 2

    # Auto-assigned cells remain untouched during manual assignments in OFF mode
    assert view._unit_status_map[auto_gid] == "Auto-assigned"

    # 4. Toggle Active Learning back ON
    view.btn_toggle_active_learning.click()
    assert view.active_learning_enabled is True
    assert view.btn_toggle_active_learning.isChecked() is True
    assert "Active Learning: ON" in view.btn_toggle_active_learning.text()

    view.close()

def test_return_to_active_candidate_button(qapp, qtbot):
    """Verify returning to suggested active learning cell via btn_return_to_candidate after inspecting other rows."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    qtbot.waitUntil(
        lambda: (view._active_learning_worker is None or not view._active_learning_worker.isRunning())
        and view.active_candidate is not None,
        timeout=10000,
    )
    assert view.active_candidate is not None
    cand_gid = view.active_candidate.global_unit_id

    # 1. Find a row different from the active candidate
    other_row = -1
    for r in range(view.table_pooled_units.rowCount()):
        if view._get_gid_for_row(r) != cand_gid:
            other_row = r
            break
    assert other_row >= 0
    other_gid = view._get_gid_for_row(other_row)

    # 2. Click the other row in table_pooled_units
    view.table_pooled_units.selectRow(other_row)
    view._on_pooled_unit_selected()

    assert view._current_selected_gid == other_gid
    assert view._current_selected_gid != cand_gid
    assert hasattr(view, "btn_return_to_candidate")
    assert view.btn_return_to_candidate.isEnabled() is True

    # 3. Click btn_return_to_candidate
    view.btn_return_to_candidate.click()

    # Focus must return to suggested active learning cell
    assert view._current_selected_gid == cand_gid
    assert view._get_gid_for_row(view.table_pooled_units.currentRow()) == cand_gid

    view.close()

def test_cluster_card_click_opens_and_closes_comparison_pane(qapp):
    """Verify clicking cluster card opens collapsible member comparison pane without changing cell selection, and re-clicking or close button closes it."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)

    # Initially comparison pane is hidden
    assert hasattr(view, "comparison_pane")
    assert hasattr(view, "comparison_scroll_area")
    assert hasattr(view, "comparison_viewers")
    assert view.comparison_pane.isHidden()
    assert view._comparison_cluster_id is None

    # Select row 0
    view.table_pooled_units.selectRow(0)
    view._on_pooled_unit_selected()
    init_selected_gid = view._current_selected_gid
    assert init_selected_gid is not None

    # Click Cluster 0 card
    card_0 = view._cluster_cards[0]
    view._on_cluster_card_clicked(0)

    # Comparison pane should open
    assert not view.comparison_pane.isHidden()
    assert view._comparison_cluster_id == 0
    assert card_0._is_comparing is True
    assert "Comparing: Cluster 0" in view.lbl_comparison_title.text()

    # Selected cell in table must NOT change
    assert view._current_selected_gid == init_selected_gid

    # Re-clicking the same cluster card closes it
    view._on_cluster_card_clicked(0)
    assert view.comparison_pane.isHidden()
    assert view._comparison_cluster_id is None
    assert card_0._is_comparing is False

    # Clicking another card (e.g. Cluster 1) opens it
    card_1 = view._cluster_cards[1]
    view._on_cluster_card_clicked(1)
    assert not view.comparison_pane.isHidden()
    assert view._comparison_cluster_id == 1
    assert card_1._is_comparing is True
    assert card_0._is_comparing is False

    # Clicking btn_comp_close closes it
    view.btn_comp_close.click()
    assert view.comparison_pane.isHidden()
    assert view._comparison_cluster_id is None
    assert card_1._is_comparing is False

    view.close()

def test_marginal_distributions_population_color_and_cluster_overlay(qapp):
    """Verify MarginalDistributionsWidget renders full dataset in light gray (#94A3B8) and overlays comparison cluster in cluster's color."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)

    # 1. Population curves are rendered in light gray (#94A3B8)
    for key, curve in view.marginal_widget.curves.items():
        pen = curve.opts.get("pen")
        assert pen is not None
        assert pen.color().name().upper() == "#94A3B8"
        assert curve.xData is not None and len(curve.xData) > 0

    # 2. Cluster curves are initially empty
    for key, c_curve in view.marginal_widget.cluster_curves.items():
        assert c_curve.xData is None or len(c_curve.xData) == 0

    # 3. Open comparison for cluster 0
    c0_color = view._cluster_colors[0]
    view.set_cluster_comparison(0)

    for key, c_curve in view.marginal_widget.cluster_curves.items():
        assert c_curve.xData is not None and len(c_curve.xData) > 0
        pen = c_curve.opts.get("pen")
        assert pen is not None
        assert pen.color().name().upper() == c0_color.upper()

    # 4. Close comparison clears overlay
    view.clear_cluster_comparison()
    for key, c_curve in view.marginal_widget.cluster_curves.items():
        assert c_curve.xData is None or len(c_curve.xData) == 0

    view.close()

def test_cluster_cards_overlay_candidate_psth(qapp):
    """Verify every ClusterCardWidget overlays the selected/suggested cell's PSTH in COLOR_TEXT (#1F2328, width 1.2)."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)

    # 1. Verify candidate_psth_curve exists on every card with correct styling
    for cid, card in view._cluster_cards.items():
        assert hasattr(card, "candidate_psth_curve")
        pen = card.candidate_psth_curve.opts.get("pen")
        assert pen is not None
        assert pen.color().name().upper() == "#1F2328"
        assert pytest.approx(pen.widthF(), 0.05) == 1.2

    # 2. Select a row -> candidate PSTH should be populated on all cards
    view.table_pooled_units.selectRow(0)
    view._on_pooled_unit_selected()

    for cid, card in view._cluster_cards.items():
        assert card.candidate_psth_curve.xData is not None
        assert len(card.candidate_psth_curve.xData) > 0
        assert card.candidate_psth_curve.yData is not None
        assert len(card.candidate_psth_curve.yData) > 0

    # 3. Clearing unit PSTH resets overlay curves
    view._update_cluster_cards_candidate_psth(None)
    for cid, card in view._cluster_cards.items():
        assert card.candidate_psth_curve.xData is None or len(card.candidate_psth_curve.xData) == 0

    view.close()


def test_auto_assignment_card_count_synchronization_and_revert(qapp):
    """Verify cluster card counts, grand-average summaries, and side-window stay 100% synchronized on auto-assignment and revert."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    assert hasattr(view, "chk_auto_assign")
    assert hasattr(view, "btn_revert_auto")

    # Pick cluster 0
    c0_card = view._cluster_cards.get(0)
    assert c0_card is not None
    initial_c0_count = c0_card.count

    # Simulate auto-assignment of 5 unassigned units to cluster 0
    unassigned_gids = [
        gid for gid, rec in view._pooled_units_map.items()
        if rec.get("cluster_id", -1) != 0 and view._unit_status_map.get(gid) != "Curated"
    ][:5]
    assert len(unassigned_gids) == 5
    for gid in unassigned_gids:
        view._pooled_units_map[gid]["cluster_id"] = -1
        view._unit_status_map[gid] = "Unassigned"

    auto_dict = {gid: 0 for gid in unassigned_gids}
    confs = {gid: 0.95 for gid in unassigned_gids}
    view._apply_active_learning_results([], auto_dict, confs)
    qapp.processEvents()

    # Verify cluster 0 card count is synchronized to initial + 5
    assert view._cluster_cards[0].count == initial_c0_count + 5

    # Verify side comparison pane count matches
    view.set_cluster_comparison(0)
    assert f"N={initial_c0_count + 5}" in view.lbl_comparison_title.text()

    # Revert auto-assignments
    reverted = view.revert_auto_assignments()
    assert reverted >= 5
    qapp.processEvents()

    # Card count should return to initial_c0_count
    assert view._cluster_cards[0].count == initial_c0_count
    assert f"N={initial_c0_count}" in view.lbl_comparison_title.text()

    # Test toggling chk_auto_assign
    view.chk_auto_assign.setChecked(False)
    assert view.auto_assign_enabled is False

    view.close()


def test_empty_cluster_is_automatically_removed(qapp):
    """Verify that when the last unit in a cluster is moved to another cluster, the empty cluster is automatically removed."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    qapp.processEvents()

    # Find a unit and assign it to a brand new cluster
    all_gids = list(view._pooled_units_map.keys())
    sample_gid = all_gids[0]

    # Select this unit and create a new cluster for it
    view.select_pooled_unit(sample_gid)
    new_cid = view.create_new_cluster_for_candidate()
    qapp.processEvents()

    # Verify new cluster was created and has 1 member
    assert new_cid in view._cluster_card_order
    assert new_cid in view._cluster_cards
    assert view._pooled_units_map[sample_gid]["cluster_id"] == new_cid
    assert view._cluster_cards[new_cid].count == 1

    # Open comparison pane on this new cluster
    view.set_cluster_comparison(new_cid)
    assert view._comparison_cluster_id == new_cid
    assert not view.comparison_pane.isHidden()

    # Now move that one cell to cluster 0
    success = view.assign_candidate_to_cluster(0, unit_id=sample_gid)
    assert success is True
    qapp.processEvents()

    # The new_cid now has 0 members and MUST be removed
    assert new_cid not in view._cluster_card_order, f"Cluster {new_cid} should be removed from card order since it has 0 members"
    assert new_cid not in view._cluster_cards, f"Cluster {new_cid} card should be deleted"
    if view.current_clustering_result is not None:
        assert new_cid not in view.current_clustering_result.cluster_summaries

    # Comparison pane for new_cid must be closed
    assert view._comparison_cluster_id is None
    assert view.comparison_pane.isHidden()

    # Filter combobox should not contain new_cid
    combo_cids = [view.combo_filter_cluster.itemData(i) for i in range(view.combo_filter_cluster.count())]
    assert new_cid not in combo_cids

    view.close()


def test_auto_assigned_cells_visible_in_cluster_inspector(qapp):
    """Verify Auto-assigned cells appear in the cluster inspector alongside Curated/Seeded cells."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    try:
        target_cid = view._cluster_card_order[0]
        all_gids = list(view._pooled_units_map.keys())
        curated_gid = next(
            g for g in all_gids
            if view._pooled_units_map[g].get("cluster_id") == target_cid
        )
        view._unit_status_map[curated_gid] = "Curated"

        # Find two unassigned/other units and mark one Auto-assigned to target_cid
        other_gid = next(
            g for g in all_gids
            if g != curated_gid and view._pooled_units_map[g].get("cluster_id") != target_cid
        )
        view._pooled_units_map[other_gid]["cluster_id"] = target_cid
        view._unit_status_map[other_gid] = "Auto-assigned"

        expected_members = [
            g for g, rec in view._pooled_units_map.items()
            if rec.get("cluster_id") == target_cid
        ]
        assert len(expected_members) >= 2

        view.set_cluster_comparison(target_cid)
        qapp.processEvents()

        assert len(view.comparison_viewers) == len(expected_members)
        assert other_gid in view._comparison_member_viewers_map
        assert curated_gid in view._comparison_member_viewers_map
    finally:
        view.close()


def test_async_queue_advances_immediately_and_syncs_on_assign_and_skip(qapp, qtbot):
    """In production async mode, assign and skip in queue focus must advance immediately and sync with worker results (Bug UI3)."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    try:
        qtbot.waitUntil(
            lambda: (view._active_learning_worker is None or not view._active_learning_worker.isRunning())
            and view.active_candidate is not None
            and len(view.ranked_candidates) >= 3,
            timeout=10000,
        )

        # 1. Focus top active-learning queue candidate and Skip in default async mode
        cand0 = view.active_candidate
        assert cand0 is not None
        view.select_pooled_unit(cand0.global_unit_id, source="queue")
        assert view._current_selected_gid == cand0.global_unit_id

        view.skip_current_candidate()
        assert view._current_selected_gid != cand0.global_unit_id

        qtbot.waitUntil(
            lambda: view._active_learning_worker is None or not view._active_learning_worker.isRunning(),
            timeout=10000,
        )
        qapp.processEvents()
        assert view.active_candidate is not None
        assert view._current_selected_gid == view.active_candidate.global_unit_id
        assert view._current_selected_gid != cand0.global_unit_id

        # 2. Assign current queue candidate to Cluster 0 in default async mode
        cand1 = view.active_candidate
        view.select_pooled_unit(cand1.global_unit_id, source="queue")
        assert view.assign_candidate_to_cluster(0) is True
        assert view._unit_status_map[cand1.global_unit_id] == "Curated"
        assert view._current_selected_gid != cand1.global_unit_id

        qtbot.waitUntil(
            lambda: view._active_learning_worker is None or not view._active_learning_worker.isRunning(),
            timeout=10000,
        )
        qapp.processEvents()
        assert view.active_candidate is not None
        assert view._current_selected_gid == view.active_candidate.global_unit_id
        assert view._current_selected_gid != cand1.global_unit_id
    finally:
        view.close()
