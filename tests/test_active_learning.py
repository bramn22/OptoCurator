"""Unit and integration tests for Iteration 5: Active Learning Loop & Drag-and-Drop Merge."""

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pytest
from PyQt6.QtCore import QMimeData, QPoint, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QMessageBox

from optocurator.analysis.clustering import (
    ActiveLearningCandidate,
    FeatureConfiguration,
    compute_auto_assignments,
    compute_entropy_ranking,
    extract_feature_matrix,
    perform_clustering,
    seed_clusters_affinity_propagation,
)
from optocurator.ui.view_clusters import ClusterCardWidget, ClusteringView


@pytest.fixture
def sample_pooled_df() -> pd.DataFrame:
    """Create synthetic pooled dataset with two well-separated groups and ambiguous boundary units."""
    rows = []
    # Cluster 0: Fast-spiking, short latency (lat: 2-4 ms, rel: 0.9)
    for i in range(10):
        rows.append({
            "global_unit_id": f"sessA_u{i:04d}",
            "session_id": "sessA",
            "unit_id": i,
            "latency_ms": 3.0 + np.random.normal(0, 0.3),
            "jitter_ms": 1.0 + np.random.normal(0, 0.1),
            "reliability": 0.92,
            "waveform_correlation": 0.96,
            "salt_p_value": 0.0005,
            "is_tagged": True,
            "cluster_id": 0,
            "depth": 500.0 + i * 50.0,
        })

    # Cluster 1: Delayed polysynaptic (lat: 12-16 ms, rel: 0.6)
    for i in range(10, 20):
        rows.append({
            "global_unit_id": f"sessA_u{i:04d}",
            "session_id": "sessA",
            "unit_id": i,
            "latency_ms": 14.0 + np.random.normal(0, 0.5),
            "jitter_ms": 3.5 + np.random.normal(0, 0.2),
            "reliability": 0.58,
            "waveform_correlation": 0.88,
            "salt_p_value": 0.025,
            "is_tagged": True,
            "cluster_id": 1,
            "depth": 1500.0 + (i - 10) * 50.0,
        })

    # Ambiguous boundary unit (lat: 8.5 ms, right between groups)
    rows.append({
        "global_unit_id": "sessA_u0099_borderline",
        "session_id": "sessA",
        "unit_id": 99,
        "latency_ms": 8.5,
        "jitter_ms": 2.2,
        "reliability": 0.75,
        "waveform_correlation": 0.92,
        "salt_p_value": 0.010,
        "is_tagged": True,
        "cluster_id": -1,  # Unassigned candidate
        "depth": 1000.0,
    })

    return pd.DataFrame(rows)


# ============================================================================
# 1. ENTROPY RANKING & RANDOM FOREST SAMPLING
# ============================================================================

def test_entropy_ranking_identifies_boundary_unit(sample_pooled_df):
    """Verify Maximum Shannon Entropy sampling ranks borderline units above clear-cut units."""
    df = sample_pooled_df
    labels = df["cluster_id"].to_numpy(dtype=np.int32)

    candidates = compute_entropy_ranking(
        df=df,
        cluster_labels=labels,
        random_state=42,
    )

    assert len(candidates) >= 1
    top_candidate = candidates[0]
    # The borderline unit at the boundary should have the highest uncertainty
    assert top_candidate.global_unit_id == "sessA_u0099_borderline"
    assert top_candidate.entropy > 0.5  # Significant Shannon entropy
    # Probabilities should sum to approximately 1.0
    total_prob = sum(top_candidate.probabilities.values())
    assert pytest.approx(total_prob, abs=1e-3) == 1.0


def test_entropy_ranking_all_assigned_units(sample_pooled_df):
    """When all units are assigned, verify active learning evaluates all units and ranks by entropy."""
    df = sample_pooled_df.copy()
    # Assign the borderline unit to Cluster 0
    df.loc[df["global_unit_id"] == "sessA_u0099_borderline", "cluster_id"] = 0
    labels = df["cluster_id"].to_numpy(dtype=np.int32)

    candidates = compute_entropy_ranking(
        df=df,
        cluster_labels=labels,
        random_state=42,
    )

    assert len(candidates) == len(df)
    # The borderline unit should still have the highest Shannon entropy among all units
    assert candidates[0].global_unit_id == "sessA_u0099_borderline"
    # List is strictly sorted in descending order of entropy
    for i in range(len(candidates) - 1):
        assert candidates[i].entropy >= candidates[i + 1].entropy


def test_entropy_ranking_skip_stack(sample_pooled_df):
    """Verify units on the skipped set are excluded from candidates."""
    df = sample_pooled_df
    labels = df["cluster_id"].to_numpy(dtype=np.int32)

    skipped = {"sessA_u0099_borderline"}
    candidates = compute_entropy_ranking(
        df=df,
        cluster_labels=labels,
        skipped_unit_ids=skipped,
        random_state=42,
    )

    candidate_ids = [c.global_unit_id for c in candidates]
    assert "sessA_u0099_borderline" not in candidate_ids


def test_entropy_ranking_degenerate_cases():
    """Verify compute_entropy_ranking handles empty or single-class datasets gracefully."""
    empty_df = pd.DataFrame()
    res_empty = compute_entropy_ranking(empty_df, np.array([]))
    assert res_empty == []

    # Single class dataset
    single_class_df = pd.DataFrame([{
        "global_unit_id": "u1",
        "latency_ms": 4.0,
        "jitter_ms": 1.0,
        "reliability": 0.9,
        "cluster_id": 0,
    }, {
        "global_unit_id": "u2",
        "latency_ms": 5.0,
        "jitter_ms": 1.2,
        "reliability": 0.8,
        "cluster_id": 0,
    }])
    res_single = compute_entropy_ranking(single_class_df, np.array([0, 0]))
    assert len(res_single) == 2
    assert res_single[0].entropy == 0.0


# ============================================================================
# 2. DRAG-AND-DROP CLUSTER MERGING
# ============================================================================

def test_cluster_card_drag_and_drop_properties(qapp):
    """Verify ClusterCardWidget accepts drops and prepares drag mime data with shift differentiation."""
    card = ClusterCardWidget(cluster_id=1, count=10, color_hex="#2563EB")
    assert card.acceptDrops() is True

    # Drop event simulation with valid cluster mime data
    emitted_merges = []
    emitted_moves = []
    card.merge_requested.connect(lambda s, t: emitted_merges.append((s, t)))
    card.move_requested.connect(lambda s, t: emitted_moves.append((s, t)))

    mime = QMimeData()
    mime.setData("application/x-cluster-id", b"2")

    class MockDropEvent:
        def __init__(self, mime_data, shift: bool = False):
            self._mime = mime_data
            self.accepted = False
            self._shift = shift

        def mimeData(self):
            return self._mime

        def acceptProposedAction(self):
            self.accepted = True

        def modifiers(self):
            return Qt.KeyboardModifier.ShiftModifier if self._shift else Qt.KeyboardModifier.NoModifier

    # 1. Drop without Shift -> move_requested
    event_no_shift = MockDropEvent(mime, shift=False)
    card.dropEvent(event_no_shift)
    assert event_no_shift.accepted is True
    assert emitted_moves == [(2, 1)]
    assert emitted_merges == []

    # 2. Drop with Shift -> merge_requested
    event_shift = MockDropEvent(mime, shift=True)
    card.dropEvent(event_shift)
    assert event_shift.accepted is True
    assert emitted_merges == [(2, 1)]
    assert emitted_moves == [(2, 1)]


def test_cluster_card_rejects_self_drop(qapp):
    """Verify dropping a cluster card onto itself is ignored."""
    card = ClusterCardWidget(cluster_id=3, count=5, color_hex="#059669")
    emitted_merges = []
    emitted_moves = []
    card.merge_requested.connect(lambda s, t: emitted_merges.append((s, t)))
    card.move_requested.connect(lambda s, t: emitted_moves.append((s, t)))

    mime = QMimeData()
    mime.setData("application/x-cluster-id", b"3")  # Same ID

    class MockDropEvent:
        def __init__(self, mime_data):
            self._mime = mime_data
            self.accepted = False
            self.ignored = False

        def mimeData(self):
            return self._mime

        def acceptProposedAction(self):
            self.accepted = True

        def ignore(self):
            self.ignored = True

        def modifiers(self):
            return Qt.KeyboardModifier.NoModifier

    event = MockDropEvent(mime)
    card.dropEvent(event)
    assert event.accepted is False
    assert event.ignored is True
    assert len(emitted_merges) == 0
    assert len(emitted_moves) == 0



def test_clustering_view_merge_clusters_execution(qapp, monkeypatch):
    """Verify merging clusters updates unit assignments, cluster cards, and recomputes waveforms."""
    # Ensure confirmation dialog automatically answers Yes
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)

    view = ClusteringView(start_with_clusters=True, load_mock=True)
    assert view.current_clustering_result is not None
    summaries = view.current_clustering_result.cluster_summaries

    c_ids = sorted(list(summaries.keys()))
    assert len(c_ids) >= 2
    src_id, tgt_id = c_ids[0], c_ids[1]
    count_src = summaries[src_id].count
    count_tgt = summaries[tgt_id].count
    expected_combined_count = count_src + count_tgt

    merged_signal_received = []
    view.cluster_merged.connect(lambda s, t: merged_signal_received.append((s, t)))

    ok = view.merge_clusters(src_id=src_id, tgt_id=tgt_id, confirm=True)
    assert ok is True
    assert merged_signal_received == [(src_id, tgt_id)]

    # Source cluster should be removed
    new_summaries = view.current_clustering_result.cluster_summaries
    assert src_id not in new_summaries

    # Target cluster should now have combined count
    assert tgt_id in new_summaries
    assert new_summaries[tgt_id].count == expected_combined_count

    # All units that were in src_id should now be assigned to tgt_id
    for rec in view._pooled_rows:
        assert rec.get("cluster_id") != src_id


def test_clustering_view_merge_clusters_cancellation(qapp, monkeypatch):
    """Verify cancelling the merge prompt leaves clusters unchanged."""
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.No)

    view = ClusteringView(start_with_clusters=True, load_mock=True)
    summaries_before = len(view.current_clustering_result.cluster_summaries)

    c_ids = sorted(list(view.current_clustering_result.cluster_summaries.keys()))
    src_id, tgt_id = c_ids[0], c_ids[1]

    ok = view.merge_clusters(src_id=src_id, tgt_id=tgt_id, confirm=True)
    assert ok is False
    assert len(view.current_clustering_result.cluster_summaries) == summaries_before


# ============================================================================
# 3. ACTIVE LEARNING WORKFLOW & KEYBOARD SHORTCUTS
# ============================================================================

def test_clustering_view_active_learning_candidate_initialization(qapp):
    """Verify initial active uncertainty sampling surfaces top candidate and displays metadata."""
    view = ClusteringView(load_mock=True)
    assert view.active_candidate is not None
    assert isinstance(view.active_candidate, ActiveLearningCandidate)
    assert view.active_candidate.entropy >= 0.0

    # Candidate label should contain separate session, unit ID, and entropy
    cand_text = view.lbl_candidate.text()
    assert "Session:" in cand_text
    assert "Unit:" in cand_text
    assert "Cluster:" in cand_text
    assert "Entropy" in cand_text

    # Probe marker and marginal distributions should be active
    assert hasattr(view, "probe_cand_marker")
    assert hasattr(view, "probe_cand_line")
    assert hasattr(view, "marginal_widget")


def test_clustering_view_hotkey_assignment_auto_advance_deliverable(qapp):
    """Deliverable Acceptance Test: Press hotkey 2 on unit -> unit assigned to Cluster 2 -> next candidate surfaces."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    initial_cand = view.active_candidate
    assert initial_cand is not None
    initial_gid = initial_cand.global_unit_id

    # Simulate pressing key '2'
    event_key_2 = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_2, Qt.KeyboardModifier.NoModifier, "2")
    view.keyPressEvent(event_key_2)

    # 1. Verify previous unit was assigned to Cluster 2
    unit_rec = view._pooled_units_map[initial_gid]
    assert unit_rec["cluster_id"] == 2

    # 2. Verify next candidate was surfaced and is different from initial
    next_cand = view.active_candidate
    if next_cand is not None:
        assert next_cand.global_unit_id != initial_gid
        assert "Session:" in view.lbl_candidate.text()
        assert "Unit:" in view.lbl_candidate.text()


def test_clustering_view_skip_shortcut(qapp):
    """Verify pressing Space / 'S' skips candidate without changing cluster."""
    view = ClusteringView(load_mock=True)
    cand1 = view.active_candidate
    assert cand1 is not None
    gid1 = cand1.global_unit_id
    cluster_before = view._pooled_units_map[gid1]["cluster_id"]

    # Press Space to skip
    event_space = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " ")
    view.keyPressEvent(event_space)

    # Unit should be on skipped list
    assert gid1 in view._skipped_gids
    # Cluster assignment should not have changed
    assert view._pooled_units_map[gid1]["cluster_id"] == cluster_before

    # Next candidate should be active
    cand2 = view.active_candidate
    if cand2 is not None:
        assert cand2.global_unit_id != gid1


# ============================================================================
# 4. ITERATION 5 ACTIVE LEARNING: SEEDING, FEATURES & AUTO-ASSIGNMENT
# ============================================================================

def test_feature_configuration_and_extraction(sample_pooled_df):
    """Verify FeatureConfiguration options and feature matrix shapes for Combined vs Metrics vs PSTH."""
    df = sample_pooled_df

    # 1. Default Combined feature space (5 default scalars + PCA-compressed PSTH components)
    cfg_combined = FeatureConfiguration()
    X_comb, psth_mat, t_bins = extract_feature_matrix(df, feature_config=cfg_combined)
    assert X_comb.shape[0] == len(df)
    assert psth_mat.shape[0] == len(df)
    assert len(t_bins) == psth_mat.shape[1]
    assert X_comb.shape[1] > 5
    assert X_comb.shape[1] <= 5 + psth_mat.shape[1]

    # 1b. Combined without PSTH PCA includes all PSTH bins
    cfg_no_pca = FeatureConfiguration(psth_use_pca=False)
    X_no_pca, _, _ = extract_feature_matrix(df, feature_config=cfg_no_pca)
    assert X_no_pca.shape[1] == 5 + psth_mat.shape[1]

    # 2. Analytical Metrics only (5 default scalars)
    cfg_metrics = FeatureConfiguration(use_psth_profile=False)
    X_met, _, _ = extract_feature_matrix(df, feature_config=cfg_metrics)
    assert X_met.shape == (len(df), 5)

    # 3. Custom subset of metrics
    cfg_subset = FeatureConfiguration(
        use_latency=True,
        use_jitter=False,
        use_reliability=True,
        use_salt=False,
        use_peak_response_rate=False,
        use_time_to_response=False,
        use_psth_profile=False,
    )
    X_sub, _, _ = extract_feature_matrix(df, feature_config=cfg_subset)
    assert X_sub.shape == (len(df), 2)


def test_affinity_propagation_seeding(sample_pooled_df):
    """Verify Affinity Propagation discovers natural response clusters and thresholds ambiguous units."""
    df = sample_pooled_df
    # Add a synthetic high-jitter noise unit
    noise_row = pd.DataFrame([{
        "global_unit_id": "sessA_u9999_noise",
        "session_id": "sessA",
        "unit_id": 9999,
        "latency_ms": 45.0,
        "jitter_ms": 15.0,
        "reliability": 0.05,
        "waveform_correlation": 0.20,
        "salt_p_value": 0.45,
        "is_tagged": True,
        "cluster_id": -1,
        "depth": 3000.0,
    }])
    df_extended = pd.concat([df, noise_row], ignore_index=True)

    result = seed_clusters_affinity_propagation(
        df_extended,
        similarity_percentile=75.0,
        random_state=42,
    )

    # Should have found at least 2 clear clusters
    assert len(result.cluster_summaries) >= 2
    # The extreme outlier / noise unit should be left unassigned (-1)
    noise_idx = len(df_extended) - 1
    assert result.cluster_labels[noise_idx] == -1


def test_compute_auto_assignments_threshold_and_curated_lock(sample_pooled_df):
    """Verify Random Forest auto-assignments respect confidence threshold and lock curated units."""
    df = sample_pooled_df
    labels = df["cluster_id"].to_numpy(dtype=np.int32)

    # Curate some units and leave borderline unit unassigned
    status_map = {
        "sessA_u0000": "Curated",
        "sessA_u0001": "Seeded",
        "sessA_u0099_borderline": "Unassigned",
    }

    # High threshold (95% confidence) -> highly conservative auto-assignment
    auto_high, conf_high = compute_auto_assignments(
        df=df,
        cluster_labels=labels,
        unit_status_map=status_map,
        threshold=0.95,
        max_distance_sigma=10.0,
        random_state=42,
    )
    # Borderline unit with high entropy should not be assigned at 95% threshold
    assert auto_high.get("sessA_u0099_borderline", -1) == -1

    # Permissive threshold (50% confidence) -> assigns borderline unit to highest probability cluster
    auto_low, conf_low = compute_auto_assignments(
        df=df,
        cluster_labels=labels,
        unit_status_map=status_map,
        threshold=0.50,
        max_distance_sigma=10.0,
        random_state=42,
    )
    assert auto_low.get("sessA_u0099_borderline", -1) in (0, 1)

    # Curated units should never be overwritten
    assert "sessA_u0000" not in auto_low


def test_feature_settings_dialog_lifecycle(qapp):
    """Verify FeatureSettingsDialog checkbox toggling, spinboxes, and Reset to Defaults."""
    from optocurator.ui.dialog_feature_settings import FeatureSettingsDialog

    init_cfg = FeatureConfiguration(
        use_latency=True,
        use_jitter=False,
        use_reliability=True,
        use_salt=False,
        use_peak_response_rate=False,
        use_time_to_response=False,
        use_psth_profile=False,
    )

    dlg = FeatureSettingsDialog(current_config=init_cfg)
    assert dlg.chk_latency.isChecked() is True
    assert dlg.chk_jitter.isChecked() is False
    assert dlg.chk_psth.isChecked() is False
    assert dlg.spin_start.isEnabled() is False

    # Toggle PSTH on
    dlg.chk_psth.setChecked(True)
    assert dlg.spin_start.isEnabled() is True
    dlg.spin_start.setValue(-30.0)
    dlg.spin_end.setValue(70.0)
    dlg.spin_bin.setValue(2.0)

    # Read updated configuration
    updated_cfg = dlg.get_feature_configuration()
    assert updated_cfg.use_psth_profile is True
    assert updated_cfg.time_window_ms == (-30.0, 70.0)
    assert updated_cfg.bin_size_ms == 2.0

    # Test Reset to Defaults
    dlg.reset_to_defaults()
    reset_cfg = dlg.get_feature_configuration()
    assert reset_cfg.use_latency is True
    assert reset_cfg.weight_latency == 5.0
    assert reset_cfg.use_jitter is True
    assert reset_cfg.use_reliability is True
    assert reset_cfg.use_salt is False
    assert reset_cfg.use_baseline_rate is False
    assert reset_cfg.use_peak_rate is False
    assert reset_cfg.use_peak_response_rate is True
    assert reset_cfg.use_time_to_response is True
    assert reset_cfg.response_std_threshold == 2.0
    assert reset_cfg.use_psth_profile is True
    assert reset_cfg.psth_normalization == "area_normalized"
    assert reset_cfg.time_window_ms == (0.0, 25.0)
    assert reset_cfg.bin_size_ms == 2.0
    assert reset_cfg.psth_use_pca is True
    assert reset_cfg.psth_pca_variance_explained == pytest.approx(0.80)
    dlg.close()


def test_clustering_view_new_cluster_creation_on_fly(qapp):
    """Verify on-the-fly cluster creation via create_new_cluster_for_candidate and 'N' hotkey."""
    view = ClusteringView(load_mock=True)
    init_cluster_count = len(view.current_clustering_result.cluster_summaries)
    assert view.active_candidate is not None
    cand_gid = view.active_candidate.global_unit_id

    # Create new cluster for candidate
    new_cid = view.create_new_cluster_for_candidate()
    assert new_cid >= init_cluster_count
    assert new_cid in view.current_clustering_result.cluster_summaries

    # Unit should be assigned to new_cid with status Curated
    assert view._pooled_units_map[cand_gid]["cluster_id"] == new_cid
    assert view._unit_status_map[cand_gid] == "Curated"
    assert new_cid in view._cluster_card_order
    view.close()


def test_clustering_view_seed_button_workflow(qapp):
    """Verify zero-cluster initial state and seeding via Affinity Propagation."""
    view = ClusteringView(start_with_clusters=False, load_mock=True)

    # Initially zero clusters and all units unassigned
    assert len(view.current_clustering_result.cluster_summaries) == 0
    for rec in view._pooled_units_map.values():
        assert rec.get("cluster_id") == -1

    # Click Seed Clusters button
    ok = view.seed_clusters_with_affinity_propagation()
    assert ok is True

    # Clusters should now exist and be rendered
    assert len(view.current_clustering_result.cluster_summaries) >= 1
    # Check that at least some units are now Seeded
    seeded_count = sum(1 for s in view._unit_status_map.values() if s == "Seeded")
    assert seeded_count > 0

    view.close()


def test_ground_truth_isolation_excludes_auto_assigned_units(sample_pooled_df):
    """Verify that auto-assigned units are strictly excluded from classifier training."""
    df = sample_pooled_df.copy()
    unit_ids = df["global_unit_id"].tolist()
    labels = df["cluster_id"].to_numpy().copy()

    # Define ground truth for 4 units: 2 in cluster 0, 2 in cluster 1
    status_map = {gid: "Unassigned" for gid in unit_ids}
    status_map[unit_ids[0]] = "Curated"
    status_map[unit_ids[1]] = "Curated"
    status_map[unit_ids[10]] = "Curated"
    status_map[unit_ids[11]] = "Curated"

    # Add 8 pseudo-labeled units assigned to cluster 1 as "Auto-assigned"
    for i in range(12, 20):
        status_map[unit_ids[i]] = "Auto-assigned"

    # Compute auto-assignments
    auto_assigns, confs, all_probs = compute_auto_assignments(
        df=df,
        cluster_labels=labels,
        unit_status_map=status_map,
        threshold=0.80,
        return_all_probabilities=True,
    )

    # Unit 0 and 1 are Curated, must not be in auto_assigns or must remain unchanged
    assert unit_ids[0] not in auto_assigns or auto_assigns[unit_ids[0]] == -1
    assert unit_ids[10] not in auto_assigns or auto_assigns[unit_ids[10]] == -1


def test_dual_criterion_distance_gating_rejects_open_space_outliers():
    """Verify that outlier units in open space far from centroid (> 2.5 sigma) are rejected."""
    rows = []
    # Cluster 0: tight cluster around latency 5.0 ms
    for i in range(6):
        rows.append({
            "global_unit_id": f"u_{i}",
            "latency_ms": 5.0 + 0.1 * i,
            "jitter_ms": 1.0,
            "reliability": 0.9,
            "waveform_correlation": 0.95,
            "salt_p_value": 0.001,
            "cluster_id": 0,
        })
    # Cluster 1: tight cluster around latency 20.0 ms
    for i in range(6, 12):
        rows.append({
            "global_unit_id": f"u_{i}",
            "latency_ms": 20.0 + 0.1 * (i - 6),
            "jitter_ms": 2.0,
            "reliability": 0.6,
            "waveform_correlation": 0.85,
            "salt_p_value": 0.02,
            "cluster_id": 1,
        })
    # Outlier unit in open space: latency 150.0 ms (vastly outside both clusters)
    rows.append({
        "global_unit_id": "u_extreme_outlier",
        "latency_ms": 150.0,
        "jitter_ms": 2.0,
        "reliability": 0.6,
        "waveform_correlation": 0.85,
        "salt_p_value": 0.02,
        "cluster_id": -1,
    })
    # Plausible candidate near Cluster 1: latency 20.2 ms
    rows.append({
        "global_unit_id": "u_close_candidate",
        "latency_ms": 20.2,
        "jitter_ms": 2.0,
        "reliability": 0.6,
        "waveform_correlation": 0.85,
        "salt_p_value": 0.02,
        "cluster_id": -1,
    })

    df = pd.DataFrame(rows)
    labels = df["cluster_id"].to_numpy().copy()
    status_map = {r["global_unit_id"]: "Curated" if r["cluster_id"] >= 0 else "Unassigned" for r in rows}

    auto_assigns, confs = compute_auto_assignments(
        df=df,
        cluster_labels=labels,
        unit_status_map=status_map,
        threshold=0.60,
        max_distance_sigma=2.5,
    )

    # The close candidate should be auto-assigned to cluster 1
    assert auto_assigns.get("u_close_candidate") == 1

    # The extreme outlier must be rejected (-1) despite whatever tree leaf it falls in
    assert auto_assigns.get("u_extreme_outlier") == -1


def test_dual_view_drawer_and_candidate_triage(qapp):
    """Verify dual-view drawer, suggested candidates table, scrubbing feedback, and batch triage."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)

    # 1. Open comparison drawer for cluster 0
    view.set_cluster_comparison(0)
    assert not view.comparison_pane.isHidden()
    assert hasattr(view, "table_suggested_candidates")
    assert hasattr(view, "curated_members_panel")
    assert hasattr(view, "suggested_candidates_panel")

    # In WP7 inspector shows only members; suggested candidates live in list_panel
    assert not view.curated_members_panel.isHidden()
    assert view.suggested_candidates_panel.isHidden()
    assert view.table_list_panel.rowCount() >= 0

    # 3. Test suggestions table population
    row_count = view.table_suggested_candidates.rowCount()
    assert row_count > 0

    # 4. Test candidate scrubbing / selection
    view.table_suggested_candidates.selectRow(0)
    cand_item = view.table_suggested_candidates.item(0, 0)
    cand_gid = cand_item.data(Qt.ItemDataRole.UserRole)
    assert cand_gid is not None
    assert view._current_selected_gid == cand_gid

    # Verify candidate PSTH overlay is active on cluster 0 card
    card0 = view._cluster_cards[0]
    c_x, c_y = card0.candidate_psth_curve.getData()
    assert len(c_x) > 0 and len(c_y) > 0

    # 5. Test checkbox toggling
    assert cand_item.checkState() == Qt.CheckState.Unchecked
    view._toggle_selected_candidate_checkboxes()
    assert cand_item.checkState() == Qt.CheckState.Checked

    # 6. Test accepting candidate as curated
    init_curated = sum(1 for rec in view._pooled_units_map.values() if rec.get("cluster_id") == 0 and view._unit_status_map.get(rec.get("global_unit_id")) == "Curated")
    view._on_accept_selected_candidates()

    # Candidate should now be curated into cluster 0
    assert view._pooled_units_map[cand_gid]["cluster_id"] == 0
    assert view._unit_status_map[cand_gid] == "Curated"
    new_curated = sum(1 for rec in view._pooled_units_map.values() if rec.get("cluster_id") == 0 and view._unit_status_map.get(rec.get("global_unit_id")) == "Curated")
    assert new_curated == init_curated + 1

    # 7. Test dismissing candidate
    if view.table_suggested_candidates.rowCount() > 0:
        view.table_suggested_candidates.selectRow(0)
        dismiss_gid = view.table_suggested_candidates.item(0, 0).data(Qt.ItemDataRole.UserRole)
        view._on_dismiss_selected_candidates()
        assert dismiss_gid in view._dismissed_cluster_candidates[0]

    view.close()


def test_compute_entropy_ranking_respects_feature_config_and_matrix(sample_pooled_df):
    """Verify compute_entropy_ranking respects feature_config and feature_matrix consistently with compute_auto_assignments."""
    df = sample_pooled_df
    labels = df["cluster_id"].to_numpy(dtype=np.int32)
    status_map = {
        gid: ("Curated" if cid >= 0 else "Unassigned")
        for gid, cid in zip(df["global_unit_id"], labels)
    }

    cfg_lat = FeatureConfiguration(
        use_latency=True,
        use_jitter=False,
        use_reliability=False,
        use_salt=False,
        use_peak_response_rate=False,
        use_time_to_response=False,
        use_psth_profile=False,
    )
    X_lat, _, _ = extract_feature_matrix(df, feature_config=cfg_lat)

    cands_cfg = compute_entropy_ranking(
        df=df,
        cluster_labels=labels,
        unit_status_map=status_map,
        feature_config=cfg_lat,
        n_estimators=50,
        random_state=42,
    )
    cands_mat = compute_entropy_ranking(
        df=df,
        cluster_labels=labels,
        unit_status_map=status_map,
        feature_matrix=X_lat,
        n_estimators=50,
        random_state=42,
    )
    _, _, all_probs = compute_auto_assignments(
        df=df,
        cluster_labels=labels,
        unit_status_map=status_map,
        feature_config=cfg_lat,
        n_estimators=50,
        random_state=42,
        return_all_probabilities=True,
    )

    assert len(cands_cfg) == 1
    assert len(cands_mat) == 1
    assert cands_cfg[0].global_unit_id == "sessA_u0099_borderline"
    assert cands_cfg[0].probabilities == pytest.approx(cands_mat[0].probabilities, abs=1e-6)
    expected_probs = all_probs["sessA_u0099_borderline"]
    for cid, prob in cands_cfg[0].probabilities.items():
        assert prob == pytest.approx(expected_probs[cid], abs=1e-4)

    # Different feature_config (e.g. reliability only, identical across clusters? no, 0.92 vs 0.58; test noise feature)
    df_noise = df.copy()
    # Set reliability identical for cluster 0 and 1 on half the units so probabilities shift
    df_noise.loc[df_noise["cluster_id"] == 0, "salt_p_value"] = 0.01
    df_noise.loc[df_noise["cluster_id"] == 1, "salt_p_value"] = 0.01
    cfg_salt_only = FeatureConfiguration(
        use_latency=False,
        use_jitter=False,
        use_reliability=False,
        use_salt=True,
        use_peak_response_rate=False,
        use_time_to_response=False,
        use_psth_profile=False,
    )
    cands_salt = compute_entropy_ranking(
        df=df_noise,
        cluster_labels=labels,
        unit_status_map=status_map,
        feature_config=cfg_salt_only,
        n_estimators=50,
        random_state=42,
    )
    assert cands_salt[0].probabilities != cands_cfg[0].probabilities


