"""Unit tests for Cluster PSTH Normalization (Peak, Probability, Raw) in OptoCurator Phase 2."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from optocurator.analysis.clustering import (
    FeatureConfiguration,
    ClusteringParameters,
    ClusterSummary,
    compute_cluster_summaries,
    normalize_psth_waveform,
)
from optocurator.io.project import (
    ClusteringProjectData,
    serialize_model_parameters,
    deserialize_model_parameters,
    save_clustering_project,
    load_clustering_project,
)
from optocurator.ui.app import MainWindow
from optocurator.ui.view_clusters import ClusteringView, ClusterCardWidget
from tests.test_detection_workflow import create_synthetic_experiment_folder


def test_normalize_psth_waveform_modes():
    """Verify normalize_psth_waveform across peak, probability, and raw modes."""
    waveform = np.array([0.0, 10.0, 50.0, 20.0, 0.0])

    # 1. Peak mode
    norm_peak = normalize_psth_waveform(waveform, norm_mode="peak")
    assert np.max(norm_peak) == pytest.approx(1.0)
    assert norm_peak[2] == 1.0
    assert norm_peak[1] == pytest.approx(0.2)
    assert norm_peak[0] == 0.0

    # 2. Probability mode
    norm_prob = normalize_psth_waveform(waveform, norm_mode="probability")
    assert np.sum(norm_prob) == pytest.approx(1.0)
    assert norm_prob[2] == pytest.approx(50.0 / 80.0)

    # 3. Raw mode
    norm_raw = normalize_psth_waveform(waveform, norm_mode="raw")
    np.testing.assert_array_equal(norm_raw, waveform)

    # 4. Edge case: all zeros
    zeros = np.zeros(10)
    assert np.all(normalize_psth_waveform(zeros, norm_mode="peak") == 0.0)
    assert np.all(normalize_psth_waveform(zeros, norm_mode="probability") == 0.0)
    assert np.all(normalize_psth_waveform(zeros, norm_mode="raw") == 0.0)


def test_compute_cluster_summaries_individual_normalization():
    """Verify compute_cluster_summaries normalizes member cell PSTHs individually first."""
    # Two units in cluster 0 with vastly different absolute firing rates but identical shape
    # Unit 1: peak at 100 Hz
    # Unit 2: peak at 10 Hz
    psth_u1 = np.array([0.0, 20.0, 100.0, 40.0, 0.0])
    psth_u2 = np.array([0.0, 2.0, 10.0, 4.0, 0.0])
    psth_matrix = np.vstack([psth_u1, psth_u2])
    time_bins = np.linspace(-10.0, 30.0, 5)

    df_active = pd.DataFrame([
        {"global_unit_id": "u1", "cluster_id": 0},
        {"global_unit_id": "u2", "cluster_id": 0},
    ])
    labels = np.array([0, 0])

    # 1. Peak normalization
    summaries_peak = compute_cluster_summaries(
        df_active, labels, psth_matrix, time_bins, norm_mode="peak"
    )
    assert 0 in summaries_peak
    summary_peak = summaries_peak[0]
    assert summary_peak.norm_mode == "peak"
    # Both units have identical normalized peak = 1.0 at index 2, so mean peak trace is exactly 1.0!
    assert summary_peak.mean_psth[2] == pytest.approx(1.0)
    assert summary_peak.mean_psth[1] == pytest.approx(0.2)
    # SEM at peak should be 0 since both individual normalized values are 1.0
    assert summary_peak.sem_psth[2] == pytest.approx(0.0)

    # 2. Probability normalization
    summaries_prob = compute_cluster_summaries(
        df_active, labels, psth_matrix, time_bins, norm_mode="probability"
    )
    summary_prob = summaries_prob[0]
    assert summary_prob.norm_mode == "probability"
    assert np.sum(summary_prob.mean_psth) == pytest.approx(1.0)

    # 3. Raw mode
    summaries_raw = compute_cluster_summaries(
        df_active, labels, psth_matrix, time_bins, norm_mode="raw"
    )
    summary_raw = summaries_raw[0]
    assert summary_raw.norm_mode == "raw"
    # Raw mean peak is (100 + 10) / 2 = 55.0 Hz
    assert summary_raw.mean_psth[2] == pytest.approx(55.0)


def test_cluster_card_widget_norm_mode_and_candidate_overlay(qapp):
    """Verify ClusterCardWidget sets Y-axis label and normalizes candidate overlay waveform."""
    t_bins = np.array([-10.0, 0.0, 10.0, 20.0])
    mean_trace = np.array([0.0, 1.0, 0.5])
    sem_trace = np.array([0.0, 0.05, 0.05])
    summary = ClusterSummary(
        cluster_id=0,
        name="Cluster 0",
        count=5,
        color_hex="#2563EB",
        time_bins_ms=t_bins,
        mean_psth=mean_trace,
        sem_psth=sem_trace,
        norm_mode="peak",
    )

    # 1. Peak mode widget
    card_peak = ClusterCardWidget(summary, norm_mode="peak")
    assert card_peak.norm_mode == "peak"
    assert card_peak.plot_psth.getAxis("left").labelText == "Norm. rate"

    # Overlay candidate with peak 80 Hz -> in peak mode, should normalize to 1.0
    candidate_trace = np.array([0.0, 80.0, 20.0])
    card_peak.set_candidate_psth(t_bins, candidate_trace)
    x_data, y_data = card_peak.candidate_psth_curve.getData()
    assert np.max(y_data) == pytest.approx(1.0)

    # 2. Probability mode widget
    card_prob = ClusterCardWidget(summary, norm_mode="probability")
    assert card_prob.norm_mode == "probability"
    assert card_prob.plot_psth.getAxis("left").labelText == "Probability"

    card_prob.set_candidate_psth(t_bins, candidate_trace)
    _, y_data_prob = card_prob.candidate_psth_curve.getData()
    assert np.sum(y_data_prob) == pytest.approx(1.0)

    # 3. Raw mode widget
    card_raw = ClusterCardWidget(summary, norm_mode="raw")
    assert card_raw.norm_mode == "raw"
    assert card_raw.plot_psth.getAxis("left").labelText == "Firing Rate (Hz)"

    card_raw.set_candidate_psth(t_bins, candidate_trace)
    _, y_data_raw = card_raw.candidate_psth_curve.getData()
    assert np.max(y_data_raw) == pytest.approx(80.0)

    card_peak.close()
    card_prob.close()
    card_raw.close()


def test_clustering_project_serialization_cluster_psth_norm(tmp_path):
    """Verify cluster_psth_norm is serialized and restored across project save & load."""
    clust_params = ClusteringParameters(cluster_psth_norm="probability")
    serialized = serialize_model_parameters(clustering_params=clust_params)
    assert serialized["cluster_psth_norm"] == "probability"

    deserialized = deserialize_model_parameters(serialized)
    assert deserialized["cluster_psth_norm"] == "probability"
    assert deserialized["clustering_params"].cluster_psth_norm == "probability"

    # Full project manifest round-trip
    proj_dir = tmp_path / "test_norm_project"
    proj_dir.mkdir(parents=True)
    proj_data = ClusteringProjectData(
        model_parameters=serialized,
        experiments=[],
        clusters={"names": {}, "colors": {}, "card_order": []},
        active_learning_state={},
        unit_assignments=[],
    )
    save_clustering_project(proj_dir, proj_data)
    loaded_data = load_clustering_project(proj_dir)
    loaded_params = deserialize_model_parameters(loaded_data.model_parameters)
    assert loaded_params["cluster_psth_norm"] == "probability"


def test_clustering_view_norm_mode_switching(qapp, tmp_path):
    """Verify ClusteringView UI combo and set_psth_norm_mode recomputes summaries with new norm."""
    exp_dir = create_synthetic_experiment_folder(tmp_path / "exp_norm")
    view = ClusteringView(start_with_clusters=True)
    view.show()
    view.add_experiment_folder(exp_dir)

    assert hasattr(view, "combo_psth_norm")
    assert view.cluster_psth_norm == "probability"
    assert view.combo_psth_norm.currentText() == "Probability"

    # Switch to Peak [0–1] mode via combo (index 0)
    view.combo_psth_norm.setCurrentIndex(0)
    assert view.cluster_psth_norm == "peak"
    assert view.active_clustering_params.cluster_psth_norm == "peak"
    for card in view._cluster_cards.values():
        assert card.norm_mode == "peak"
        assert card.plot_psth.getAxis("left").labelText == "Norm. rate"

    # Switch back to Probability mode via set_psth_norm_mode
    view.set_psth_norm_mode("probability")
    assert view.cluster_psth_norm == "probability"
    assert view.combo_psth_norm.currentIndex() == 1
    for card in view._cluster_cards.values():
        assert card.norm_mode == "probability"
        assert card.plot_psth.getAxis("left").labelText == "Probability"

    # Switch to Raw mode via set_psth_norm_mode
    view.set_psth_norm_mode("raw")
    assert view.cluster_psth_norm == "raw"
    assert view.combo_psth_norm.currentIndex() == 2
    for card in view._cluster_cards.values():
        assert card.norm_mode == "raw"
        assert card.plot_psth.getAxis("left").labelText == "Firing Rate (Hz)"

    view.close()


def test_main_window_cluster_psth_norm_menu(qapp):
    """Verify MainWindow View -> Cluster PSTH Normalization menu actions synchronize with ClusteringView."""
    win = MainWindow()
    win.show()

    assert hasattr(win, "group_psth_norm")
    assert hasattr(win, "psth_norm_actions")
    assert win.psth_norm_actions["probability"].isChecked() is True
    assert win.psth_norm_actions["peak"].isChecked() is False

    # 1. Trigger Peak from menu
    win.psth_norm_actions["peak"].trigger()
    assert win.clustering_view.cluster_psth_norm == "peak"
    assert win.clustering_view.combo_psth_norm.currentText() == "Peak [0–1]"
    assert win.psth_norm_actions["peak"].isChecked() is True
    assert win.psth_norm_actions["probability"].isChecked() is False

    # 2. Change from ClusteringView combo -> menu updates
    win.clustering_view.combo_psth_norm.setCurrentIndex(2)  # Raw
    assert win.psth_norm_actions["raw"].isChecked() is True
    assert win.psth_norm_actions["peak"].isChecked() is False
    assert win.psth_norm_actions["probability"].isChecked() is False

    win.close()
