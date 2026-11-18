"""Unit tests for Elephant Gaussian-smoothed PSTH firing rate calculations and project persistence."""

from pathlib import Path
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from PyQt6.QtWidgets import QInputDialog

from optocurator.analysis.metrics import compute_gaussian_psth, compute_psth_and_raster
from optocurator.analysis.clustering import (
    FeatureConfiguration,
    ClusteringParameters,
    generate_psth_matrix,
)
from optocurator.io.project import (
    ClusteringProjectData,
    serialize_model_parameters,
    deserialize_model_parameters,
    save_clustering_project,
    load_clustering_project,
)
from optocurator.ui.app import MainWindow
from optocurator.ui.view_clusters import ClusteringView
from optocurator.ui.view_detection import DetectionView
from tests.test_detection_workflow import create_synthetic_experiment_folder


def test_gaussian_psth_elephant_computation():
    """Verify that compute_gaussian_psth accurately estimates smoothed firing rate using Elephant."""
    # 4 trials with a spike at +5 ms relative to onset
    raster_x = np.array([5.0, 5.0, 5.0, 5.0])
    n_trials = 4
    window_ms = (-20.0, 40.0)
    bin_size_ms = 2.0
    gaussian_sigma_ms = 2.0

    bins, rates = compute_gaussian_psth(
        raster_x=raster_x,
        n_trials=n_trials,
        window_ms=window_ms,
        bin_size_ms=bin_size_ms,
        gaussian_sigma_ms=gaussian_sigma_ms,
    )

    assert len(rates) == len(bins) - 1
    assert np.all(rates >= 0.0)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    peak_idx = int(np.argmax(rates))
    assert abs(bin_centers[peak_idx] - 5.0) <= 2.0
    # Peak rate should be high (> 50 Hz)
    assert rates[peak_idx] > 50.0


def test_compute_psth_and_raster_smoothing_defaults():
    """Verify compute_psth_and_raster returns smoothed rates by default and provides raw rates."""
    pulse_onsets = np.array([1.0, 2.0, 3.0, 4.0])
    spike_times = np.array([1.005, 2.005, 3.005, 4.005])  # Exactly 5 ms latency

    res = compute_psth_and_raster(
        spike_times=spike_times,
        pulse_onsets=pulse_onsets,
        window_ms=(-20.0, 40.0),
        bin_size_ms=2.0,
        baseline_window_ms=(-20.0, 0.0),
    )

    assert res["is_smooth"] is True
    assert res["gaussian_sigma_ms"] == 2.0
    assert "psth_rates_raw" in res
    assert len(res["psth_rates"]) == len(res["psth_bins"]) - 1
    assert len(res["psth_rates_raw"]) == len(res["psth_bins"]) - 1

    # Raw rate at bin [4, 6] should be 4 spikes / (4 trials * 0.002 s) = 500 Hz
    bins = res["psth_bins"]
    idx_5ms = np.where((bins[:-1] <= 5.0) & (bins[1:] > 5.0))[0][0]
    assert pytest.approx(res["psth_rates_raw"][idx_5ms], abs=1e-3) == 500.0


def test_generate_psth_matrix_smoothing():
    """Verify generate_psth_matrix accepts smoothing parameters and returns valid matrix."""
    df = pd.DataFrame([
        {"global_unit_id": "u1", "latency_ms": 4.0, "reliability": 0.9},
        {"global_unit_id": "u2", "latency_ms": 10.0, "reliability": 0.7},
    ])

    psth_cache = {}
    psth_matrix, time_bins = generate_psth_matrix(
        df,
        time_window_ms=(-20.0, 40.0),
        bin_size_ms=2.0,
        psth_cache=psth_cache,
        smooth=True,
        gaussian_sigma_ms=2.5,
    )

    assert psth_matrix.shape == (2, len(time_bins))
    assert np.all(psth_matrix >= 0.0)
    # Check that cache keys include smoothing parameters
    for k in psth_cache:
        assert len(k) == 5
        assert k[3] is True  # smooth
        assert k[4] == 2.5   # gaussian_sigma_ms


def test_project_model_parameters_smoothing_roundtrip(tmp_path):
    """Verify that psth_smooth and psth_gaussian_sigma_ms serialize and deserialize accurately."""
    fc = FeatureConfiguration(psth_smooth=True, psth_gaussian_sigma_ms=3.5)
    cp = ClusteringParameters(psth_smooth=True, psth_gaussian_sigma_ms=3.5)

    serialized = serialize_model_parameters(
        feature_config=fc,
        clustering_params=cp,
        psth_smooth=True,
        psth_gaussian_sigma_ms=3.5,
    )

    assert serialized["psth_smooth"] is True
    assert serialized["psth_gaussian_sigma_ms"] == 3.5

    deserialized = deserialize_model_parameters(serialized)
    assert deserialized["psth_smooth"] is True
    assert deserialized["psth_gaussian_sigma_ms"] == 3.5
    assert deserialized["feature_config"].psth_smooth is True
    assert deserialized["feature_config"].psth_gaussian_sigma_ms == 3.5
    assert deserialized["clustering_params"].psth_smooth is True
    assert deserialized["clustering_params"].psth_gaussian_sigma_ms == 3.5

    # Test complete project save/load YAML roundtrip
    proj_data = ClusteringProjectData(
        schema_version="1.0",
        model_parameters=serialized,
    )
    yaml_path = save_clustering_project(tmp_path, proj_data)
    loaded_data = load_clustering_project(yaml_path)

    loaded_params = deserialize_model_parameters(loaded_data.model_parameters)
    assert loaded_params["psth_smooth"] is True
    assert loaded_params["psth_gaussian_sigma_ms"] == 3.5


def test_detection_view_psth_smoothing_interaction(qapp, tmp_path):
    """Verify that toggling smoothing and tuning sigma in DetectionView updates state and re-renders PSTH."""
    exp_folder = tmp_path / "exp_session"
    create_synthetic_experiment_folder(exp_folder, n_units=3, n_pulses=15)

    view = DetectionView()
    success = view.load_experiment_folder(exp_folder)
    assert success is True

    # Initial state defaults
    assert view.psth_smooth is True
    assert view.chk_smooth.isChecked() is True
    assert view.psth_gaussian_sigma_ms == 2.0
    assert view.spin_sigma.value() == 2.0
    assert view.spin_sigma.isEnabled() is True

    # Select Unit 0 to trigger rendering
    view.table_units.selectRow(0)

    # When smoothing is enabled, PSTH plot should have a visible continuous curve and no visible BarGraphItem
    items = view.psth_plot.items if isinstance(view.psth_plot.items, list) else view.psth_plot.items()
    bar_items = [it for it in items if isinstance(it, pg.BarGraphItem) and it.isVisible()]
    curve_items = [it for it in items if isinstance(it, (pg.PlotDataItem, pg.PlotCurveItem)) and it.isVisible()]
    assert len(bar_items) == 0
    assert len(curve_items) >= 1

    # Toggle smoothing off via chk_smooth
    view.chk_smooth.setChecked(False)
    assert view.psth_smooth is False
    assert view.spin_sigma.isEnabled() is False

    # PSTH plot should now have a visible BarGraphItem
    items = view.psth_plot.items if isinstance(view.psth_plot.items, list) else view.psth_plot.items()
    bar_items = [it for it in items if isinstance(it, pg.BarGraphItem) and it.isVisible()]
    assert len(bar_items) >= 1

    # Toggle smoothing back on
    view.chk_smooth.setChecked(True)
    assert view.psth_smooth is True
    assert view.spin_sigma.isEnabled() is True
    items = view.psth_plot.items if isinstance(view.psth_plot.items, list) else view.psth_plot.items()
    bar_items = [it for it in items if isinstance(it, pg.BarGraphItem) and it.isVisible()]
    assert len(bar_items) == 0

    # Adjust Gaussian sigma via spin_sigma
    view.spin_sigma.setValue(4.5)
    assert view.psth_gaussian_sigma_ms == 4.5

    view.close()


def test_clustering_view_psth_smoothing_interaction(qapp):
    """Verify that toggling smoothing and tuning sigma in ClusteringView updates model params and re-renders cluster cards."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)

    # Initial defaults
    assert view.psth_smooth is True
    assert view.chk_smooth.isChecked() is True
    assert view.psth_gaussian_sigma_ms == 2.0
    assert view.spin_sigma.value() == 2.0
    assert len(view._cluster_cards) > 0

    # Toggle smoothing off
    view.chk_smooth.setChecked(False)
    assert view.psth_smooth is False
    assert view.feature_config.psth_smooth is False
    assert view.active_clustering_params.psth_smooth is False
    assert len(view._cluster_cards) > 0

    # Adjust Gaussian sigma
    view.spin_sigma.setValue(3.5)
    assert view.psth_gaussian_sigma_ms == 3.5
    assert view.feature_config.psth_gaussian_sigma_ms == 3.5
    assert view.active_clustering_params.psth_gaussian_sigma_ms == 3.5
    assert len(view._cluster_cards) > 0

    # Check ImageViewer preview updates
    if view.table_pooled_units.rowCount() > 0:
        view.table_pooled_units.selectRow(0)
        assert view.image_viewer is not None

    view.close()


def test_main_window_psth_smoothing_menu_delegation(qapp, monkeypatch):
    """Verify MainWindow View -> PSTH Smoothing menu actions delegate to DetectionView and ClusteringView."""
    win = MainWindow()
    win.show()

    assert hasattr(win, "act_smooth_toggle")
    assert hasattr(win, "sigma_actions")
    assert win.act_smooth_toggle.isChecked() is True

    # 1. Toggle smoothing action
    win.act_smooth_toggle.setChecked(False)
    win.act_smooth_toggle.triggered.emit(False)
    assert win.detection_view.psth_smooth is False
    assert win.clustering_view.psth_smooth is False
    assert win.detection_view.chk_smooth.isChecked() is False
    assert win.clustering_view.chk_smooth.isChecked() is False

    win.act_smooth_toggle.setChecked(True)
    win.act_smooth_toggle.triggered.emit(True)
    assert win.detection_view.psth_smooth is True
    assert win.clustering_view.psth_smooth is True

    # 2. Select preset sigma (5.0 ms)
    assert 5.0 in win.sigma_actions
    win.sigma_actions[5.0].trigger()
    assert win.detection_view.psth_gaussian_sigma_ms == 5.0
    assert win.clustering_view.psth_gaussian_sigma_ms == 5.0
    assert win.detection_view.spin_sigma.value() == 5.0
    assert win.clustering_view.spin_sigma.value() == 5.0
    assert win.sigma_actions[5.0].isChecked() is True

    # 3. Custom sigma prompt dialog
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *args, **kwargs: (3.5, True))
    win._on_custom_psth_sigma()
    assert win.detection_view.psth_gaussian_sigma_ms == 3.5
    assert win.clustering_view.psth_gaussian_sigma_ms == 3.5
    assert win.detection_view.spin_sigma.value() == 3.5
    assert win.clustering_view.spin_sigma.value() == 3.5

    win.close()
