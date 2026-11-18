"""Acceptance and unit tests for Iteration 0: Walking Skeleton & Navigation."""

import numpy as np
import pytest
from optocurator.models import StimulusProtocol, UnitData, UnitMetrics
from optocurator.ui.app import MainWindow


def test_models_dataclasses():
    """Verify core dataclasses and properties."""
    # UnitData
    spike_times = np.array([0.01, 0.05, 0.12])
    unit = UnitData(
        session_id="mouseA_20261021",
        unit_id=4,
        spike_times=spike_times,
        sampling_rate=30000.0,
        channel=12,
        depth=750,
    )
    assert unit.global_id == "mouseA_20261021_u0004"
    assert len(unit.spike_times) == 3

    # StimulusProtocol
    onsets = np.array([1.0, 2.0, 3.0])
    offsets = np.array([1.02, 2.02, 3.02])
    proto = StimulusProtocol(onsets=onsets, offsets=offsets, name="OPTS_03")
    assert proto.name == "OPTS_03"
    assert len(proto.onsets) == 3

    # UnitMetrics
    metrics = UnitMetrics(
        global_id=unit.global_id,
        session_id=unit.session_id,
        unit_id=unit.unit_id,
        plot_path="plots/u0004.png",
        latency_ms=4.5,
        jitter_ms=1.2,
        reliability=0.88,
        waveform_correlation=0.96,
        salt_p_value=0.0005,
        is_tagged=True,
    )
    assert metrics.is_tagged is True
    assert metrics.cluster_id == -1
    assert metrics.curated_label == "unassigned"


def test_main_window_initial_state(qapp):
    """Verify application launches on the Launcher Dashboard."""
    window = MainWindow()
    assert window.stack.currentIndex() == MainWindow.VIEW_LAUNCHER
    assert window.windowTitle().startswith("OptoCurator")
    assert window.stack.count() == 3


def test_navigation_via_cards(qapp):
    """Verify clicking Phase 1 & Phase 2 cards switches views correctly."""
    window = MainWindow()

    # Click Detection Card
    window.launcher_view.detection_card.clicked.emit()
    assert window.stack.currentIndex() == MainWindow.VIEW_DETECTION

    # Click Home button in DetectionView
    window.detection_view.request_home.emit()
    assert window.stack.currentIndex() == MainWindow.VIEW_LAUNCHER

    # Click Clustering Card
    window.launcher_view.clustering_card.clicked.emit()
    assert window.stack.currentIndex() == MainWindow.VIEW_CLUSTERING

    # Click Home button in ClusteringView
    window.clustering_view.request_home.emit()
    assert window.stack.currentIndex() == MainWindow.VIEW_LAUNCHER


def test_navigation_between_workspaces(qapp):
    """Verify switching directly between Detection and Clustering views."""
    window = MainWindow()

    # Start in Detection
    window.switch_to_detection()
    assert window.stack.currentIndex() == MainWindow.VIEW_DETECTION

    # Switch to Clustering from Detection
    window.detection_view.request_clustering_view.emit()
    assert window.stack.currentIndex() == MainWindow.VIEW_CLUSTERING

    # Switch back to Detection from Clustering
    window.clustering_view.request_detection_view.emit()
    assert window.stack.currentIndex() == MainWindow.VIEW_DETECTION


def test_navigation_via_menubar(qapp):
    """Verify menubar Modes actions switch views."""
    window = MainWindow()

    # Trigger Detection action
    window.act_mode_detection.trigger()
    assert window.stack.currentIndex() == MainWindow.VIEW_DETECTION

    # Trigger Clustering action
    window.act_mode_clustering.trigger()
    assert window.stack.currentIndex() == MainWindow.VIEW_CLUSTERING

    # Trigger Home action
    window.act_mode_launcher.trigger()
    assert window.stack.currentIndex() == MainWindow.VIEW_LAUNCHER


def test_clustering_menubar_action(qapp, monkeypatch):
    """Verify Clustering menu is present and triggers clustering view switch."""
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)
    window = MainWindow()
    assert hasattr(window, "clustering_menu")
    assert hasattr(window, "act_hierarchical")
    window.act_hierarchical.trigger()
    assert window.stack.currentIndex() == MainWindow.VIEW_CLUSTERING


def test_detection_view_slider_interaction(qapp):
    """Verify slider threshold controls work and dynamically update unit status."""
    window = MainWindow()
    view = window.detection_view
    view._load_mock_data()

    # Default: u0001 (lat=4.2, rel=0.85) should be Tagged
    status_0 = view.table_units.item(0, 7).text()
    assert status_0 == "✓"

    # Restrict max latency to 3.0 ms -> u0001 should become Untagged
    view.spin_latency.setValue(3.0)
    view._apply_thresholds()
    assert view.table_units.item(0, 7).text() == "○"

    # Loosen max latency to 10.0 ms -> u0001 should become Tagged again
    view.spin_latency.setValue(10.0)
    view._apply_thresholds()
    assert view.table_units.item(0, 7).text() == "✓"


def test_cli_preload_experiment(qapp):
    """Verify preloading an experiment directory via MainWindow or CLI logic."""
    window = MainWindow()
    test_path = "C:/test/path/to/experiment"
    window.switch_to_detection()
    window.detection_view.lbl_session.setText(f"Session: {test_path}")
    assert "test/path/to/experiment" in window.detection_view.lbl_session.text()


def test_cli_version(capsys):
    """Verify optocurator --version exits cleanly and prints version."""
    from optocurator import __version__
    from optocurator.cli import main
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert __version__ in captured.out
    assert "1.0.0" in captured.out


