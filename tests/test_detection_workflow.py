"""End-to-end integration and workflow tests for Phase 1 MVP Detection Workspace."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from optocurator.ui.app import MainWindow
from optocurator.ui.theme import COLOR_TEXT
from optocurator.ui.view_detection import DetectionView


def create_synthetic_experiment_folder(folder: Path, n_units: int = 5, n_pulses: int = 20) -> Path:
    """Helper to create a synthetic recording directory."""
    folder.mkdir(parents=True, exist_ok=True)
    sr = 30000.0
    pulse_onsets_sec = np.linspace(10.0, 30.0, n_pulses)
    pulse_offsets_sec = pulse_onsets_sec + 0.020

    # Save pulses in samples
    pulses_samples = np.column_stack([
        (pulse_onsets_sec * sr).astype(np.uint32),
        (pulse_offsets_sec * sr).astype(np.uint32),
    ])
    np.save(folder / "opto_idxs_OPTS_03.npy", pulses_samples)

    # Spikes
    all_spikes_samples = []
    all_clusters = []

    for u in range(n_units):
        if u == 0:
            # Direct tagged unit: latency ~ 4ms, high reliability
            spk_sec = pulse_onsets_sec + 0.004
        elif u == 1:
            # Borderline unit: latency ~ 7.5ms
            spk_sec = pulse_onsets_sec + 0.0075
        elif u == 2:
            # Slow untagged unit: latency ~ 18ms
            spk_sec = pulse_onsets_sec + 0.018
        else:
            # Low reliability or spontaneous
            spk_sec = np.linspace(5.0, 35.0, 30)

        spk_samples = (spk_sec * sr).astype(np.uint64)
        all_spikes_samples.extend(spk_samples)
        all_clusters.extend([u] * len(spk_samples))

    sort_order = np.argsort(all_spikes_samples)
    st = np.array(all_spikes_samples)[sort_order]
    sc = np.array(all_clusters)[sort_order]

    np.save(folder / "spike_times.npy", st)
    np.save(folder / "spike_clusters.npy", sc)

    (folder / "params.py").write_text(f"sample_rate = {sr}\n")

    tsv_lines = ["id\tch\tdepth\n"] + [f"{u}\t{u * 2}\t{u * 100}\n" for u in range(n_units)]
    (folder / "cluster_info.tsv").write_text("".join(tsv_lines))

    return folder


def test_detection_view_end_to_end_synthetic(qapp, tmp_path: Path):
    """Verify loading synthetic folder, tuning thresholds, and exporting CSV."""
    exp_folder = tmp_path / "synthetic_mouse_session"
    exp_folder.mkdir()
    create_synthetic_experiment_folder(exp_folder, n_units=4, n_pulses=25)

    view = DetectionView()
    success = view.load_experiment_folder(exp_folder)
    assert success is True
    assert len(view.units) == 4
    assert view.session_id == "synthetic_mouse_session"
    assert view.table_units.rowCount() == 4
    assert "OPTS_03" in view.combo_proto.currentText()

    # Unit 0 (lat ~ 4ms) should be Tagged
    assert view.table_units.item(0, 7).text() == "✓"
    # Unit 2 (lat ~ 18ms) should be Untagged
    assert view.table_units.item(2, 7).text() == "○"

    # Select Unit 0 and verify unified combined raster and PSTH render
    view.table_units.selectRow(0)
    assert not view.raster_plot.getPlotItem().titleLabel.text
    # PSTH has no separate heading
    assert not view.psth_plot.getPlotItem().titleLabel.text
    # X-axes are linked (shared x-axis)
    assert view.psth_plot.getViewBox().linkedView(0) is view.raster_plot.getViewBox()

    # Verify raster plot displays dark ink vertical line ticks
    import pyqtgraph as pg
    raw_items = view.raster_plot.items if isinstance(view.raster_plot.items, list) else view.raster_plot.items()
    scatter_items = [it for it in raw_items if isinstance(it, pg.ScatterPlotItem)]
    assert len(scatter_items) == 1
    assert scatter_items[0].opts.get("symbol") == "|"
    assert scatter_items[0].opts.get("pen").color().name().lower() == COLOR_TEXT.lower()
    assert scatter_items[0].opts.get("brush").color().name().lower() == COLOR_TEXT.lower()

    # Test slider threshold adjustment:
    # Set max latency to 3.0ms -> Unit 0 (lat=4.0ms) should become Untagged
    view.spin_latency.setValue(3.0)
    view._apply_thresholds()
    assert view.table_units.item(0, 7).text() == "○"

    # Relax max latency to 8.0ms -> Unit 0 should become Tagged again
    view.spin_latency.setValue(8.0)
    view._apply_thresholds()
    assert view.table_units.item(0, 7).text() == "✓"

    # Test SALT p-value threshold adjustment:
    assert view.spin_salt.value() == 0.050
    # Restrict max SALT p-value cutoff -> Unit 0 (p ~ 0.0002) becomes Untagged if cutoff < p
    view.spin_salt.setValue(0.00001)
    view._apply_thresholds()
    assert view.table_units.item(0, 7).text() == "○"

    # Relax max SALT p-value cutoff back to 0.050 -> Unit 0 becomes Tagged again
    view.spin_salt.setValue(0.050)
    view._apply_thresholds()
    assert view.table_units.item(0, 7).text() == "✓"

    # Test manual override toggle
    view.table_units.selectRow(2)  # Unit 2 is Untagged
    assert view.table_units.item(2, 7).text() == "○"
    view._toggle_selected_unit_tag()
    assert view.table_units.item(2, 7).text() == "✓ M"

    # Export session to CSV
    export_dir = tmp_path / "exports"
    csv_path = view.export_curated_session(export_dir, show_dialog=False, precompute_pngs=True)
    assert csv_path is not None
    assert csv_path.exists()
    assert csv_path.name == "synthetic_mouse_session_opto_curated.csv"

    # Validate exported CSV contents
    df = pd.read_csv(csv_path)
    assert len(df) == 4
    expected_cols = [
        "global_unit_id",
        "session_id",
        "unit_id",
        "latency_ms",
        "jitter_ms",
        "reliability",
        "waveform_correlation",
        "salt_p_value",
        "is_tagged",
        "cluster_id",
        "curated_label",
        "plot_path",
    ]
    for col in expected_cols:
        assert col in df.columns

    # Verify unit 0 is tagged and unit 2 has manual override
    assert bool(df.loc[df["unit_id"] == 0, "is_tagged"].iloc[0]) is True
    assert bool(df.loc[df["unit_id"] == 2, "is_tagged"].iloc[0]) is True

    # Verify pre-rendered plots gallery in <session_id>_plots/
    plots_dir = export_dir / "synthetic_mouse_session_plots"
    assert plots_dir.is_dir()
    plot_files = list(plots_dir.glob("*.png"))
    assert len(plot_files) >= 2
    for _, row in df[df["is_tagged"]].iterrows():
        rel_path = row["plot_path"]
        assert (export_dir / rel_path).exists()
        assert (export_dir / rel_path).stat().st_size > 1000


def test_detection_view_real_dataset_workflow(qapp, tmp_path: Path):
    """Verify loading real experimental dataset if OPTOCURATOR_TEST_DATA_DIR is configured."""
    data_dir = os.environ.get("OPTOCURATOR_TEST_DATA_DIR", "").strip()
    if not data_dir:
        pytest.skip("OPTOCURATOR_TEST_DATA_DIR not set.")
    real_path = Path(data_dir)
    if not real_path.exists():
        pytest.skip(f"OPTOCURATOR_TEST_DATA_DIR path does not exist: {real_path}")

    window = MainWindow()
    window.switch_to_detection()
    view = window.detection_view

    success = view.load_experiment_folder(real_path)
    assert success is True
    assert len(view.units) == 619
    assert view.session_id == "test_OptoCurator"

    # Check protocol
    assert "OPTS_03" in view.combo_proto.currentText()
    assert view.table_units.rowCount() == 619

    # Check tagged candidates exist
    tagged_items = [
        view.table_units.item(r, 7).text()
        for r in range(view.table_units.rowCount())
        if view.table_units.item(r, 7).text().startswith("✓")
    ]
    assert len(tagged_items) > 0

    # Test selecting a unit renders real raster and PSTH
    view.table_units.selectRow(0)
    assert not view.raster_plot.getPlotItem().titleLabel.text

    # Test export
    export_dir = tmp_path / "real_export"
    csv_file = view.export_curated_session(export_dir)
    assert csv_file is not None
    assert csv_file.exists()

    df = pd.read_csv(csv_file)
    assert len(df) == 619
    assert df["is_tagged"].sum() > 0

    real_plots_dir = export_dir / "test_OptoCurator_plots"
    assert real_plots_dir.is_dir()
    real_pngs = list(real_plots_dir.glob("*.png"))
    assert len(real_pngs) > 0


def test_btn_load_folder_clicked_signal(qapp, monkeypatch, tmp_path: Path):
    """Verify clicking btn_load_folder (which emits checked: bool) does not raise TypeError."""
    from PyQt6.QtWidgets import QFileDialog

    exp_folder = tmp_path / "mock_session"
    exp_folder.mkdir()
    create_synthetic_experiment_folder(exp_folder, n_units=2, n_pulses=5)

    # Mock QFileDialog.getExistingDirectory to return exp_folder
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(exp_folder))

    view = DetectionView()
    # Simulate clicking button directly (passes checked=False to slot)
    view.btn_load_folder.click()
    assert view.session_id == "mock_session"
    assert view.table_units.rowCount() == 2

    # Simulate clicking export button directly
    export_mock = tmp_path / "export_click"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(export_mock))
    view.current_folder = export_mock
    view.btn_export.click()
    assert (export_mock / "mock_session_opto_curated.csv").exists()


def test_detection_view_visualization_options(qapp, tmp_path: Path):
    """Verify changing visualized window and pulse ordering in DetectionView and MainWindow."""
    exp_folder = tmp_path / "view_test_session"
    exp_folder.mkdir()
    create_synthetic_experiment_folder(exp_folder, n_units=3, n_pulses=15)

    main_win = MainWindow()
    detection_view = main_win.detection_view
    success = detection_view.load_experiment_folder(exp_folder)
    assert success is True

    # 1. Default initial state
    assert detection_view.current_window_ms == (-30.0, 50.0)
    assert detection_view.edit_win_start.text() == "-30"
    assert detection_view.edit_win_end.text() in ("50", "+50")
    assert detection_view.raster_order_by == "presentation"
    assert detection_view.combo_order.currentText() == "Presentation order"

    # 2. Change window via method
    detection_view.set_visualized_window((-10.0, 30.0))
    assert detection_view.current_window_ms == (-10.0, 30.0)
    assert detection_view.edit_win_start.text() == "-10"
    assert detection_view.edit_win_end.text() in ("30", "+30")
    # Check raster plot view range
    view_box = detection_view.raster_plot.getViewBox()
    x_range = view_box.viewRange()[0]
    assert x_range[0] <= -9.0 and x_range[1] >= 29.0

    # 3. Change pulse ordering to latency
    detection_view.set_pulse_ordering("latency")
    assert detection_view.raster_order_by == "latency"
    assert detection_view.combo_order.currentText() == "By first-spike latency"
    assert not detection_view.raster_plot.getPlotItem().titleLabel.text

    # 4. Change window via text boxes directly (arbitrary user input)
    detection_view.edit_win_start.setText("-25")
    detection_view.edit_win_end.setText("45")
    detection_view._on_window_text_edited()
    assert detection_view.current_window_ms == (-25.0, 45.0)

    # 5. Change pulse ordering via combo box
    detection_view.combo_order.setCurrentIndex(0)
    assert detection_view.raster_order_by == "presentation"


def test_detection_view_table_column_sorting(qapp, tmp_path: Path):
    """Verify ordering cells table based on column headers with toggling sort indicator triangle."""
    from PyQt6.QtCore import Qt

    view = DetectionView(load_mock=True)
    header = view.table_units.horizontalHeader()

    assert header.isSortIndicatorShown() is True
    assert header.sectionsClickable() is True

    # 1. Test mock data sorting by Latency (column 1)
    view.sort_by_column(1, Qt.SortOrder.AscendingOrder)
    assert header.sortIndicatorSection() == 1
    assert header.sortIndicatorOrder() == Qt.SortOrder.AscendingOrder
    latencies_asc = [float(view.table_units.item(r, 1).text()) for r in range(view.table_units.rowCount())]
    assert latencies_asc == sorted(latencies_asc)
    assert latencies_asc[0] == 4.2
    assert latencies_asc[-1] > 10.0

    # Toggle to Descending on column 1
    view.sort_by_column(1)
    assert header.sortIndicatorOrder() == Qt.SortOrder.DescendingOrder
    latencies_desc = [float(view.table_units.item(r, 1).text()) for r in range(view.table_units.rowCount())]
    assert latencies_desc == sorted(latencies_desc, reverse=True)
    assert latencies_desc[0] == latencies_asc[-1]

    # 2. Test sorting by SALT p-value (column 4)
    view.sort_by_column(4, Qt.SortOrder.AscendingOrder)
    assert header.sortIndicatorSection() == 4
    assert header.sortIndicatorOrder() == Qt.SortOrder.AscendingOrder
    salt_asc = [view.table_units.item(r, 4).sort_key for r in range(view.table_units.rowCount())]
    assert salt_asc == sorted(salt_asc)
    assert salt_asc[0] <= 0.01

    # Toggle to Descending on column 4
    view.sort_by_column(4)
    assert header.sortIndicatorOrder() == Qt.SortOrder.DescendingOrder
    salt_desc = [view.table_units.item(r, 4).sort_key for r in range(view.table_units.rowCount())]
    assert salt_desc == sorted(salt_desc, reverse=True)
    assert salt_desc[0] > 0.1

    # 3. Test sorting by Status (column 7): Tagged first in ascending
    view.sort_by_column(7, Qt.SortOrder.AscendingOrder)
    status_asc = [view.table_units.item(r, 7).text() for r in range(view.table_units.rowCount())]
    assert status_asc[0] == "✓"
    assert status_asc[-1] == "○"

    view.sort_by_column(7)  # toggle to descending -> Untagged first
    status_desc = [view.table_units.item(r, 7).text() for r in range(view.table_units.rowCount())]
    assert status_desc[0] == "○"
    assert status_desc[-1] == "✓"

    # 4. Test real experiment folder loading and sorting
    exp_folder = tmp_path / "sort_session"
    exp_folder.mkdir()
    create_synthetic_experiment_folder(exp_folder, n_units=4, n_pulses=15)
    view.load_experiment_folder(exp_folder)

    # Sort by unit ID (column 0) ascending and descending
    view.sort_by_column(0, Qt.SortOrder.AscendingOrder)
    uids_asc = [int(view.table_units.item(r, 0).text().replace("u", "")) for r in range(view.table_units.rowCount())]
    assert uids_asc == [0, 1, 2, 3]

    view.sort_by_column(0)
    uids_desc = [int(view.table_units.item(r, 0).text().replace("u", "")) for r in range(view.table_units.rowCount())]
    assert uids_desc == [3, 2, 1, 0]

    # Verify selecting a row after sorting renders the corresponding unit correctly
    view.table_units.selectRow(0)  # unit 3
    assert not view.raster_plot.getPlotItem().titleLabel.text


def test_stats_menu_and_dialog_integration(qapp, monkeypatch):
    """Verify Stats menu exists between View and Clustering and triggers dialog."""
    from optocurator.models import StatsParameters
    from optocurator.ui.dialog_stats import StatsParametersDialog

    window = MainWindow()
    menubar = window.menuBar()
    actions = menubar.actions()
    menu_titles = [a.text().replace("&", "") for a in actions]

    # Verify menu order: File, Modes, View, Stats, Clustering, Help
    assert "Stats" in menu_titles
    assert "View" in menu_titles
    assert "Clustering" in menu_titles

    view_idx = menu_titles.index("View")
    stats_idx = menu_titles.index("Stats")
    clustering_idx = menu_titles.index("Clustering")

    assert view_idx < stats_idx < clustering_idx, "Stats menu must be between View and Clustering"

    # Verify 'Metric parameters…' action exists in Stats menu (renamed per UI_OVERHAUL_SPEC § 11.1 / § 13.3)
    stats_actions = window.stats_menu.actions()
    action_texts = [a.text() for a in stats_actions]
    assert "Metric parameters…" in action_texts

    # Test dialog directly
    dlg = StatsParametersDialog(current_params=StatsParameters.default(), parent=window)
    # Reliability
    assert dlg.edit_rel_start.text() == "0.0"
    assert dlg.edit_rel_end.text() == "20.0"
    assert dlg.edit_rel_trials.text() == "All"
    # Latency
    assert dlg.edit_lat_start.text() == "0.0"
    assert dlg.edit_lat_end.text() == "20.0"
    assert dlg.edit_lat_trials.text() == "All"
    # Jitter
    assert dlg.edit_jit_start.text() == "0.0"
    assert dlg.edit_jit_end.text() == "20.0"
    assert dlg.edit_jit_trials.text() == "All"
    # SALT
    assert dlg.edit_salt_start.text() == "0.0"
    assert dlg.edit_salt_end.text() == "10.0"
    assert dlg.edit_salt_bins.text() == "10"
    assert dlg.edit_salt_perms.text() == "100"

    # Modify parameters separately for each metric
    dlg.edit_rel_start.setText("0.0")
    dlg.edit_rel_end.setText("25.0")
    dlg.edit_rel_trials.setText("40")

    dlg.edit_lat_start.setText("1.0")
    dlg.edit_lat_end.setText("15.0")
    dlg.edit_lat_trials.setText("50")

    dlg.edit_jit_start.setText("0.5")
    dlg.edit_jit_end.setText("12.0")
    dlg.edit_jit_trials.setText("30")

    dlg.edit_salt_start.setText("0.5")
    dlg.edit_salt_end.setText("8.0")
    dlg.edit_salt_bins.setText("12")
    dlg.edit_salt_perms.setText("200")

    # Apply
    dlg._on_apply_clicked()
    res = dlg.get_parameters()
    # Check separate reliability
    assert res.reliability_window_start_ms == 0.0
    assert res.reliability_window_end_ms == 25.0
    assert res.reliability_trials == 40
    # Check separate latency
    assert res.latency_window_start_ms == 1.0
    assert res.latency_window_end_ms == 15.0
    assert res.latency_trials == 50
    # Check separate jitter
    assert res.jitter_window_start_ms == 0.5
    assert res.jitter_window_end_ms == 12.0
    assert res.jitter_trials == 30
    # Check SALT
    assert res.salt_window_start_ms == 0.5
    assert res.salt_window_end_ms == 8.0
    assert res.salt_n_bins == 12
    assert res.salt_n_permutations == 200

    # Test reset defaults
    dlg._reset_defaults()
    assert dlg.edit_rel_start.text() == "0.0"
    assert dlg.edit_rel_end.text() == "20.0"
    assert dlg.edit_rel_trials.text() == "All"
    assert dlg.edit_lat_start.text() == "0.0"
    assert dlg.edit_lat_end.text() == "20.0"
    assert dlg.edit_lat_trials.text() == "All"
    assert dlg.edit_jit_start.text() == "0.0"
    assert dlg.edit_jit_end.text() == "20.0"
    assert dlg.edit_jit_trials.text() == "All"


def test_stats_dialog_validation(qapp):
    """Verify input validation and warnings in StatsParametersDialog."""
    from optocurator.ui.dialog_stats import StatsParametersDialog

    dlg = StatsParametersDialog()

    # Invalid latency start >= end
    dlg.edit_lat_start.setText("25.0")
    dlg.edit_lat_end.setText("10.0")
    dlg._on_apply_clicked()
    # Shouldn't change parameters if invalid
    assert dlg.params.latency_window_start_ms == 0.0

    # Invalid non-numeric input
    dlg.edit_lat_start.setText("invalid_number")
    dlg._on_apply_clicked()
    assert dlg.params.latency_window_start_ms == 0.0

    # Invalid reliability start >= end
    dlg.edit_rel_start.setText("30.0")
    dlg.edit_rel_end.setText("10.0")
    dlg._on_apply_clicked()
    assert dlg.params.reliability_window_start_ms == 0.0

    # SALT exclusion window controls
    assert dlg.chk_salt_exclusion.isChecked() is False
    assert dlg.edit_salt_excl_start.isEnabled() is False
    assert dlg.edit_salt_excl_end.isEnabled() is False
    assert dlg.edit_salt_excl_start.text() == "-1.0"
    assert dlg.edit_salt_excl_end.text() == "1.0"

    # Toggle exclusion checkbox
    dlg.chk_salt_exclusion.setChecked(True)
    assert dlg.edit_salt_excl_start.isEnabled() is True
    assert dlg.edit_salt_excl_end.isEnabled() is True

    # Invalid exclusion start >= end
    dlg.edit_salt_excl_start.setText("2.0")
    dlg.edit_salt_excl_end.setText("1.0")
    dlg._on_apply_clicked()
    assert dlg.params.salt_exclusion_enabled is False

    # Valid exclusion window applied
    dlg.edit_lat_start.setText("0.0")
    dlg.edit_lat_end.setText("20.0")
    dlg.edit_rel_start.setText("0.0")
    dlg.edit_rel_end.setText("20.0")
    dlg.edit_salt_excl_start.setText("-1.0")
    dlg.edit_salt_excl_end.setText("1.0")
    dlg._on_apply_clicked()
    assert dlg.params.salt_exclusion_enabled is True
    assert dlg.params.salt_exclusion_window_start_ms == -1.0
    assert dlg.params.salt_exclusion_window_end_ms == 1.0
    assert dlg.params.salt_exclusion_window_ms == (-1.0, 1.0)


def test_detection_view_update_stats_parameters(qapp, tmp_path: Path):
    """Verify update_stats_parameters triggers recomputation on loaded session."""
    from optocurator.models import StatsParameters

    synthetic_dir = create_synthetic_experiment_folder(tmp_path / "synthetic_test", n_units=3, n_pulses=20)
    window = MainWindow()
    window.switch_to_detection()
    view = window.detection_view

    # Load session
    ok = view.load_experiment_folder(synthetic_dir)
    assert ok is True
    assert len(view.units) == 3

    # Initial reliability for unit 0
    init_rel = view.metrics[0].reliability
    assert init_rel > 0.0

    # Create new stats parameters with max 10 trials for reliability, 15 for latency, 12 for jitter
    new_params = StatsParameters(
        reliability_window_start_ms=0.0,
        reliability_window_end_ms=18.0,
        reliability_trials=10,
        latency_window_start_ms=0.0,
        latency_window_end_ms=12.0,
        latency_trials=15,
        jitter_window_start_ms=0.0,
        jitter_window_end_ms=10.0,
        jitter_trials=12,
    )
    view.update_stats_parameters(new_params)

    # Check params stored
    assert view.stats_params.reliability_trials == 10
    assert view.stats_params.latency_trials == 15
    assert view.stats_params.jitter_trials == 12
    assert view.stats_params.reliability_window_end_ms == 18.0
    assert view.stats_params.latency_window_end_ms == 12.0
    assert 0 in view.metrics


def test_render_raster_vertical_lines(qapp):
    """Verify render_raster creates a ScatterPlotItem with vertical line symbol '|' and black styling."""
    import pyqtgraph as pg
    from optocurator.ui.plots import render_raster

    widget = pg.PlotWidget()
    raster_x = np.array([1.0, 2.5, -4.0])
    raster_y = np.array([0, 1, 2])
    render_raster(widget, raster_x, raster_y, n_trials=3)

    raw_items = widget.items if isinstance(widget.items, list) else widget.items()
    scatter_items = [it for it in raw_items if isinstance(it, pg.ScatterPlotItem)]
    assert len(scatter_items) == 1
    scatter = scatter_items[0]
    assert scatter.opts.get("symbol") == "|"
    assert scatter.opts.get("size") == 3.0
    pen = scatter.opts.get("pen")
    brush = scatter.opts.get("brush")
    assert pen.color().name().lower() == COLOR_TEXT.lower()
    assert brush.color().name().lower() == COLOR_TEXT.lower()


def test_raster_marker_size_configuration(qapp, tmp_path: Path):
    """Verify raster marker size customization in DetectionView."""
    import pyqtgraph as pg
    from optocurator.ui.plots import render_raster

    # 1. Custom marker_size parameter in render_raster
    widget = pg.PlotWidget()
    render_raster(widget, np.array([1.0, 2.0]), np.array([0, 1]), n_trials=2, marker_size=5.5)
    raw_items = widget.items if isinstance(widget.items, list) else widget.items()
    scatter_items = [it for it in raw_items if isinstance(it, pg.ScatterPlotItem)]
    assert scatter_items[0].opts.get("size") == 5.5

    # 2. DetectionView marker size setting & re-render
    exp_folder = tmp_path / "test_session_marker"
    exp_folder.mkdir()
    create_synthetic_experiment_folder(exp_folder, n_units=2, n_pulses=5)

    window = MainWindow()
    window.switch_to_detection()
    view = window.detection_view
    view.load_experiment_folder(exp_folder)
    view.table_units.selectRow(0)

    # Default marker size is 3.0
    assert view.raster_marker_size == 3.0
    raw_plot_items = view.raster_plot.items if isinstance(view.raster_plot.items, list) else view.raster_plot.items()
    scatters = [it for it in raw_plot_items if isinstance(it, pg.ScatterPlotItem)]
    assert scatters[0].opts.get("size") == 3.0

    # Change marker size via set_raster_marker_size
    view.set_raster_marker_size(5.0)
    assert view.raster_marker_size == 5.0
    raw_plot_items2 = view.raster_plot.items if isinstance(view.raster_plot.items, list) else view.raster_plot.items()
    scatters2 = [it for it in raw_plot_items2 if isinstance(it, pg.ScatterPlotItem)]
    assert scatters2[0].opts.get("size") == 5.0


def test_raster_psth_y_axis_alignment(qapp, tmp_path: Path):
    """Verify y-axis line alignment between raster plot and PSTH plot."""
    import pyqtgraph as pg
    from optocurator.ui.plots import render_raster, render_psth

    # 1. Verify render_raster and render_psth set left axis width
    widget_raster = pg.PlotWidget()
    widget_psth = pg.PlotWidget()

    render_raster(
        widget_raster,
        np.array([1.0, 2.0]),
        np.array([0, 1]),
        n_trials=2,
        y_axis_width=58,
    )
    render_psth(
        widget_psth,
        np.array([0.0, 1.0, 2.0]),
        np.array([10.0, 20.0]),
        baseline_mean=5.0,
        baseline_3sigma=15.0,
        y_axis_width=58,
    )
    assert widget_raster.getAxis("left").width() == widget_psth.getAxis("left").width()
    widget_raster.close()
    widget_psth.close()

    # 2. Verify DetectionView layout alignment
    exp_folder = tmp_path / "test_session_alignment"
    exp_folder.mkdir()
    create_synthetic_experiment_folder(exp_folder, n_units=2, n_pulses=10)

    view = DetectionView()
    view.load_experiment_folder(exp_folder)
    view.table_units.selectRow(0)
    view.show()
    qapp.processEvents()

    assert view.raster_plot.getAxis("left").width() == view.psth_plot.getAxis("left").width()
    assert view.raster_plot.getViewBox().geometry().x() == view.psth_plot.getViewBox().geometry().x()
    view.close()


def test_detection_view_checkboxes_and_textboxes(qapp, tmp_path: Path):
    """Verify that threshold controls use textboxes and checkboxes to enable/disable filtering."""
    exp_folder = tmp_path / "test_session_thresh_cb"
    exp_folder.mkdir()
    create_synthetic_experiment_folder(exp_folder, n_units=4, n_pulses=20)

    view = DetectionView()
    success = view.load_experiment_folder(exp_folder)
    assert success is True

    # 1. Verify default values and checked states
    assert view.chk_latency.isChecked() is True
    assert view.chk_jitter.isChecked() is True
    assert view.chk_rel.isChecked() is True
    assert view.chk_salt.isChecked() is True

    assert view.edit_latency.text() == "8.0"
    assert view.edit_jitter.text() == "2.0"
    assert view.edit_rel.text() == "0.60"
    assert float(view.edit_salt.text()) == 0.05

    assert view.edit_latency.isEnabled() is True
    assert view.edit_jitter.isEnabled() is True
    assert view.edit_rel.isEnabled() is True
    assert view.edit_salt.isEnabled() is True

    # Initial state: Unit 0 tagged, Unit 2 (lat ~ 18ms) untagged
    assert view.table_units.item(0, 7).text() == "✓"
    assert view.table_units.item(2, 7).text() == "○"

    # 2. Test editing text box value
    # Unit 1 has latency ~ 7.5ms (tagged with default 8.0ms)
    assert view.table_units.item(1, 7).text() == "✓"
    view.edit_latency.setText("6.0")
    view._apply_thresholds()
    # Unit 1 (lat ~ 7.5ms) now fails the 6.0ms cutoff -> Untagged
    assert view.table_units.item(1, 7).text() == "○"

    # 3. Test disabling threshold via checkbox
    # When latency checkbox is unchecked:
    view.chk_latency.setChecked(False)
    assert view.edit_latency.isEnabled() is False
    # Unit 1 should now be Tagged because latency filter is disabled!
    assert view.table_units.item(1, 7).text() == "✓"

    # Re-enable latency checkbox:
    view.chk_latency.setChecked(True)
    assert view.edit_latency.isEnabled() is True
    # Unit 1 fails latency again (since cutoff is still 6.0ms) -> Untagged
    assert view.table_units.item(1, 7).text() == "○"

    # Restore latency to 8.0 -> Unit 1 Tagged again
    view.edit_latency.setText("8.0")
    view._apply_thresholds()
    assert view.table_units.item(1, 7).text() == "✓"

    # Unit 2 has lat ~ 18ms (> 8ms) and salt p ~ 1.0 (> 0.05).
    # Disabling both latency and SALT thresholds makes Unit 2 Tagged:
    assert view.table_units.item(2, 7).text() == "○"
    view.chk_latency.setChecked(False)
    view.chk_salt.setChecked(False)
    assert view.table_units.item(2, 7).text() == "✓"
    # Restore checkboxes
    view.chk_latency.setChecked(True)
    view.chk_salt.setChecked(True)
    assert view.table_units.item(2, 7).text() == "○"

    # 4. Test disabling SALT p-value threshold
    view.edit_salt.setText("0.00001")
    view._apply_thresholds()
    assert view.table_units.item(0, 7).text() == "○"

    # Uncheck SALT threshold -> Unit 0 should become Tagged even with 0.00001 in text box
    view.chk_salt.setChecked(False)
    assert view.edit_salt.isEnabled() is False
    assert view.table_units.item(0, 7).text() == "✓"

    # 5. Test manual override takes precedence over checkbox / textbox
    view.table_units.selectRow(0)
    view._toggle_selected_unit_tag()  # Manually Untag Unit 0
    assert view.table_units.item(0, 7).text() == "○ M"

    # Changing thresholds or checkboxes does not overwrite manual override
    view.chk_salt.setChecked(True)
    view.edit_salt.setText("0.0500")
    view._apply_thresholds()
    assert view.table_units.item(0, 7).text() == "○ M"

    view.close()


def test_detection_view_status_filtering(qapp):
    """Validate that combo_filter_status properly filters rows in DetectionView."""
    view = DetectionView(load_mock=True)
    # Mock data initializes with 5 units: 3 Tagged (rows 0, 1, 3) and 2 Untagged (rows 2, 4)
    assert view.table_units.rowCount() == 5
    assert hasattr(view, "combo_filter_status")

    # Initial state: "All units" -> no row hidden
    assert view.combo_filter_status.currentText() == "All units"
    for r in range(5):
        assert not view.table_units.isRowHidden(r)
    assert view.lbl_unit_summary.text() == "5 units"

    # Filter: "Tagged" -> hides untagged rows, shows tagged rows
    idx_tagged = view.combo_filter_status.findText("Tagged")
    assert idx_tagged >= 0
    view.combo_filter_status.setCurrentIndex(idx_tagged)

    for r in range(5):
        status = view.table_units.item(r, 7).text()
        if status.startswith("✓"):
            assert not view.table_units.isRowHidden(r)
        else:
            assert view.table_units.isRowHidden(r)
    assert view.lbl_unit_summary.text() == "3 of 5 units"

    # Filter: "Untagged" -> hides tagged rows, shows untagged rows
    idx_untagged = view.combo_filter_status.findText("Untagged")
    assert idx_untagged >= 0
    view.combo_filter_status.setCurrentIndex(idx_untagged)

    for r in range(5):
        status = view.table_units.item(r, 7).text()
        if status.startswith("○"):
            assert not view.table_units.isRowHidden(r)
        else:
            assert view.table_units.isRowHidden(r)
    assert view.lbl_unit_summary.text() == "2 of 5 units"

    # Filter: "All units" -> unhides all rows
    idx_all = view.combo_filter_status.findText("All units")
    view.combo_filter_status.setCurrentIndex(idx_all)
    for r in range(5):
        assert not view.table_units.isRowHidden(r)
    assert view.lbl_unit_summary.text() == "5 units"

    # Test sorting preserves row hiddenness while filter is active
    view.combo_filter_status.setCurrentIndex(idx_tagged)
    assert view.lbl_unit_summary.text() == "3 of 5 units"

    # Header section click (column 1: Latency)
    view._on_header_section_clicked(1)
    for r in range(5):
        status = view.table_units.item(r, 7).text()
        if status.startswith("✓"):
            assert not view.table_units.isRowHidden(r)
        else:
            assert view.table_units.isRowHidden(r)

    # Programmatic sort_by_column (column 0: Unit ID)
    view.sort_by_column(0)
    for r in range(5):
        status = view.table_units.item(r, 7).text()
        if status.startswith("✓"):
            assert not view.table_units.isRowHidden(r)
        else:
            assert view.table_units.isRowHidden(r)

    # Sorting with "Untagged" active
    view.combo_filter_status.setCurrentIndex(idx_untagged)
    view._on_header_section_clicked(2)
    for r in range(5):
        status = view.table_units.item(r, 7).text()
        if status.startswith("○"):
            assert not view.table_units.isRowHidden(r)
        else:
            assert view.table_units.isRowHidden(r)

    # Verify selection desync fix: selection shifts to first visible row if currently selected row is hidden
    first_visible = -1
    for r in range(5):
        if not view.table_units.isRowHidden(r):
            first_visible = r
            break
    assert first_visible >= 0
    view.table_units.selectRow(first_visible)
    assert view.table_units.currentRow() == first_visible

    # Switching filter to "Tagged" hides the untagged selection and shifts to a visible tagged row
    view.combo_filter_status.setCurrentIndex(idx_tagged)
    curr_selected = view.table_units.currentRow()
    assert curr_selected >= 0
    assert not view.table_units.isRowHidden(curr_selected)
    assert view.table_units.item(curr_selected, 7).text().startswith("✓")

    view.close()


def test_detection_view_protocol_default_stats_and_dialog_reset(qapp, tmp_path: Path):
    """Verify DetectionView defaults StatsParameters windows to [0, D_pulse] and StatsParametersDialog resets to default_params."""
    from optocurator.models import StatsParameters
    from optocurator.ui.dialog_stats import StatsParametersDialog

    # Build synthetic folder with 10 ms pulse duration (distinct from 20 ms fallback)
    exp_folder = tmp_path / "pulse10ms_session"
    create_synthetic_experiment_folder(exp_folder, n_units=3, n_pulses=15)
    sr = 30000.0
    onsets_sec = np.linspace(10.0, 30.0, 15)
    offsets_sec = onsets_sec + 0.010  # 10 ms pulse duration
    pulses_samples = np.column_stack([
        (onsets_sec * sr).astype(np.uint32),
        (offsets_sec * sr).astype(np.uint32),
    ])
    np.save(exp_folder / "opto_idxs_OPTS_03.npy", pulses_samples)

    view = DetectionView()
    assert view.load_experiment_folder(exp_folder) is True

    # DetectionView stats_params should default all evoked windows to [0.0, 10.0]
    for start_attr, end_attr in (
        ("reliability_window_start_ms", "reliability_window_end_ms"),
        ("latency_window_start_ms", "latency_window_end_ms"),
        ("jitter_window_start_ms", "jitter_window_end_ms"),
        ("salt_window_start_ms", "salt_window_end_ms"),
        ("wf_evoked_start_ms", "wf_evoked_end_ms"),
    ):
        assert getattr(view.stats_params, start_attr) == pytest.approx(0.0)
        assert getattr(view.stats_params, end_attr) == pytest.approx(10.0)

    proto_defaults = view.default_stats_parameters()
    assert proto_defaults.latency_window_end_ms == pytest.approx(10.0)

    # Open StatsParametersDialog with customized values and verify Reset Defaults restores default_params
    customized = StatsParameters(
        reliability_window_start_ms=1.0,
        reliability_window_end_ms=25.0,
        latency_window_start_ms=1.0,
        latency_window_end_ms=25.0,
        jitter_window_start_ms=1.0,
        jitter_window_end_ms=25.0,
        salt_window_start_ms=1.0,
        salt_window_end_ms=25.0,
        wf_evoked_start_ms=1.0,
        wf_evoked_end_ms=25.0,
    )
    dlg = StatsParametersDialog(current_params=customized, default_params=proto_defaults)
    assert dlg.edit_lat_end.text() == "25.0"

    dlg._reset_defaults()
    assert dlg.edit_lat_end.text() == "10.0"
    assert dlg.edit_rel_end.text() == "10.0"
    dlg._on_apply_clicked()
    reset_params = dlg.get_parameters()
    assert reset_params.reliability_window_start_ms == pytest.approx(0.0)
    assert reset_params.reliability_window_end_ms == pytest.approx(10.0)
    assert reset_params.latency_window_start_ms == pytest.approx(0.0)
    assert reset_params.latency_window_end_ms == pytest.approx(10.0)
    assert reset_params.jitter_window_start_ms == pytest.approx(0.0)
    assert reset_params.jitter_window_end_ms == pytest.approx(10.0)
    assert reset_params.salt_window_start_ms == pytest.approx(0.0)
    assert reset_params.salt_window_end_ms == pytest.approx(10.0)
    assert reset_params.wf_evoked_start_ms == pytest.approx(0.0)
    assert reset_params.wf_evoked_end_ms == pytest.approx(10.0)

    dlg.close()
    view.close()



