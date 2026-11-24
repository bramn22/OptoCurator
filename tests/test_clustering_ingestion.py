"""Tests for Iteration 3: Phase 2 MVP — Multi-Experiment Ingestion & Instant Preview."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import QHeaderView, QMessageBox

from optocurator.io.validation import (
    InvalidPhase1ExportError,
    MissingPhase1ExportError,
    Phase1ExportMetadata,
    validate_phase1_export,
)
from optocurator.ui.app import MainWindow
from optocurator.ui.plots import UnitResponsePlot
from optocurator.ui.view_clusters import ClusteringView


def _create_dummy_png(target_file: Path, width: int = 120, height: int = 80) -> Path:
    """Create a minimal valid PNG file on disk for image viewer tests."""
    target_file.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    im = Image.new("RGB", (width, height), color=(37, 99, 235))
    im.save(str(target_file), "PNG")
    return target_file


def _build_test_phase1_export(
    folder: Path,
    session_id: str = "session_01",
    n_units: int = 4,
    n_tagged: int = 2,
    with_plots: bool = True,
) -> Path:
    """Helper to build a valid Phase 1 exported recording folder."""
    session_dir = folder / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = session_dir / f"{session_id}_plots"
    if with_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(1, n_units + 1):
        is_tag = (i <= n_tagged)
        gid = f"{session_id}_u{i:04d}"
        plot_rel = f"{session_id}_plots/{gid}_psth_raster.png"
        rows.append({
            "global_unit_id": gid,
            "session_id": session_id,
            "unit_id": i,
            "latency_ms": 3.0 + i * 1.2,
            "jitter_ms": 1.1 + i * 0.1,
            "reliability": 0.85 - i * 0.1,
            "waveform_correlation": 0.94,
            "salt_p_value": 0.002 * i,
            "is_tagged": is_tag,
            "cluster_id": -1,
            "curated_label": "Tagged" if is_tag else "unassigned",
            "plot_path": plot_rel,
        })
        if with_plots:
            _create_dummy_png(session_dir / plot_rel)

    df = pd.DataFrame(rows)
    csv_path = session_dir / f"{session_id}_opto_curated.csv"
    df.to_csv(csv_path, index=False)
    np.save(session_dir / "opto_idxs_OPTS03.npy", np.array([30000, 60000, 90000]))

    # Include real spike arrays + pulse_onsets in <session_id>_spikes.npz matching Phase 1 export
    arrays_dir = session_dir / f"{session_id}_arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    pulse_onsets = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    spikes_payload = {"pulse_onsets": pulse_onsets}
    for i in range(1, n_units + 1):
        spikes_payload[f"unit_{i}"] = np.sort(
            np.concatenate([pulse_onsets - 0.015, pulse_onsets + 0.003 + 0.0012 * i])
        )
    np.savez_compressed(arrays_dir / f"{session_id}_spikes.npz", **spikes_payload)
    return session_dir


# --- 1. Validation Tests ---


def test_validate_phase1_export_success(tmp_path: Path):
    """Verify validation passes on a properly exported Phase 1 folder."""
    session_dir = _build_test_phase1_export(tmp_path, "mouseA_session1", n_units=5, n_tagged=2)
    meta = validate_phase1_export(session_dir)

    assert isinstance(meta, Phase1ExportMetadata)
    assert meta.session_id == "mouseA_session1"
    assert meta.total_units == 5
    assert meta.tagged_units == 2
    assert meta.protocol_name == "OPTS03"
    assert meta.csv_path.is_file()
    assert meta.plots_dir.is_dir()
    assert len(meta.df) == 5


def test_validate_phase1_export_missing_csv_raises(tmp_path: Path):
    """Verify validate_phase1_export raises MissingPhase1ExportError when CSV is absent."""
    uncurated_dir = tmp_path / "raw_recording_no_export"
    uncurated_dir.mkdir()
    np.save(uncurated_dir / "opto_idxs_OPTS03.npy", np.array([1, 2, 3]))

    with pytest.raises(MissingPhase1ExportError) as exc_info:
        validate_phase1_export(uncurated_dir)

    assert "does not contain a Phase 1 curation export file" in str(exc_info.value)
    assert "Export Detection Session" in str(exc_info.value)


def test_validate_phase1_export_invalid_csv_schema(tmp_path: Path):
    """Verify validate_phase1_export raises InvalidPhase1ExportError on malformed columns."""
    bad_dir = tmp_path / "bad_session"
    bad_dir.mkdir()
    bad_df = pd.DataFrame({"global_unit_id": ["u001"], "session_id": ["bad"]})
    bad_df.to_csv(bad_dir / "bad_session_opto_curated.csv", index=False)

    with pytest.raises(InvalidPhase1ExportError) as exc_info:
        validate_phase1_export(bad_dir)

    assert "missing required Phase 1 columns" in str(exc_info.value)


def test_validate_phase1_export_nonexistent_dir():
    """Verify FileNotFoundError is raised if path does not exist."""
    with pytest.raises(FileNotFoundError):
        validate_phase1_export(Path("non_existent_directory_12345"))


def test_validate_phase1_export_string_boolean_is_tagged(tmp_path: Path):
    """validate_phase1_export must parse string is_tagged booleans consistently with tagged_df (Bug IO7)."""
    session_dir = _build_test_phase1_export(tmp_path, "str_bool_session", n_units=4, n_tagged=1)
    csv_path = session_dir / "str_bool_session_opto_curated.csv"
    df = pd.read_csv(csv_path)
    df["is_tagged"] = pd.Series(["True ", "False ", "0", "no"], dtype=object)
    df.to_csv(csv_path, index=False)

    meta = validate_phase1_export(session_dir)
    assert len(meta.tagged_df) == 1
    assert meta.tagged_units == 1


# --- 2. ClusteringView Multi-Experiment Ingestion Tests ---


def test_clustering_view_initial_mock_state(qapp):
    """Verify ClusteringView launches with responsive mock data."""
    view = ClusteringView(load_mock=True)
    assert view._is_mock is True
    assert view.table_exps.rowCount() == 3
    assert view.table_pooled_units.rowCount() > 0
    assert view.image_viewer is not None


def test_clustering_view_rejects_uncurated_folder(qapp, tmp_path: Path, monkeypatch):
    """Verify warning dialog triggers and state is preserved when uncurated folder is added."""
    warn_calls = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: warn_calls.append(args))

    view = ClusteringView(load_mock=True)
    uncurated_dir = tmp_path / "uncurated_session"
    uncurated_dir.mkdir()

    success = view.add_experiment_folder(uncurated_dir)
    assert success is False
    assert len(warn_calls) == 1
    # Check warning text explains missing export
    assert "Missing Phase 1 Export" in str(warn_calls[0])
    # Mock state remains intact
    assert view._is_mock is True
    assert view.table_exps.rowCount() == 3


def test_clustering_view_ingests_valid_experiment(qapp, tmp_path: Path):
    """Verify valid experiment is parsed, registered, and unit preview is displayed."""
    session_dir = _build_test_phase1_export(tmp_path, "mouseA_20261021", n_units=4, n_tagged=2)

    view = ClusteringView()
    success = view.add_experiment_folder(session_dir)

    assert success is True
    assert view._is_mock is False
    assert len(view.loaded_experiments) == 1
    assert "mouseA_20261021" in view.loaded_experiments

    # Check Active Experiments table
    assert view.table_exps.rowCount() == 1
    assert view.table_exps.item(0, 0).text() == "mouseA_20261021"
    assert view.table_exps.item(0, 1).text() == "OPTS03"
    assert "2 tagged" in view.table_exps.item(0, 2).text()

    # Check Pooled Units table (only 2 tagged units loaded out of 4 total)
    assert view.table_pooled_units.rowCount() == 2
    assert view.table_pooled_units.item(0, 0).text() == "1"
    assert view.table_pooled_units.item(0, 1).toolTip() == "mouseA_20261021"
    assert view._get_gid_for_row(0) == "mouseA_20261021_u0001"
    assert view._get_gid_for_row(1) == "mouseA_20261021_u0002"

    # Row 0 is automatically selected and rendered in UnitResponsePlot
    assert "mouseA_20261021" in view.lbl_candidate.text()
    assert "Unit: 1" in view.lbl_candidate.text()
    assert isinstance(view.unit_response_plot, UnitResponsePlot)
    assert not view.unit_response_plot.raster_plot.isHidden()
    assert not view.unit_response_plot.psth_plot.isHidden()
    assert view.unit_response_plot.raster_plot.getPlotItem()._raster_cache["scatter"].isVisible() is True
    assert view.unit_response_plot.psth_plot.getPlotItem()._psth_cache["smooth_curve"].isVisible() is True


def test_clustering_view_multi_experiment_pooling_and_swapping(qapp, tmp_path: Path):
    """Verify adding multiple experiments pools tagged units and clicking swaps plots instantly."""
    dir1 = _build_test_phase1_export(tmp_path, "session_01", n_units=3, n_tagged=1)
    dir2 = _build_test_phase1_export(tmp_path, "session_02", n_units=4, n_tagged=2)

    view = ClusteringView()
    view.add_experiment_folder(dir1)
    view.add_experiment_folder(dir2)

    assert len(view.loaded_experiments) == 2
    assert view.table_exps.rowCount() == 2
    # 1 tagged from dir1 + 2 tagged from dir2 = 3 pooled units
    assert view.table_pooled_units.rowCount() == 3

    # Select unit from second session in table (row index 1 or 2 is from session_02)
    view.table_pooled_units.selectRow(1)
    assert "session_02" in view.lbl_candidate.text()
    assert view.unit_response_plot.raster_plot.getPlotItem()._raster_cache["scatter"].isVisible() is True
    assert view.unit_response_plot.psth_plot.getPlotItem()._psth_cache["smooth_curve"].isVisible() is True

    # Column header sorting preserves lookup
    view.table_pooled_units.horizontalHeader().sectionClicked.emit(3)  # Sort by Latency
    view.table_pooled_units.selectRow(0)
    current_gid = view._get_gid_for_row(0)
    sess_part, uid_part = current_gid.rsplit("_u", 1)
    assert sess_part in view.lbl_candidate.text()
    assert str(int(uid_part)) in view.lbl_candidate.text()


def test_clustering_view_excludes_untagged_units(qapp, tmp_path: Path):
    """Verify that un-tagged units from Phase 1 are strictly excluded from Phase 2 pooling."""
    session_dir = _build_test_phase1_export(tmp_path, "session_mixed", n_units=6, n_tagged=2)
    view = ClusteringView()
    view.add_experiment_folder(session_dir)

    assert view.table_pooled_units.rowCount() == 2
    gids = [view._get_gid_for_row(r) for r in range(view.table_pooled_units.rowCount())]
    assert gids == ["session_mixed_u0001", "session_mixed_u0002"]
    assert "session_mixed_u0003" not in gids
    assert "session_mixed_u0004" not in gids


def test_clustering_view_remove_experiment(qapp, tmp_path: Path):
    """Verify removing an experiment updates unit counts and cleans state."""
    dir1 = _build_test_phase1_export(tmp_path, "session_A", n_units=3, n_tagged=1)
    dir2 = _build_test_phase1_export(tmp_path, "session_B", n_units=2, n_tagged=1)

    view = ClusteringView()
    view.add_experiment_folder(dir1)
    view.add_experiment_folder(dir2)
    assert view.table_exps.rowCount() == 2
    assert view.table_pooled_units.rowCount() == 2

    # Select session_A (row 0) and remove it
    view.table_exps.selectRow(0)
    removed = view.remove_selected_experiment()
    assert removed is True
    assert len(view.loaded_experiments) == 1
    assert "session_A" not in view.loaded_experiments
    assert "session_B" in view.loaded_experiments

    # Table reflects remaining session
    assert view.table_exps.rowCount() == 1
    assert view.table_pooled_units.rowCount() == 1

    # Remove remaining session
    view.table_exps.selectRow(0)
    view.remove_selected_experiment()
    assert len(view.loaded_experiments) == 0
    assert view.table_exps.rowCount() == 0
    assert view.table_pooled_units.rowCount() == 0
    raster_scatter = view.unit_response_plot.raster_plot.getPlotItem()._raster_cache["scatter"]
    psth_curve = view.unit_response_plot.psth_plot.getPlotItem()._psth_cache["smooth_curve"]
    rx, _ = raster_scatter.getData()
    px, _ = psth_curve.getData()
    assert len(rx) == 0
    assert len(px) == 0
    assert raster_scatter.isVisible() is False
    assert psth_curve.isVisible() is False


def test_main_window_phase2_integration(qapp, tmp_path: Path):
    """Verify MainWindow File menu action and status bar integration for Phase 2."""
    session_dir = _build_test_phase1_export(tmp_path, "mouse_mw_test", n_units=3, n_tagged=1)

    win = MainWindow()
    win.switch_to_clustering()
    assert win.stack.currentIndex() == win.VIEW_CLUSTERING

    # Ingest experiment
    ok = win.clustering_view.add_experiment_folder(session_dir)
    assert ok is True
    assert "mouse_mw_test" in win.status.currentMessage()

    # Reset layout action
    assert hasattr(win, "act_reset_phase2_layout")
    win.act_reset_phase2_layout.trigger()


def test_ingested_experiment_cell_list_schema_and_resizing(qapp, tmp_path: Path):
    """Verify ingested experiment data strictly conforms to 7-column schema, GID UserRole, and Interactive resize mode."""
    session_dir = _build_test_phase1_export(tmp_path, "session_schema_test", n_units=4, n_tagged=2)
    view = ClusteringView()
    ok = view.add_experiment_folder(session_dir)
    assert ok is True

    # 1. Column count is 7
    assert view.table_pooled_units.columnCount() == 7

    # 2. Exact column ordering
    expected_headers = ["Unit", "Session", "Cluster", "Lat", "Rel.", "SALT p", "Status"]
    actual_headers = [
        view.table_pooled_units.horizontalHeaderItem(c).text()
        for c in range(view.table_pooled_units.columnCount())
    ]
    assert actual_headers == expected_headers
    assert "Global ID" not in actual_headers

    # 3. All columns have ResizeMode.Interactive
    header = view.table_pooled_units.horizontalHeader()
    for col in range(7):
        assert header.sectionResizeMode(col) == QHeaderView.ResizeMode.Interactive

    # 4. Row contents and GID retrieval from UserRole
    assert view.table_pooled_units.rowCount() == 2
    for r in range(2):
        unit_item = view.table_pooled_units.item(r, 0)
        assert unit_item.text() == str(r + 1)
        sess_item = view.table_pooled_units.item(r, 1)
        assert sess_item.toolTip() == "session_schema_test"
        gid = view._get_gid_for_row(r)
        assert gid == f"session_schema_test_u{r + 1:04d}"
        assert unit_item.data(Qt.ItemDataRole.UserRole) == gid
        assert view.table_pooled_units.item(r, 2).text() == "—"
        assert view.table_pooled_units.item(r, 6).toolTip() == "Unassigned"

    # 5. clusters_container allows free horizontal splitter movement
    assert hasattr(view, "clusters_container")
    assert view.clusters_container.minimumSize().width() <= 300

    view.close()


def test_protocol_name_persisted_and_fallback_uses_sampling_rate(qapp, tmp_path: Path):
    """DetectionView export must persist protocol_name and Phase 2 fallback must pass sampling_rate (Bug IO4)."""
    from optocurator.models import StimulusProtocol, UnitData
    from optocurator.ui.view_detection import DetectionView

    # 1. DetectionView export with custom protocol name and no opto_idxs_*.npy on disk
    sess_dir = tmp_path / "sess_custom_proto"
    sess_dir.mkdir(parents=True)
    onsets = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    proto = StimulusProtocol(onsets=onsets, offsets=onsets + 0.01, name="LASER_473")
    units = [
        UnitData(
            session_id="sess_custom_proto",
            unit_id=1,
            spike_times=np.sort(np.concatenate([onsets - 0.02, onsets + 0.004])),
            sampling_rate=20000.0,
        )
    ]

    det = DetectionView()
    det.current_folder = sess_dir
    det.session_id = "sess_custom_proto"
    det.units = units
    det.units_by_id = {u.unit_id: u for u in units}
    det.protocols = {"LASER_473": proto}
    det.current_protocol = proto
    det._recompute_session_metrics()
    det.export_curated_session(export_dir=sess_dir, show_dialog=False, precompute_pngs=False)
    det.close()

    npz_path = sess_dir / "sess_custom_proto_arrays" / "sess_custom_proto_spikes.npz"
    with np.load(npz_path, allow_pickle=True) as data:
        assert "protocol_name" in data.files
        assert str(np.asarray(data["protocol_name"]).item()) == "LASER_473"

    meta = validate_phase1_export(sess_dir)
    assert meta.protocol_name == "LASER_473"

    # 2. Fallback disk pulse loader in ClusteringView must pass session sampling_rate (20000 Hz)
    sess_20k = tmp_path / "sess_20k"
    sess_20k.mkdir(parents=True)
    sr = 20000.0
    (sess_20k / "params.py").write_text(f"sample_rate = {sr}\n", encoding="utf-8")
    np.save(sess_20k / "spike_times.npy", np.array([20080, 40080], dtype=np.uint64))
    np.save(sess_20k / "spike_clusters.npy", np.array([1, 1], dtype=np.int32))
    np.save(sess_20k / "opto_idxs_BLUE20K.npy", np.array([[20000, 20200], [40000, 40200]], dtype=np.uint32))
    (sess_20k / "sess_20k_plots").mkdir()
    _create_dummy_png(sess_20k / "sess_20k_plots" / "sess_20k_u0001_psth_raster.png")
    pd.DataFrame([
        {
            "global_unit_id": "sess_20k_u0001",
            "session_id": "sess_20k",
            "unit_id": 1,
            "latency_ms": 4.0,
            "jitter_ms": 0.5,
            "reliability": 1.0,
            "waveform_correlation": 0.95,
            "salt_p_value": 0.01,
            "is_tagged": True,
            "cluster_id": -1,
            "curated_label": "Tagged",
            "plot_path": "sess_20k_plots/sess_20k_u0001_psth_raster.png",
        }
    ]).to_csv(sess_20k / "sess_20k_opto_curated.csv", index=False)

    clust = ClusteringView()
    assert clust.add_experiment_folder(sess_20k) is True
    onsets_loaded = clust._experiment_stimulus["sess_20k"]
    np.testing.assert_allclose(onsets_loaded, [1.0, 2.0])
    clust.close()


