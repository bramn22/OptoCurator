"""Tests for NWB file inspection, spike/stimulus reading, and DetectionView integration."""

import os
from pathlib import Path
import h5py
import numpy as np
import pytest

from optocurator.io.nwb import (
    NwbSpikeReader,
    NwbStimulusReader,
    inspect_nwb,
    load_nwb_probe_geometry,
)
from optocurator.ui.app import MainWindow
from optocurator.ui.view_detection import DetectionView


def _create_synthetic_nwb(nwb_path: Path) -> Path:
    """Create a minimal Allen-style HDF5 NWB file with two probes, QC flags, and opto intervals."""
    with h5py.File(nwb_path, "w") as hf:
        hf.create_dataset("identifier", data=b"ses-synth-nwb-001")

        # Electrodes table: 4 sites on ProbeA, 2 sites on ProbeB
        ephys = hf.require_group("general/extracellular_ephys")
        el = ephys.create_group("electrodes")
        el.create_dataset("id", data=np.array([0, 1, 2, 3, 4, 5], dtype=np.int64))
        el.create_dataset("rel_x", data=np.array([0.0, 32.0, 0.0, 32.0, 0.0, 32.0], dtype=np.float64))
        el.create_dataset("rel_y", data=np.array([20.0, 20.0, 40.0, 40.0, 100.0, 120.0], dtype=np.float64))
        el.create_dataset(
            "group_name",
            data=np.array([b"ProbeA", b"ProbeA", b"ProbeA", b"ProbeA", b"ProbeB", b"ProbeB"]),
        )

        # Units table: 3 units (2 ProbeA [1 QC True, 1 QC False], 1 ProbeB [QC True])
        u = hf.create_group("units")
        u.create_dataset("id", data=np.array([0, 1, 2], dtype=np.int64))
        u.create_dataset("ks_unit_id", data=np.array([10, 11, 20], dtype=np.int64))
        u.create_dataset("device_name", data=np.array([b"ProbeA", b"ProbeA", b"ProbeB"]))
        u.create_dataset("default_qc", data=np.array([True, False, True], dtype=bool))
        u.create_dataset("depth", data=np.array([21.0, 39.0, 119.0], dtype=np.float64))
        u.create_dataset("shank", data=np.array([b"0", b"0", b"1"]))

        onsets = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float64)
        sp_u0 = np.sort(np.concatenate([onsets + 0.004, onsets + 0.204, onsets + 0.404]))
        sp_u1 = np.array([0.5, 2.5, 4.5], dtype=np.float64)
        sp_u2 = np.sort(onsets + 0.006)
        all_sp = np.concatenate([sp_u0, sp_u1, sp_u2])
        u.create_dataset("spike_times", data=all_sp)
        u.create_dataset(
            "spike_times_index",
            data=np.array([len(sp_u0), len(sp_u0) + len(sp_u1), len(all_sp)], dtype=np.uint32),
        )

        all_els = np.array([0, 1, 2, 3, 0, 1, 2, 3, 4, 5], dtype=np.int64)
        u.create_dataset("electrodes", data=all_els)
        u.create_dataset("electrodes_index", data=np.array([4, 8, 10], dtype=np.uint32))

        # Intervals: 5 Hz pulse train (10 ms wide) and raised cosine
        iv = hf.create_group("intervals")
        g5 = iv.create_group("5 hz pulse train_presentations")
        g5.create_dataset("start_time", data=onsets)
        g5.create_dataset("stop_time", data=onsets + 1.0)
        g5.create_dataset("condition", data=np.array([b"Each pulse is 10 ms wide"] * len(onsets)))

        gr = iv.create_group("raised_cosine_presentations")
        gr.create_dataset("start_time", data=np.array([10.0, 13.0], dtype=np.float64))
        gr.create_dataset("stop_time", data=np.array([11.0, 14.0], dtype=np.float64))
        gr.create_dataset("condition", data=np.array([b"half-period of a cosine wave"] * 2))

    return nwb_path


def test_nwb_inspect_and_readers(tmp_path: Path):
    """Verify inspect_nwb, NwbSpikeReader probe/QC filtering, and NwbStimulusReader."""
    nwb_file = _create_synthetic_nwb(tmp_path / "synthetic.nwb")

    summary = inspect_nwb(nwb_file)
    assert summary.session_id == "ses-synth-nwb-001"
    assert summary.n_total_units == 3
    assert summary.probe_unit_counts == {"ProbeA": 2, "ProbeB": 1}
    assert summary.probe_qc_counts == {"ProbeA": 1, "ProbeB": 1}
    assert [p.display_name for p in summary.protocols] == ["5 hz pulse train", "raised_cosine"]

    # ProbeA with qc_only=True -> 1 unit (ks_unit_id=10)
    units_a = NwbSpikeReader(probe_name="ProbeA", qc_only=True).load(nwb_file)
    assert len(units_a) == 1
    assert units_a[0].unit_id == 10
    assert units_a[0].depth == pytest.approx(21.0)
    assert len(units_a[0].spike_times) == 12

    # ProbeA with qc_only=False -> 2 units
    units_a_all = NwbSpikeReader(probe_name="ProbeA", qc_only=False).load(nwb_file)
    assert [u.unit_id for u in units_a_all] == [10, 11]

    # Probe geometry filtered to ProbeB (2 sites)
    pos_b, _ = load_nwb_probe_geometry(nwb_file, probe_name="ProbeB")
    assert pos_b is not None and pos_b.shape == (2, 2)

    # Pulse-train expansion: 4 trains * 5 pulses = 20 pulses @ 10 ms width
    protos_expanded = NwbStimulusReader(expand_pulse_trains=True).load_all(nwb_file)
    p5 = next(p for p in protos_expanded if p.name == "5 hz pulse train")
    assert len(p5.onsets) == 20
    assert np.allclose((p5.offsets - p5.onsets) * 1000.0, 10.0)

    # Without expansion: 4 train intervals @ 1000 ms
    protos_train = NwbStimulusReader(expand_pulse_trains=False).load_all(nwb_file)
    p5_train = next(p for p in protos_train if p.name == "5 hz pulse train")
    assert len(p5_train.onsets) == 4
    assert np.allclose((p5_train.offsets - p5_train.onsets) * 1000.0, 1000.0)


def test_detection_view_load_nwb_and_menu_action(qapp, tmp_path: Path):
    """Verify DetectionView.load_nwb_file and MainWindow File -> Load NWB file action."""
    nwb_file = _create_synthetic_nwb(tmp_path / "synthetic_ui.nwb")

    win = MainWindow()
    win.switch_to_detection()
    qapp.processEvents()
    assert any(a.text() == "Load NWB file…" for a in win.file_menu.actions() if a.isVisible())

    ok = win.detection_view.load_nwb_file(
        nwb_file,
        probe_name="ProbeA",
        qc_only=True,
        expand_pulse_trains=True,
    )
    assert ok is True
    assert len(win.detection_view.units) == 1
    assert win.detection_view.session_id == "ses-synth-nwb-001_ProbeA"
    assert "5 hz pulse train" in win.detection_view.protocols
    assert tuple(win.detection_view.raster_plot.getPlotItem()._raster_cache["pulse_band"].getRegion()) == (0.0, 10.0)
    win.close()


def test_real_openscope_nwb_smoke(qapp):
    """Smoke-test real OpenScope Psycode NWB file if OPTOCURATOR_TEST_NWB_DIR is configured."""
    env_nwb = os.environ.get("OPTOCURATOR_TEST_NWB_DIR", "").strip()
    if not env_nwb:
        pytest.skip("OPTOCURATOR_TEST_NWB_DIR not set.")
    nwb_dir = Path(env_nwb)
    files = sorted(nwb_dir.glob("*.nwb")) if nwb_dir.is_dir() else []
    if not files:
        pytest.skip(f"No .nwb files found in OPTOCURATOR_TEST_NWB_DIR: {nwb_dir}")

    summary = inspect_nwb(files[0])
    assert summary.n_total_units > 0
    assert len(summary.probe_unit_counts) >= 1
    assert len(summary.protocols) >= 1

    first_probe = next(iter(summary.probe_unit_counts))
    det = DetectionView()
    ok = det.load_nwb_file(
        files[0],
        probe_name=first_probe,
        qc_only=True,
        expand_pulse_trains=True,
    )
    assert ok is True
    assert len(det.units) > 0
    assert det._channel_positions is not None and len(det._channel_positions) > 0
    det.close()


def test_nwb_export_persists_channel_positions_for_clustering(qapp, tmp_path: Path):
    """Exporting a loaded NWB session must persist _channel_positions into <session_id>_spikes.npz (Bug IO6)."""
    from optocurator.ui.view_clusters import ClusteringView

    nwb_file = _create_synthetic_nwb(tmp_path / "synthetic_probe_export.nwb")
    det = DetectionView()
    assert det.load_nwb_file(nwb_file, probe_name="ProbeA", qc_only=False, expand_pulse_trains=True) is True
    expected_pos = np.asarray(det._channel_positions, dtype=np.float64)
    assert expected_pos.shape == (4, 2)

    export_dir = tmp_path / "nwb_export"
    csv_path = det.export_curated_session(export_dir, show_dialog=False, precompute_pngs=False)
    assert csv_path is not None and csv_path.is_file()

    npz_path = export_dir / f"{det.session_id}_arrays" / f"{det.session_id}_spikes.npz"
    assert npz_path.is_file()
    with np.load(npz_path, allow_pickle=True) as npz_data:
        assert "channel_positions" in npz_data.files
        np.testing.assert_allclose(npz_data["channel_positions"], expected_pos)

    cview = ClusteringView()
    assert cview.add_experiment_folder(export_dir) is True
    np.testing.assert_allclose(cview._experiment_probes[det.session_id], expected_pos)
    cview.close()
    det.close()

