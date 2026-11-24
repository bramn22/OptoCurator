"""Unit tests for Phy/Kilosort and stimulus pulse I/O readers."""

import os
from pathlib import Path
import numpy as np
import pytest
import scipy.io

from optocurator.io import PhyKilosortReader, MatNpyStimulusReader
from optocurator.models import StimulusProtocol, UnitData


def test_phy_reader_synthetic(tmp_path: Path):
    """Test PhyKilosortReader on synthetic sorting directory."""
    # Create fake Kilosort files
    sr = 30000.0
    # Unit 0 has 3 spikes, Unit 1 has 2 spikes
    spike_times_samples = np.array([30000, 30300, 30600, 60000, 60600], dtype=np.uint64)
    spike_clusters = np.array([0, 0, 0, 1, 1], dtype=np.int32)

    np.save(tmp_path / "spike_times.npy", spike_times_samples)
    np.save(tmp_path / "spike_clusters.npy", spike_clusters)

    # params.py
    params_content = f"sample_rate = {sr}\nn_channels_dat = 384\n"
    (tmp_path / "params.py").write_text(params_content)

    # cluster_info.tsv
    tsv_content = "id\tch\tdepth\n0\t12\t450\n1\t15\t550\n"
    (tmp_path / "cluster_info.tsv").write_text(tsv_content)

    reader = PhyKilosortReader()
    units = reader.load(tmp_path, session_id="test_session")

    assert len(units) == 2
    u0 = units[0]
    assert u0.session_id == "test_session"
    assert u0.unit_id == 0
    assert u0.global_id == "test_session_u0000"
    assert u0.channel == 12
    assert u0.depth == 450
    assert len(u0.spike_times) == 3
    # Check timestamps converted to seconds (30000 samples / 30000 Hz = 1.0s)
    np.testing.assert_allclose(u0.spike_times, [1.0, 1.01, 1.02])

    u1 = units[1]
    assert u1.unit_id == 1
    assert u1.global_id == "test_session_u0001"
    assert u1.channel == 15
    assert u1.depth == 550
    assert len(u1.spike_times) == 2
    np.testing.assert_allclose(u1.spike_times, [2.0, 2.02])


def test_phy_reader_missing_files_error(tmp_path: Path):
    """Test PhyKilosortReader raises FileNotFoundError when essential files are missing."""
    reader = PhyKilosortReader()
    with pytest.raises(FileNotFoundError):
        reader.load(tmp_path)


def test_stimulus_reader_npy_2d(tmp_path: Path):
    """Test MatNpyStimulusReader with 2D sample-indexed .npy file."""
    sr = 30000.0
    # 3 pulses: onsets & offsets in samples
    pulses = np.array([
        [30000, 30600],   # 1.0s to 1.02s
        [60000, 60600],   # 2.0s to 2.02s
        [90000, 90600],   # 3.0s to 3.02s
    ], dtype=np.uint32)

    file_path = tmp_path / "opto_idxs_OPTS_03.npy"
    np.save(file_path, pulses)

    reader = MatNpyStimulusReader()
    discovered = reader.discover(tmp_path)
    assert len(discovered) == 1
    assert discovered[0].name == "opto_idxs_OPTS_03.npy"

    proto = reader.load(file_path, sampling_rate=sr)
    assert proto.name == "OPTS_03"
    assert len(proto.onsets) == 3
    assert len(proto.offsets) == 3
    np.testing.assert_allclose(proto.onsets, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(proto.offsets, [1.02, 2.02, 3.02])


def test_stimulus_reader_mat(tmp_path: Path):
    """Test MatNpyStimulusReader with .mat file."""
    # 2 pulses in seconds
    pulses_sec = np.array([[10.5, 10.52], [20.5, 20.52]])
    mat_file = tmp_path / "opto_pulses.mat"
    scipy.io.savemat(str(mat_file), {"opto": pulses_sec})

    reader = MatNpyStimulusReader()
    proto = reader.load(mat_file)
    assert len(proto.onsets) == 2
    np.testing.assert_allclose(proto.onsets, [10.5, 20.5])
    np.testing.assert_allclose(proto.offsets, [10.52, 20.52])


def test_real_dataset_readers():
    """Verify readers on real dataset if OPTOCURATOR_TEST_DATA_DIR is configured."""
    data_dir = os.environ.get("OPTOCURATOR_TEST_DATA_DIR", "").strip()
    if not data_dir:
        pytest.skip("OPTOCURATOR_TEST_DATA_DIR not set.")
    real_path = Path(data_dir)
    if not real_path.exists():
        pytest.skip(f"OPTOCURATOR_TEST_DATA_DIR path does not exist: {real_path}")

    s_reader = MatNpyStimulusReader()
    pulse_files = s_reader.discover(real_path)
    assert len(pulse_files) >= 1
    proto = s_reader.load(pulse_files[0])
    assert proto.name == "OPTS_03"
    assert len(proto.onsets) == 500

    phy_reader = PhyKilosortReader()
    units = phy_reader.load(real_path)
    assert len(units) > 100
    assert units[0].unit_id == 0
    assert len(units[0].spike_times) > 0


def test_timestamps_dtype_strict_samples_vs_seconds(tmp_path: Path):
    """Integer arrays are sample indices; float arrays are seconds without >10000 guessing (Bug IO1)."""
    # 1. Long recording (> 10,000 s) in float64 must NOT be divided by sampling_rate
    long_spikes = np.array([100.0, 10500.0, 20000.0], dtype=np.float64)
    np.save(tmp_path / "spike_times.npy", long_spikes)
    np.save(tmp_path / "spike_clusters.npy", np.array([0, 0, 0], dtype=np.int32))
    (tmp_path / "params.py").write_text("sample_rate = 30000.0\n", encoding="utf-8")

    units = PhyKilosortReader().load(tmp_path, session_id="long_float")
    assert len(units) == 1
    np.testing.assert_allclose(units[0].spike_times, [100.0, 10500.0, 20000.0])

    # 2. Small integer sample indices (<= 10,000) must still be divided by sampling_rate
    int_spikes = np.array([300, 600], dtype=np.int64)
    np.save(tmp_path / "spike_times.npy", int_spikes)
    np.save(tmp_path / "spike_clusters.npy", np.array([0, 0], dtype=np.int32))
    units_int = PhyKilosortReader().load(tmp_path, session_id="short_int")
    np.testing.assert_allclose(units_int[0].spike_times, [0.01, 0.02])

    # 3. Float stimulus timestamps > 10,000 s must stay seconds (not divided by sampling_rate)
    stim_reader = MatNpyStimulusReader()
    float_pulses_path = tmp_path / "opto_float_long.npy"
    np.save(float_pulses_path, np.array([[100.0, 100.02], [10500.0, 10500.02]], dtype=np.float64))
    proto_float = stim_reader.load(float_pulses_path, sampling_rate=30000.0)
    np.testing.assert_allclose(proto_float.onsets, [100.0, 10500.0])
    np.testing.assert_allclose(proto_float.offsets, [100.02, 10500.02])

    # 4. Float stimulus values that happen to be whole numbers stay seconds
    float_whole_path = tmp_path / "opto_float_whole.npy"
    np.save(float_whole_path, np.array([[300.0, 300.5], [15000.0, 15000.5]], dtype=np.float64))
    proto_whole = stim_reader.load(float_whole_path, sampling_rate=30000.0)
    np.testing.assert_allclose(proto_whole.onsets, [300.0, 15000.0])
    np.testing.assert_allclose(proto_whole.offsets, [300.5, 15000.5])

    # 5. Integer stimulus arrays are divided by sampling_rate
    int_pulses_path = tmp_path / "opto_idxs_SHORT.npy"
    np.save(int_pulses_path, np.array([[300, 600], [900, 1200]], dtype=np.int32))
    proto_int = stim_reader.load(int_pulses_path, sampling_rate=30000.0)
    np.testing.assert_allclose(proto_int.onsets, [0.01, 0.03])
    np.testing.assert_allclose(proto_int.offsets, [0.02, 0.04])


def test_stimulus_reader_mat_row_vector_and_transposed_2xN(tmp_path: Path):
    """MATLAB (1, N) row vectors and (2, N) transposed matrices must load all N pulses (Bug IO2)."""
    reader = MatNpyStimulusReader(default_pulse_duration_s=0.020)

    # 1. MATLAB (1, 100) row vector -> all 100 pulses loaded
    onsets_100 = np.linspace(1.0, 50.0, 100)
    mat_row = tmp_path / "opto_row.mat"
    scipy.io.savemat(str(mat_row), {"pulses": onsets_100.reshape(1, 100)})
    proto_row = reader.load(mat_row)
    assert len(proto_row.onsets) == 100
    np.testing.assert_allclose(proto_row.onsets, onsets_100)
    np.testing.assert_allclose(proto_row.offsets, onsets_100 + 0.020)

    # 2. MATLAB (2, 10) [onsets; offsets] -> all 10 pulses with exact offsets
    on_10 = np.arange(1.0, 11.0, 1.0)
    off_10 = on_10 + 0.005
    mat_2xn = tmp_path / "opto_2xn.mat"
    scipy.io.savemat(str(mat_2xn), {"pulses": np.vstack([on_10, off_10])})
    proto_2xn = reader.load(mat_2xn)
    assert len(proto_2xn.onsets) == 10
    np.testing.assert_allclose(proto_2xn.onsets, on_10)
    np.testing.assert_allclose(proto_2xn.offsets, off_10)

    # 3. Single-element (1, 1) row vector -> 1 pulse
    npy_1x1 = tmp_path / "opto_single.npy"
    np.save(npy_1x1, np.array([[5.0]], dtype=np.float64))
    proto_1x1 = reader.load(npy_1x1)
    assert len(proto_1x1.onsets) == 1
    np.testing.assert_allclose(proto_1x1.onsets, [5.0])


def test_phy_reader_scientific_notation_and_parent_params_py(tmp_path: Path):
    """_read_sampling_rate must parse scientific/underscore literals and fallback to parent params.py (Bug IO3)."""
    # 1. Scientific notation sample_rate = 3e4 in params.py
    sci_dir = tmp_path / "sci"
    sci_dir.mkdir()
    (sci_dir / "params.py").write_text("sample_rate = 3e4\nn_channels_dat = 64\n", encoding="utf-8")
    np.save(sci_dir / "spike_times.npy", np.array([30000], dtype=np.uint64))
    np.save(sci_dir / "spike_clusters.npy", np.array([0], dtype=np.int32))

    units_sci = PhyKilosortReader().load(sci_dir, session_id="sci")
    assert units_sci[0].sampling_rate == pytest.approx(30000.0)
    np.testing.assert_allclose(units_sci[0].spike_times, [1.0])

    # 2. Parent params.py fallback with underscore literal (25_000) when sorted_dir has no params.py
    parent_dir = tmp_path / "rec"
    sub_dir = parent_dir / "sorted_dir"
    sub_dir.mkdir(parents=True)
    (parent_dir / "params.py").write_text("sample_rate = 25_000\n", encoding="utf-8")
    np.save(sub_dir / "spike_times.npy", np.array([25000], dtype=np.uint64))
    np.save(sub_dir / "spike_clusters.npy", np.array([0], dtype=np.int32))

    units_sub = PhyKilosortReader().load(sub_dir)
    assert units_sub[0].sampling_rate == pytest.approx(25000.0)
    np.testing.assert_allclose(units_sub[0].spike_times, [1.0])


def test_phy_reader_templates_and_non_contiguous_channel_map(tmp_path: Path):
    """PhyKilosortReader must infer units from templates without cluster_info/spike_templates and index probe_positions by template/site index (Bug IO9)."""
    # Case 1: Only spike_times, spike_clusters, templates, channel_positions, channel_map (no cluster_info.tsv or spike_templates.npy)
    d1 = tmp_path / "tmpl_only"
    d1.mkdir()
    np.save(d1 / "spike_times.npy", np.array([30000, 60000], dtype=np.uint64))
    np.save(d1 / "spike_clusters.npy", np.array([0, 1], dtype=np.int32))
    np.save(d1 / "channel_positions.npy", np.array([[0.0, 0.0], [32.0, 200.0], [16.0, 400.0], [48.0, 600.0]]))
    np.save(d1 / "channel_map.npy", np.array([2, 10, 20, 30], dtype=np.int32))

    # 2 templates x 5 timepoints x 4 channels
    templates = np.zeros((2, 5, 4), dtype=np.float32)
    templates[0, 2, 1] = -10.0  # Unit 0 peaks at site index 1 -> hw=10, pos=[32.0, 200.0]
    templates[1, 2, 0] = -10.0  # Unit 1 peaks at site index 0 -> hw=2,  pos=[0.0, 0.0] (must NOT fall back to probe_positions[2])
    np.save(d1 / "templates.npy", templates)

    units = PhyKilosortReader().load(d1, session_id="tmpl_only")
    assert len(units) == 2
    assert units[0].channel == 10
    assert units[0].probe_x == pytest.approx(32.0)
    assert units[0].depth == pytest.approx(200.0)
    assert units[1].channel == 2
    assert units[1].probe_x == pytest.approx(0.0)
    assert units[1].depth == pytest.approx(0.0)

    # Case 2: spike_templates.npy + non-contiguous channel_map where template 1 peaks at site 0=(0.0, 0.0) with hw=2
    d2 = tmp_path / "non_contig_fallback"
    d2.mkdir()
    np.save(d2 / "spike_times.npy", np.array([30000, 60000], dtype=np.uint64))
    np.save(d2 / "spike_clusters.npy", np.array([0, 1], dtype=np.int32))
    np.save(d2 / "spike_templates.npy", np.array([0, 1], dtype=np.int32))
    np.save(d2 / "channel_positions.npy", np.array([[0.0, 0.0], [32.0, 200.0], [16.0, 400.0], [48.0, 600.0]]))
    np.save(d2 / "channel_map.npy", np.array([2, 10, 20, 30], dtype=np.int32))
    np.save(d2 / "templates.npy", templates)

    units2 = PhyKilosortReader().load(d2, session_id="non_contig")
    assert units2[1].channel == 2
    assert units2[1].probe_x == pytest.approx(0.0)
    assert units2[1].depth == pytest.approx(0.0)
