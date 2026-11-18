"""Unit tests for Phy/Kilosort and stimulus pulse I/O readers."""

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
    """Verify readers on real dataset if path exists."""
    real_path = Path("E:/OptotaggingCuration/test_OptoCurator/sorted_dir")
    if not real_path.exists():
        pytest.skip("Real test dataset not available on this machine.")

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
