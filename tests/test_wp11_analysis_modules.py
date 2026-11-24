"""Unit tests for WP11 isolated analysis modules (waveform and likelihood)."""

from pathlib import Path
import numpy as np
import pytest

import optocurator.analysis.metrics as metrics_mod
from optocurator.analysis.likelihood import (
    CriterionSpec,
    compute_population_spreads,
    compute_session_likelihoods,
    compute_unit_likelihood,
)
from optocurator.analysis.metrics import compute_session_metrics, compute_unit_metrics
from optocurator.analysis.waveform import (
    compute_unit_waveform_correlation,
    compute_wf_r,
    extract_evoked_and_spontaneous_snippets,
    extract_session_waveforms,
    split_evoked_and_spontaneous_spikes,
    subsample_indices,
)
from optocurator.models import StatsParameters, StimulusProtocol, UnitData, UnitMetrics


def test_old_waveform_correlation_removed_from_metrics():
    """Old compute_waveform_correlation must be removed from metrics.py."""
    assert not hasattr(metrics_mod, "compute_waveform_correlation")


def test_split_evoked_and_spontaneous_spikes_and_fallback():
    """Verify evoked vs baseline split and fallback when baseline window is empty."""
    onsets = np.array([1.0, 2.0])
    spikes = np.array([0.500, 0.980, 1.005, 1.970, 2.010, 2.500])

    ev, sp = split_evoked_and_spontaneous_spikes(
        spikes, onsets, evoked_window_ms=(0.0, 20.0), baseline_window_ms=(-50.0, -5.0)
    )
    np.testing.assert_array_equal(ev, [2, 4])
    np.testing.assert_array_equal(sp, [1, 3])

    # When baseline window has no spikes, fall back to all non-evoked spikes
    ev2, sp2 = split_evoked_and_spontaneous_spikes(
        spikes, onsets, evoked_window_ms=(0.0, 20.0), baseline_window_ms=(-5.0, -1.0)
    )
    np.testing.assert_array_equal(ev2, [2, 4])
    np.testing.assert_array_equal(sp2, [0, 1, 3, 5])


def test_split_evoked_and_spontaneous_spikes_half_open_window():
    """waveform.py must match metrics.py [start, end) half-open window convention (Bug M5)."""
    from optocurator.analysis.metrics import compute_first_spike_latency, compute_response_reliability

    onsets = np.array([1.000])
    # 0.950 (-50 ms, base start included), 0.995 (-5 ms, base end excluded),
    # 1.000 (0 ms, evoked start included), 1.020 (+20 ms, evoked end excluded)
    spikes = np.array([0.950, 0.995, 1.000, 1.020])
    ev, sp = split_evoked_and_spontaneous_spikes(
        spikes, onsets, evoked_window_ms=(0.0, 20.0), baseline_window_ms=(-50.0, -5.0)
    )
    np.testing.assert_array_equal(ev, [2])
    np.testing.assert_array_equal(sp, [0])

    # Consistency with metrics.py [start, end) boundary behavior at +20.0 ms
    edge_only = np.array([1.020])
    ev_edge, _ = split_evoked_and_spontaneous_spikes(
        edge_only, onsets, evoked_window_ms=(0.0, 20.0), baseline_window_ms=(-50.0, -5.0)
    )
    assert len(ev_edge) == 0
    assert compute_response_reliability(edge_only, onsets, eval_window_ms=(0.0, 20.0)) == 0.0
    assert np.isnan(compute_first_spike_latency(edge_only, onsets, eval_window_ms=(0.0, 20.0)))


def test_subsample_indices_deterministic():
    """Verify deterministic evenly-spaced subsampling."""
    idx = np.arange(100, dtype=np.int64)
    sub = subsample_indices(idx, 10)
    assert len(sub) == 10
    assert sub[0] == 0
    assert sub[-1] == 99
    assert np.all(np.diff(sub) > 0)

    short = np.array([3, 7, 9], dtype=np.int64)
    np.testing.assert_array_equal(subsample_indices(short, 10), short)


def test_extract_snippets_peak_channel_and_invalid_inputs():
    """Verify 3D peak-channel selection and None returns for invalid inputs."""
    onsets = np.array([1.0, 2.0])
    spikes = np.array([0.98, 1.005, 1.98, 2.005])
    t = np.linspace(-1.0, 1.0, 40)
    shape = -np.exp(-(t**2) / 0.1)

    # 3D array: channel 1 has largest peak-to-peak amplitude
    wf_3d = np.zeros((4, 3, 40), dtype=np.float64)
    wf_3d[:, 0, :] = 0.2 * shape
    wf_3d[:, 1, :] = 2.0 * shape
    wf_3d[:, 2, :] = 0.5 * shape

    snippets = extract_evoked_and_spontaneous_snippets(wf_3d, spikes, onsets, max_spikes_per_group=40)
    assert snippets is not None
    ev_s, sp_s = snippets
    assert ev_s.shape == (2, 40)
    assert sp_s.shape == (2, 40)
    assert np.isclose(compute_wf_r(ev_s, sp_s), 1.0, atol=1e-6)

    # Invalid / missing inputs return None or NaN
    assert extract_evoked_and_spontaneous_snippets(None, spikes, onsets) is None
    assert extract_evoked_and_spontaneous_snippets(np.ones(40), spikes, onsets) is None
    assert extract_evoked_and_spontaneous_snippets(np.ones((2, 40)), spikes, onsets) is None
    assert np.isnan(compute_unit_waveform_correlation(None, spikes, onsets))
    assert np.isnan(compute_wf_r(np.ones((2, 20)), np.ones((2, 20))))


def test_extract_session_waveforms_from_synthetic_dat(tmp_path: Path):
    """Extract peak-channel snippets from synthetic int16 .dat + params.py."""
    fs = 10000.0
    n_channels = 4
    duration_s = 3.0
    n_samples = int(fs * duration_s)

    t_wf = np.linspace(-1.0, 1.0, 30)
    ap_template = (-500.0 * np.exp(-(t_wf**2) / 0.12)).astype(np.int16)

    onsets = np.array([1.0, 2.0])
    spikes = np.array([0.970, 1.005, 1.970, 2.005])

    data = np.zeros((n_samples, n_channels), dtype=np.int16)
    peak_ch = 2
    for spk in spikes:
        center = int(round(spk * fs))
        start = center - 10
        end = center + 20
        data[start:end, peak_ch] = ap_template

    dat_path = tmp_path / "recording.dat"
    data.tofile(dat_path)

    params_path = tmp_path / "params.py"
    params_path.write_text(
        "dat_path = 'recording.dat'\n"
        f"n_channels_dat = {n_channels}\n"
        "dtype = 'int16'\n"
        f"sample_rate = {fs}\n",
        encoding="utf-8",
    )

    unit = UnitData(
        session_id="sess_dat",
        unit_id=1,
        spike_times=spikes,
        sampling_rate=fs,
        channel=peak_ch,
        waveforms=None,
    )
    sp = StatsParameters(wf_window_start_ms=-1.0, wf_window_end_ms=2.0, wf_max_spikes=40)
    extract_session_waveforms(tmp_path, [unit], onsets, stats_params=sp)

    assert unit.waveforms is not None
    assert unit.waveforms.shape == (len(spikes), 30)
    r = compute_unit_waveform_correlation(unit.waveforms, unit.spike_times, onsets)
    assert r > 0.99

    # Missing binary folder leaves unit.waveforms = None cleanly
    unit_missing = UnitData(
        session_id="sess_empty",
        unit_id=2,
        spike_times=spikes,
        sampling_rate=fs,
        channel=0,
        waveforms=None,
    )
    empty_dir = tmp_path / "no_bin"
    empty_dir.mkdir()
    extract_session_waveforms(empty_dir, [unit_missing], onsets, stats_params=sp)
    assert unit_missing.waveforms is None


def test_extract_session_waveforms_excludes_clipped_and_unextracted_rows(tmp_path: Path):
    """Boundary-clipped and unextracted spikes must be NaN and excluded from snippets (Bug M3)."""
    fs = 10000.0
    n_channels = 2
    n_samples = int(fs * 2.0)

    t_wf = np.linspace(-1.0, 1.0, 30)
    ap_template = (-500.0 * np.exp(-(t_wf**2) / 0.12)).astype(np.int16)

    # Pulse 0 near t=0 -> evoked spike at 0.0002 s (sample 2) is clipped (start = 2 - 10 = -8 < 0)
    # Pulse 1 at 1.0 s -> valid spontaneous at 0.970 s, valid evoked at 1.005 s, valid evoked at 1.008 s
    onsets = np.array([0.0001, 1.0000])
    spikes = np.array([0.0002, 0.9700, 1.0050, 1.0080])

    data = np.zeros((n_samples, n_channels), dtype=np.int16)
    peak_ch = 1
    for spk in spikes[1:]:
        center = int(round(spk * fs))
        data[center - 10 : center + 20, peak_ch] = ap_template

    dat_path = tmp_path / "clipped.dat"
    data.tofile(dat_path)
    (tmp_path / "params.py").write_text(
        "dat_path = 'clipped.dat'\nn_channels_dat = 2\ndtype = 'int16'\nsample_rate = 10000.0\n",
        encoding="utf-8",
    )

    unit = UnitData(
        session_id="sess_clip",
        unit_id=1,
        spike_times=spikes,
        sampling_rate=fs,
        channel=peak_ch,
        waveforms=None,
    )
    sp = StatsParameters(wf_window_start_ms=-1.0, wf_window_end_ms=2.0, wf_max_spikes=2)
    extract_session_waveforms(tmp_path, [unit], onsets, stats_params=sp)

    assert unit.waveforms is not None
    # Row 0 (clipped at sample 2) and row 2 (skipped by max_spikes=2 subsampling of [0, 2, 3] -> [0, 3]) must be NaN
    assert np.all(np.isnan(unit.waveforms[0]))
    assert np.all(np.isnan(unit.waveforms[2]))

    # Extracting snippets up to 40 must exclude both NaN rows, keeping only the 1 valid finite evoked spike (row 3)
    snippets = extract_evoked_and_spontaneous_snippets(
        unit.waveforms, unit.spike_times, onsets, max_spikes_per_group=40
    )
    assert snippets is not None
    ev_s, sp_s = snippets
    assert ev_s.shape == (1, 30)
    assert sp_s.shape == (1, 30)
    assert np.all(np.isfinite(ev_s))
    np.testing.assert_allclose(ev_s[0], ap_template.astype(np.float64), atol=1e-4)

    # When only non-finite evoked rows remain, return None
    wf_only_bad_ev = unit.waveforms.copy()
    wf_only_bad_ev[3, :] = np.nan
    assert (
        extract_evoked_and_spontaneous_snippets(
            wf_only_bad_ev, unit.spike_times, onsets, max_spikes_per_group=40
        )
        is None
    )


def test_missing_waveforms_do_not_block_tagging():
    """Decision 1: units without waveforms get NaN WF r and can still be tagged."""
    onsets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    proto = StimulusProtocol(onsets=onsets, offsets=onsets + 0.02)
    unit = UnitData(
        session_id="sess1",
        unit_id=7,
        spike_times=onsets + 0.004,
        sampling_rate=30000.0,
        waveforms=None,
    )

    m = compute_unit_metrics(unit, proto, max_latency_ms=8.0, min_reliability=0.6, min_waveform_r=0.9)
    assert np.isnan(m.waveform_correlation)
    assert m.is_tagged is True

    ms = compute_session_metrics([unit], proto, max_latency_ms=8.0, min_reliability=0.6, min_waveform_r=0.9)
    assert len(ms) == 1
    assert np.isnan(ms[0].waveform_correlation)
    assert ms[0].is_tagged is True


def _make_metrics(
    gid: str,
    lat: float,
    jit: float,
    rel: float,
    wf_r: float,
    salt_p: float,
) -> UnitMetrics:
    return UnitMetrics(
        global_id=gid,
        session_id="s1",
        unit_id=int(gid.split("_u")[-1]),
        plot_path="",
        latency_ms=lat,
        jitter_ms=jit,
        reliability=rel,
        waveform_correlation=wf_r,
        salt_p_value=salt_p,
    )


def test_likelihood_boundary_monotonicity_and_missing_wf():
    """Verify threshold ~0.5 boundary, monotone scoring, log SALT p, and skipped NaN WF r."""
    criteria = [
        CriterionSpec("latency_ms", threshold=8.0, direction="le"),
        CriterionSpec("jitter_ms", threshold=2.0, direction="le"),
        CriterionSpec("reliability", threshold=0.6, direction="ge"),
        CriterionSpec("salt_p_value", threshold=0.05, direction="le", is_log=True),
        CriterionSpec("waveform_correlation", threshold=0.9, direction="ge"),
    ]

    # Single-criterion exact threshold check -> ~0.5
    single_crit = [CriterionSpec("latency_ms", threshold=8.0, direction="le")]
    u_thresh = _make_metrics("s1_u0001", lat=8.0, jit=1.0, rel=0.8, wf_r=0.95, salt_p=0.01)
    spreads_single = compute_population_spreads([u_thresh], single_crit)
    assert pytest.approx(compute_unit_likelihood(u_thresh, single_crit, spreads_single), abs=1e-6) == 0.5

    # Population with strong, borderline, and weak units
    u_strong = _make_metrics("s1_u0001", lat=3.0, jit=0.5, rel=0.95, wf_r=0.98, salt_p=0.001)
    u_mid = _make_metrics("s1_u0002", lat=7.5, jit=1.8, rel=0.65, wf_r=0.91, salt_p=0.04)
    u_weak = _make_metrics("s1_u0003", lat=18.0, jit=5.0, rel=0.10, wf_r=0.20, salt_p=0.80)
    u_no_wf = _make_metrics("s1_u0004", lat=3.0, jit=0.5, rel=0.95, wf_r=float("nan"), salt_p=0.001)

    scores = compute_session_likelihoods([u_strong, u_mid, u_weak, u_no_wf], criteria)
    assert 0.0 <= scores[u_weak.global_id] < scores[u_mid.global_id] < scores[u_strong.global_id] <= 1.0
    # Missing WF r is skipped (not penalized) and matches 4-criterion score
    assert scores[u_no_wf.global_id] > scores[u_strong.global_id]
    spreads_4 = compute_population_spreads([u_strong, u_mid, u_weak, u_no_wf], criteria[:-1])
    assert pytest.approx(scores[u_no_wf.global_id], abs=1e-9) == compute_unit_likelihood(
        u_no_wf, criteria[:-1], spreads_4
    )
    assert u_strong.likelihood == scores[u_strong.global_id]

    # Disabled criteria -> 0.0
    disabled = [CriterionSpec("latency_ms", threshold=8.0, direction="le", enabled=False)]
    assert compute_unit_likelihood(u_strong, disabled, {"latency_ms": 1.0}) == 0.0


def test_likelihood_excludes_999_sentinel_units():
    """Verify silent 999.0 ms sentinel units do not inflate MAD/std spread or receive non-zero likelihood (Bug M1)."""
    criteria = [CriterionSpec("latency_ms", threshold=8.0, direction="le")]
    units = [
        _make_metrics("s1_u0001", lat=4.0, jit=0.8, rel=0.9, wf_r=0.95, salt_p=0.001),
        _make_metrics("s1_u0002", lat=6.0, jit=1.0, rel=0.8, wf_r=0.95, salt_p=0.005),
        _make_metrics("s1_u0003", lat=20.0, jit=2.0, rel=0.4, wf_r=0.85, salt_p=0.10),
    ] + [
        _make_metrics(f"s1_u{i:04d}", lat=999.0, jit=999.0, rel=0.0, wf_r=float("nan"), salt_p=1.0)
        for i in range(4, 11)
    ]

    spreads = compute_population_spreads(units, criteria)
    assert spreads["latency_ms"] < 10.0

    scores = compute_session_likelihoods(units, criteria)
    assert scores["s1_u0001"] > 0.95
    assert scores["s1_u0003"] < 1e-4
    for i in range(4, 11):
        assert scores[f"s1_u{i:04d}"] == 0.0
