"""Unit tests for optotagging analytical metrics (PSTH, latency, jitter, reliability)."""

import numpy as np
import pytest

from optocurator.analysis.metrics import (
    compute_first_spike_latency_and_jitter,
    compute_psth_and_raster,
    compute_session_metrics,
    compute_unit_metrics,
)
from optocurator.models import StimulusProtocol, UnitData


def test_first_spike_latency_and_jitter_exact():
    """Verify exact first spike latency, jitter, and reliability calculations."""
    pulse_onsets = np.array([1.0, 2.0, 3.0, 4.0])

    # Spikes with latencies:
    # Trial 1 (t=1.0): 1.005s -> latency = 5.0ms
    # Trial 2 (t=2.0): 2.007s -> latency = 7.0ms
    # Trial 3 (t=3.0): 3.003s -> latency = 3.0ms
    # Trial 4 (t=4.0): 4.005s -> latency = 5.0ms
    # Also add a second spike in trial 1 to ensure only first spike is counted
    spike_times = np.array([1.005, 1.015, 2.007, 3.003, 4.005])

    latency, jitter, reliability = compute_first_spike_latency_and_jitter(
        spike_times, pulse_onsets, eval_window_ms=(0.0, 20.0)
    )

    # Median of [5.0, 7.0, 3.0, 5.0] is 5.0
    assert pytest.approx(latency, abs=1e-3) == 5.0
    # Reliability: 4 / 4 = 1.0
    assert pytest.approx(reliability, abs=1e-3) == 1.0
    # Jitter: std of [5.0, 7.0, 3.0, 5.0]
    expected_std = float(np.std([5.0, 7.0, 3.0, 5.0]))
    assert pytest.approx(jitter, abs=1e-3) == expected_std


def test_first_spike_partial_reliability():
    """Verify reliability when some trials do not evoke spikes in [0, 20ms]."""
    pulse_onsets = np.array([1.0, 2.0, 3.0, 4.0])
    # Only trial 1 and trial 3 have spikes in [0, 20ms]
    spike_times = np.array([1.006, 2.040, 3.008])  # 2.040 is 40ms post-stimulus (outside [0, 20ms])

    latency, jitter, reliability = compute_first_spike_latency_and_jitter(
        spike_times, pulse_onsets, eval_window_ms=(0.0, 20.0)
    )

    # 2 out of 4 trials have spikes in window
    assert pytest.approx(reliability, abs=1e-3) == 0.50
    # Median of [6.0, 8.0] is 7.0
    assert pytest.approx(latency, abs=1e-3) == 7.0


def test_first_spike_empty_spikes():
    """Verify metrics return nan and 0 reliability when no spikes exist."""
    pulse_onsets = np.array([1.0, 2.0])
    spike_times = np.empty(0)

    latency, jitter, reliability = compute_first_spike_latency_and_jitter(
        spike_times, pulse_onsets
    )

    assert np.isnan(latency)
    assert np.isnan(jitter)
    assert reliability == 0.0


def test_psth_and_raster_computation():
    """Verify trial-aligned spike raster and PSTH rate calculation."""
    pulse_onsets = np.array([1.0, 2.0])
    # Trial 1 has spikes at -10ms (0.990), +5ms (1.005)
    # Trial 2 has spike at +5ms (2.005)
    spike_times = np.array([0.990, 1.005, 2.005])

    psth = compute_psth_and_raster(
        spike_times,
        pulse_onsets,
        window_ms=(-50.0, 50.0),
        bin_size_ms=10.0,
        baseline_window_ms=(-50.0, 0.0),
    )

    assert psth["n_trials"] == 2
    assert len(psth["raster_x"]) == 3
    # Check relative timestamps in ms: -10, 5, 5
    np.testing.assert_allclose(psth["raster_x"], [-10.0, 5.0, 5.0], atol=1e-3)
    np.testing.assert_array_equal(psth["raster_y"], [0, 0, 1])

    # Check PSTH bin counts
    # Bins: [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
    # Bin [0, 10] ms has 2 spikes -> raw rate = 2 / (2 trials * 0.010 s) = 100 Hz
    bins = psth["psth_bins"]
    rates_raw = psth["psth_rates_raw"]
    idx_evoked = np.where((bins[:-1] == 0.0) & (bins[1:] == 10.0))[0][0]
    assert pytest.approx(rates_raw[idx_evoked], abs=1e-3) == 100.0
    assert psth["is_smooth"] is True
    assert psth["psth_rates"][idx_evoked] > 0.0

    # Test explicit unsmoothed mode
    psth_raw = compute_psth_and_raster(
        spike_times,
        pulse_onsets,
        window_ms=(-50.0, 50.0),
        bin_size_ms=10.0,
        baseline_window_ms=(-50.0, 0.0),
        smooth=False,
    )
    assert psth_raw["is_smooth"] is False
    assert pytest.approx(psth_raw["psth_rates"][idx_evoked], abs=1e-3) == 100.0


def test_compute_gaussian_psth_elephant():
    """Verify Elephant-based Gaussian smoothed instantaneous firing rate estimation."""
    from optocurator.analysis.metrics import compute_gaussian_psth

    raster_x = np.array([-10.0, 5.0, 5.0])
    n_trials = 2
    window_ms = (-30.0, 50.0)
    psth_bins, psth_rates = compute_gaussian_psth(
        raster_x=raster_x,
        n_trials=n_trials,
        window_ms=window_ms,
        bin_size_ms=2.0,
        gaussian_sigma_ms=2.0,
    )
    assert len(psth_rates) == len(psth_bins) - 1
    assert np.all(psth_rates >= 0.0)
    bin_centers = (psth_bins[:-1] + psth_bins[1:]) / 2.0
    peak_idx = int(np.argmax(psth_rates))
    assert abs(bin_centers[peak_idx] - 5.0) <= 2.0


def test_unit_and_session_metrics():
    """Verify UnitMetrics threshold classification."""
    pulse_onsets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    proto = StimulusProtocol(onsets=pulse_onsets, offsets=pulse_onsets + 0.02, name="test_proto")

    # Fast, reliable tagged unit: 4.0ms latency, 100% reliability
    spikes_tagged = pulse_onsets + 0.004
    unit_tagged = UnitData(
        session_id="test_sess",
        unit_id=1,
        spike_times=spikes_tagged,
        sampling_rate=30000.0,
    )

    # Slow, untagged unit: 16.0ms latency (> 8.0ms)
    spikes_untagged = pulse_onsets + 0.016
    unit_untagged = UnitData(
        session_id="test_sess",
        unit_id=2,
        spike_times=spikes_untagged,
        sampling_rate=30000.0,
    )

    m1 = compute_unit_metrics(unit_tagged, proto, max_latency_ms=8.0, min_reliability=0.60)
    assert m1.is_tagged is True
    assert pytest.approx(m1.latency_ms, abs=1e-2) == 4.0
    assert pytest.approx(m1.reliability, abs=1e-2) == 1.0

    m2 = compute_unit_metrics(unit_untagged, proto, max_latency_ms=8.0, min_reliability=0.60)
    assert m2.is_tagged is False
    assert pytest.approx(m2.latency_ms, abs=1e-2) == 16.0

    # Session metrics
    session_metrics = compute_session_metrics([unit_tagged, unit_untagged], proto)
    assert len(session_metrics) == 2
    assert session_metrics[0].is_tagged is True
    assert session_metrics[1].is_tagged is False


def test_psth_and_raster_latency_ordering():
    """Verify that optogenetic pulses can be ordered by first-spike latency."""
    pulse_onsets = np.array([1.0, 2.0, 3.0, 4.0])
    # Pulse 0 (t=1.0): spike at 1.012 -> latency = 12.0 ms
    # Pulse 1 (t=2.0): no evoked spike
    # Pulse 2 (t=3.0): spike at 3.003 -> latency = 3.0 ms
    # Pulse 3 (t=4.0): spike at 4.007 -> latency = 7.0 ms
    # Also add a pre-stimulus spike to trial 2 at 2.990 (-10 ms)
    spike_times = np.array([1.012, 2.990, 3.003, 4.007])

    # 1. Presentation order
    psth_pres = compute_psth_and_raster(
        spike_times, pulse_onsets, window_ms=(-50.0, 50.0), order_by="presentation"
    )
    assert np.array_equal(psth_pres["trial_display_order"], [0, 1, 2, 3])
    # Pulse 0 spike (12ms) is at trial 0
    # Pulse 2 spikes (-10ms, 3ms) are at trial 2
    # Pulse 3 spike (7ms) is at trial 3
    idx_p0 = np.where(np.isclose(psth_pres["raster_x"], 12.0, atol=1e-3))[0][0]
    assert psth_pres["raster_y"][idx_p0] == 0

    idx_p2_ev = np.where(np.isclose(psth_pres["raster_x"], 3.0, atol=1e-3))[0][0]
    assert psth_pres["raster_y"][idx_p2_ev] == 2

    idx_p3 = np.where(np.isclose(psth_pres["raster_x"], 7.0, atol=1e-3))[0][0]
    assert psth_pres["raster_y"][idx_p3] == 3

    # 2. Latency order:
    # Sorted evoked: Trial 2 (3ms), Trial 3 (7ms), Trial 0 (12ms)
    # Unevoked: Trial 1
    # Expected display row assignment:
    # Row 0 -> Trial 2
    # Row 1 -> Trial 3
    # Row 2 -> Trial 0
    # Row 3 -> Trial 1
    psth_lat = compute_psth_and_raster(
        spike_times, pulse_onsets, window_ms=(-50.0, 50.0), order_by="latency"
    )
    assert np.array_equal(psth_lat["trial_display_order"], [2, 3, 0, 1])

    # Inverted row verification
    idx_lat_p2 = np.where(np.isclose(psth_lat["raster_x"], 3.0, atol=1e-3))[0][0]
    assert psth_lat["raster_y"][idx_lat_p2] == 0  # Trial 2 is at row 0

    idx_lat_p3 = np.where(np.isclose(psth_lat["raster_x"], 7.0, atol=1e-3))[0][0]
    assert psth_lat["raster_y"][idx_lat_p3] == 1  # Trial 3 is at row 1

    idx_lat_p0 = np.where(np.isclose(psth_lat["raster_x"], 12.0, atol=1e-3))[0][0]
    assert psth_lat["raster_y"][idx_lat_p0] == 2  # Trial 0 is at row 2

    # Baseline spike of Trial 2 (-10ms) also moved to row 0
    idx_lat_p2_base = np.where(np.isclose(psth_lat["raster_x"], -10.0, atol=1e-3))[0][0]
    assert psth_lat["raster_y"][idx_lat_p2_base] == 0


def test_psth_and_raster_custom_window():
    """Verify custom analysis window e.g. [-10ms, +30ms]."""
    pulse_onsets = np.array([1.0, 2.0])
    # Spikes:
    # -20ms (0.980 - outside [-10, 30])
    # -5ms (0.995 - inside baseline)
    # +15ms (1.015 - inside evoked)
    # +40ms (1.040 - outside [-10, 30])
    spike_times = np.array([0.980, 0.995, 1.015, 1.040])

    psth = compute_psth_and_raster(
        spike_times,
        pulse_onsets,
        window_ms=(-10.0, 30.0),
        bin_size_ms=5.0,
    )

    # Bin edges: [-10, -5, 0, 5, 10, 15, 20, 25, 30]
    assert psth["psth_bins"][0] == -10.0
    assert psth["psth_bins"][-1] == 30.0

    # Only spikes within [-10, 30] should be in raster
    assert len(psth["raster_x"]) == 2
    np.testing.assert_allclose(psth["raster_x"], [-5.0, 15.0], atol=1e-3)

    # Baseline is [-10, 0] ms: contains spike at -5ms
    assert psth["baseline_mean"] > 0.0


def test_decoupled_salt_window_and_per_unit_seeding():
    """Verify SALT runs when salt_window has spikes despite narrow latency_window, and seeding depends only on unit_id (Bug M4)."""
    from optocurator.models import StatsParameters

    pulse_onsets = np.arange(1.0, 31.0, 1.0)
    proto = StimulusProtocol(onsets=pulse_onsets, offsets=pulse_onsets + 0.02, name="test_proto")

    # Unit fires at +8 ms (outside 0..5 ms latency window, inside 0..20 ms SALT window)
    rng = np.random.default_rng(7)
    bg_spikes = np.concatenate([pulse_onsets - 0.030, pulse_onsets - 0.012])
    u_8ms = UnitData(
        session_id="sess",
        unit_id=17,
        spike_times=np.sort(np.concatenate([bg_spikes, pulse_onsets + 0.008])),
        sampling_rate=30000.0,
    )
    # Preceding noisy active unit with different unit_id=3
    u_noise = UnitData(
        session_id="sess",
        unit_id=3,
        spike_times=np.sort(np.concatenate([bg_spikes, pulse_onsets + rng.uniform(0.001, 0.019, len(pulse_onsets))])),
        sampling_rate=30000.0,
    )

    params = StatsParameters(
        latency_window_start_ms=0.0,
        latency_window_end_ms=5.0,
        reliability_window_start_ms=0.0,
        reliability_window_end_ms=5.0,
        salt_window_start_ms=0.0,
        salt_window_end_ms=20.0,
    )

    m_single = compute_unit_metrics(u_8ms, proto, stats_params=params)
    assert m_single.salt_p_value < 0.05

    m_sess_only = compute_session_metrics([u_8ms], proto, stats_params=params, n_workers=1)
    assert m_sess_only[0].salt_p_value == m_single.salt_p_value

    # Preceding active unit must not shift per-unit SALT RNG seed or p-value
    m_pair_a = compute_session_metrics([u_noise], proto, stats_params=StatsParameters(), n_workers=1)[0]
    m_pair_b = compute_session_metrics([u_8ms], proto, stats_params=StatsParameters(salt_window_end_ms=20.0), n_workers=1)[0]
    u_both_late = UnitData(
        session_id="sess",
        unit_id=17,
        spike_times=np.sort(np.concatenate([bg_spikes, pulse_onsets + 0.004 + rng.normal(0, 0.004, len(pulse_onsets))])),
        sampling_rate=30000.0,
    )
    m_alone = compute_unit_metrics(u_both_late, proto)
    m_after = compute_session_metrics([u_noise, u_both_late], proto, n_workers=1)
    assert m_after[1].salt_p_value == m_alone.salt_p_value
    assert m_pair_a.salt_p_value >= 0.0 and m_pair_b.salt_p_value >= 0.0


