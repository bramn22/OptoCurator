"""Unit tests for Stimulus-Associated Spike Latency Test (SALT)."""

import numpy as np
import pytest

from optocurator.analysis.salt import compute_salt_p_value, jensen_shannon_divergence


def test_jensen_shannon_divergence():
    """Verify properties of Jensen-Shannon divergence."""
    # Identical distributions have zero divergence
    p = np.array([0.2, 0.5, 0.3])
    assert pytest.approx(jensen_shannon_divergence(p, p), abs=1e-6) == 0.0

    # Completely disjoint distributions have max divergence (1.0 bit)
    p1 = np.array([1.0, 0.0])
    p2 = np.array([0.0, 1.0])
    assert pytest.approx(jensen_shannon_divergence(p1, p2), abs=1e-6) == 1.0


def test_salt_p_value_direct_opto_evoked():
    """Verify that sharp, reliable light-evoked spikes produce a statistically significant SALT p-value."""
    # 50 stimulus pulses spaced by 1 second
    n_pulses = 50
    pulse_onsets = np.arange(10.0, 10.0 + n_pulses, 1.0)

    # Evoked spikes with tight latency (3.5 ms ± 0.3 ms) on almost every trial
    rng = np.random.default_rng(123)
    evoked_latencies = rng.normal(0.0035, 0.0003, size=n_pulses)
    evoked_spikes = pulse_onsets + evoked_latencies

    # Low spontaneous background firing rate (5 Hz)
    spont_spikes = rng.uniform(0.0, 70.0, size=200)

    all_spikes = np.sort(np.concatenate([evoked_spikes, spont_spikes]))

    p_val = compute_salt_p_value(
        all_spikes,
        pulse_onsets,
        eval_window_ms=(0.0, 10.0),
        baseline_window_ms=(-50.0, 0.0),
        n_permutations=200,
        random_state=42,
    )

    # Clear stimulus locking must produce p < 0.01
    assert p_val < 0.01


def test_salt_p_value_spontaneous_noise():
    """Verify that pure spontaneous Poisson/uniform firing produces a non-significant SALT p-value."""
    n_pulses = 40
    pulse_onsets = np.arange(10.0, 10.0 + n_pulses, 1.0)

    # Pure random spikes completely unrelated to stimulus pulses
    rng = np.random.default_rng(999)
    spikes = np.sort(rng.uniform(5.0, 55.0, size=400))

    p_val = compute_salt_p_value(
        spikes,
        pulse_onsets,
        eval_window_ms=(0.0, 10.0),
        baseline_window_ms=(-50.0, 0.0),
        n_permutations=200,
        random_state=42,
    )

    # Non-locked firing should have p > 0.05
    assert p_val > 0.05


def test_salt_p_value_empty():
    """Verify that an inactive unit with 0 spikes yields p = 1.0."""
    pulse_onsets = np.array([1.0, 2.0, 3.0])
    spike_times = np.empty(0)

    p_val = compute_salt_p_value(spike_times, pulse_onsets)
    assert p_val == 1.0


def test_compute_salt_p_values_parallel():
    """Verify multiprocessing parallel SALT computation across multiple spike trains."""
    from optocurator.analysis.salt import compute_salt_p_values_parallel

    n_pulses = 30
    pulse_onsets = np.arange(5.0, 5.0 + n_pulses, 1.0)

    # Unit 1: locked opto spikes
    spk1 = np.sort(np.concatenate([pulse_onsets + 0.003, np.arange(0.0, 35.0, 0.5)]))
    # Unit 2: spontaneous Poisson noise
    rng = np.random.default_rng(42)
    spk2 = np.sort(rng.uniform(0.0, 35.0, 150))
    # Unit 3: empty
    spk3 = np.empty(0)

    p_vals = compute_salt_p_values_parallel([spk1, spk2, spk3], pulse_onsets, n_permutations=100)

    assert len(p_vals) == 3
    assert p_vals[0] < 0.05
    assert p_vals[1] > 0.05
    assert p_vals[2] == 1.0


def test_salt_exclusion_window_artifact_rejection():
    """Verify that exclusion window rejects optical onset/offset artifacts on spontaneous units."""
    n_pulses = 40
    pulse_onsets = np.arange(10.0, 10.0 + n_pulses, 1.0)
    pulse_dur = 0.010  # 10 ms pulse
    pulse_offsets = pulse_onsets + pulse_dur

    rng = np.random.default_rng(777)
    # Background spontaneous firing
    spont_spikes = rng.uniform(5.0, 55.0, size=300)

    # Photoelectric / optical artifact spikes at onset (+0.3 ms) and offset (+0.2 ms after offset = 10.2 ms)
    onset_artifacts = pulse_onsets + 0.0003
    offset_artifacts = pulse_offsets + 0.0002

    corrupted_spikes = np.sort(np.concatenate([spont_spikes, onset_artifacts, offset_artifacts]))

    # Without exclusion window: artifacts are falsely interpreted as stimulus-locked spikes (p < 0.01)
    p_no_excl = compute_salt_p_value(
        corrupted_spikes,
        pulse_onsets,
        pulse_offsets=pulse_offsets,
        eval_window_ms=(0.0, 10.0),
        baseline_window_ms=(-50.0, 0.0),
        exclusion_window_ms=None,
        n_permutations=100,
        random_state=42,
    )
    assert p_no_excl < 0.01

    # With exclusion window (-1.0 to +1.0 ms around onsets and offsets):
    # Both onset and offset artifacts are filtered out, revealing true spontaneous nature (p > 0.05)
    p_with_excl = compute_salt_p_value(
        corrupted_spikes,
        pulse_onsets,
        pulse_offsets=pulse_offsets,
        eval_window_ms=(0.0, 10.0),
        baseline_window_ms=(-50.0, 0.0),
        exclusion_window_ms=(-1.0, 1.0),
        n_permutations=100,
        random_state=42,
    )
    assert p_with_excl > 0.05


def test_salt_exclusion_window_true_opto_preserved():
    """Verify that true light-evoked spikes outside exclusion window are reliably detected."""
    n_pulses = 40
    pulse_onsets = np.arange(10.0, 10.0 + n_pulses, 1.0)
    pulse_dur = 0.010  # 10 ms pulse
    pulse_offsets = pulse_onsets + pulse_dur

    rng = np.random.default_rng(888)
    # True evoked spikes at 4.0 ms (well outside [-1, 1] ms around onset and offset)
    evoked_latencies = rng.normal(0.004, 0.0004, size=n_pulses)
    evoked_spikes = pulse_onsets + evoked_latencies

    # Artifact spikes at onset (+0.2 ms)
    onset_artifacts = pulse_onsets + 0.0002
    spont_spikes = rng.uniform(5.0, 55.0, size=150)

    spikes = np.sort(np.concatenate([evoked_spikes, onset_artifacts, spont_spikes]))

    p_val = compute_salt_p_value(
        spikes,
        pulse_onsets,
        pulse_offsets=pulse_offsets,
        eval_window_ms=(0.0, 10.0),
        baseline_window_ms=(-50.0, 0.0),
        exclusion_window_ms=(-1.0, 1.0),
        n_permutations=100,
        random_state=42,
    )

    # True evoked response at 4.0 ms must remain highly statistically significant (p < 0.01)
    assert p_val < 0.01


def test_salt_baseline_subwindow_no_overflow_and_zero_eval_dur():
    """Verify baseline sub-windows never spill into post-stimulus window and degenerate windows return 1.0 (Bug M2)."""
    pulse_onsets = np.arange(1.0, 31.0, 1.0)
    evoked_spikes = pulse_onsets + 0.003  # deterministic +3.0 ms evoked spike, 0 spontaneous spikes

    # 55 ms / 20 ms = 2.75 -> round() used 3 sub-windows, pushing m=2 to [-15, +5] ms
    p_non_int = compute_salt_p_value(
        evoked_spikes,
        pulse_onsets,
        eval_window_ms=(0.0, 20.0),
        baseline_window_ms=(-55.0, 0.0),
        n_permutations=100,
        random_state=42,
    )
    assert p_non_int < 0.05

    # Baseline shorter than eval_dur must still clamp sub-window end to baseline_window_ms[1] (0.0 ms)
    p_short_base = compute_salt_p_value(
        evoked_spikes,
        pulse_onsets,
        eval_window_ms=(0.0, 20.0),
        baseline_window_ms=(-15.0, 0.0),
        n_permutations=100,
        random_state=42,
    )
    assert p_short_base < 0.05

    # With 0 pre-stimulus spikes, (-55, 0) / (0, 20) must use floor(2.75)=2 strictly pre-stimulus
    # windows (same M=2 empty baseline as (-40, 0)), never spilling into [0, +5) ms to capture +3 ms spikes
    sparse_evoked = pulse_onsets[:2] + 0.003
    p_55 = compute_salt_p_value(
        sparse_evoked,
        pulse_onsets,
        eval_window_ms=(0.0, 20.0),
        baseline_window_ms=(-55.0, 0.0),
        n_permutations=100,
        random_state=42,
    )
    p_40 = compute_salt_p_value(
        sparse_evoked,
        pulse_onsets,
        eval_window_ms=(0.0, 20.0),
        baseline_window_ms=(-40.0, 0.0),
        n_permutations=100,
        random_state=42,
    )
    assert p_55 == pytest.approx(p_40)

    # Degenerate zero-width evaluation or baseline windows must return 1.0 instead of raising ZeroDivisionError
    assert compute_salt_p_value(evoked_spikes, pulse_onsets, eval_window_ms=(0.0, 0.0)) == 1.0
    assert compute_salt_p_value(evoked_spikes, pulse_onsets, baseline_window_ms=(0.0, 0.0)) == 1.0
