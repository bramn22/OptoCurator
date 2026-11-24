"""Stimulus-Associated Spike Latency Test (SALT).

Implements the non-parametric latency distribution test from Kvitsiani et al. (Nature 2013)
comparing first-spike post-stimulus latencies against baseline spontaneous intervals using
the Jensen-Shannon divergence (Information Radius).
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from typing import Any, Callable, List, Optional, Sequence, Tuple
import numpy as np


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence (Information Radius) between two discrete probability distributions.

    Args:
        p: 1D probability distribution array (sums to 1.0).
        q: 1D probability distribution array (sums to 1.0).

    Returns:
        JS divergence in bits (bounded in [0, 1]).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    sum_p = np.sum(p)
    sum_q = np.sum(q)
    if sum_p > 0:
        p = p / sum_p
    if sum_q > 0:
        q = q / sum_q

    m = 0.5 * (p + q)

    mask_p = (p > 0) & (m > 0)
    mask_q = (q > 0) & (m > 0)
    kl_p = np.sum(p[mask_p] * np.log2(p[mask_p] / m[mask_p])) if np.any(mask_p) else 0.0
    kl_q = np.sum(q[mask_q] * np.log2(q[mask_q] / m[mask_q])) if np.any(mask_q) else 0.0

    return float(0.5 * kl_p + 0.5 * kl_q)


def compute_salt_p_value(
    spike_times: np.ndarray,
    pulse_onsets: np.ndarray,
    pulse_offsets: Optional[np.ndarray] = None,
    eval_window_ms: Tuple[float, float] = (0.0, 10.0),
    baseline_window_ms: Tuple[float, float] = (-50.0, 0.0),
    exclusion_window_ms: Optional[Tuple[float, float]] = None,
    n_bins: int = 10,
    n_permutations: int = 100,
    random_state: Optional[int] = 42,
) -> float:
    """Compute the non-parametric SALT p-value for a spike train across stimulus pulses.

    Args:
        spike_times: 1D array of sorted spike timestamps in seconds.
        pulse_onsets: 1D array of stimulus pulse start times in seconds.
        pulse_offsets: Optional 1D array of stimulus pulse end times in seconds.
        eval_window_ms: (start_ms, end_ms) post-stimulus evaluation window (default 0 to 10ms).
        baseline_window_ms: (start_ms, end_ms) pre-stimulus baseline interval (default -50 to 0ms).
        exclusion_window_ms: Optional (start_ms, end_ms) exclusion window around onsets and offsets
                             (e.g. (-1.0, 1.0) ms) to ignore optogenetic stimulation artifacts.
        n_bins: Number of latency bins within the evaluation window.
        n_permutations: Number of permutation iterations to build the null distribution.
        random_state: Random seed for reproducible permutation testing.

    Returns:
        p-value in [0.0, 1.0]. Lower values indicate significant stimulus-locked latency distribution.
    """
    n_trials = len(pulse_onsets)
    if n_trials == 0 or len(spike_times) == 0:
        return 1.0

    eval_dur_s = (eval_window_ms[1] - eval_window_ms[0]) / 1000.0
    eval_start_s = eval_window_ms[0] / 1000.0

    base_total_dur_s = (baseline_window_ms[1] - baseline_window_ms[0]) / 1000.0
    if eval_dur_s <= 0.0 or base_total_dur_s <= 0.0:
        return 1.0
    m_baseline_intervals = max(1, int(np.floor((base_total_dur_s + 1e-9) / eval_dur_s)))

    # Step 1: Extract post-stimulus first-spike latencies
    stim_starts = pulse_onsets + eval_start_s
    stim_ends = stim_starts + eval_dur_s

    stim_idx_s = np.searchsorted(spike_times, stim_starts)
    stim_idx_e = np.searchsorted(spike_times, stim_ends)

    if exclusion_window_ms is None:
        has_stim_spike = stim_idx_e > stim_idx_s
        if not np.any(has_stim_spike):
            return 1.0

        # Values in [0, eval_dur_s], sentinel for no spike = eval_dur_s + 1e-4
        stim_latencies = np.full(n_trials, eval_dur_s + 1e-4, dtype=np.float64)
        v_idx = np.where(has_stim_spike)[0]
        stim_latencies[v_idx] = spike_times[stim_idx_s[v_idx]] - stim_starts[v_idx]

        # Step 2: Extract baseline first-spike latencies across M baseline intervals (vectorized)
        base_window_start_s = baseline_window_ms[0] / 1000.0
        base_window_end_s = baseline_window_ms[1] / 1000.0
        baseline_latencies = np.full((m_baseline_intervals, n_trials), eval_dur_s + 1e-4, dtype=np.float64)

        for m in range(m_baseline_intervals):
            b_sub_start = pulse_onsets + base_window_start_s + (m * eval_dur_s)
            b_sub_end = np.maximum(
                b_sub_start,
                np.minimum(b_sub_start + eval_dur_s, pulse_onsets + base_window_end_s),
            )

            b_idx_s = np.searchsorted(spike_times, b_sub_start)
            b_idx_e = np.searchsorted(spike_times, b_sub_end)

            b_has = b_idx_e > b_idx_s
            b_v = np.where(b_has)[0]
            if len(b_v) > 0:
                baseline_latencies[m, b_v] = spike_times[b_idx_s[b_v]] - b_sub_start[b_v]
    else:
        excl_start_s = float(exclusion_window_ms[0]) / 1000.0
        excl_end_s = float(exclusion_window_ms[1]) / 1000.0

        stim_latencies = np.full(n_trials, eval_dur_s + 1e-4, dtype=np.float64)
        base_window_start_s = baseline_window_ms[0] / 1000.0
        base_window_end_s = baseline_window_ms[1] / 1000.0
        baseline_latencies = np.full((m_baseline_intervals, n_trials), eval_dur_s + 1e-4, dtype=np.float64)

        has_offsets = pulse_offsets is not None and len(pulse_offsets) == n_trials

        # Precompute baseline intervals and spike index bounds
        b_starts_all = []
        b_idx_s_all = []
        b_idx_e_all = []
        for m in range(m_baseline_intervals):
            b_sub_start = pulse_onsets + base_window_start_s + (m * eval_dur_s)
            b_sub_end = np.maximum(
                b_sub_start,
                np.minimum(b_sub_start + eval_dur_s, pulse_onsets + base_window_end_s),
            )
            b_starts_all.append(b_sub_start)
            b_idx_s_all.append(np.searchsorted(spike_times, b_sub_start))
            b_idx_e_all.append(np.searchsorted(spike_times, b_sub_end))

        for i in range(n_trials):
            on_i = pulse_onsets[i]
            off_i = pulse_offsets[i] if has_offsets else None
            dur_i = (off_i - on_i) if has_offsets else None

            # 1. Search post-stimulus test window for first valid non-artifact spike
            s_idx = stim_idx_s[i]
            e_idx = stim_idx_e[i]
            for idx in range(s_idx, e_idx):
                t_spk = spike_times[idx]
                # Check absolute exclusion around pulse onset
                if (on_i + excl_start_s) <= t_spk <= (on_i + excl_end_s):
                    continue
                # Check absolute exclusion around pulse offset
                if has_offsets and (off_i + excl_start_s) <= t_spk <= (off_i + excl_end_s):
                    continue

                tau = t_spk - stim_starts[i]
                # Check relative exclusion around onset in test window
                if excl_start_s <= (tau + eval_start_s) <= excl_end_s:
                    continue
                # Check relative exclusion around offset in test window
                if has_offsets and excl_start_s <= (tau + eval_start_s - dur_i) <= excl_end_s:
                    continue

                stim_latencies[i] = tau
                break

            # 2. Search baseline intervals for first valid spike under identical temporal censoring
            for m in range(m_baseline_intervals):
                b_s = b_idx_s_all[m][i]
                b_e = b_idx_e_all[m][i]
                b_start = b_starts_all[m][i]

                for idx in range(b_s, b_e):
                    t_spk = spike_times[idx]
                    # Absolute exclusion around pulse onset
                    if (on_i + excl_start_s) <= t_spk <= (on_i + excl_end_s):
                        continue
                    # Absolute exclusion around pulse offset
                    if has_offsets and (off_i + excl_start_s) <= t_spk <= (off_i + excl_end_s):
                        continue

                    tau = t_spk - b_start
                    # Relative exclusion matching test window onset
                    if excl_start_s <= (tau + eval_start_s) <= excl_end_s:
                        continue
                    # Relative exclusion matching test window offset
                    if has_offsets and excl_start_s <= (tau + eval_start_s - dur_i) <= excl_end_s:
                        continue

                    baseline_latencies[m, i] = tau
                    break

        if not np.any(stim_latencies <= eval_dur_s):
            return 1.0

    # Discretize continuous latencies into integer bin indices in [0, n_bins]
    # Bin index n_bins represents "no spike"
    bin_edges = np.linspace(0.0, eval_dur_s, n_bins + 1)
    stim_bins = np.clip(np.digitize(stim_latencies, bin_edges) - 1, 0, n_bins)
    base_bins = np.clip(np.digitize(baseline_latencies, bin_edges) - 1, 0, n_bins)

    num_classes = n_bins + 1
    p_stim = np.bincount(stim_bins, minlength=num_classes) / float(n_trials)
    p_base_list = [np.bincount(base_bins[m], minlength=num_classes) / float(n_trials) for m in range(m_baseline_intervals)]
    p_base_avg = np.mean(p_base_list, axis=0)

    js_test = jensen_shannon_divergence(p_stim, p_base_avg)
    if js_test <= 1e-6:
        return 1.0

    # Step 3: Fast permutation test using bincount
    pooled_bins = np.concatenate([stim_bins] + [base_bins[m] for m in range(m_baseline_intervals)])
    base_total_trials = m_baseline_intervals * n_trials

    rng = np.random.default_rng(random_state)
    greater_count = 0

    for _ in range(n_permutations):
        permuted = rng.permutation(pooled_bins)
        p_p_stim = np.bincount(permuted[:n_trials], minlength=num_classes) / float(n_trials)
        p_p_base = np.bincount(permuted[n_trials:], minlength=num_classes) / float(base_total_trials)

        js_null = jensen_shannon_divergence(p_p_stim, p_p_base)
        if js_null >= js_test:
            greater_count += 1

    p_value = (greater_count + 1) / (n_permutations + 1)
    return float(p_value)


def _salt_single_unit_worker(args: Tuple[Any, ...]) -> float:
    """Top-level module worker function for parallel SALT permutation testing.

    Args:
        args: Tuple containing unit parameters.

    Returns:
        Computed SALT p-value.
    """
    (
        spike_times,
        pulse_onsets,
        pulse_offsets,
        eval_window_ms,
        baseline_window_ms,
        exclusion_window_ms,
        n_bins,
        n_permutations,
        random_state,
    ) = args

    return compute_salt_p_value(
        spike_times=spike_times,
        pulse_onsets=pulse_onsets,
        pulse_offsets=pulse_offsets,
        eval_window_ms=eval_window_ms,
        baseline_window_ms=baseline_window_ms,
        exclusion_window_ms=exclusion_window_ms,
        n_bins=n_bins,
        n_permutations=n_permutations,
        random_state=random_state,
    )


def compute_salt_p_values_parallel(
    spike_times_list: List[np.ndarray],
    pulse_onsets: np.ndarray,
    pulse_offsets: Optional[np.ndarray] = None,
    eval_window_ms: Tuple[float, float] = (0.0, 10.0),
    baseline_window_ms: Tuple[float, float] = (-50.0, 0.0),
    exclusion_window_ms: Optional[Tuple[float, float]] = None,
    n_bins: int = 10,
    n_permutations: int = 100,
    n_workers: Optional[int] = None,
    random_state: Optional[int] = 42,
    random_states: Optional[Sequence[Optional[int]]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[float]:
    """Compute SALT p-values for multiple spike trains in parallel using multiprocessing.

    Args:
        spike_times_list: List of 1D arrays of spike timestamps.
        pulse_onsets: 1D array of stimulus pulse timestamps.
        pulse_offsets: Optional 1D array of stimulus pulse offset timestamps.
        eval_window_ms: Post-stimulus latency evaluation window in ms.
        baseline_window_ms: Pre-stimulus baseline interval in ms.
        exclusion_window_ms: Optional (start_ms, end_ms) exclusion window around onsets and offsets.
        n_bins: Number of latency bins.
        n_permutations: Number of permutations for null distribution.
        n_workers: Number of worker processes (defaults to available CPU cores).
        random_state: Base random seed for reproducible permutations (used if random_states is None).
        random_states: Optional per-unit random seeds matching spike_times_list.
        progress_callback: Optional callback receiving (completed_count, total_count).

    Returns:
        List of float p-values matching order of spike_times_list.
    """
    n_units = len(spike_times_list)
    if n_units == 0:
        return []
    if random_states is not None and len(random_states) != n_units:
        raise ValueError(f"random_states length ({len(random_states)}) must match spike_times_list ({n_units})")

    # Prepare argument tuples
    tasks = [
        (
            spike_times,
            pulse_onsets,
            pulse_offsets,
            eval_window_ms,
            baseline_window_ms,
            exclusion_window_ms,
            n_bins,
            n_permutations,
            (random_states[idx] if random_states is not None else (random_state + idx if random_state is not None else None)),
        )
        for idx, spike_times in enumerate(spike_times_list)
    ]

    # For tiny batches or single worker, run sequentially without process pool overhead
    if n_units <= 2 or n_workers == 1:
        results = []
        for i, t in enumerate(tasks):
            results.append(_salt_single_unit_worker(t))
            if progress_callback is not None:
                progress_callback(i + 1, n_units)
        return results

    cpu_cores = os.cpu_count() or 4
    workers = min(n_workers if n_workers is not None else cpu_cores, n_units)
    chunk = max(1, n_units // (workers * 4))

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            if progress_callback is not None:
                futures = {executor.submit(_salt_single_unit_worker, t): i for i, t in enumerate(tasks)}
                results = [1.0] * n_units
                completed = 0
                for f in as_completed(futures):
                    orig_idx = futures[f]
                    results[orig_idx] = f.result()
                    completed += 1
                    progress_callback(completed, n_units)
                return results
            else:
                return list(executor.map(_salt_single_unit_worker, tasks, chunksize=chunk))
    except Exception:
        # Graceful fallback to sequential processing if process pool fails
        results = []
        for i, t in enumerate(tasks):
            results.append(_salt_single_unit_worker(t))
            if progress_callback is not None:
                progress_callback(i + 1, n_units)
        return results
