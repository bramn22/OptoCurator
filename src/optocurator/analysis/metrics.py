"""Analytical metrics for optotagging evaluation (PSTH, latency, jitter, reliability, waveform r, SALT)."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from optocurator.analysis.salt import (
    compute_salt_p_value,
    compute_salt_p_values_parallel,
)
from optocurator.analysis.waveform import compute_unit_waveform_correlation
from optocurator.models import StatsParameters, StimulusProtocol, UnitData, UnitMetrics


def compute_response_reliability(
    spike_times: np.ndarray,
    pulse_onsets: np.ndarray,
    eval_window_ms: Tuple[float, float] = (0.0, 20.0),
    max_trials: Optional[int] = None,
) -> float:
    """Compute evoked response probability [0.0, 1.0] across stimulus pulses.

    Args:
        spike_times: 1D array of sorted spike timestamps in seconds.
        pulse_onsets: 1D array of pulse start times in seconds.
        eval_window_ms: (start_ms, end_ms) post-stimulus interval in ms.
        max_trials: Optional maximum number of pulses to evaluate.

    Returns:
        Fraction of evaluated pulses eliciting at least one spike in the window.
    """
    if max_trials is not None and max_trials > 0 and max_trials < len(pulse_onsets):
        pulse_onsets = pulse_onsets[:max_trials]

    n_trials = len(pulse_onsets)
    if n_trials == 0 or len(spike_times) == 0:
        return 0.0

    w_start_s = eval_window_ms[0] / 1000.0
    w_end_s = eval_window_ms[1] / 1000.0
    starts = pulse_onsets + w_start_s
    ends = pulse_onsets + w_end_s

    idx_starts = np.searchsorted(spike_times, starts)
    idx_ends = np.searchsorted(spike_times, ends)
    has_spike = idx_ends > idx_starts
    return float(np.mean(has_spike))


def compute_first_spike_latency(
    spike_times: np.ndarray,
    pulse_onsets: np.ndarray,
    eval_window_ms: Tuple[float, float] = (0.0, 20.0),
    max_trials: Optional[int] = None,
) -> float:
    """Compute median first-spike latency in milliseconds across evoked pulses.

    Args:
        spike_times: 1D array of sorted spike timestamps in seconds.
        pulse_onsets: 1D array of pulse start times in seconds.
        eval_window_ms: (start_ms, end_ms) post-stimulus interval in ms.
        max_trials: Optional maximum number of pulses to evaluate.

    Returns:
        Median first-spike latency in ms, or np.nan if no evoked spikes.
    """
    if max_trials is not None and max_trials > 0 and max_trials < len(pulse_onsets):
        pulse_onsets = pulse_onsets[:max_trials]

    n_trials = len(pulse_onsets)
    if n_trials == 0 or len(spike_times) == 0:
        return np.nan

    w_start_s = eval_window_ms[0] / 1000.0
    w_end_s = eval_window_ms[1] / 1000.0
    starts = pulse_onsets + w_start_s
    ends = pulse_onsets + w_end_s

    idx_starts = np.searchsorted(spike_times, starts)
    idx_ends = np.searchsorted(spike_times, ends)

    first_latencies: List[float] = []
    for s_idx, e_idx, t_pulse in zip(idx_starts, idx_ends, pulse_onsets):
        if e_idx > s_idx:
            first_spk = spike_times[s_idx]
            lat_ms = (first_spk - t_pulse) * 1000.0
            first_latencies.append(lat_ms)

    if len(first_latencies) > 0:
        return float(np.median(first_latencies))
    return np.nan


def compute_first_spike_jitter(
    spike_times: np.ndarray,
    pulse_onsets: np.ndarray,
    eval_window_ms: Tuple[float, float] = (0.0, 20.0),
    max_trials: Optional[int] = None,
) -> float:
    """Compute first-spike jitter (standard deviation of first latencies in ms).

    Args:
        spike_times: 1D array of sorted spike timestamps in seconds.
        pulse_onsets: 1D array of pulse start times in seconds.
        eval_window_ms: (start_ms, end_ms) post-stimulus interval in ms.
        max_trials: Optional maximum number of pulses to evaluate.

    Returns:
        Standard deviation in ms, or np.nan if no evoked spikes, or 0.0 if only 1 spike.
    """
    if max_trials is not None and max_trials > 0 and max_trials < len(pulse_onsets):
        pulse_onsets = pulse_onsets[:max_trials]

    n_trials = len(pulse_onsets)
    if n_trials == 0 or len(spike_times) == 0:
        return np.nan

    w_start_s = eval_window_ms[0] / 1000.0
    w_end_s = eval_window_ms[1] / 1000.0
    starts = pulse_onsets + w_start_s
    ends = pulse_onsets + w_end_s

    idx_starts = np.searchsorted(spike_times, starts)
    idx_ends = np.searchsorted(spike_times, ends)

    first_latencies: List[float] = []
    for s_idx, e_idx, t_pulse in zip(idx_starts, idx_ends, pulse_onsets):
        if e_idx > s_idx:
            first_spk = spike_times[s_idx]
            lat_ms = (first_spk - t_pulse) * 1000.0
            first_latencies.append(lat_ms)

    if len(first_latencies) > 1:
        return float(np.std(first_latencies))
    elif len(first_latencies) == 1:
        return 0.0
    return np.nan


def compute_first_spike_latency_and_jitter(
    spike_times: np.ndarray,
    pulse_onsets: np.ndarray,
    eval_window_ms: Tuple[float, float] = (0.0, 20.0),
    max_trials: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Compute first-spike latency, jitter, and response reliability across stimulus pulses.

    Args:
        spike_times: 1D array of sorted spike timestamps in seconds.
        pulse_onsets: 1D array of pulse start times in seconds.
        eval_window_ms: (start_ms, end_ms) post-stimulus window to evaluate (default 0 to 20ms).
        max_trials: Optional maximum number of pulses/trials on which to evaluate reliability and latency.

    Returns:
        tuple (latency_ms, jitter_ms, reliability):
            - latency_ms: Median first-spike latency in milliseconds (np.nan if no trials evoked spikes).
            - jitter_ms: Standard deviation of first-spike latency across evoked trials in ms.
            - reliability: Fraction of stimulus pulses eliciting at least one spike in the window [0, 1].
    """
    if max_trials is not None and max_trials > 0 and max_trials < len(pulse_onsets):
        pulse_onsets = pulse_onsets[:max_trials]

    n_trials = len(pulse_onsets)
    if n_trials == 0 or len(spike_times) == 0:
        return np.nan, np.nan, 0.0

    w_start_s = eval_window_ms[0] / 1000.0
    w_end_s = eval_window_ms[1] / 1000.0

    starts = pulse_onsets + w_start_s
    ends = pulse_onsets + w_end_s

    # Binary search for fast spike lookup
    idx_starts = np.searchsorted(spike_times, starts)
    idx_ends = np.searchsorted(spike_times, ends)

    has_spike = idx_ends > idx_starts
    reliability = float(np.mean(has_spike))

    first_latencies: List[float] = []
    for s_idx, e_idx, t_pulse, had in zip(idx_starts, idx_ends, pulse_onsets, has_spike):
        if had:
            first_spk = spike_times[s_idx]
            lat_ms = (first_spk - t_pulse) * 1000.0
            first_latencies.append(lat_ms)

    if len(first_latencies) > 0:
        lat_arr = np.asarray(first_latencies, dtype=np.float64)
        latency_ms = float(np.median(lat_arr))
        jitter_ms = float(np.std(lat_arr)) if len(lat_arr) > 1 else 0.0
    else:
        latency_ms = np.nan
        jitter_ms = np.nan

    return latency_ms, jitter_ms, reliability


def compute_gaussian_psth(
    raster_x: np.ndarray,
    n_trials: int,
    window_ms: Tuple[float, float],
    bin_size_ms: float = 2.0,
    gaussian_sigma_ms: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute smooth firing rate PSTH based on a Gaussian kernel using Elephant.

    Args:
        raster_x: 1D array of trial-relative spike timestamps in milliseconds.
        n_trials: Number of stimulus pulses / trials.
        window_ms: (start_ms, end_ms) window in milliseconds.
        bin_size_ms: Histogram bin / sampling period width in milliseconds.
        gaussian_sigma_ms: Gaussian kernel standard deviation in milliseconds.

    Returns:
        Tuple of (bins, psth_rates):
            - bins: 1D array of bin edges in ms.
            - psth_rates: 1D array of smoothed firing rates in Hz (length = len(bins) - 1).
    """
    bins = np.arange(window_ms[0], window_ms[1] + bin_size_ms * 0.5, bin_size_ms)
    n_bins = max(0, len(bins) - 1)
    if n_bins == 0 or n_trials <= 0 or len(raster_x) == 0:
        return bins, np.zeros(n_bins, dtype=np.float64)

    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    try:
        import quantities as pq
        from neo.core import SpikeTrain
        from elephant.statistics import instantaneous_rate
        from elephant.kernels import GaussianKernel

        # Filter spikes strictly within window [start_ms, end_ms]
        valid_mask = (raster_x >= window_ms[0]) & (raster_x <= window_ms[1])
        spk_times = np.sort(raster_x[valid_mask].astype(np.float64))

        if len(spk_times) == 0:
            return bins, np.zeros(n_bins, dtype=np.float64)

        t_start = float(window_ms[0]) * pq.ms
        t_stop = (float(window_ms[1]) + 1e-4) * pq.ms

        st = SpikeTrain(spk_times * pq.ms, t_start=t_start, t_stop=t_stop)
        kernel = GaussianKernel(sigma=max(0.1, float(gaussian_sigma_ms)) * pq.ms)
        sampling_period = float(bin_size_ms) * pq.ms

        rate_signal = instantaneous_rate(
            st,
            sampling_period=sampling_period,
            kernel=kernel,
            t_start=t_start,
            t_stop=float(window_ms[1]) * pq.ms,
        )

        rates_hz = np.asarray(rate_signal.rescale(pq.Hz).magnitude).flatten() / float(n_trials)
        signal_times_ms = np.asarray(rate_signal.times.rescale(pq.ms).magnitude).flatten()

        if len(rates_hz) == n_bins:
            psth_rates = rates_hz
        else:
            psth_rates = np.interp(bin_centers, signal_times_ms, rates_hz)

        return bins, psth_rates.astype(np.float64)

    except Exception:
        try:
            from scipy.ndimage import gaussian_filter1d
            counts, _ = np.histogram(raster_x, bins=bins)
            bin_width_s = bin_size_ms / 1000.0
            raw_rates = counts.astype(np.float64) / (n_trials * bin_width_s)
            sigma_bins = max(0.1, gaussian_sigma_ms / bin_size_ms)
            smoothed = gaussian_filter1d(raw_rates, sigma=sigma_bins, mode="nearest")
            return bins, smoothed.astype(np.float64)
        except Exception:
            counts, _ = np.histogram(raster_x, bins=bins)
            bin_width_s = bin_size_ms / 1000.0
            raw_rates = counts.astype(np.float64) / (n_trials * bin_width_s)
            return bins, raw_rates


def compute_psth_and_raster(
    spike_times: np.ndarray,
    pulse_onsets: np.ndarray,
    window_ms: Tuple[float, float] = (-50.0, 50.0),
    bin_size_ms: float = 2.0,
    baseline_window_ms: Optional[Tuple[float, float]] = None,
    order_by: str = "presentation",
    smooth: bool = True,
    gaussian_sigma_ms: float = 2.0,
) -> Dict[str, Any]:
    """Compute trial-aligned spike raster and firing-rate PSTH histogram.

    Args:
        spike_times: 1D array of sorted spike timestamps in seconds.
        pulse_onsets: 1D array of pulse start times in seconds.
        window_ms: (start_ms, end_ms) around pulse onset (t=0).
        bin_size_ms: PSTH histogram bin width in milliseconds.
        baseline_window_ms: Baseline interval in ms for mean and 3-sigma threshold calculation.
                            If None, defaults to (window_ms[0], 0.0) if window_ms[0] < 0 else (0.0, 0.0).
        order_by: Raster trial ordering mode. Options:
                  - "presentation": Chronological order of stimulus pulses (0 to n_trials - 1).
                  - "latency": Sorted by first-spike latency post-stimulus onset (t >= 0).
                    Pulses with no evoked spikes are grouped at the top.
        smooth: If True, compute smooth firing rate PSTH using Elephant Gaussian kernel.
        gaussian_sigma_ms: Gaussian kernel standard deviation in milliseconds for smoothing.

    Returns:
        Dict with keys:
            - 'raster_x': 1D array of relative spike times in ms.
            - 'raster_y': 1D array of trial indices (0 to n_trials-1, ordered according to order_by).
            - 'psth_bins': 1D array of histogram bin edges in ms.
            - 'psth_rates': 1D array of firing rates in Hz per bin (smoothed if smooth=True).
            - 'psth_rates_raw': 1D array of raw histogram firing rates in Hz per bin.
            - 'is_smooth': Boolean flag indicating if smoothing was applied.
            - 'gaussian_sigma_ms': Gaussian kernel sigma used for smoothing.
            - 'baseline_mean': Mean baseline firing rate in Hz.
            - 'baseline_std': Standard deviation of baseline firing rate.
            - 'baseline_3sigma': Baseline mean + 3*std threshold in Hz.
            - 'n_trials': Number of stimulus pulses.
            - 'first_latencies': 1D array of first spike latency per trial (or NaN).
            - 'trial_display_order': 1D array mapping display row (0..N-1) to original pulse index.
    """
    n_trials = len(pulse_onsets)
    if baseline_window_ms is None:
        baseline_window_ms = (window_ms[0], 0.0) if window_ms[0] < 0.0 else (0.0, 0.0)

    bins = np.arange(window_ms[0], window_ms[1] + bin_size_ms * 0.5, bin_size_ms)
    bin_width_s = bin_size_ms / 1000.0

    if n_trials == 0 or len(spike_times) == 0:
        empty_rates = np.zeros(max(0, len(bins) - 1), dtype=np.float64)
        return {
            "raster_x": np.empty(0, dtype=np.float64),
            "raster_y": np.empty(0, dtype=np.int32),
            "psth_bins": bins,
            "psth_rates": empty_rates,
            "psth_rates_raw": empty_rates,
            "is_smooth": smooth,
            "gaussian_sigma_ms": gaussian_sigma_ms,
            "baseline_mean": 0.0,
            "baseline_std": 0.0,
            "baseline_3sigma": 0.0,
            "n_trials": n_trials,
            "first_latencies": np.full(n_trials, np.nan, dtype=np.float64),
            "trial_display_order": np.arange(n_trials, dtype=np.int32),
        }

    w_start_s = window_ms[0] / 1000.0
    w_end_s = window_ms[1] / 1000.0

    starts = pulse_onsets + w_start_s
    ends = pulse_onsets + w_end_s

    idx_starts = np.searchsorted(spike_times, starts)
    idx_ends = np.searchsorted(spike_times, ends)

    # Collect per-trial spikes and compute first-spike latency post onset (t >= 0)
    trial_spikes_list: List[np.ndarray] = []
    first_latencies = np.full(n_trials, np.nan, dtype=np.float64)

    for trial_idx, (s_idx, e_idx, t_pulse) in enumerate(zip(idx_starts, idx_ends, pulse_onsets)):
        if e_idx > s_idx:
            t_spk_ms = (spike_times[s_idx:e_idx] - t_pulse) * 1000.0
            trial_spikes_list.append(t_spk_ms)
            post_spikes = t_spk_ms[t_spk_ms >= 0.0]
            if len(post_spikes) > 0:
                first_latencies[trial_idx] = post_spikes[0]
        else:
            trial_spikes_list.append(np.empty(0, dtype=np.float64))

    # Compute trial row mapping based on order_by
    if order_by == "latency":
        has_evoked = ~np.isnan(first_latencies)
        evoked_indices = np.where(has_evoked)[0]
        sorted_evoked = evoked_indices[np.argsort(first_latencies[evoked_indices], kind="stable")]
        unevoked_indices = np.where(~has_evoked)[0]
        trial_display_order = np.concatenate([sorted_evoked, unevoked_indices]).astype(np.int32)

        trial_to_row = np.empty(n_trials, dtype=np.int32)
        trial_to_row[trial_display_order] = np.arange(n_trials, dtype=np.int32)
    else:  # "presentation"
        trial_display_order = np.arange(n_trials, dtype=np.int32)
        trial_to_row = np.arange(n_trials, dtype=np.int32)

    # Build raster arrays
    raster_x_list: List[np.ndarray] = []
    raster_y_list: List[np.ndarray] = []

    for trial_idx, spks in enumerate(trial_spikes_list):
        if len(spks) > 0:
            raster_x_list.append(spks)
            row_val = trial_to_row[trial_idx]
            raster_y_list.append(np.full(len(spks), row_val, dtype=np.int32))

    if raster_x_list:
        raster_x = np.concatenate(raster_x_list)
        raster_y = np.concatenate(raster_y_list)
    else:
        raster_x = np.empty(0, dtype=np.float64)
        raster_y = np.empty(0, dtype=np.int32)

    # PSTH histogram
    counts, _ = np.histogram(raster_x, bins=bins)
    psth_rates_raw = counts.astype(np.float64) / (n_trials * bin_width_s)

    if smooth:
        _, psth_rates = compute_gaussian_psth(
            raster_x=raster_x,
            n_trials=n_trials,
            window_ms=window_ms,
            bin_size_ms=bin_size_ms,
            gaussian_sigma_ms=gaussian_sigma_ms,
        )
    else:
        psth_rates = psth_rates_raw

    # Baseline statistics on active psth_rates over baseline window
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    baseline_mask = (bin_centers >= baseline_window_ms[0]) & (bin_centers < baseline_window_ms[1])

    if np.any(baseline_mask):
        baseline_mean = float(np.mean(psth_rates[baseline_mask]))
        baseline_std = float(np.std(psth_rates[baseline_mask]))
    else:
        baseline_mean = 0.0
        baseline_std = 0.0

    baseline_3sigma = baseline_mean + 3.0 * baseline_std

    return {
        "raster_x": raster_x,
        "raster_y": raster_y,
        "psth_bins": bins,
        "psth_rates": psth_rates,
        "psth_rates_raw": psth_rates_raw,
        "is_smooth": smooth,
        "gaussian_sigma_ms": gaussian_sigma_ms,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "baseline_3sigma": baseline_3sigma,
        "n_trials": n_trials,
        "first_latencies": first_latencies,
        "trial_display_order": trial_display_order,
    }


def compute_unit_metrics(
    unit: UnitData,
    stimulus: StimulusProtocol,
    eval_window_ms: Tuple[float, float] = (0.0, 20.0),
    max_latency_ms: float = 8.0,
    max_jitter_ms: float = 2.0,
    min_reliability: float = 0.60,
    min_waveform_r: float = 0.90,
    max_salt_p: float = 0.05,
    plot_rel_path: str = "",
    stats_params: Optional[StatsParameters] = None,
) -> UnitMetrics:
    """Compute statistical metrics and optotagged status for a single unit."""
    max_trials = None
    salt_eval_win = (0.0, 10.0)
    salt_base_win = (-50.0, 0.0)
    salt_bins = 10
    salt_perms = 100
    wf_eval_win = eval_window_ms
    wf_spont_win = (-50.0, -5.0)

    if stats_params is not None:
        rel = compute_response_reliability(
            unit.spike_times,
            stimulus.onsets,
            eval_window_ms=(stats_params.reliability_window_start_ms, stats_params.reliability_window_end_ms),
            max_trials=stats_params.reliability_trials,
        )
        lat_ms = compute_first_spike_latency(
            unit.spike_times,
            stimulus.onsets,
            eval_window_ms=(stats_params.latency_window_start_ms, stats_params.latency_window_end_ms),
            max_trials=stats_params.latency_trials,
        )
        jit_ms = compute_first_spike_jitter(
            unit.spike_times,
            stimulus.onsets,
            eval_window_ms=(stats_params.jitter_window_start_ms, stats_params.jitter_window_end_ms),
            max_trials=stats_params.jitter_trials,
        )
        salt_eval_win = (stats_params.salt_window_start_ms, stats_params.salt_window_end_ms)
        salt_base_win = (stats_params.salt_baseline_start_ms, stats_params.salt_baseline_end_ms)
        salt_bins = stats_params.salt_n_bins
        salt_perms = stats_params.salt_n_permutations
        salt_excl_win = stats_params.salt_exclusion_window_ms
        wf_eval_win = (stats_params.wf_evoked_start_ms, stats_params.wf_evoked_end_ms)
        wf_spont_win = (stats_params.wf_baseline_start_ms, stats_params.wf_baseline_end_ms)
    else:
        lat_ms, jit_ms, rel = compute_first_spike_latency_and_jitter(
            unit.spike_times, stimulus.onsets, eval_window_ms=eval_window_ms, max_trials=max_trials
        )
        salt_eval_win = (0.0, 10.0)
        salt_base_win = (-50.0, 0.0)
        salt_bins = 10
        salt_perms = 100
        salt_excl_win = None
        wf_eval_win = eval_window_ms
        wf_spont_win = (-50.0, -5.0)

    wf_max_spikes = stats_params.wf_max_spikes if stats_params is not None else 40
    waveform_correlation = compute_unit_waveform_correlation(
        unit.waveforms,
        unit.spike_times,
        stimulus.onsets,
        eval_window_ms=wf_eval_win,
        baseline_window_ms=wf_spont_win,
        max_spikes_per_group=wf_max_spikes,
    )

    if rel > 0.0 and not np.isnan(lat_ms):
        salt_p = compute_salt_p_value(
            unit.spike_times,
            stimulus.onsets,
            pulse_offsets=stimulus.offsets,
            eval_window_ms=salt_eval_win,
            baseline_window_ms=salt_base_win,
            exclusion_window_ms=salt_excl_win,
            n_bins=salt_bins,
            n_permutations=salt_perms,
        )
    else:
        salt_p = 1.0

    is_tagged = bool(
        not np.isnan(lat_ms)
        and (lat_ms <= max_latency_ms)
        and (rel >= min_reliability)
        and (np.isnan(jit_ms) or jit_ms <= max_jitter_ms)
        and ((not np.isfinite(waveform_correlation)) or (waveform_correlation >= min_waveform_r))
        and (salt_p <= max_salt_p)
    )

    if not plot_rel_path:
        plot_rel_path = f"{unit.session_id}_plots/{unit.global_id}_psth_raster.png"

    return UnitMetrics(
        global_id=unit.global_id,
        session_id=unit.session_id,
        unit_id=unit.unit_id,
        plot_path=plot_rel_path,
        latency_ms=lat_ms if not np.isnan(lat_ms) else 999.0,
        jitter_ms=jit_ms if not np.isnan(jit_ms) else 999.0,
        reliability=rel,
        waveform_correlation=waveform_correlation,
        salt_p_value=salt_p,
        is_tagged=is_tagged,
    )


def compute_session_metrics(
    units: List[UnitData],
    stimulus: StimulusProtocol,
    eval_window_ms: Tuple[float, float] = (0.0, 20.0),
    max_latency_ms: float = 8.0,
    max_jitter_ms: float = 2.0,
    min_reliability: float = 0.60,
    min_waveform_r: float = 0.90,
    max_salt_p: float = 0.05,
    stats_params: Optional[StatsParameters] = None,
    n_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[UnitMetrics]:
    """Compute optotagging metrics across all units in a recording session using multiprocessing for SALT.

    Args:
        units: List of UnitData instances.
        stimulus: StimulusProtocol with pulse onsets.
        eval_window_ms: Latency evaluation window in ms.
        max_latency_ms: Cutoff for maximum first-spike latency.
        max_jitter_ms: Cutoff for maximum jitter.
        min_reliability: Cutoff for minimum response reliability.
        min_waveform_r: Cutoff for minimum waveform correlation.
        max_salt_p: Cutoff for maximum SALT p-value.
        stats_params: Optional StatsParameters instance containing detailed analytical hyperparameters.
        n_workers: Number of worker processes (defaults to os.cpu_count()).
        progress_callback: Optional callback receiving (completed_count, total_count).

    Returns:
        List of UnitMetrics for all units.
    """
    if not units:
        return []

    max_trials = None
    salt_eval_win = (0.0, 10.0)
    salt_base_win = (-50.0, 0.0)
    salt_bins = 10
    salt_perms = 100
    wf_eval_win = eval_window_ms
    wf_spont_win = (-50.0, -5.0)

    if stats_params is not None:
        rel_win = (stats_params.reliability_window_start_ms, stats_params.reliability_window_end_ms)
        rel_trials = stats_params.reliability_trials
        lat_win = (stats_params.latency_window_start_ms, stats_params.latency_window_end_ms)
        lat_trials = stats_params.latency_trials
        jit_win = (stats_params.jitter_window_start_ms, stats_params.jitter_window_end_ms)
        jit_trials = stats_params.jitter_trials

        salt_eval_win = (stats_params.salt_window_start_ms, stats_params.salt_window_end_ms)
        salt_base_win = (stats_params.salt_baseline_start_ms, stats_params.salt_baseline_end_ms)
        salt_bins = stats_params.salt_n_bins
        salt_perms = stats_params.salt_n_permutations
        salt_excl_win = stats_params.salt_exclusion_window_ms
        wf_eval_win = (stats_params.wf_evoked_start_ms, stats_params.wf_evoked_end_ms)
        wf_spont_win = (stats_params.wf_baseline_start_ms, stats_params.wf_baseline_end_ms)
    else:
        rel_win = eval_window_ms
        rel_trials = max_trials
        lat_win = eval_window_ms
        lat_trials = max_trials
        jit_win = eval_window_ms
        jit_trials = max_trials
        salt_eval_win = (0.0, 10.0)
        salt_base_win = (-50.0, 0.0)
        salt_bins = 10
        salt_perms = 100
        salt_excl_win = None

    wf_max_spikes = stats_params.wf_max_spikes if stats_params is not None else 40
    # 1. Fast vectorized pass for latency, jitter, reliability, and waveform correlation
    preliminary: List[Tuple[float, float, float, float]] = []
    active_indices: List[int] = []
    active_spikes: List[np.ndarray] = []

    for idx, u in enumerate(units):
        if stats_params is not None:
            rel = compute_response_reliability(
                u.spike_times, stimulus.onsets, eval_window_ms=rel_win, max_trials=rel_trials
            )
            lat_ms = compute_first_spike_latency(
                u.spike_times, stimulus.onsets, eval_window_ms=lat_win, max_trials=lat_trials
            )
            jit_ms = compute_first_spike_jitter(
                u.spike_times, stimulus.onsets, eval_window_ms=jit_win, max_trials=jit_trials
            )
        else:
            lat_ms, jit_ms, rel = compute_first_spike_latency_and_jitter(
                u.spike_times, stimulus.onsets, eval_window_ms=eval_window_ms, max_trials=max_trials
            )
        wv_r = compute_unit_waveform_correlation(
            u.waveforms,
            u.spike_times,
            stimulus.onsets,
            eval_window_ms=wf_eval_win,
            baseline_window_ms=wf_spont_win,
            max_spikes_per_group=wf_max_spikes,
        )
        preliminary.append((lat_ms, jit_ms, rel, wv_r))

        if rel > 0.0 and not np.isnan(lat_ms):
            active_indices.append(idx)
            active_spikes.append(u.spike_times)

    # 2. Parallel computation of SALT p-values for all active units
    salt_results: List[float] = []
    if active_spikes:
        salt_results = compute_salt_p_values_parallel(
            active_spikes,
            stimulus.onsets,
            pulse_offsets=stimulus.offsets,
            eval_window_ms=salt_eval_win,
            baseline_window_ms=salt_base_win,
            exclusion_window_ms=salt_excl_win,
            n_bins=salt_bins,
            n_permutations=salt_perms,
            n_workers=n_workers,
            progress_callback=progress_callback,
        )

    salt_p_map: Dict[int, float] = {}
    for unit_idx, p_val in zip(active_indices, salt_results):
        salt_p_map[unit_idx] = p_val

    # 3. Assemble UnitMetrics for all units
    metrics_list: List[UnitMetrics] = []
    for idx, u in enumerate(units):
        lat_ms, jit_ms, rel, wv_r = preliminary[idx]
        salt_p = salt_p_map.get(idx, 1.0)

        is_tagged = bool(
            not np.isnan(lat_ms)
            and (lat_ms <= max_latency_ms)
            and (rel >= min_reliability)
            and (np.isnan(jit_ms) or jit_ms <= max_jitter_ms)
            and ((not np.isfinite(wv_r)) or (wv_r >= min_waveform_r))
            and (salt_p <= max_salt_p)
        )

        plot_rel = f"{u.session_id}_plots/{u.global_id}_psth_raster.png"
        metrics_list.append(
            UnitMetrics(
                global_id=u.global_id,
                session_id=u.session_id,
                unit_id=u.unit_id,
                plot_path=plot_rel,
                latency_ms=lat_ms if not np.isnan(lat_ms) else 999.0,
                jitter_ms=jit_ms if not np.isnan(jit_ms) else 999.0,
                reliability=rel,
                waveform_correlation=wv_r,
                salt_p_value=salt_p,
                is_tagged=is_tagged,
            )
        )

    return metrics_list


def compute_psth_time_ticks(w_start: float, w_end: float) -> List[float]:
    """Compute standard PSTH time ticks guaranteeing ticks at 5.0 and 10.0 ms if within range.

    Args:
        w_start: Start of time window in ms (e.g. -30.0).
        w_end: End of time window in ms (e.g. 50.0).

    Returns:
        Sorted unique list of float tick values.
    """
    span = float(w_end) - float(w_start)
    step = 10.0 if span <= 90.0 else (20.0 if span <= 200.0 else 50.0)

    start_tick = np.ceil(w_start / step) * step
    raw_ticks = list(np.arange(start_tick, w_end + 1e-6, step))

    if w_start <= 0.0 <= w_end and 0.0 not in raw_ticks:
        raw_ticks.append(0.0)

    for special in (5.0, 10.0):
        if w_start <= special <= w_end and special not in raw_ticks:
            raw_ticks.append(special)

    unique_ticks = sorted(list(set(round(float(t), 4) for t in raw_ticks)))
    return unique_ticks

