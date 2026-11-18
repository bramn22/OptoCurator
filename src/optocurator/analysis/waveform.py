"""Waveform snippet extraction and evoked-vs-spontaneous correlation analysis."""

import ast
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np

from optocurator.models import StatsParameters, UnitData


def split_evoked_and_spontaneous_spikes(
    spike_times: np.ndarray,
    pulse_onsets: np.ndarray,
    evoked_window_ms: Tuple[float, float] = (0.0, 20.0),
    baseline_window_ms: Tuple[float, float] = (-50.0, -5.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Partition spike indices into light-evoked and spontaneous groups.

    Args:
        spike_times: 1D array of spike timestamps in seconds.
        pulse_onsets: 1D array of stimulus pulse onset times in seconds.
        evoked_window_ms: (start_ms, end_ms) post-stimulus evoked window in ms.
        baseline_window_ms: (start_ms, end_ms) pre-stimulus spontaneous window in ms.

    Returns:
        Tuple of (evoked_indices, spontaneous_indices) as 1D int64 arrays.
        Spontaneous spikes fall within ``baseline_window_ms`` (and outside any
        evoked window); if none fall in ``baseline_window_ms``, falls back to all
        spikes outside any pulse's evoked window.
    """
    times = np.asarray(spike_times, dtype=np.float64)
    onsets = np.asarray(pulse_onsets, dtype=np.float64)
    n_spikes = len(times)
    if n_spikes == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    if onsets.size == 0:
        return np.empty(0, dtype=np.int64), np.arange(n_spikes, dtype=np.int64)

    is_sorted = bool(n_spikes <= 1 or np.all(times[:-1] <= times[1:]))
    if is_sorted:
        sorted_times = times
        order = None
    else:
        order = np.argsort(times, kind="mergesort")
        sorted_times = times[order]

    ev_start_s = evoked_window_ms[0] / 1000.0
    ev_end_s = evoked_window_ms[1] / 1000.0
    base_start_s = baseline_window_ms[0] / 1000.0
    base_end_s = baseline_window_ms[1] / 1000.0

    evoked_sorted_mask = np.zeros(n_spikes, dtype=bool)
    base_sorted_mask = np.zeros(n_spikes, dtype=bool)

    for onset in onsets:
        e_lo = int(np.searchsorted(sorted_times, onset + ev_start_s, side="left"))
        e_hi = int(np.searchsorted(sorted_times, onset + ev_end_s, side="right"))
        if e_hi > e_lo:
            evoked_sorted_mask[e_lo:e_hi] = True

        b_lo = int(np.searchsorted(sorted_times, onset + base_start_s, side="left"))
        b_hi = int(np.searchsorted(sorted_times, onset + base_end_s, side="right"))
        if b_hi > b_lo:
            base_sorted_mask[b_lo:b_hi] = True

    if order is not None:
        evoked_mask = np.zeros(n_spikes, dtype=bool)
        base_mask = np.zeros(n_spikes, dtype=bool)
        evoked_mask[order] = evoked_sorted_mask
        base_mask[order] = base_sorted_mask
    else:
        evoked_mask = evoked_sorted_mask
        base_mask = base_sorted_mask

    evoked_indices = np.flatnonzero(evoked_mask).astype(np.int64, copy=False)
    spont_mask = base_mask & (~evoked_mask)
    spont_indices = np.flatnonzero(spont_mask).astype(np.int64, copy=False)

    if spont_indices.size == 0:
        spont_indices = np.flatnonzero(~evoked_mask).astype(np.int64, copy=False)

    return evoked_indices, spont_indices


def subsample_indices(indices: np.ndarray, max_count: int) -> np.ndarray:
    """Deterministically subsample an index array to at most ``max_count`` elements."""
    idx = np.asarray(indices, dtype=np.int64)
    if max_count <= 0 or idx.size == 0:
        return np.empty(0, dtype=np.int64)
    if len(idx) <= max_count:
        return idx
    picks = np.linspace(0, len(idx) - 1, num=max_count, dtype=np.int64)
    return idx[picks]


def extract_evoked_and_spontaneous_snippets(
    waveforms: Optional[np.ndarray],
    spike_times: np.ndarray,
    pulse_onsets: np.ndarray,
    evoked_window_ms: Tuple[float, float] = (0.0, 20.0),
    baseline_window_ms: Tuple[float, float] = (-50.0, -5.0),
    max_spikes_per_group: int = 40,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract subsampled evoked and spontaneous 2D waveform snippet arrays on peak channel.

    Returns ``None`` if waveforms are unavailable, 1D, mismatched in spike count,
    or if either spike group is empty.
    """
    if waveforms is None:
        return None
    wf = np.asarray(waveforms)
    if wf.size == 0 or wf.ndim == 1 or len(wf) != len(spike_times):
        return None

    if wf.ndim == 3:
        peak_ch = int(np.argmax(np.ptp(np.mean(wf, axis=0), axis=-1)))
        wf_2d = wf[:, peak_ch, :]
    elif wf.ndim == 2:
        wf_2d = wf
    else:
        return None

    evoked_idx, spont_idx = split_evoked_and_spontaneous_spikes(
        spike_times,
        pulse_onsets,
        evoked_window_ms=evoked_window_ms,
        baseline_window_ms=baseline_window_ms,
    )
    if evoked_idx.size == 0 or spont_idx.size == 0:
        return None

    evoked_sub = subsample_indices(evoked_idx, max_spikes_per_group)
    spont_sub = subsample_indices(spont_idx, max_spikes_per_group)
    if evoked_sub.size == 0 or spont_sub.size == 0:
        return None

    return (
        np.asarray(wf_2d[evoked_sub], dtype=np.float64),
        np.asarray(wf_2d[spont_sub], dtype=np.float64),
    )


def compute_wf_r(evoked_snippets: np.ndarray, spont_snippets: np.ndarray) -> float:
    """Compute Pearson r between mean evoked and mean spontaneous waveforms."""
    evoked = np.asarray(evoked_snippets, dtype=np.float64)
    spont = np.asarray(spont_snippets, dtype=np.float64)
    if (
        evoked.size == 0
        or spont.size == 0
        or evoked.ndim != 2
        or spont.ndim != 2
        or evoked.shape[1] != spont.shape[1]
    ):
        return float("nan")

    mean_e = np.mean(evoked, axis=0)
    mean_s = np.mean(spont, axis=0)
    std_e = float(np.std(mean_e))
    std_s = float(np.std(mean_s))

    if not np.isfinite(std_e) or not np.isfinite(std_s) or std_e <= 1e-9 or std_s <= 1e-9:
        return float("nan")

    dev_e = mean_e - np.mean(mean_e)
    dev_s = mean_s - np.mean(mean_s)
    r = float(np.dot(dev_e, dev_s) / (len(dev_e) * std_e * std_s))
    return float(np.clip(r, -1.0, 1.0))


def compute_unit_waveform_correlation(
    waveforms: Optional[np.ndarray],
    spike_times: np.ndarray,
    pulse_onsets: np.ndarray,
    eval_window_ms: Tuple[float, float] = (0.0, 20.0),
    baseline_window_ms: Tuple[float, float] = (-50.0, -5.0),
    max_spikes_per_group: int = 40,
) -> float:
    """Compute evoked-vs-spontaneous waveform Pearson r, or NaN if unavailable."""
    snippets = extract_evoked_and_spontaneous_snippets(
        waveforms=waveforms,
        spike_times=spike_times,
        pulse_onsets=pulse_onsets,
        evoked_window_ms=eval_window_ms,
        baseline_window_ms=baseline_window_ms,
        max_spikes_per_group=max_spikes_per_group,
    )
    if snippets is None:
        return float("nan")
    return compute_wf_r(snippets[0], snippets[1])


def _parse_params_py(params_path: Path) -> Dict[str, Any]:
    """Safely parse literal assignments from a Phy/Kilosort params.py file."""
    parsed: Dict[str, Any] = {}
    if not params_path.exists():
        return parsed
    try:
        tree = ast.parse(params_path.read_text(encoding="utf-8", errors="ignore"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                key = node.targets[0].id
                try:
                    parsed[key] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    continue
    except OSError:
        return parsed
    return parsed


def _resolve_raw_binary(session_folder: Path, params: Dict[str, Any]) -> Optional[Path]:
    """Resolve raw binary recording path from params.py or fallback glob in session_folder."""
    dat_val = params.get("dat_path")
    candidates = []
    if isinstance(dat_val, (list, tuple)) and dat_val:
        dat_val = dat_val[0]
    if isinstance(dat_val, str) and dat_val.strip():
        p = Path(dat_val.strip())
        if not p.is_absolute():
            p = session_folder / p
        candidates.append(p)

    candidates.extend(sorted(session_folder.glob("*.ap.bin")))
    candidates.extend(sorted(session_folder.glob("*.dat")))

    for candidate in candidates:
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        except OSError:
            continue
    return None


def extract_session_waveforms(
    session_folder: Union[str, Path],
    units: Sequence[UnitData],
    pulse_onsets: np.ndarray,
    stats_params: Optional[StatsParameters] = None,
) -> None:
    """Populate ``unit.waveforms`` from raw binary recording for units missing waveforms.

    Reads ``params.py`` (or ``*.ap.bin`` / ``*.dat``) in ``session_folder`` via
    ``np.memmap`` and extracts peak-channel snippets around ``wf_window_start_ms``
    to ``wf_window_end_ms`` for subsampled evoked and spontaneous spikes.
    If no valid raw binary is available, leaves ``unit.waveforms`` as ``None``.
    """
    missing_units = [u for u in units if u.waveforms is None]
    if not missing_units:
        return

    folder = Path(session_folder)
    if not folder.is_dir():
        return

    params = _parse_params_py(folder / "params.py")
    bin_path = _resolve_raw_binary(folder, params)
    if bin_path is None:
        return

    try:
        n_channels = int(params.get("n_channels_dat", 0))
    except (TypeError, ValueError):
        return
    if n_channels <= 0:
        return

    dtype_spec = params.get("dtype", "int16")
    try:
        dtype = np.dtype(dtype_spec)
    except (TypeError, ValueError):
        dtype = np.dtype("int16")

    try:
        offset = max(0, int(params.get("offset", 0)))
    except (TypeError, ValueError):
        offset = 0

    file_bytes = bin_path.stat().st_size - offset
    frame_bytes = n_channels * dtype.itemsize
    if file_bytes <= 0 or frame_bytes <= 0:
        return
    n_total_samples = file_bytes // frame_bytes
    if n_total_samples <= 0:
        return

    sp = stats_params if stats_params is not None else StatsParameters.default()
    evoked_win = (sp.wf_evoked_start_ms, sp.wf_evoked_end_ms)
    baseline_win = (sp.wf_baseline_start_ms, sp.wf_baseline_end_ms)
    max_spikes = sp.wf_max_spikes

    try:
        mmap = np.memmap(
            bin_path,
            dtype=dtype,
            mode="r",
            shape=(n_total_samples, n_channels),
            offset=offset,
        )
    except (OSError, ValueError):
        return

    try:
        for unit in missing_units:
            if unit.spike_times.size == 0 or unit.sampling_rate <= 0:
                continue

            pre_samples = int(round((sp.wf_window_start_ms / 1000.0) * unit.sampling_rate))
            post_samples = int(round((sp.wf_window_end_ms / 1000.0) * unit.sampling_rate))
            n_wf_samples = post_samples - pre_samples
            if n_wf_samples <= 1:
                continue

            evoked_idx, spont_idx = split_evoked_and_spontaneous_spikes(
                unit.spike_times,
                pulse_onsets,
                evoked_window_ms=evoked_win,
                baseline_window_ms=baseline_win,
            )
            evoked_sub = subsample_indices(evoked_idx, max_spikes)
            spont_sub = subsample_indices(spont_idx, max_spikes)
            if evoked_sub.size == 0 or spont_sub.size == 0:
                continue

            selected = np.unique(np.concatenate([evoked_sub, spont_sub]))
            peak_ch = int(np.clip(unit.channel, 0, n_channels - 1))

            wf_array = np.zeros((len(unit.spike_times), n_wf_samples), dtype=np.float32)
            extracted_any = False
            for idx in selected:
                center = int(round(float(unit.spike_times[idx]) * unit.sampling_rate))
                start = center + pre_samples
                end = center + post_samples
                if start < 0 or end > n_total_samples:
                    continue
                wf_array[idx, :] = mmap[start:end, peak_ch].astype(np.float32, copy=False)
                extracted_any = True

            if extracted_any:
                unit.waveforms = wf_array
    finally:
        del mmap
