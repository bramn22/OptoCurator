"""Core data structures and contracts for OptoCurator."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class UnitData:
    """Represents a single electrophysiological unit within a recording session."""
    session_id: str
    unit_id: int
    spike_times: np.ndarray          # 1D array of spike timestamps in seconds
    sampling_rate: float
    channel: int = 0
    depth: float = 0.0
    probe_x: float = 0.0
    waveforms: Optional[np.ndarray] = None  # shape: (n_spikes, n_samples) or mean template

    @property
    def global_id(self) -> str:
        return f"{self.session_id}_u{self.unit_id:04d}"


@dataclass
class StimulusProtocol:
    """Optical stimulus pulse timestamps for a session protocol."""
    onsets: np.ndarray               # 1D array of pulse start times in seconds
    offsets: np.ndarray              # 1D array of pulse end times in seconds
    name: str = "default"


@dataclass
class UnitMetrics:
    """Statistical metrics, classification status, and plot metadata for a unit."""
    global_id: str
    session_id: str
    unit_id: int
    plot_path: str                   # Relative path to pre-rendered raster/PSTH PNG
    latency_ms: float
    jitter_ms: float
    reliability: float               # Evoked response probability [0, 1]
    waveform_correlation: float      # Pearson r between spontaneous and evoked
    salt_p_value: float              # Stimulus-associated spike latency test p-value
    is_tagged: bool = False
    cluster_id: int = -1
    curated_label: str = "unassigned"
    likelihood: float = 0.0


@dataclass
class StatsParameters:
    """Hyperparameters for statistical metric calculation across optotagging workflows."""

    # 1. Response Reliability
    reliability_window_start_ms: float = 0.0
    reliability_window_end_ms: float = 20.0
    reliability_trials: Optional[int] = None  # None means all pulses

    # 2. First-Spike Latency
    latency_window_start_ms: float = 0.0
    latency_window_end_ms: float = 20.0
    latency_trials: Optional[int] = None  # None means all pulses

    # 3. First-Spike Jitter
    jitter_window_start_ms: float = 0.0
    jitter_window_end_ms: float = 20.0
    jitter_trials: Optional[int] = None  # None means all pulses

    # SALT (Stimulus-Associated Spike Latency Test)
    salt_window_start_ms: float = 0.0
    salt_window_end_ms: float = 10.0
    salt_baseline_start_ms: float = -50.0
    salt_baseline_end_ms: float = 0.0
    salt_n_bins: int = 10
    salt_n_permutations: int = 100
    salt_exclusion_enabled: bool = False
    salt_exclusion_window_start_ms: float = -1.0
    salt_exclusion_window_end_ms: float = 1.0

    @property
    def salt_exclusion_window_ms(self) -> Optional[Tuple[float, float]]:
        """Return (start_ms, end_ms) exclusion window around onsets and offsets if enabled, else None."""
        if self.salt_exclusion_enabled:
            return (self.salt_exclusion_window_start_ms, self.salt_exclusion_window_end_ms)
        return None

    # Waveform correlation
    wf_evoked_start_ms: float = 0.0
    wf_evoked_end_ms: float = 20.0
    wf_baseline_start_ms: float = -50.0
    wf_baseline_end_ms: float = -5.0
    wf_window_start_ms: float = -1.0
    wf_window_end_ms: float = 2.0
    wf_max_spikes: int = 40

    # PSTH
    psth_bin_size_ms: float = 2.0
    psth_baseline_start_ms: float = -50.0
    psth_baseline_end_ms: float = 0.0

    @classmethod
    def default(cls) -> "StatsParameters":
        """Return fresh instance with standard default parameters."""
        return cls()

