"""Core data structures and contracts for OptoCurator."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple
import numpy as np

MISSING_METRIC_SENTINEL: float = 999.0
DEFAULT_SAMPLING_RATE_HZ: float = 30000.0
DEFAULT_PULSE_DURATION_MS: float = 10.0
DEFAULT_PULSE_DURATION_S: float = DEFAULT_PULSE_DURATION_MS / 1000.0


def is_valid_metric_value(val: float) -> bool:
    """Return True when a scalar latency/jitter metric is finite and below the missing sentinel."""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and f < (MISSING_METRIC_SENTINEL - 1.0)


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

    @classmethod
    def default(cls) -> "StatsParameters":
        """Return fresh instance with standard default parameters."""
        return cls()

