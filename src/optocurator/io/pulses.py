"""Reader for stimulus pulse timestamp files (.npy and .mat)."""

from pathlib import Path
import re
from typing import List, Optional, Union
import numpy as np
import scipy.io

from optocurator.io.base import BaseStimulusReader
from optocurator.models import StimulusProtocol


class MatNpyStimulusReader(BaseStimulusReader):
    """Loads optical stimulus pulse onsets and offsets from .npy or .mat files."""

    def __init__(self, default_pulse_duration_s: float = 0.020) -> None:
        self.default_pulse_duration_s = default_pulse_duration_s

    def discover(self, folder_path: Union[str, Path]) -> List[Path]:
        """Discover stimulus pulse files in the directory (and immediate parent if sorted_dir).

        Args:
            folder_path: Directory to search.

        Returns:
            List of Path objects matching pulse filename patterns.
        """
        path = Path(folder_path)
        if not path.is_dir():
            return []

        search_dirs = [path]
        if path.name.lower() in ("sorted_dir", "kilosort", "sorter_output") and path.parent.is_dir():
            search_dirs.append(path.parent)

        found: List[Path] = []
        patterns = [
            "*opto*.npy",
            "*pulse*.npy",
            "*stim*.npy",
            "*opto*.mat",
            "*pulse*.mat",
            "*stim*.mat",
        ]

        seen = set()
        for sdir in search_dirs:
            for pat in patterns:
                for match in sdir.glob(pat):
                    if match.is_file() and match.resolve() not in seen:
                        seen.add(match.resolve())
                        found.append(match)

        # Sort so that preferred files (e.g. OPTS_03, opto_idxs) appear consistently
        found.sort(key=lambda p: p.name)
        return found

    def load(
        self,
        file_path: Union[str, Path],
        sampling_rate: float = 30000.0,
        protocol_name: Optional[str] = None,
    ) -> StimulusProtocol:
        """Load pulse timestamps from .npy or .mat file.

        Args:
            file_path: Path to stimulus file.
            sampling_rate: Recording sampling rate in Hz.
            protocol_name: Optional explicit protocol name. Inferred if None.

        Returns:
            StimulusProtocol with onsets and offsets in seconds.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If data cannot be parsed.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Stimulus file not found: {path}")

        # Infer protocol name if not supplied
        if protocol_name is None:
            protocol_name = self._infer_protocol_name(path)

        suffix = path.suffix.lower()
        if suffix == ".npy":
            raw_data = np.load(path, allow_pickle=True)
        elif suffix == ".mat":
            raw_data = self._load_mat_array(path)
        else:
            raise ValueError(f"Unsupported stimulus file format: {suffix}. Expected .npy or .mat")

        data = np.asarray(raw_data)
        if data.size == 0:
            return StimulusProtocol(
                onsets=np.empty(0, dtype=np.float64),
                offsets=np.empty(0, dtype=np.float64),
                name=protocol_name,
            )

        # Extract onsets and offsets
        if data.ndim == 2 and data.shape[1] >= 2:
            onsets_raw = data[:, 0]
            offsets_raw = data[:, 1]
        elif data.ndim == 2 and data.shape[0] >= 2 and data.shape[1] == 1:
            onsets_raw = data.flatten()
            offsets_raw = None
        elif data.ndim == 1:
            onsets_raw = data
            offsets_raw = None
        else:
            onsets_raw = data.flatten()
            offsets_raw = None

        # Convert to seconds
        # Check if values are sample indices (integers or large numbers > 10000)
        is_sample_indices = (
            np.issubdtype(onsets_raw.dtype, np.integer)
            or (len(onsets_raw) > 0 and np.nanmax(onsets_raw) > 10000.0)
        )

        if is_sample_indices:
            onsets_sec = onsets_raw.astype(np.float64) / sampling_rate
            if offsets_raw is not None:
                offsets_sec = offsets_raw.astype(np.float64) / sampling_rate
            else:
                offsets_sec = onsets_sec + self.default_pulse_duration_s
        else:
            onsets_sec = onsets_raw.astype(np.float64)
            if offsets_raw is not None:
                offsets_sec = offsets_raw.astype(np.float64)
            else:
                offsets_sec = onsets_sec + self.default_pulse_duration_s

        # Ensure sorted by onsets
        if len(onsets_sec) > 1 and not np.all(onsets_sec[:-1] <= onsets_sec[1:]):
            sort_order = np.argsort(onsets_sec)
            onsets_sec = onsets_sec[sort_order]
            offsets_sec = offsets_sec[sort_order]

        return StimulusProtocol(
            onsets=onsets_sec,
            offsets=offsets_sec,
            name=protocol_name,
        )

    def _infer_protocol_name(self, path: Path) -> str:
        """Infer clean protocol name from filename, e.g. opto_idxs_OPTS_03.npy -> OPTS_03."""
        name = path.stem
        match = re.search(r"(?:opto(?:_idxs)?_)(.+)", name, re.IGNORECASE)
        if match:
            return match.group(1)
        return name

    def _load_mat_array(self, path: Path) -> np.ndarray:
        """Extract stimulus array from .mat file dictionary."""
        mat_dict = scipy.io.loadmat(path)
        # Check known candidate keys
        candidates = ["opto", "pulses", "stim", "onsets", "stim_times", "pulse_times", "data"]
        for cand in candidates:
            for k, val in mat_dict.items():
                if cand in k.lower() and isinstance(val, np.ndarray):
                    return val

        # Fallback: find any 1D or 2D numeric array
        for k, val in mat_dict.items():
            if not k.startswith("__") and isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number):
                return val

        raise ValueError(f"No suitable numeric array found in .mat file: {path}")
