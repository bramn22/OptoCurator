"""Phy / Kilosort spike sorted data reader."""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from optocurator.analysis.waveform import _parse_params_py
from optocurator.models import DEFAULT_SAMPLING_RATE_HZ, UnitData


def timestamps_to_seconds(arr: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Convert a timestamp array to float64 seconds based strictly on dtype.

    Integer arrays are interpreted as sample indices and divided by
    ``sampling_rate``; floating-point arrays are assumed to already be in seconds.
    """
    values = np.asarray(arr)
    if np.issubdtype(values.dtype, np.integer):
        if sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be positive to convert sample indices, got {sampling_rate}")
        return values.astype(np.float64) / float(sampling_rate)
    return values.astype(np.float64)


def load_probe_and_unit_positions(
    folder: Union[str, Path],
) -> Tuple[Dict[int, int], Dict[int, float], Dict[int, float], Optional[np.ndarray]]:
    """Resolve probe geometry and (channel, depth, probe_x) for each unit in a Phy/Kilosort directory.

    Args:
        folder: Path to directory containing Kilosort/Phy files.

    Returns:
        Tuple of:
            - channel_map: Dict[unit_id, int]
            - depth_map: Dict[unit_id, float]
            - probe_x_map: Dict[unit_id, float]
            - probe_positions: Optional[(N, 2) ndarray] of electrode sites [x, y] in um.
    """
    path = Path(folder)
    channel_map: Dict[int, int] = {}
    depth_map: Dict[int, float] = {}
    probe_x_map: Dict[int, float] = {}
    probe_positions: Optional[np.ndarray] = None

    # 1. Locate channel_positions.npy (probe geometry)
    pos_file = path / "channel_positions.npy"
    if not pos_file.exists() and path.parent != path:
        parent_pos = path.parent / "channel_positions.npy"
        if parent_pos.exists():
            pos_file = parent_pos

    if pos_file.exists():
        try:
            arr = np.load(pos_file)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                probe_positions = arr.astype(np.float64)
        except Exception:
            probe_positions = None

    # Load channel_map.npy if present
    chan_map_file = path / "channel_map.npy"
    if not chan_map_file.exists() and path.parent != path:
        parent_cm = path.parent / "channel_map.npy"
        if parent_cm.exists():
            chan_map_file = parent_cm
    hardware_channels: Optional[np.ndarray] = None
    hw_to_site: Dict[int, int] = {}
    if chan_map_file.exists():
        try:
            hardware_channels = np.load(chan_map_file).flatten()
            hw_to_site = {int(hw): idx for idx, hw in enumerate(hardware_channels)}
        except Exception:
            hardware_channels = None

    # 2. Check cluster_info.tsv for unit channels and coordinates
    info_file = path / "cluster_info.tsv"
    if info_file.exists():
        try:
            df = pd.read_csv(info_file, sep="\t")
            id_col = "id" if "id" in df.columns else ("cluster_id" if "cluster_id" in df.columns else None)
            if id_col is not None:
                ch_col = "ch" if "ch" in df.columns else ("channel" if "channel" in df.columns else None)
                depth_col = "depth" if "depth" in df.columns else None
                x_col = (
                    "x"
                    if "x" in df.columns
                    else (
                        "probe_x"
                        if "probe_x" in df.columns
                        else ("pos_x" if "pos_x" in df.columns else None)
                    )
                )

                for _, row in df.iterrows():
                    try:
                        uid = int(row[id_col])
                    except (ValueError, TypeError):
                        continue

                    if ch_col and pd.notna(row[ch_col]):
                        try:
                            channel_map[uid] = int(row[ch_col])
                        except (ValueError, TypeError):
                            pass

                    if depth_col and pd.notna(row[depth_col]):
                        try:
                            depth_map[uid] = float(row[depth_col])
                        except (ValueError, TypeError):
                            pass

                    if x_col and pd.notna(row[x_col]):
                        try:
                            probe_x_map[uid] = float(row[x_col])
                        except (ValueError, TypeError):
                            pass
                    elif uid in channel_map and probe_positions is not None:
                        ch = channel_map[uid]
                        site_idx = hw_to_site.get(ch, ch) if hardware_channels is not None else ch
                        if 0 <= site_idx < len(probe_positions):
                            probe_x_map[uid] = float(probe_positions[site_idx, 0])
                            if uid not in depth_map:
                                depth_map[uid] = float(probe_positions[site_idx, 1])
        except Exception:
            pass

    # 3. If templates.npy exists, infer peak channel and coordinates for units
    templates_file = path / "templates.npy"
    if not templates_file.exists() and path.parent != path:
        parent_tmpl = path.parent / "templates.npy"
        if parent_tmpl.exists():
            templates_file = parent_tmpl

    if templates_file.exists():
        try:
            templates = np.load(templates_file, mmap_mode="r")  # (n_templates, n_timepoints, n_channels)
            cluster_to_tmpl: Dict[int, int] = {}
            st_file = path / "spike_templates.npy"
            sc_file = path / "spike_clusters.npy"
            spk_cls: Optional[np.ndarray] = None
            if sc_file.exists():
                spk_cls = np.load(sc_file).flatten()
            if st_file.exists() and spk_cls is not None:
                spk_tmpl = np.load(st_file).flatten()
                # Fast mapping of unique cluster IDs to modal template
                df_pairs = pd.DataFrame({"cls": spk_cls, "tmpl": spk_tmpl})
                for cid, group in df_pairs.groupby("cls"):
                    cluster_to_tmpl[int(cid)] = int(group["tmpl"].mode().iloc[0])

            n_templates = templates.shape[0]
            sc_uids = {int(c) for c in np.unique(spk_cls)} if spk_cls is not None else set()
            candidate_uids = set(cluster_to_tmpl.keys()) | set(channel_map.keys()) | sc_uids
            for uid in candidate_uids:
                t_idx = cluster_to_tmpl.get(uid, uid)
                if 0 <= t_idx < n_templates:
                    tmpl = templates[t_idx]  # (time, channels)
                    ptp = np.ptp(tmpl, axis=0)
                    peak_ch = int(np.argmax(ptp))
                    if uid not in channel_map:
                        if hardware_channels is not None and 0 <= peak_ch < len(hardware_channels):
                            channel_map[uid] = int(hardware_channels[peak_ch])
                        else:
                            channel_map[uid] = peak_ch
                    if probe_positions is not None and 0 <= peak_ch < len(probe_positions):
                        if uid not in probe_x_map:
                            probe_x_map[uid] = float(probe_positions[peak_ch, 0])
                        if uid not in depth_map:
                            depth_map[uid] = float(probe_positions[peak_ch, 1])
        except Exception:
            pass

    return channel_map, depth_map, probe_x_map, probe_positions


class PhyKilosortReader:
    """Loads spike sorted clusters and timestamps from Phy / Kilosort output directories."""

    def __init__(self, default_sampling_rate: float = DEFAULT_SAMPLING_RATE_HZ) -> None:
        self.default_sampling_rate = default_sampling_rate

    def load(
        self, folder_path: Union[str, Path], session_id: Optional[str] = None
    ) -> List[UnitData]:
        """Load spike sorted data from the given folder.

        Args:
            folder_path: Directory containing spike_times.npy and spike_clusters.npy.
            session_id: Optional session identifier. Inferred from folder if None.

        Returns:
            List of UnitData objects.

        Raises:
            FileNotFoundError: If spike_times.npy or spike_clusters.npy are missing.
        """
        path = Path(folder_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Directory not found: {path}")

        spike_times_file = path / "spike_times.npy"
        spike_clusters_file = path / "spike_clusters.npy"

        if not spike_times_file.exists():
            raise FileNotFoundError(f"Missing spike_times.npy in {path}")
        if not spike_clusters_file.exists():
            raise FileNotFoundError(f"Missing spike_clusters.npy in {path}")

        # Resolve session ID
        if session_id is None or not session_id.strip():
            generic_names = {"sorted_dir", "kilosort", "kilosort3", "kilosort4", "sorter_output", "output"}
            if path.name.lower() in generic_names and path.parent.name:
                session_id = path.parent.name
            else:
                session_id = path.name

        # Parse sampling rate from params.py if present
        sampling_rate = self._read_sampling_rate(path)

        # Parse channel & depth & probe_x metadata
        channel_map, depth_map, probe_x_map, probe_positions = load_probe_and_unit_positions(path)

        # Load numpy arrays
        spike_times = np.load(spike_times_file).flatten()
        spike_clusters = np.load(spike_clusters_file).flatten()

        if len(spike_times) != len(spike_clusters):
            raise ValueError(
                f"Mismatch in spike_times ({len(spike_times)}) and spike_clusters ({len(spike_clusters)}) lengths."
            )

        if len(spike_times) == 0:
            return []

        # Integer arrays are sample indices; float arrays are already in seconds.
        spike_times_sec = timestamps_to_seconds(spike_times, sampling_rate)

        # Fast grouping by unit ID using argsort and unique
        sort_indices = np.argsort(spike_clusters, kind="stable")
        sc_sorted = spike_clusters[sort_indices]
        st_sorted = spike_times_sec[sort_indices]

        unique_units, split_indices = np.unique(sc_sorted, return_index=True)
        unit_spike_arrays = np.split(st_sorted, split_indices[1:])

        units: List[UnitData] = []
        for uid, spk_times in zip(unique_units, unit_spike_arrays):
            int_uid = int(uid)
            # Ensure spike times within the unit are sorted
            if len(spk_times) > 1 and not np.all(spk_times[:-1] <= spk_times[1:]):
                spk_times = np.sort(spk_times)

            ch = int(channel_map.get(int_uid, 0))
            depth = float(depth_map.get(int_uid, 0.0))
            probe_x = float(probe_x_map.get(int_uid, 0.0))

            # Fallback to probe_positions only if coordinates were not resolved in metadata/templates
            if (
                probe_positions is not None
                and int_uid not in depth_map
                and int_uid not in probe_x_map
                and 0 <= ch < len(probe_positions)
            ):
                probe_x = float(probe_positions[ch, 0])
                depth = float(probe_positions[ch, 1])

            units.append(
                UnitData(
                    session_id=session_id,
                    unit_id=int_uid,
                    spike_times=spk_times,
                    sampling_rate=sampling_rate,
                    channel=ch,
                    depth=depth,
                    probe_x=probe_x,
                )
            )

        # Sort units by unit_id
        units.sort(key=lambda u: u.unit_id)
        return units

    def _read_sampling_rate(self, folder: Path) -> float:
        """Extract sample_rate from folder or parent params.py, or return default."""
        candidates = [folder / "params.py"]
        if folder.parent != folder:
            candidates.append(folder.parent / "params.py")
        for params_file in candidates:
            parsed = _parse_params_py(params_file)
            if "sample_rate" in parsed:
                try:
                    rate = float(parsed["sample_rate"])
                    if math.isfinite(rate) and rate > 0.0:
                        return rate
                except (TypeError, ValueError):
                    continue
        return self.default_sampling_rate

