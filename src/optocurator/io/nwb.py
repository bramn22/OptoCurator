"""Direct h5py reader for Allen/DANDI NWB electrophysiology and optotagging files."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from optocurator.io.base import BaseSpikeReader, BaseStimulusReader
from optocurator.models import StimulusProtocol, UnitData


_OPTO_INTERVAL_PATTERNS = ("pulse", "opto", "raised_cosine", "laser", "stim_train")


def _decode_str(val: Any) -> str:
    """Decode HDF5 bytes / numpy scalar strings cleanly to Python str."""
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _decode_str_array(ds: Any) -> List[str]:
    """Read an HDF5 string dataset into a list of Python str."""
    arr = ds[:]
    return [_decode_str(x) for x in arr]


@dataclass
class NwbProtocolInfo:
    """Metadata summary for a discovered NWB optogenetic stimulation table."""

    key: str
    location: str          # "intervals/<name>" or "processing/optotagging/<name>"
    display_name: str
    n_intervals: int
    median_interval_ms: float
    frequency_hz: Optional[float] = None
    pulse_width_ms: Optional[float] = None


@dataclass
class NwbSessionSummary:
    """Lightweight summary of an NWB file scanned without bulk-loading spike arrays."""

    file_path: Path
    session_id: str
    n_total_units: int
    probe_unit_counts: Dict[str, int]
    probe_qc_counts: Dict[str, int]
    protocols: List[NwbProtocolInfo]


def _parse_frequency_and_width(name: str, condition: str = "") -> Tuple[Optional[float], Optional[float]]:
    """Extract pulse train frequency (Hz) and individual pulse width (ms) from table metadata."""
    freq_hz: Optional[float] = None
    width_ms: Optional[float] = None
    m_freq = re.search(r"(\d+(?:\.\d+)?)\s*hz", name, re.IGNORECASE)
    if m_freq:
        freq_hz = float(m_freq.group(1))
    m_width = re.search(r"(\d+(?:\.\d+)?)\s*ms", condition, re.IGNORECASE)
    if not m_width:
        m_width = re.search(r"(\d+(?:\.\d+)?)\s*ms", name, re.IGNORECASE)
    if m_width:
        width_ms = float(m_width.group(1))
    return freq_hz, width_ms


def _clean_protocol_name(raw_key: str) -> str:
    """Normalize NWB interval group name for display in Protocol selector."""
    name = re.sub(r"_presentations$", "", raw_key.strip(), flags=re.IGNORECASE)
    return name or raw_key


def _discover_opto_groups(hf: h5py.File) -> List[Tuple[str, str, h5py.Group]]:
    """Find candidate optogenetic stimulus DynamicTables in an open NWB HDF5 file."""
    found: List[Tuple[str, str, h5py.Group]] = []

    if "processing" in hf and "optotagging" in hf["processing"]:
        opto_proc = hf["processing"]["optotagging"]
        for key in sorted(opto_proc.keys()):
            grp = opto_proc[key]
            if isinstance(grp, h5py.Group) and "start_time" in grp:
                found.append((key, f"processing/optotagging/{key}", grp))

    if "intervals" in hf:
        intervals = hf["intervals"]
        for key in sorted(intervals.keys()):
            low = key.lower()
            if any(pat in low for pat in _OPTO_INTERVAL_PATTERNS):
                grp = intervals[key]
                if isinstance(grp, h5py.Group) and "start_time" in grp:
                    found.append((key, f"intervals/{key}", grp))

    return found


def inspect_nwb(nwb_path: Union[str, Path]) -> NwbSessionSummary:
    """Inspect an NWB file for session ID, probes, QC unit counts, and opto protocols."""
    path = Path(nwb_path)
    if not path.is_file():
        raise FileNotFoundError(f"NWB file not found: {path}")

    with h5py.File(path, "r") as hf:
        if "identifier" in hf:
            try:
                raw_id = hf["identifier"][()]
                session_id = _decode_str(raw_id).strip() or path.stem
            except Exception:
                session_id = path.stem
        else:
            session_id = path.stem

        # Prefer concise ses-* stem if identifier is a long UUID
        if len(session_id) > 40 or session_id.count("-") >= 4:
            if path.stem:
                session_id = path.stem

        n_units = 0
        probe_counts: Dict[str, int] = {}
        qc_counts: Dict[str, int] = {}

        if "units" in hf:
            u = hf["units"]
            n_units = int(len(u["id"])) if "id" in u else 0
            if "device_name" in u and n_units > 0:
                devs = _decode_str_array(u["device_name"])
            else:
                devs = ["ProbeA"] * n_units

            if "default_qc" in u and n_units > 0:
                qc_arr = np.asarray(u["default_qc"][:], dtype=bool)
            else:
                qc_arr = np.ones(n_units, dtype=bool)

            for dev, qc in zip(devs, qc_arr):
                probe_counts[dev] = probe_counts.get(dev, 0) + 1
                if qc:
                    qc_counts[dev] = qc_counts.get(dev, 0) + 1

        protocols: List[NwbProtocolInfo] = []
        for raw_key, loc, grp in _discover_opto_groups(hf):
            starts = np.asarray(grp["start_time"][:], dtype=np.float64)
            if "stop_time" in grp:
                stops = np.asarray(grp["stop_time"][:], dtype=np.float64)
                durs_ms = (stops - starts) * 1000.0
                valid_durs = durs_ms[np.isfinite(durs_ms) & (durs_ms > 0)]
                med_dur = float(np.median(valid_durs)) if len(valid_durs) > 0 else 20.0
            else:
                med_dur = 20.0
            cond_str = ""
            if "condition" in grp and len(grp["condition"]) > 0:
                cond_str = _decode_str(grp["condition"][0])
            freq_hz, width_ms = _parse_frequency_and_width(raw_key, cond_str)
            protocols.append(
                NwbProtocolInfo(
                    key=raw_key,
                    location=loc,
                    display_name=_clean_protocol_name(raw_key),
                    n_intervals=int(len(starts)),
                    median_interval_ms=round(med_dur, 3),
                    frequency_hz=freq_hz,
                    pulse_width_ms=width_ms,
                )
            )

    return NwbSessionSummary(
        file_path=path,
        session_id=session_id,
        n_total_units=n_units,
        probe_unit_counts=dict(sorted(probe_counts.items())),
        probe_qc_counts=dict(sorted(qc_counts.items())),
        protocols=protocols,
    )


def load_nwb_probe_geometry(
    nwb_path: Union[str, Path],
    probe_name: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], Dict[int, Tuple[int, float, float]]]:
    """Load electrode positions for a probe and global-electrode-ID lookup map.

    Returns:
        (channel_positions (N, 2) or None,
         electrode_map: global_electrode_id -> (local_channel_idx, x_um, y_um))
    """
    path = Path(nwb_path)
    with h5py.File(path, "r") as hf:
        if "general" not in hf or "extracellular_ephys" not in hf["general"]:
            return None, {}
        ephys = hf["general"]["extracellular_ephys"]
        if "electrodes" not in ephys:
            return None, {}
        el = ephys["electrodes"]
        n_el = int(len(el["id"])) if "id" in el else 0
        if n_el == 0:
            return None, {}

        el_ids = np.asarray(el["id"][:], dtype=np.int64)
        if "rel_x" in el and "rel_y" in el:
            xs = np.asarray(el["rel_x"][:], dtype=np.float64)
            ys = np.asarray(el["rel_y"][:], dtype=np.float64)
        elif "x" in el and "y" in el:
            xs = np.asarray(el["x"][:], dtype=np.float64)
            ys = np.asarray(el["y"][:], dtype=np.float64)
        else:
            return None, {}

        if "group_name" in el:
            groups = _decode_str_array(el["group_name"])
        else:
            groups = ["ProbeA"] * n_el

        coords: List[List[float]] = []
        el_map: Dict[int, Tuple[int, float, float]] = {}
        local_idx = 0
        for row_i in range(n_el):
            if probe_name and probe_name != "All" and groups[row_i] != probe_name:
                continue
            eid = int(el_ids[row_i])
            x_val = float(xs[row_i]) if np.isfinite(xs[row_i]) else 0.0
            y_val = float(ys[row_i]) if np.isfinite(ys[row_i]) else 0.0
            el_map[eid] = (local_idx, x_val, y_val)
            coords.append([x_val, y_val])
            local_idx += 1

        if not coords:
            return None, {}
        return np.asarray(coords, dtype=np.float64), el_map


class NwbSpikeReader(BaseSpikeReader):
    """Loads units, spike times, and electrode coordinates from an NWB file via h5py."""

    def __init__(
        self,
        probe_name: Optional[str] = None,
        qc_only: bool = True,
        sampling_rate: float = 30000.0,
    ) -> None:
        self.probe_name = probe_name
        self.qc_only = qc_only
        self.sampling_rate = sampling_rate

    def load(
        self,
        folder_path: Union[str, Path],
        session_id: Optional[str] = None,
    ) -> List[UnitData]:
        """Load UnitData objects for the configured probe and QC filter from an NWB file."""
        path = Path(folder_path)
        if not path.is_file():
            raise FileNotFoundError(f"NWB file not found: {path}")

        if session_id is None:
            base_id = path.stem
            if self.probe_name and self.probe_name != "All":
                session_id = f"{base_id}_{self.probe_name}"
            else:
                session_id = base_id

        _, el_map = load_nwb_probe_geometry(path, probe_name=self.probe_name)

        units: List[UnitData] = []
        with h5py.File(path, "r") as hf:
            if "units" not in hf:
                return units
            u = hf["units"]
            n_units = int(len(u["id"])) if "id" in u else 0
            if n_units == 0:
                return units

            ids = np.asarray(u["id"][:], dtype=np.int64)
            ks_ids = np.asarray(u["ks_unit_id"][:], dtype=np.int64) if "ks_unit_id" in u else ids
            devs = _decode_str_array(u["device_name"]) if "device_name" in u else ["ProbeA"] * n_units
            qc_arr = np.asarray(u["default_qc"][:], dtype=bool) if "default_qc" in u else np.ones(n_units, dtype=bool)
            depths = np.asarray(u["depth"][:], dtype=np.float64) if "depth" in u else np.zeros(n_units, dtype=np.float64)

            spike_times_ds = u["spike_times"]
            spike_idx = np.asarray(u["spike_times_index"][:], dtype=np.int64)

            has_els = "electrodes" in u and "electrodes_index" in u
            els_ds = u["electrodes"] if has_els else None
            els_idx = np.asarray(u["electrodes_index"][:], dtype=np.int64) if has_els else None

            shanks_raw = _decode_str_array(u["shank"]) if "shank" in u else [""] * n_units

            seen_uids = set()
            for i in range(n_units):
                if self.probe_name and self.probe_name != "All" and devs[i] != self.probe_name:
                    continue
                if self.qc_only and not bool(qc_arr[i]):
                    continue

                raw_uid = int(ks_ids[i]) if (self.probe_name and self.probe_name != "All") else int(ids[i])
                if raw_uid in seen_uids:
                    raw_uid = int(ids[i])
                seen_uids.add(raw_uid)

                s_start = 0 if i == 0 else int(spike_idx[i - 1])
                s_end = int(spike_idx[i])
                spks = np.asarray(spike_times_ds[s_start:s_end], dtype=np.float64)
                if len(spks) > 1 and not np.all(spks[:-1] <= spks[1:]):
                    spks = np.sort(spks)

                u_depth = float(depths[i]) if np.isfinite(depths[i]) else 0.0
                u_channel = 0
                u_x = 0.0

                if has_els and els_ds is not None and els_idx is not None:
                    e_start = 0 if i == 0 else int(els_idx[i - 1])
                    e_end = int(els_idx[i])
                    if e_end > e_start:
                        unit_els = np.asarray(els_ds[e_start:e_end], dtype=np.int64)
                        matched_sites = [el_map[int(eid)] for eid in unit_els if int(eid) in el_map]
                        if matched_sites:
                            best = min(matched_sites, key=lambda tup: abs(tup[2] - u_depth))
                            u_channel, u_x, site_y = best
                            if u_depth == 0.0:
                                u_depth = site_y

                shank_val = 0
                if shanks_raw[i]:
                    m_sh = re.search(r"(\d+)", shanks_raw[i])
                    if m_sh:
                        shank_val = int(m_sh.group(1))

                units.append(
                    UnitData(
                        unit_id=raw_uid,
                        session_id=session_id,
                        spike_times=spks,
                        sampling_rate=self.sampling_rate,
                        waveforms=None,
                        channel=int(u_channel),
                        depth=float(u_depth),
                        probe_x=float(u_x),
                    )
                )

        return units


class NwbStimulusReader(BaseStimulusReader):
    """Loads optogenetic stimulation protocols from an NWB file."""

    def __init__(
        self,
        expand_pulse_trains: bool = True,
        default_pulse_duration_s: float = 0.010,
    ) -> None:
        self.expand_pulse_trains = expand_pulse_trains
        self.default_pulse_duration_s = default_pulse_duration_s

    def discover(self, folder_path: Union[str, Path]) -> List[Path]:
        """Return [nwb_path] if the path is an existing .nwb file."""
        path = Path(folder_path)
        if path.is_file() and path.suffix.lower() == ".nwb":
            return [path]
        if path.is_dir():
            return sorted(path.glob("*.nwb"))
        return []

    def load(
        self,
        file_path: Union[str, Path],
        sampling_rate: float = 30000.0,
        protocol_key: Optional[str] = None,
    ) -> StimulusProtocol:
        """Load the first (or named) optogenetic protocol from an NWB file."""
        protos = self.load_all(file_path)
        if not protos:
            return StimulusProtocol(
                onsets=np.empty(0, dtype=np.float64),
                offsets=np.empty(0, dtype=np.float64),
                name="default",
            )
        if protocol_key:
            for p in protos:
                if p.name == protocol_key or p.name == _clean_protocol_name(protocol_key):
                    return p
        return protos[0]

    def load_all(self, file_path: Union[str, Path]) -> List[StimulusProtocol]:
        """Load all discovered optogenetic protocols from an NWB file."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"NWB file not found: {path}")

        protocols: List[StimulusProtocol] = []
        with h5py.File(path, "r") as hf:
            for raw_key, _loc, grp in _discover_opto_groups(hf):
                starts = np.asarray(grp["start_time"][:], dtype=np.float64)
                if len(starts) == 0:
                    continue
                if "stop_time" in grp:
                    stops = np.asarray(grp["stop_time"][:], dtype=np.float64)
                else:
                    stops = starts + self.default_pulse_duration_s

                cond_str = ""
                if "condition" in grp and len(grp["condition"]) > 0:
                    cond_str = _decode_str(grp["condition"][0])
                freq_hz, width_ms = _parse_frequency_and_width(raw_key, cond_str)

                if (
                    self.expand_pulse_trains
                    and freq_hz is not None
                    and freq_hz > 0.0
                    and width_ms is not None
                    and width_ms > 0.0
                ):
                    period_s = 1.0 / freq_hz
                    width_s = width_ms / 1000.0
                    expanded_on: List[float] = []
                    expanded_off: List[float] = []
                    for t0, t1 in zip(starts, stops):
                        if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
                            continue
                        t = float(t0)
                        while t + width_s <= float(t1) + 1e-5:
                            expanded_on.append(t)
                            expanded_off.append(t + width_s)
                            t += period_s
                    on_arr = np.asarray(expanded_on, dtype=np.float64)
                    off_arr = np.asarray(expanded_off, dtype=np.float64)
                else:
                    mask = np.isfinite(starts) & np.isfinite(stops) & (stops > starts)
                    on_arr = starts[mask]
                    off_arr = stops[mask]

                if len(on_arr) > 1 and not np.all(on_arr[:-1] <= on_arr[1:]):
                    order = np.argsort(on_arr)
                    on_arr = on_arr[order]
                    off_arr = off_arr[order]

                protocols.append(
                    StimulusProtocol(
                        onsets=on_arr,
                        offsets=off_arr,
                        name=_clean_protocol_name(raw_key),
                    )
                )

        return protocols

