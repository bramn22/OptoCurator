"""Directory and Phase 1 export validation routines for Phase 2 multi-experiment ingestion."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from optocurator.io.phy import PhyKilosortReader, timestamps_to_seconds
from optocurator.io.pulses import MatNpyStimulusReader


class MissingPhase1ExportError(Exception):
    """Raised when an experiment directory does not contain valid Phase 1 curation export artifacts."""


class InvalidPhase1ExportError(MissingPhase1ExportError):
    """Raised when a Phase 1 export CSV is malformed or corrupted."""


def _parse_tagged_mask(df: Optional[pd.DataFrame]) -> pd.Series:
    """Return a boolean mask selecting rows where ``is_tagged`` is truthy."""
    if df is None or df.empty or "is_tagged" not in df.columns:
        return pd.Series(False, index=df.index if df is not None else None, dtype=bool)
    col = df["is_tagged"]
    if pd.api.types.is_bool_dtype(col):
        return col.fillna(False).astype(bool)
    return col.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "1.0"])


@dataclass
class Phase1ExportMetadata:
    """Metadata and parsed dataset representing a validated Phase 1 curated experiment."""

    session_id: str
    folder_path: Path
    csv_path: Path
    plots_dir: Path
    total_units: int
    tagged_units: int
    protocol_name: str = "OPTS03"
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    arrays_dir: Optional[Path] = None

    @property
    def tagged_df(self) -> pd.DataFrame:
        """Return subset of dataframe containing only confirmed optotagged units (is_tagged == True)."""
        if self.df is None or self.df.empty or "is_tagged" not in self.df.columns:
            return pd.DataFrame()
        return self.df[_parse_tagged_mask(self.df)].copy()


def validate_phase1_export(dir_path: Union[str, Path]) -> Phase1ExportMetadata:
    """Validate that a folder contains valid Phase 1 curation export artifacts.

    Args:
        dir_path: Directory path to inspect.

    Returns:
        Phase1ExportMetadata containing verified paths, unit counts, and dataframe.

    Raises:
        FileNotFoundError: If dir_path is not an existing directory.
        MissingPhase1ExportError: If no *_opto_curated.csv file exists in the directory.
        InvalidPhase1ExportError: If the export CSV is missing required columns.
    """
    path = Path(dir_path)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Directory does not exist or is not a folder: {dir_path}")

    # Search for Phase 1 export CSV (<session_id>_opto_curated.csv)
    csv_files = list(path.glob("*_opto_curated.csv"))
    if not csv_files:
        raise MissingPhase1ExportError(
            f"Directory '{path.name}' does not contain a Phase 1 curation export file (<session_id>_opto_curated.csv).\n\n"
            "Please open this recording in Phase 1 (Optotagged Cell Detection), run the curation workflow, "
            "and click 'Export Detection Session' before importing into Phase 2."
        )

    matched_csv: Optional[Path] = None
    for cf in csv_files:
        if cf.name.startswith(path.name):
            matched_csv = cf
            break
    if matched_csv is None:
        matched_csv = csv_files[0]

    try:
        df = pd.read_csv(matched_csv)
    except Exception as err:
        raise InvalidPhase1ExportError(f"Failed to read Phase 1 export CSV '{matched_csv.name}': {err}") from err

    required_cols = [
        "global_unit_id",
        "session_id",
        "unit_id",
        "latency_ms",
        "jitter_ms",
        "reliability",
        "salt_p_value",
        "is_tagged",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise InvalidPhase1ExportError(
            f"Export CSV '{matched_csv.name}' is missing required Phase 1 columns: {missing_cols}"
        )

    # Session ID derivation
    if len(df) > 0 and "session_id" in df and pd.notna(df["session_id"].iloc[0]):
        session_id = str(df["session_id"].iloc[0])
    else:
        session_id = matched_csv.stem.replace("_opto_curated", "")

    # Locate plots directory
    plots_dir = path / f"{session_id}_plots"
    if not plots_dir.is_dir():
        fallback_plots = path / "plots"
        if fallback_plots.is_dir():
            plots_dir = fallback_plots
        else:
            # Fallback to designated plots_dir path even if currently unpopulated
            plots_dir = path / f"{session_id}_plots"

    # Locate arrays directory
    arrays_dir = path / f"{session_id}_arrays"
    if not arrays_dir.is_dir():
        fallback_arrays = path / "arrays"
        if fallback_arrays.is_dir():
            arrays_dir = fallback_arrays
        else:
            arrays_dir = path / f"{session_id}_arrays"

    total_units = len(df)
    tagged_units = int(_parse_tagged_mask(df).sum())

    # Discover stimulus protocol name from persisted arrays metadata first, then stimulus files
    proto_name: Optional[str] = None
    npz_candidates = [arrays_dir / f"{session_id}_spikes.npz", path / f"{session_id}_spikes.npz"]
    if path.parent != path:
        npz_candidates.append(path.parent / f"{session_id}_spikes.npz")
    for npz_file in npz_candidates:
        if npz_file.is_file():
            try:
                with np.load(npz_file, allow_pickle=True) as npz_data:
                    if "protocol_name" in npz_data.files:
                        raw_name = str(np.asarray(npz_data["protocol_name"]).item()).strip()
                        if raw_name:
                            proto_name = raw_name
                            break
            except Exception:
                pass

    if not proto_name:
        stim_reader = MatNpyStimulusReader()
        discovered_proto = stim_reader.discover(path)
        if discovered_proto:
            proto_name = stim_reader._infer_protocol_name(discovered_proto[0])
        else:
            proto_name = "OPTS03"

    return Phase1ExportMetadata(
        session_id=session_id,
        folder_path=path,
        csv_path=matched_csv,
        plots_dir=plots_dir,
        total_units=total_units,
        tagged_units=tagged_units,
        protocol_name=proto_name,
        df=df,
        arrays_dir=arrays_dir,
    )


def load_phase1_session_arrays(
    meta: Phase1ExportMetadata,
) -> Tuple[Dict[int, np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """Load unit spike trains and stimulus pulses from ``_spikes.npz`` or raw Kilosort/stimulus fallback."""
    spikes_dict: Dict[int, np.ndarray] = {}
    pulse_onsets: Optional[np.ndarray] = None
    pulse_offsets: Optional[np.ndarray] = None

    npz_candidates: List[Path] = []
    if meta.arrays_dir and meta.arrays_dir.is_dir():
        npz_candidates.extend(meta.arrays_dir.glob(f"{meta.session_id}_spikes.npz"))
        if not npz_candidates:
            npz_candidates.extend(meta.arrays_dir.glob("*_spikes.npz"))
    if not npz_candidates:
        arr_sub = meta.folder_path / f"{meta.session_id}_arrays"
        if arr_sub.is_dir():
            npz_candidates.extend(arr_sub.glob(f"{meta.session_id}_spikes.npz"))
            if not npz_candidates:
                npz_candidates.extend(arr_sub.glob("*_spikes.npz"))
    if not npz_candidates:
        npz_candidates.extend(meta.folder_path.glob(f"{meta.session_id}_spikes.npz"))
    if not npz_candidates:
        npz_candidates.extend(meta.folder_path.glob("*_spikes.npz"))
    if not npz_candidates and meta.folder_path.parent.is_dir():
        npz_candidates.extend(meta.folder_path.parent.glob(f"{meta.session_id}_spikes.npz"))

    npz_dict: Dict[str, Any] = {}
    if npz_candidates:
        try:
            with np.load(npz_candidates[0], allow_pickle=True) as data:
                for k in data.files:
                    npz_dict[k] = data[k]
                if "pulse_onsets" in npz_dict:
                    pulse_onsets = np.asarray(npz_dict["pulse_onsets"], dtype=np.float64)
                if "pulse_offsets" in npz_dict:
                    pulse_offsets = np.asarray(npz_dict["pulse_offsets"], dtype=np.float64)
                for k in npz_dict:
                    if k.startswith("unit_"):
                        try:
                            spikes_dict[int(k.replace("unit_", ""))] = np.asarray(npz_dict[k], dtype=np.float64)
                        except ValueError:
                            pass
        except Exception:
            pass

    if not spikes_dict or pulse_onsets is None:
        try:
            raw_dirs = [meta.folder_path]
            if (
                meta.folder_path.name.lower() in ("sorted_dir", "kilosort", "sorter_output")
                and meta.folder_path.parent.is_dir()
            ):
                raw_dirs.append(meta.folder_path.parent)

            phy_reader = PhyKilosortReader()
            s_rate = phy_reader._read_sampling_rate(raw_dirs[0])
            for r_dir in raw_dirs:
                if (r_dir / "params.py").is_file():
                    s_rate = phy_reader._read_sampling_rate(r_dir)
                    break

            st_file = sc_file = st_dir = None
            for r_dir in raw_dirs:
                if (r_dir / "spike_times.npy").is_file() and (r_dir / "spike_clusters.npy").is_file():
                    st_file = r_dir / "spike_times.npy"
                    sc_file = r_dir / "spike_clusters.npy"
                    st_dir = r_dir
                    break

            if st_file is not None and not spikes_dict:
                s_rate = phy_reader._read_sampling_rate(st_dir)
                st_raw = np.load(st_file).flatten()
                sc_raw = np.load(sc_file).flatten()
                st_sec = timestamps_to_seconds(st_raw, s_rate)

                target_uids = set()
                target_df = meta.tagged_df if not meta.tagged_df.empty else meta.df
                if "unit_id" in target_df.columns:
                    for u in target_df["unit_id"].dropna().unique():
                        try:
                            target_uids.add(int(u))
                        except (ValueError, TypeError):
                            pass

                for uid in target_uids:
                    spks = st_sec[sc_raw == uid]
                    if len(spks) > 1 and not np.all(spks[:-1] <= spks[1:]):
                        spks = np.sort(spks)
                    spikes_dict[uid] = spks

            if pulse_onsets is None or pulse_offsets is None:
                stim_reader = MatNpyStimulusReader()
                discovered = stim_reader.discover(meta.folder_path)
                if discovered:
                    stim_proto = stim_reader.load(discovered[0], sampling_rate=s_rate)
                    if pulse_onsets is None:
                        pulse_onsets = np.asarray(stim_proto.onsets, dtype=np.float64)
                    if pulse_offsets is None and getattr(stim_proto, "offsets", None) is not None:
                        pulse_offsets = np.asarray(stim_proto.offsets, dtype=np.float64)

            if spikes_dict and pulse_onsets is not None:
                try:
                    save_arrays_dir = meta.folder_path / f"{meta.session_id}_arrays"
                    save_arrays_dir.mkdir(parents=True, exist_ok=True)
                    save_npz_path = save_arrays_dir / f"{meta.session_id}_spikes.npz"
                    spikes_export: Dict[str, Any] = {"pulse_onsets": pulse_onsets}
                    if pulse_offsets is not None:
                        spikes_export["pulse_offsets"] = pulse_offsets
                    if meta.protocol_name:
                        spikes_export["protocol_name"] = np.array(str(meta.protocol_name))
                    for uid_val, spk_arr in spikes_dict.items():
                        spikes_export[f"unit_{uid_val}"] = spk_arr
                    np.savez_compressed(save_npz_path, **spikes_export)
                except Exception:
                    pass
        except Exception:
            pass

    return spikes_dict, pulse_onsets, pulse_offsets, npz_dict
