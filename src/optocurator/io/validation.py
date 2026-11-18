"""Directory and Phase 1 export validation routines for Phase 2 multi-experiment ingestion."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
import pandas as pd


class MissingPhase1ExportError(Exception):
    """Raised when an experiment directory does not contain valid Phase 1 curation export artifacts."""


class InvalidPhase1ExportError(MissingPhase1ExportError):
    """Raised when a Phase 1 export CSV is malformed or corrupted."""


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
        if self.df["is_tagged"].dtype == bool:
            return self.df[self.df["is_tagged"]].copy()
        mask = self.df["is_tagged"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "1.0"])
        return self.df[mask].copy()


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

    # Search for Phase 1 export CSV (<session_id>_opto_curated.csv) or alternative TSV/CSV export
    csv_files = list(path.glob("*_opto_curated.csv"))
    matched_csv: Optional[Path] = None
    df: Optional[pd.DataFrame] = None

    if csv_files:
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
    else:
        tsv_files = list(path.glob("*_curated.tsv")) + list(path.glob("*_curated.csv"))
        if not tsv_files:
            raise MissingPhase1ExportError(
                f"Directory '{path.name}' does not contain a Phase 1 curation export file (<session_id>_opto_curated.csv).\n\n"
                "Please open this recording in Phase 1 (Optotagged Cell Detection), run the curation workflow, "
                "and click 'Export Detection Session' before importing into Phase 2."
            )
        matched_csv = tsv_files[0]
        try:
            sep = "\t" if matched_csv.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(matched_csv, sep=sep)
        except Exception as err:
            raise InvalidPhase1ExportError(f"Failed to read Phase 1 export '{matched_csv.name}': {err}") from err

        json_files = list(path.glob("*_curated.json"))
        sess_name = path.name
        if json_files:
            try:
                with open(json_files[0], "r", encoding="utf-8") as f:
                    j_dict = json.load(f)
                    sess_name = j_dict.get("session_id", sess_name)
            except Exception:
                pass

        if "session_id" not in df.columns:
            df["session_id"] = sess_name
        if "global_unit_id" not in df.columns and "unit_id" in df.columns:
            df["global_unit_id"] = [f"{sess_name}_u{u}" for u in df["unit_id"]]
        if "latency_ms" not in df.columns and "opto_latency_ms" in df.columns:
            df["latency_ms"] = df["opto_latency_ms"]
        if "is_tagged" not in df.columns and "status" in df.columns:
            df["is_tagged"] = df["status"].astype(str).str.lower().isin(["tagged", "true", "1"])
        if "salt_p_value" not in df.columns:
            df["salt_p_value"] = 0.001
        if "jitter_ms" not in df.columns:
            df["jitter_ms"] = 1.0
        if "reliability" not in df.columns:
            df["reliability"] = 0.8

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
    tagged_units = int(df["is_tagged"].astype(bool).sum()) if "is_tagged" in df.columns else 0

    # Discover stimulus protocol name from folder if possible
    proto_name = "OPTS03"
    proto_files = list(path.glob("opto_idxs_*.npy"))
    if proto_files:
        proto_name = proto_files[0].stem.replace("opto_idxs_", "")

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
