"""Clustering Project save and load routines for Phase 2 multi-experiment workspaces."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import yaml

from optocurator.analysis.clustering import (
    AffinityPropagationParameters,
    ClusteringParameters,
    EmbeddingParameters,
    FeatureConfiguration,
)


class ProjectError(Exception):
    """Base exception for OptoCurator project operations."""


class ProjectNotFoundError(ProjectError):
    """Raised when a clustering project directory or manifest file is not found."""


class InvalidProjectConfigError(ProjectError):
    """Raised when a clustering project configuration or manifest is invalid or corrupted."""


class MissingExperimentDirectoryError(ProjectError):
    """Raised when a referenced experiment directory cannot be resolved or found."""


@dataclass
class ExperimentRef:
    """Reference to an ingested Phase 1 experiment in a clustering project."""

    session_id: str
    path: str
    protocol_name: str
    total_units: int
    tagged_units: int


@dataclass
class UnitAssignmentRecord:
    """Record of a unit's cluster assignment and curation state."""

    global_unit_id: str
    session_id: str
    unit_id: int
    cluster_id: int
    curated_status: str
    auto_confidence: Optional[float] = None


@dataclass
class ClusteringProjectData:
    """Root metadata contract for a Phase 2 response clustering project."""

    schema_version: str = "1.0"
    project_type: str = "optocurator_clustering_project"
    created_at: str = ""
    updated_at: str = ""
    experiments: List[ExperimentRef] = field(default_factory=list)
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    clusters: Dict[str, Any] = field(default_factory=dict)
    active_learning_state: Dict[str, Any] = field(default_factory=dict)
    unit_assignments: List[UnitAssignmentRecord] = field(default_factory=list)


def _to_serializable(val: Any) -> Any:
    """Recursively convert dataclasses, tuples, and numpy primitives to standard YAML-safe types."""
    if hasattr(val, "item") and callable(getattr(val, "item")):
        return val.item()
    if hasattr(val, "__dataclass_fields__"):
        return _to_serializable(asdict(val))
    if isinstance(val, (tuple, list)):
        return [_to_serializable(x) for x in val]
    if isinstance(val, dict):
        return {
            (k if isinstance(k, (int, float, bool)) else str(k)): _to_serializable(v)
            for k, v in val.items()
        }
    return val


def compute_portable_path(project_dir: Path, exp_dir: Path) -> str:
    """Return relative POSIX path if same drive/root, otherwise absolute POSIX string."""
    proj = Path(project_dir).resolve()
    exp = Path(exp_dir).resolve()
    try:
        if proj.drive.lower() != exp.drive.lower():
            return exp.as_posix()
        rel = os.path.relpath(exp, proj)
        return Path(rel).as_posix()
    except (ValueError, Exception):
        return exp.as_posix()


def resolve_experiment_path(project_dir: Path, path_str: str) -> Path:
    """Check relative to project_dir first, then absolute. If neither exists, raise MissingExperimentDirectoryError."""
    proj = Path(project_dir).resolve()

    # Check relative to project_dir first
    candidate_rel = (proj / path_str).resolve()
    if candidate_rel.is_dir():
        return candidate_rel

    # Check as absolute path
    candidate_abs = Path(path_str).resolve()
    if candidate_abs.is_dir():
        return candidate_abs

    raise MissingExperimentDirectoryError(
        f"Experiment directory '{path_str}' could not be resolved relative to project directory "
        f"'{proj}' or as an absolute path."
    )


def serialize_model_parameters(
    feature_config: Optional[FeatureConfiguration] = None,
    embedding_params: Optional[EmbeddingParameters] = None,
    ap_params: Optional[AffinityPropagationParameters] = None,
    clustering_params: Optional[ClusteringParameters] = None,
    window_ms: Optional[Tuple[float, float]] = None,
    bin_size_ms: Optional[float] = None,
    raster_order_by: Optional[str] = None,
    raster_marker_size: Optional[float] = None,
    auto_assign_threshold: Optional[float] = None,
    auto_assign_enabled: Optional[bool] = None,
    active_learning_enabled: Optional[bool] = None,
    psth_smooth: Optional[bool] = None,
    psth_gaussian_sigma_ms: Optional[float] = None,
    cluster_psth_norm: Optional[str] = None,
) -> Dict[str, Any]:
    """Serialize all model, view, and active learning parameters to a dictionary."""
    fc = feature_config or FeatureConfiguration()
    emb = embedding_params or EmbeddingParameters()
    ap = ap_params or AffinityPropagationParameters()
    clust = clustering_params or ClusteringParameters(feature_config=fc)

    clust_dict = asdict(clust)
    if "feature_config" in clust_dict:
        clust_dict.pop("feature_config", None)

    smooth_val = bool(psth_smooth) if psth_smooth is not None else getattr(fc, "psth_smooth", True)
    sigma_val = float(psth_gaussian_sigma_ms) if psth_gaussian_sigma_ms is not None else getattr(fc, "psth_gaussian_sigma_ms", 2.0)
    norm_val = str(cluster_psth_norm) if cluster_psth_norm is not None else getattr(clust, "cluster_psth_norm", "probability")

    return {
        "feature_config": asdict(fc),
        "embedding_params": asdict(emb),
        "ap_params": asdict(ap),
        "clustering_params": clust_dict,
        "window_ms": list(window_ms) if window_ms is not None else [-30.0, 50.0],
        "bin_size_ms": float(bin_size_ms) if bin_size_ms is not None else 2.0,
        "raster_order_by": str(raster_order_by) if raster_order_by is not None else "presentation",
        "raster_marker_size": float(raster_marker_size) if raster_marker_size is not None else 3.0,
        "auto_assign_threshold": float(auto_assign_threshold) if auto_assign_threshold is not None else 0.85,
        "auto_assign_enabled": bool(auto_assign_enabled) if auto_assign_enabled is not None else False,
        "active_learning_enabled": bool(active_learning_enabled) if active_learning_enabled is not None else True,
        "psth_smooth": smooth_val,
        "psth_gaussian_sigma_ms": sigma_val,
        "cluster_psth_norm": norm_val,
    }


def deserialize_model_parameters(data: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct all model, view, and active learning parameters from serialized dictionary."""
    # 1. FeatureConfiguration
    fc_dict = dict(data.get("feature_config", {}))
    if "time_window_ms" in fc_dict and isinstance(fc_dict["time_window_ms"], (list, tuple)):
        fc_dict["time_window_ms"] = tuple(float(x) for x in fc_dict["time_window_ms"])
    if "psth_normalization" in fc_dict and str(fc_dict["psth_normalization"]).lower() in ("zscore_population", "baseline_zscore"):
        fc_dict["psth_normalization"] = "area_normalized"
    valid_fc_fields = set(FeatureConfiguration.__dataclass_fields__.keys())
    filtered_fc = {k: v for k, v in fc_dict.items() if k in valid_fc_fields}
    feature_config = FeatureConfiguration(**filtered_fc)

    # 2. EmbeddingParameters
    emb_dict = dict(data.get("embedding_params", {}))
    valid_emb_fields = set(EmbeddingParameters.__dataclass_fields__.keys())
    filtered_emb = {k: v for k, v in emb_dict.items() if k in valid_emb_fields}
    embedding_params = EmbeddingParameters(**filtered_emb)

    # 3. AffinityPropagationParameters
    ap_dict = dict(data.get("ap_params", {}))
    valid_ap_fields = set(AffinityPropagationParameters.__dataclass_fields__.keys())
    filtered_ap = {k: v for k, v in ap_dict.items() if k in valid_ap_fields}
    ap_params = AffinityPropagationParameters(**filtered_ap)

    # 4. ClusteringParameters
    clust_dict = dict(data.get("clustering_params", {}))
    if "presented_window_ms" in clust_dict and isinstance(clust_dict["presented_window_ms"], (list, tuple)):
        clust_dict["presented_window_ms"] = tuple(float(x) for x in clust_dict["presented_window_ms"])
    valid_clust_fields = set(ClusteringParameters.__dataclass_fields__.keys())
    filtered_clust = {k: v for k, v in clust_dict.items() if k in valid_clust_fields}
    filtered_clust["feature_config"] = feature_config
    clustering_params = ClusteringParameters(**filtered_clust)

    # 5. View parameters
    raw_win = data.get("window_ms", data.get("view_params", {}).get("window_ms", (-30.0, 50.0)))
    if isinstance(raw_win, (list, tuple)):
        window_ms = tuple(float(x) for x in raw_win)
    else:
        window_ms = (-30.0, 50.0)

    bin_size_ms = float(data.get("bin_size_ms", data.get("view_params", {}).get("bin_size_ms", 2.0)))
    raster_order_by = str(data.get("raster_order_by", data.get("view_params", {}).get("raster_order_by", "presentation")))
    raster_marker_size = float(data.get("raster_marker_size", data.get("view_params", {}).get("raster_marker_size", 3.0)))

    # 6. Active learning parameters
    auto_assign_threshold = float(data.get("auto_assign_threshold", data.get("active_learning_params", {}).get("auto_assign_threshold", 0.85)))
    auto_assign_enabled = bool(data.get("auto_assign_enabled", data.get("active_learning_params", {}).get("auto_assign_enabled", False)))
    active_learning_enabled = bool(data.get("active_learning_enabled", data.get("active_learning_params", {}).get("active_learning_enabled", True)))

    # 7. Smoothing parameters
    psth_smooth = bool(data.get("psth_smooth", getattr(feature_config, "psth_smooth", True)))
    psth_gaussian_sigma_ms = float(data.get("psth_gaussian_sigma_ms", getattr(feature_config, "psth_gaussian_sigma_ms", 2.0)))

    # 8. Normalization parameters
    cluster_psth_norm = str(data.get("cluster_psth_norm", getattr(clustering_params, "cluster_psth_norm", "probability")))
    clustering_params.cluster_psth_norm = cluster_psth_norm

    return {
        "feature_config": feature_config,
        "embedding_params": embedding_params,
        "ap_params": ap_params,
        "clustering_params": clustering_params,
        "window_ms": window_ms,
        "bin_size_ms": bin_size_ms,
        "raster_order_by": raster_order_by,
        "raster_marker_size": raster_marker_size,
        "auto_assign_threshold": auto_assign_threshold,
        "auto_assign_enabled": auto_assign_enabled,
        "active_learning_enabled": active_learning_enabled,
        "psth_smooth": psth_smooth,
        "psth_gaussian_sigma_ms": psth_gaussian_sigma_ms,
        "cluster_psth_norm": cluster_psth_norm,
    }


def save_clustering_project(
    project_dir: Union[str, Path],
    data: ClusteringProjectData,
    pooled_df: Optional[pd.DataFrame] = None,
) -> Path:
    """Save clustering project manifest and optional curated units CSV to project_dir.

    Args:
        project_dir: Target folder to store project files.
        data: ClusteringProjectData containing project configuration and state.
        pooled_df: Optional pooled curated units DataFrame to write as pooled_units_curated.csv.

    Returns:
        Path to the saved clustering_project.yaml manifest.
    """
    target_dir = Path(project_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest_dict = {
        "schema_version": data.schema_version,
        "project_type": data.project_type,
        "created_at": data.created_at or datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "experiments": [
            {
                "session_id": exp.session_id,
                "path": exp.path,
                "protocol_name": exp.protocol_name,
                "total_units": exp.total_units,
                "tagged_units": exp.tagged_units,
            }
            if isinstance(exp, ExperimentRef) else exp
            for exp in data.experiments
        ],
        "model_parameters": data.model_parameters,
        "clusters": data.clusters,
        "active_learning_state": data.active_learning_state,
        "unit_assignments": [
            {
                "global_unit_id": u.global_unit_id,
                "session_id": u.session_id,
                "unit_id": u.unit_id,
                "cluster_id": u.cluster_id,
                "curated_status": u.curated_status,
                "auto_confidence": u.auto_confidence,
            }
            if isinstance(u, UnitAssignmentRecord) else u
            for u in data.unit_assignments
        ],
    }

    clean_manifest = _to_serializable(manifest_dict)
    yaml_file = target_dir / "clustering_project.yaml"
    temp_yaml = target_dir / "clustering_project.yaml.tmp"
    with open(temp_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(clean_manifest, f, default_flow_style=False, sort_keys=False)
    os.replace(temp_yaml, yaml_file)

    if pooled_df is not None:
        csv_path = target_dir / "pooled_units_curated.csv"
        temp_csv = target_dir / "pooled_units_curated.csv.tmp"
        pooled_df.to_csv(temp_csv, index=False)
        os.replace(temp_csv, csv_path)

    return yaml_file


def load_clustering_project(project_dir: Union[str, Path]) -> ClusteringProjectData:
    """Load and validate a clustering project from directory or YAML file.

    Args:
        project_dir: Directory containing clustering_project.yaml, or direct path to YAML file.

    Returns:
        ClusteringProjectData deserialized from the manifest.

    Raises:
        ProjectNotFoundError: If project directory or manifest file cannot be found.
        InvalidProjectConfigError: If project manifest is malformed or invalid.
    """
    path = Path(project_dir).resolve()
    if path.is_file():
        yaml_file = path
    elif path.is_dir():
        yaml_file = path / "clustering_project.yaml"
    else:
        raise ProjectNotFoundError(f"Project directory or file does not exist: {path}")

    if not yaml_file.exists():
        raise ProjectNotFoundError(f"Clustering project manifest '{yaml_file.name}' not found at: {yaml_file}")

    try:
        with open(yaml_file, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except Exception as err:
        raise InvalidProjectConfigError(f"Failed to parse project manifest '{yaml_file.name}': {err}") from err

    if not isinstance(raw, dict):
        raise InvalidProjectConfigError(f"Invalid project manifest '{yaml_file.name}': expected dictionary mapping.")

    proj_type = raw.get("project_type")
    if proj_type != "optocurator_clustering_project":
        raise InvalidProjectConfigError(
            f"Invalid project_type '{proj_type}' in manifest '{yaml_file.name}'. "
            "Expected 'optocurator_clustering_project'."
        )

    if "schema_version" not in raw:
        raise InvalidProjectConfigError(f"Missing 'schema_version' in project manifest '{yaml_file.name}'.")

    exp_list: List[ExperimentRef] = []
    for exp_raw in raw.get("experiments", []):
        if not isinstance(exp_raw, dict):
            continue
        exp_list.append(ExperimentRef(
            session_id=str(exp_raw.get("session_id", "")),
            path=str(exp_raw.get("path", "")),
            protocol_name=str(exp_raw.get("protocol_name", "OPTS03")),
            total_units=int(exp_raw.get("total_units", 0)),
            tagged_units=int(exp_raw.get("tagged_units", 0)),
        ))

    unit_list: List[UnitAssignmentRecord] = []
    for u_raw in raw.get("unit_assignments", []):
        if not isinstance(u_raw, dict):
            continue
        conf_val = u_raw.get("auto_confidence")
        unit_list.append(UnitAssignmentRecord(
            global_unit_id=str(u_raw.get("global_unit_id", "")),
            session_id=str(u_raw.get("session_id", "")),
            unit_id=int(u_raw.get("unit_id", 0)),
            cluster_id=int(u_raw.get("cluster_id", -1)),
            curated_status=str(u_raw.get("curated_status", "Unassigned")),
            auto_confidence=float(conf_val) if conf_val is not None else None,
        ))

    clusters_dict = dict(raw.get("clusters", {}))
    if "names" in clusters_dict and isinstance(clusters_dict["names"], dict):
        clusters_dict["names"] = {
            (int(k) if str(k).lstrip("-").isdigit() else k): str(v)
            for k, v in clusters_dict["names"].items()
        }
    if "colors" in clusters_dict and isinstance(clusters_dict["colors"], dict):
        clusters_dict["colors"] = {
            (int(k) if str(k).lstrip("-").isdigit() else k): str(v)
            for k, v in clusters_dict["colors"].items()
        }
    if "hotkeys" in clusters_dict and isinstance(clusters_dict["hotkeys"], dict):
        clusters_dict["hotkeys"] = {
            (int(k) if str(k).lstrip("-").isdigit() else k): str(v)
            for k, v in clusters_dict["hotkeys"].items()
        }
    if "card_order" in clusters_dict and isinstance(clusters_dict["card_order"], list):
        clusters_dict["card_order"] = [
            (int(x) if str(x).lstrip("-").isdigit() else x)
            for x in clusters_dict["card_order"]
        ]

    return ClusteringProjectData(
        schema_version=str(raw.get("schema_version", "1.0")),
        project_type=str(raw.get("project_type", "optocurator_clustering_project")),
        created_at=str(raw.get("created_at", "")),
        updated_at=str(raw.get("updated_at", "")),
        experiments=exp_list,
        model_parameters=dict(raw.get("model_parameters", {})),
        clusters=clusters_dict,
        active_learning_state=dict(raw.get("active_learning_state", {})),
        unit_assignments=unit_list,
    )
