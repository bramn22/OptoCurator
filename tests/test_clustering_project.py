"""Comprehensive unit and integration tests for Phase 2 Clustering Project Save & Load functionality."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import pytest
import yaml

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from optocurator.analysis.clustering import (
    AffinityPropagationParameters,
    ClusteringParameters,
    EmbeddingParameters,
    FeatureConfiguration,
)
from optocurator.io.project import (
    ClusteringProjectData,
    ExperimentRef,
    InvalidProjectConfigError,
    MissingExperimentDirectoryError,
    ProjectError,
    ProjectNotFoundError,
    UnitAssignmentRecord,
    compute_portable_path,
    deserialize_model_parameters,
    load_clustering_project,
    resolve_experiment_path,
    save_clustering_project,
    serialize_model_parameters,
)
from optocurator.ui.app import MainWindow
from optocurator.ui.view_clusters import ClusteringView


def _create_dummy_png(target_file: Path, width: int = 120, height: int = 80) -> Path:
    """Create a minimal valid PNG file on disk for unit plot inspection."""
    target_file.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    im = Image.new("RGB", (width, height), color=(37, 99, 235))
    im.save(str(target_file), "PNG")
    return target_file


def _build_test_phase1_export(
    folder: Path,
    session_id: str = "session_01",
    n_units: int = 4,
    n_tagged: int = 2,
    with_plots: bool = True,
) -> Path:
    """Helper to build a valid Phase 1 exported recording directory."""
    session_dir = folder / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = session_dir / f"{session_id}_plots"
    if with_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(1, n_units + 1):
        is_tag = (i <= n_tagged)
        gid = f"{session_id}_u{i:04d}"
        plot_rel = f"{session_id}_plots/{gid}_psth_raster.png"
        rows.append({
            "global_unit_id": gid,
            "session_id": session_id,
            "unit_id": i,
            "latency_ms": 3.0 + i * 1.2,
            "jitter_ms": 1.1 + i * 0.1,
            "reliability": 0.85 - i * 0.1,
            "waveform_correlation": 0.94,
            "salt_p_value": 0.002 * i,
            "is_tagged": is_tag,
            "cluster_id": -1,
            "curated_label": "Tagged" if is_tag else "unassigned",
            "plot_path": plot_rel,
            "channel": i,
            "depth": 200.0 * i,
        })
        if with_plots:
            _create_dummy_png(session_dir / plot_rel)

    df = pd.DataFrame(rows)
    csv_path = session_dir / f"{session_id}_opto_curated.csv"
    df.to_csv(csv_path, index=False)
    np.save(session_dir / "opto_idxs_OPTS03.npy", np.array([50, 100, 150]))
    return session_dir


# ============================================================================
# 1. UNIT TESTS: optocurator.io.project
# ============================================================================


def test_portable_and_resolved_paths(tmp_path: Path):
    """Verify compute_portable_path produces relative paths and resolve_experiment_path handles paths."""
    proj_dir = tmp_path / "workspaces" / "project_01"
    proj_dir.mkdir(parents=True, exist_ok=True)

    # 1. Same-drive relative path
    exp_dir1 = tmp_path / "recordings" / "mouseA_session1"
    exp_dir1.mkdir(parents=True, exist_ok=True)
    portable1 = compute_portable_path(proj_dir, exp_dir1)
    assert not Path(portable1).is_absolute()
    assert portable1.replace("\\", "/") == "../../recordings/mouseA_session1"

    # Subdirectory of project
    exp_dir2 = proj_dir / "sub_experiments" / "mouseB_session2"
    exp_dir2.mkdir(parents=True, exist_ok=True)
    portable2 = compute_portable_path(proj_dir, exp_dir2)
    assert portable2.replace("\\", "/") == "sub_experiments/mouseB_session2"

    # 2. Resolving relative paths
    resolved_rel1 = resolve_experiment_path(proj_dir, portable1)
    assert resolved_rel1 == exp_dir1.resolve()

    resolved_rel2 = resolve_experiment_path(proj_dir, portable2)
    assert resolved_rel2 == exp_dir2.resolve()

    # 3. Resolving absolute paths
    resolved_abs = resolve_experiment_path(proj_dir, str(exp_dir1.resolve()))
    assert resolved_abs == exp_dir1.resolve()

    # 4. Non-existent path raises MissingExperimentDirectoryError
    with pytest.raises(MissingExperimentDirectoryError) as exc_info:
        resolve_experiment_path(proj_dir, "nonexistent/rel/path")
    assert "could not be resolved" in str(exc_info.value)


def test_portable_path_preserves_symlinks_and_resolves_backslashes(tmp_path: Path):
    """compute_portable_path must preserve symlink/junction relative paths and resolve_experiment_path handles backslashes (Bug IO8)."""
    import os

    proj_dir = tmp_path / "workspace" / "project_01"
    proj_dir.mkdir(parents=True, exist_ok=True)
    target_dir = tmp_path / "external_drive" / "session_target"
    target_dir.mkdir(parents=True, exist_ok=True)
    linked_dir = proj_dir / "linked_session"

    try:
        os.symlink(target_dir, linked_dir, target_is_directory=True)
    except OSError:
        if os.name == "nt":
            import _winapi

            _winapi.CreateJunction(str(target_dir), str(linked_dir))
        else:
            pytest.skip("Symlinks not supported in this environment")

    portable = compute_portable_path(proj_dir, linked_dir)
    assert portable == "linked_session"
    assert resolve_experiment_path(proj_dir, portable) == target_dir.resolve()
    assert resolve_experiment_path(proj_dir, "..\\..\\external_drive\\session_target") == target_dir.resolve()


def test_model_parameters_serialization_roundtrip():
    """Verify serialization and deserialization of all model, view, and active learning parameters."""
    fc = FeatureConfiguration(
        use_latency=True,
        use_jitter=False,
        use_reliability=True,
        use_salt=True,
        use_peak_response_rate=True,
        use_time_to_response=False,
        response_std_threshold=3.0,
        use_psth_profile=True,
        psth_normalization="area_normalized",
        psth_use_pca=True,
        psth_pca_variance_explained=0.85,
        time_window_ms=(-20.0, 40.0),
        bin_size_ms=1.5,
        weight_latency=2.0,
        weight_reliability=1.5,
        weight_salt=0.5,
        weight_psth_profile=1.2,
    )
    emb = EmbeddingParameters(
        method="PCA",
        pca_component_x=1,
        pca_component_y=2,
        umap_n_neighbors=25,
        umap_min_dist=0.15,
        umap_metric="cosine",
    )
    ap = AffinityPropagationParameters(
        damping=0.85,
        similarity_percentile=40.0,
        min_cluster_size=3,
        preserve_curated=False,
    )
    clust = ClusteringParameters(
        algorithm="hierarchical",
        n_clusters=4,
        linkage="complete",
        metric="cosine",
        presented_window_ms=(-25.0, 55.0),
        presented_bin_size_ms=1.5,
        feature_config=fc,
    )

    serialized = serialize_model_parameters(
        feature_config=fc,
        embedding_params=emb,
        ap_params=ap,
        clustering_params=clust,
        window_ms=(-25.0, 55.0),
        bin_size_ms=1.5,
        raster_order_by="latency",
        raster_marker_size=4.5,
        auto_assign_threshold=0.90,
        auto_assign_enabled=True,
        active_learning_enabled=False,
    )

    assert isinstance(serialized, dict)
    assert serialized["window_ms"] == [-25.0, 55.0]
    assert serialized["bin_size_ms"] == 1.5
    assert serialized["raster_order_by"] == "latency"
    assert serialized["raster_marker_size"] == 4.5
    assert serialized["auto_assign_threshold"] == 0.90
    assert serialized["auto_assign_enabled"] is True
    assert serialized["active_learning_enabled"] is False

    deserialized = deserialize_model_parameters(serialized)

    assert deserialized["feature_config"] == fc
    assert deserialized["embedding_params"] == emb
    assert deserialized["ap_params"] == ap

    # ClusteringParameters equality
    d_clust = deserialized["clustering_params"]
    assert d_clust.algorithm == clust.algorithm
    assert d_clust.n_clusters == clust.n_clusters
    assert d_clust.linkage == clust.linkage
    assert d_clust.metric == clust.metric
    assert d_clust.presented_window_ms == clust.presented_window_ms
    assert d_clust.presented_bin_size_ms == clust.presented_bin_size_ms
    assert d_clust.feature_config == fc

    # View and active learning equality
    assert deserialized["window_ms"] == (-25.0, 55.0)
    assert isinstance(deserialized["window_ms"], tuple)
    assert deserialized["bin_size_ms"] == 1.5
    assert deserialized["raster_order_by"] == "latency"
    assert deserialized["raster_marker_size"] == 4.5
    assert deserialized["auto_assign_threshold"] == 0.90
    assert deserialized["auto_assign_enabled"] is True
    assert deserialized["active_learning_enabled"] is False


def test_save_and_load_clustering_project(tmp_path: Path):
    """Verify end-to-end saving and loading of a clustering project manifest and curated CSV."""
    proj_dir = tmp_path / "sample_clustering_project"

    exp_ref = ExperimentRef(
        session_id="session_01",
        path="data/session_01",
        protocol_name="OPTS03",
        total_units=8,
        tagged_units=4,
    )
    records = [
        UnitAssignmentRecord("session_01_u0001", "session_01", 1, 0, "Curated", 0.95),
        UnitAssignmentRecord("session_01_u0002", "session_01", 2, 1, "Auto-assigned", 0.88),
        UnitAssignmentRecord("session_01_u0003", "session_01", 3, -1, "Unassigned", None),
    ]
    clusters = {
        "names": {0: "Narrow Spike", 1: "Wide Burst"},
        "colors": {0: "#2563EB", 1: "#DC2626"},
        "card_order": [1, 0],
    }
    al_state = {
        "skipped_gids": ["session_01_u0003"],
        "active_learning_enabled": True,
        "auto_assign_enabled": True,
        "auto_assign_threshold": 0.85,
    }
    model_params = serialize_model_parameters()

    project_data = ClusteringProjectData(
        schema_version="1.0",
        project_type="optocurator_clustering_project",
        experiments=[exp_ref],
        model_parameters=model_params,
        clusters=clusters,
        active_learning_state=al_state,
        unit_assignments=records,
    )

    sample_df = pd.DataFrame([
        {
            "global_unit_id": "session_01_u0001",
            "session_id": "session_01",
            "unit_id": 1,
            "cluster_id": 0,
            "curated_status": "Curated",
            "latency_ms": 3.4,
        },
        {
            "global_unit_id": "session_01_u0002",
            "session_id": "session_01",
            "unit_id": 2,
            "cluster_id": 1,
            "curated_status": "Auto-assigned",
            "latency_ms": 12.1,
        },
    ])

    saved_yaml = save_clustering_project(proj_dir, project_data, pooled_df=sample_df)
    assert saved_yaml.is_file()
    assert (proj_dir / "clustering_project.yaml").is_file()
    assert (proj_dir / "pooled_units_curated.csv").is_file()

    # Verify reload from directory
    loaded = load_clustering_project(proj_dir)
    assert loaded.schema_version == "1.0"
    assert loaded.project_type == "optocurator_clustering_project"
    assert len(loaded.experiments) == 1
    assert loaded.experiments[0].session_id == "session_01"
    assert loaded.experiments[0].path == "data/session_01"
    assert loaded.experiments[0].protocol_name == "OPTS03"
    assert loaded.experiments[0].total_units == 8
    assert loaded.experiments[0].tagged_units == 4

    assert len(loaded.unit_assignments) == 3
    assert loaded.unit_assignments[0].global_unit_id == "session_01_u0001"
    assert loaded.unit_assignments[0].cluster_id == 0
    assert loaded.unit_assignments[0].curated_status == "Curated"
    assert pytest.approx(loaded.unit_assignments[0].auto_confidence) == 0.95
    assert loaded.unit_assignments[2].auto_confidence is None

    assert loaded.clusters["names"] == {0: "Narrow Spike", 1: "Wide Burst"}
    assert loaded.clusters["colors"] == {0: "#2563EB", 1: "#DC2626"}
    assert loaded.clusters["card_order"] == [1, 0]
    assert loaded.active_learning_state["skipped_gids"] == ["session_01_u0003"]

    # Verify reload directly from file path
    loaded_file = load_clustering_project(saved_yaml)
    assert loaded_file.project_type == "optocurator_clustering_project"
    assert len(loaded_file.experiments) == 1


def test_project_errors(tmp_path: Path):
    """Verify proper ProjectError exceptions are raised for invalid directories or manifests."""
    # 1. ProjectNotFoundError: empty directory
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(ProjectNotFoundError):
        load_clustering_project(empty_dir)

    # 2. ProjectNotFoundError: non-existent path
    with pytest.raises(ProjectNotFoundError):
        load_clustering_project(tmp_path / "does_not_exist")

    # 3. InvalidProjectConfigError: malformed YAML
    bad_yaml_dir = tmp_path / "bad_yaml"
    bad_yaml_dir.mkdir()
    with open(bad_yaml_dir / "clustering_project.yaml", "w", encoding="utf-8") as f:
        f.write("::: invalid yaml syntax :::")
    with pytest.raises(InvalidProjectConfigError) as exc_info:
        load_clustering_project(bad_yaml_dir)
    assert "Failed to parse project manifest" in str(exc_info.value)

    # 4. InvalidProjectConfigError: YAML root not a dictionary
    list_yaml_dir = tmp_path / "list_yaml"
    list_yaml_dir.mkdir()
    with open(list_yaml_dir / "clustering_project.yaml", "w", encoding="utf-8") as f:
        f.write("- item1\n- item2\n")
    with pytest.raises(InvalidProjectConfigError) as exc_info:
        load_clustering_project(list_yaml_dir)
    assert "expected dictionary mapping" in str(exc_info.value)

    # 5. InvalidProjectConfigError: wrong project_type
    wrong_type_dir = tmp_path / "wrong_type"
    wrong_type_dir.mkdir()
    with open(wrong_type_dir / "clustering_project.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"schema_version": "1.0", "project_type": "some_other_type"}, f)
    with pytest.raises(InvalidProjectConfigError) as exc_info:
        load_clustering_project(wrong_type_dir)
    assert "Invalid project_type" in str(exc_info.value)

    # 6. InvalidProjectConfigError: missing schema_version
    no_schema_dir = tmp_path / "no_schema"
    no_schema_dir.mkdir()
    with open(no_schema_dir / "clustering_project.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"project_type": "optocurator_clustering_project"}, f)
    with pytest.raises(InvalidProjectConfigError) as exc_info:
        load_clustering_project(no_schema_dir)
    assert "Missing 'schema_version'" in str(exc_info.value)


# ============================================================================
# 2. GUI INTEGRATION TESTS: ClusteringView & MainWindow
# ============================================================================


def test_clustering_view_save_and_load_project_headless(qapp, tmp_path: Path):
    """Verify ClusteringView saves and fully restores state from disk in headless Qt environment."""
    exp_dir = _build_test_phase1_export(tmp_path, "mouseA_session1", n_units=4, n_tagged=2)

    # 1. Initialize view and ingest experiment
    view = ClusteringView(load_mock=False)
    success = view.add_experiment_folder(exp_dir)
    assert success is True
    assert view.table_pooled_units.rowCount() == 2
    assert "mouseA_session1" in view.loaded_experiments

    # 2. Assign units to clusters and customize metadata
    view.assign_candidate_to_cluster(0, unit_id="mouseA_session1_u0001")
    view.assign_candidate_to_cluster(1, unit_id="mouseA_session1_u0002")

    assert view._pooled_units_map["mouseA_session1_u0001"]["cluster_id"] == 0
    assert view._pooled_units_map["mouseA_session1_u0002"]["cluster_id"] == 1
    assert view._unit_status_map["mouseA_session1_u0001"] == "Curated"
    assert view._unit_status_map["mouseA_session1_u0002"] == "Curated"

    view.apply_cluster_customizations(
        new_order=[1, 0],
        new_names={0: "Fast Regular", 1: "Delayed Burst"},
        new_colors={0: "#10B981", 1: "#F59E0B"},
    )
    assert view._cluster_names[0] == "Fast Regular"
    assert view._cluster_names[1] == "Delayed Burst"
    assert view._cluster_colors[0] == "#10B981"
    assert view._cluster_colors[1] == "#F59E0B"
    assert view._cluster_card_order == [1, 0]

    # Skip unit in active learning
    view._skipped_gids.add("mouseA_session1_u0001")

    # Adjust view parameters
    view.set_visualized_window((-15.0, 35.0))
    view.set_cluster_bin_size(1.0)
    view.set_pulse_ordering("latency")
    view.set_raster_marker_size(4.0)

    # 3. Save project
    target_proj_dir = tmp_path / "saved_test_project"
    saved_path = view.save_project(target_proj_dir)
    assert saved_path is not None
    assert saved_path.is_file()
    assert (target_proj_dir / "clustering_project.yaml").is_file()
    assert (target_proj_dir / "pooled_units_curated.csv").is_file()
    view.close()

    # 4. Instantiate fresh ClusteringView and load project
    new_view = ClusteringView(load_mock=False)
    assert len(new_view.loaded_experiments) == 0
    assert new_view.table_pooled_units.rowCount() == 0

    ok_load = new_view.load_project(target_proj_dir)
    assert ok_load is True

    # 5. Assert all states restored
    # Ingested experiments
    assert "mouseA_session1" in new_view.loaded_experiments
    assert new_view.table_exps.rowCount() == 1
    assert new_view.table_pooled_units.rowCount() == 2

    # Unit assignments and statuses
    assert new_view._pooled_units_map["mouseA_session1_u0001"]["cluster_id"] == 0
    assert new_view._pooled_units_map["mouseA_session1_u0002"]["cluster_id"] == 1
    assert new_view._unit_status_map["mouseA_session1_u0001"] == "Curated"
    assert new_view._unit_status_map["mouseA_session1_u0002"] == "Curated"

    # Cluster customizations
    assert new_view._cluster_names[0] == "Fast Regular"
    assert new_view._cluster_names[1] == "Delayed Burst"
    assert new_view._cluster_colors[0] == "#10B981"
    assert new_view._cluster_colors[1] == "#F59E0B"
    assert new_view._cluster_card_order == [1, 0]

    # Active learning state
    assert "mouseA_session1_u0001" in new_view._skipped_gids

    # Model and view parameters
    assert new_view.current_window_ms == (-15.0, 35.0)
    assert new_view.current_bin_size_ms == 1.0
    assert new_view.raster_order_by == "latency"
    assert new_view.raster_marker_size == 4.0

    # UI elements updated without crashing
    assert len(new_view._cluster_cards) == 2
    assert new_view.scatter_item is not None

    new_view.close()


def test_clustering_view_prompt_save_and_load_dialogs(qapp, tmp_path: Path, monkeypatch):
    """Verify prompt_save_project and prompt_load_project handle user directory selection."""
    exp_dir = _build_test_phase1_export(tmp_path, "mouseB_session1", n_units=4, n_tagged=2)
    view = ClusteringView(load_mock=False)
    view.add_experiment_folder(exp_dir)
    view.assign_candidate_to_cluster(0, unit_id="mouseB_session1_u0001")

    # 1. User cancels save dialog
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: "")
    res_cancel = view.prompt_save_project()
    assert res_cancel is None

    # 2. User selects save directory
    save_dir = tmp_path / "prompt_save_dir"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(save_dir))
    res_save = view.prompt_save_project()
    assert res_save is not None
    assert res_save.is_file()
    assert (save_dir / "clustering_project.yaml").is_file()

    # 3. User cancels load dialog
    new_view = ClusteringView(load_mock=False)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: "")
    res_load_cancel = new_view.prompt_load_project()
    assert res_load_cancel is False
    assert len(new_view.loaded_experiments) == 0

    # 4. User accepts load dialog
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(save_dir))
    res_load_ok = new_view.prompt_load_project()
    assert res_load_ok is True
    assert "mouseB_session1" in new_view.loaded_experiments
    assert new_view.table_pooled_units.rowCount() == 2

    view.close()
    new_view.close()


def test_main_window_save_load_project_actions(qapp, tmp_path: Path, monkeypatch):
    """Verify MainWindow's act_open_project and act_save_project actions trigger correctly."""
    win = MainWindow()
    win.show()

    assert hasattr(win, "act_open_project")
    assert hasattr(win, "act_save_project")

    # Build and ingest experiment in clustering_view
    exp_dir = _build_test_phase1_export(tmp_path, "mouseC_session1", n_units=3, n_tagged=2)
    win.clustering_view.add_experiment_folder(exp_dir)
    win.clustering_view.assign_candidate_to_cluster(0, unit_id="mouseC_session1_u0001")

    # Trigger Save Clustering Project
    save_dir = tmp_path / "main_window_project"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(save_dir))
    win.act_save_project.trigger()

    assert win.clustering_view.current_project_dir == save_dir.resolve()
    assert (save_dir / "clustering_project.yaml").is_file()

    # Switch away to Launcher
    win.switch_to_launcher()
    assert win.stack.currentIndex() == MainWindow.VIEW_LAUNCHER

    # Trigger Open Clustering Project (should switch to Clustering and load)
    win.act_open_project.trigger()
    assert win.stack.currentIndex() == MainWindow.VIEW_CLUSTERING
    assert "mouseC_session1" in win.clustering_view.loaded_experiments

    win.close()


def test_legacy_project_loading_ignores_removed_fields_and_maps_norm(tmp_path: Path):
    """Verify legacy project manifests with removed fields and deprecated PSTH norm names load cleanly."""
    for legacy_norm in ("zscore_population", "baseline_zscore"):
        legacy_model_params = {
            "feature_config": {
                "use_latency": True,
                "weight_latency": 1.0,
                "use_waveform_correlation": True,
                "weight_waveform_correlation": 1.0,
                "use_peak_normalized_psth": False,
                "use_time_to_peak": False,
                "use_evoked_auc": False,
                "use_salt": True,
                "psth_normalization": legacy_norm,
                "time_window_ms": [-50.0, 50.0],
                "bin_size_ms": 2.0,
            },
            "embedding_params": {
                "method": "UMAP",
                "n_pca_components": 3,
                "umap_spread": 1.2,
                "random_state": 42,
            },
            "ap_params": {
                "damping": 0.8,
                "preference_mode": "quantile",
                "preference_quantile": 0.5,
                "preference_custom": -10.0,
            },
            "clustering_params": {
                "algorithm": "hierarchical",
                "n_clusters": 4,
                "feature_space": "Combined",
                "input_representation": "raw",
            },
        }

        # 1. Direct deserialize_model_parameters check
        restored = deserialize_model_parameters(legacy_model_params)
        fc = restored["feature_config"]
        assert fc.psth_normalization == "area_normalized"
        assert fc.weight_latency == 1.0
        assert fc.use_salt is True
        assert fc.time_window_ms == (-50.0, 50.0)
        assert not hasattr(fc, "use_waveform_correlation")

        # 2. Full project YAML save & load round-trip
        proj_dir = tmp_path / f"legacy_proj_{legacy_norm}"
        proj_data = ClusteringProjectData(
            model_parameters=legacy_model_params,
            experiments=[],
            clusters={"names": {}, "colors": {}, "card_order": []},
            active_learning_state={},
            unit_assignments=[],
        )
        save_clustering_project(proj_dir, proj_data)
        loaded = load_clustering_project(proj_dir)
        restored_loaded = deserialize_model_parameters(loaded.model_parameters)
        assert restored_loaded["feature_config"].psth_normalization == "area_normalized"
        assert restored_loaded["clustering_params"].n_clusters == 4


def test_project_roundtrip_persists_al_hyperparams_view_options_and_defers_refresh(qapp, tmp_path: Path, monkeypatch):
    """Project save/load must round-trip AL params, view options, combo_embedding_method, and defer batch refresh (Bug IO5)."""
    s1 = _build_test_phase1_export(tmp_path / "data", "sess_io5_1", n_units=4, n_tagged=2)
    s2 = _build_test_phase1_export(tmp_path / "data", "sess_io5_2", n_units=4, n_tagged=2)
    proj_dir = tmp_path / "proj_io5"

    view = ClusteringView()
    assert view.add_experiment_folder(s1) is True
    assert view.add_experiment_folder(s2) is True

    view.al_n_estimators = 250
    view.al_random_state = 99
    view.al_max_distance_sigma = 1.5
    view.overlay_focused_unit_enabled = False
    view.shared_y_axis_enabled = True
    view._cluster_sort_mode = "size"
    view.embedding_params.method = "UMAP"

    saved_yaml = view.save_project(proj_dir)
    assert saved_yaml is not None and saved_yaml.is_file()
    view.close()

    reopened = ClusteringView()
    reset_calls = 0
    orig_reset = reopened.reset_to_unassigned

    def counted_reset(*args, **kwargs):
        nonlocal reset_calls
        reset_calls += 1
        return orig_reset(*args, **kwargs)

    monkeypatch.setattr(reopened, "reset_to_unassigned", counted_reset)
    assert reopened.load_project(proj_dir) is True

    # Deferred refresh: only clear_all_experiments calls reset_to_unassigned (1); add_experiment_folder loop calls 0
    assert reset_calls == 1

    assert reopened.al_n_estimators == 250
    assert reopened.al_random_state == 99
    assert reopened.al_max_distance_sigma == pytest.approx(1.5)
    assert reopened.overlay_focused_unit_enabled is False
    assert reopened.act_overlay_focused_unit.isChecked() is False
    assert reopened.shared_y_axis_enabled is True
    assert reopened.act_shared_y_axis.isChecked() is True
    assert reopened._cluster_sort_mode == "size"
    assert reopened._sort_actions["size"].isChecked() is True
    assert reopened.embedding_params.method == "UMAP"
    assert reopened.combo_embedding_method.currentData() == "UMAP"
    reopened.close()


def test_optoclustering_project2_roundtrip_preserves_all_fields(qapp, tmp_path):
    """Copy of OptoClusteringProject2 opens with identical state and re-saving loses no field."""
    import shutil
    import subprocess
    import sys

    src_project = Path(__file__).resolve().parents[1] / "data" / "OptoClusteringProject2"
    src_yaml_path = src_project / "clustering_project.yaml"
    if not src_yaml_path.is_file():
        pytest.skip("data/OptoClusteringProject2 not available")

    orig_yaml = yaml.safe_load(src_yaml_path.read_text(encoding="utf-8"))

    tmp_project = tmp_path / "OptoClusteringProject2"
    shutil.copytree(src_project, tmp_project)

    created_links: List[Path] = []
    view: Optional[ClusteringView] = None
    reopened: Optional[ClusteringView] = None
    try:
        for exp in orig_yaml.get("experiments", []):
            rel_str = exp["path"]
            real_exp = (src_project / rel_str).resolve()
            if not real_exp.is_dir():
                pytest.skip(f"Referenced session folder missing: {real_exp}")
            link_path = (tmp_project / rel_str).resolve()
            if not link_path.exists():
                link_path.parent.mkdir(parents=True, exist_ok=True)
                if sys.platform == "win32":
                    subprocess.run(
                        ["cmd", "/c", "mklink", "/J", str(link_path), str(real_exp)],
                        check=True,
                        capture_output=True,
                    )
                else:
                    link_path.symlink_to(real_exp, target_is_directory=True)
                created_links.append(link_path)

        view = ClusteringView()
        assert view.load_project(tmp_project) is True

        orig_clusters = orig_yaml["clusters"]
        orig_names = {int(k): v for k, v in (orig_clusters.get("names") or {}).items()}
        orig_colors = {int(k): v for k, v in (orig_clusters.get("colors") or {}).items()}
        orig_hotkeys = {int(k): v for k, v in (orig_clusters.get("hotkeys") or {}).items()}
        assert {int(k): v for k, v in view._cluster_names.items() if v} == orig_names
        for cid, hex_color in orig_colors.items():
            assert view._cluster_colors[cid] == hex_color
        assert {int(k): v for k, v in view._cluster_hotkeys.items() if v} == orig_hotkeys
        assert [int(x) for x in view._cluster_card_order] == [int(x) for x in orig_clusters["card_order"]]

        expected_units = {
            rec["global_unit_id"]: (int(rec["cluster_id"]), str(rec["curated_status"]))
            for rec in orig_yaml["unit_assignments"]
        }
        assert len(view._pooled_units_map) == len(expected_units)
        for gid, (exp_cid, exp_status) in expected_units.items():
            assert int(view._pooled_units_map[gid]["cluster_id"]) == exp_cid
            assert str(view._unit_status_map.get(gid, "Unassigned")) == exp_status

        orig_mp = orig_yaml["model_parameters"]
        assert view.embedding_params.method == orig_mp["embedding_params"]["method"]
        assert view.active_clustering_params.algorithm == orig_mp["clustering_params"]["algorithm"]
        assert view.active_clustering_params.n_clusters == orig_mp["clustering_params"]["n_clusters"]
        assert view.auto_assign_threshold == pytest.approx(orig_mp["auto_assign_threshold"])
        assert view.auto_assign_enabled is orig_mp["auto_assign_enabled"]
        assert view.active_learning_enabled is orig_mp["active_learning_enabled"]
        assert list(view.current_window_ms) == [float(x) for x in orig_mp["window_ms"]]
        assert view.current_bin_size_ms == pytest.approx(orig_mp["bin_size_ms"])

        saved_path = view.save_project(tmp_project)
        assert saved_path is not None and saved_path.is_file()
        saved_yaml = yaml.safe_load(saved_path.read_text(encoding="utf-8"))

        # Every original top-level section and nested key/value must survive re-save
        assert [e["session_id"] for e in saved_yaml["experiments"]] == [
            e["session_id"] for e in orig_yaml["experiments"]
        ]
        for section_key, orig_section in orig_mp.items():
            if isinstance(orig_section, dict):
                for sub_key, orig_val in orig_section.items():
                    assert saved_yaml["model_parameters"][section_key][sub_key] == orig_val
            else:
                assert saved_yaml["model_parameters"][section_key] == orig_section

        assert saved_yaml["clusters"]["names"] == orig_clusters["names"]
        for cid, hex_color in orig_colors.items():
            assert saved_yaml["clusters"]["colors"][cid] == hex_color
        assert saved_yaml["clusters"]["hotkeys"] == orig_clusters["hotkeys"]
        assert saved_yaml["clusters"]["card_order"] == orig_clusters["card_order"]
        assert saved_yaml["active_learning_state"] == orig_yaml["active_learning_state"]

        saved_units = {
            rec["global_unit_id"]: (int(rec["cluster_id"]), str(rec["curated_status"]))
            for rec in saved_yaml["unit_assignments"]
        }
        assert saved_units == expected_units

        reopened = ClusteringView()
        assert reopened.load_project(tmp_project) is True
        assert [int(x) for x in reopened._cluster_card_order] == [int(x) for x in orig_clusters["card_order"]]
        assert {
            gid: (int(row["cluster_id"]), str(reopened._unit_status_map.get(gid, "Unassigned")))
            for gid, row in reopened._pooled_units_map.items()
        } == expected_units
    finally:
        if reopened is not None:
            reopened.close()
        if view is not None:
            view.close()
        for link_path in created_links:
            if link_path.exists():
                if sys.platform == "win32":
                    subprocess.run(["cmd", "/c", "rmdir", str(link_path)], check=False, capture_output=True)
                else:
                    link_path.unlink(missing_ok=True)


