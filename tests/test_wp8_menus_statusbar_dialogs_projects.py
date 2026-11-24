"""WP8 tests for menus, status bar synchronization, § 12 dialogs, and multi-experiment project roundtrip."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import QDialogButtonBox

from optocurator.ui.app import MainWindow
from optocurator.ui.dialog_clustering import (
    EmbeddingSettingsDialog,
    HierarchicalClusteringDialog,
)
from optocurator.ui.dialog_clustering_settings import (
    ActiveLearningSettingsDialog,
    ConfidenceThresholdDialog,
    ResetClusteringDialog,
    RevertAutoAssignmentsDialog,
    VisualizationSettingsDialog,
)
from optocurator.ui.dialog_clusters_customize import CustomizeClustersDialog
from optocurator.ui.dialog_feature_settings import FeatureSettingsDialog
from optocurator.ui.dialog_seed_clusters import SeedClustersDialog
from optocurator.ui.theme import STATUS_BAR_HEIGHT
from optocurator.ui.view_clusters import ClusteringView


def _create_dummy_png(target_file: Path, width: int = 120, height: int = 80) -> Path:
    """Create a minimal valid PNG file on disk for unit plot inspection."""
    target_file.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (width, height), color=(37, 99, 235))
    im.save(str(target_file), "PNG")
    return target_file


def _build_phase1_export(
    folder: Path,
    session_id: str,
    n_units: int = 4,
    n_tagged: int = 3,
) -> Path:
    """Build a minimal valid Phase 1 exported recording directory."""
    session_dir = folder / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = session_dir / f"{session_id}_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(1, n_units + 1):
        is_tag = i <= n_tagged
        gid = f"{session_id}_u{i:04d}"
        plot_rel = f"{session_id}_plots/{gid}_psth_raster.png"
        rows.append(
            {
                "global_unit_id": gid,
                "session_id": session_id,
                "unit_id": i,
                "latency_ms": 3.0 + i * 1.1,
                "jitter_ms": 0.9 + i * 0.1,
                "reliability": max(0.1, 0.9 - i * 0.1),
                "waveform_correlation": 0.95,
                "salt_p_value": 0.001 * i,
                "is_tagged": is_tag,
                "cluster_id": -1,
                "curated_label": "Tagged" if is_tag else "unassigned",
                "plot_path": plot_rel,
                "channel": i,
                "depth": 150.0 * i,
            }
        )
        _create_dummy_png(session_dir / plot_rel)

    pd.DataFrame(rows).to_csv(session_dir / f"{session_id}_opto_curated.csv", index=False)
    np.save(session_dir / "opto_idxs_OPTS03.npy", np.array([50, 100, 150]))
    return session_dir


def _visible_action_texts(menu) -> List[str]:
    """Return non-empty texts of visible actions in a QMenu."""
    return [act.text() for act in menu.actions() if act.isVisible() and not act.isSeparator()]


# ============================================================================
# 1. § 11.1 MENUS & MODE-AWARE VISIBILITY
# ============================================================================


def test_wp8_menus_and_mode_aware_visibility(qapp):
    """Verify File, View, Stats/Clustering, and Help menus and mode-aware visibility."""
    win = MainWindow()
    try:
        win.resize(1360, 860)
        win.show()

        # --- Clustering mode ---
        win.switch_to_clustering()
        qapp.processEvents()

        assert win.clustering_menu.menuAction().isVisible() is True
        assert win.stats_menu.menuAction().isVisible() is False

        file_texts_clustering = _visible_action_texts(win.file_menu)
        for expected in (
            "New project…",
            "Open project…",
            "Recent",
            "Save",
            "Save as…",
            "Add experiment…",
            "Remove experiment…",
            "Close project",
        ):
            assert expected in file_texts_clustering

        assert win.act_new_project.shortcut() == QKeySequence("Ctrl+N")
        assert win.act_open_project.shortcut() == QKeySequence("Ctrl+O")
        assert win.act_save_project.shortcut() == QKeySequence("Ctrl+S")
        assert win.act_save_project_as.shortcut() == QKeySequence("Ctrl+Shift+S")
        assert win.act_add_experiment.shortcut() == QKeySequence("Ctrl+E")
        assert win.act_close_project.shortcut() == QKeySequence("Ctrl+W")
        assert all("export" not in act.text().lower() for act in win.file_menu.actions())

        # View menu (Clustering mode)
        view_texts_clustering = _visible_action_texts(win.view_menu)
        for expected in (
            "Overlay focused unit",
            "Shared y-axis",
            "Configure visualizations…",
            "Reset panel layout",
        ):
            assert expected in view_texts_clustering

        # Clustering menu items & AL gating
        clustering_texts = _visible_action_texts(win.clustering_menu)
        for expected in (
            "Seed clusters…",
            "Run clustering…",
            "Feature settings…",
            "Customize clusters…",
            "Active learning",
            "Go to suggested unit",
            "Active learning settings…",
            "Auto-assign",
            "Confidence threshold…",
            "Revert auto-assignments…",
            "Reset clustering…",
        ):
            assert expected in clustering_texts

        assert win.act_go_to_suggested_unit.shortcut() == QKeySequence("G")
        win.clustering_view.set_active_learning_mode(False)
        qapp.processEvents()
        assert win.act_go_to_suggested_unit.isEnabled() is False

        win.clustering_view.set_active_learning_mode(True)
        qapp.processEvents()
        assert win.act_go_to_suggested_unit.isEnabled() is True

        # Help menu items
        help_texts = _visible_action_texts(win.help_menu)
        for expected in (
            "Keyboard shortcuts",
            "Documentation",
            "How to cite OptoCurator",
            "About OptoCurator",
        ):
            assert expected in help_texts

        # --- Detection mode ---
        win.switch_to_detection()
        qapp.processEvents()

        assert win.stats_menu.menuAction().isVisible() is True
        assert win.clustering_menu.menuAction().isVisible() is False

        file_texts_detection = _visible_action_texts(win.file_menu)
        for expected in (
            "Load experiment folder…",
            "Recent",
            "Save",
            "Close session",
        ):
            assert expected in file_texts_detection
        assert all("export" not in act.text().lower() for act in win.file_menu.actions())

        # Triggering Reset panel layout in Detection mode must stay in Detection mode
        win.act_reset_panel_layout.trigger()
        qapp.processEvents()
        assert win.stack.currentIndex() == MainWindow.VIEW_DETECTION
    finally:
        win.close()


# ============================================================================
# 2. § 11.2 STATUS BAR (24 PX) & SYNCHRONIZATION
# ============================================================================


def test_wp8_statusbar_layout_sync_and_dirty_indicator(qapp, tmp_path: Path):
    """Verify 24px status bar widgets, bidirectional AL/Auto-assign sync, and save indicator."""
    win = MainWindow()
    try:
        win.resize(1360, 860)
        win.show()
        win.switch_to_clustering()
        qapp.processEvents()

        # Height is STATUS_BAR_HEIGHT (24 px)
        assert STATUS_BAR_HEIGHT == 24
        assert win.status.height() == 24

        # Left-side clustering controls
        assert win.btn_status_threshold.isVisible()
        assert "%" in win.btn_status_threshold.text()
        assert win.btn_status_auto_assign.isCheckable()
        assert win.btn_status_active_learning.isCheckable()
        assert win.btn_status_al_settings.text() == "⚙"

        # Right-side indicators
        assert win.lbl_status_version.text().startswith("v")
        assert win.lbl_status_save_state.text() == "● Saved"

        # Bidirectional sync: Auto-assign (status bar <-> menu)
        initial_aa = win.btn_status_auto_assign.isChecked()
        win.btn_status_auto_assign.setChecked(not initial_aa)
        qapp.processEvents()
        assert win.act_auto_assign.isChecked() is (not initial_aa)

        win.act_auto_assign.setChecked(initial_aa)
        qapp.processEvents()
        assert win.btn_status_auto_assign.isChecked() is initial_aa

        # Bidirectional sync: Active learning (status bar <-> menu + Go to suggested unit)
        win.btn_status_active_learning.setChecked(False)
        qapp.processEvents()
        assert win.act_active_learning.isChecked() is False
        assert win.act_go_to_suggested_unit.isEnabled() is False

        win.act_active_learning.setChecked(True)
        qapp.processEvents()
        assert win.btn_status_active_learning.isChecked() is True
        assert win.act_go_to_suggested_unit.isEnabled() is True

        # Dirty state synchronization on curation edit & reset on save
        exp_dir = _build_phase1_export(tmp_path, "status_exp", n_units=3, n_tagged=2)
        assert win.clustering_view.add_experiment_folder(exp_dir) is True
        win.clustering_view.assign_candidate_to_cluster(0, unit_id="status_exp_u0001")
        qapp.processEvents()

        assert win.clustering_view._is_dirty is True
        assert win.lbl_status_save_state.text() == "● Unsaved changes"

        proj_dir = tmp_path / "status_project"
        saved_path = win.clustering_view.save_project(proj_dir)
        qapp.processEvents()
        assert saved_path is not None and saved_path.is_file()
        assert win.clustering_view._is_dirty is False
        assert win.lbl_status_save_state.text() == "● Saved"
    finally:
        win.close()


# ============================================================================
# 3. § 12 DIALOGS
# ============================================================================


def test_wp8_section12_dialogs_contracts(qapp):
    """Verify widget presence, constraints, and primary button labels across all § 12 dialogs."""
    # EmbeddingSettingsDialog: PCA & UMAP only (no t-SNE), "Apply"
    dlg_emb = EmbeddingSettingsDialog()
    try:
        methods_text = [dlg_emb.combo_method.itemText(i) for i in range(dlg_emb.combo_method.count())]
        methods_data = [dlg_emb.combo_method.itemData(i) for i in range(dlg_emb.combo_method.count())]
        assert methods_data == ["PCA", "UMAP"]
        assert all("tsne" not in t.lower() and "t-sne" not in t.lower() for t in methods_text)
        assert dlg_emb.btn_apply.text() == "Apply"
    finally:
        dlg_emb.close()

    # HierarchicalClusteringDialog
    dlg_hier = HierarchicalClusteringDialog()
    try:
        assert dlg_hier.chk_use_embedding_features is not None
        assert dlg_hier.radio_keep_curated is not None
        assert dlg_hier.radio_replace_all is not None
        assert dlg_hier.btn_run.text() == "Run clustering"
    finally:
        dlg_hier.close()

    # SeedClustersDialog
    dlg_seed = SeedClustersDialog()
    try:
        assert dlg_seed.chk_use_embedding_features is not None
        assert not hasattr(dlg_seed, "spin_random_seed")
        assert dlg_seed.spin_damping is not None
        assert dlg_seed.spin_percentile is not None
        assert dlg_seed.spin_min_size is not None
        assert dlg_seed.radio_keep_curated is not None
        assert dlg_seed.radio_replace_all is not None
        assert dlg_seed.btn_run.text() == "Seed clusters"
    finally:
        dlg_seed.close()

    # FeatureSettingsDialog
    dlg_feat = FeatureSettingsDialog()
    try:
        ok_btn = dlg_feat.button_box.button(QDialogButtonBox.StandardButton.Ok)
        assert ok_btn is not None
        assert ok_btn.text() == "Apply"
    finally:
        dlg_feat.close()

    # CustomizeClustersDialog
    dlg_cust = CustomizeClustersDialog(
        cluster_order=[0, 1],
        cluster_names={0: "Cluster 0", 1: "Cluster 1"},
        cluster_colors={0: "#2563EB", 1: "#DC2626"},
        cluster_counts={0: 3, 1: 4},
        cluster_hotkeys={0: "0", 1: "1"},
    )
    try:
        assert dlg_cust.btn_apply.text() == "Done"
    finally:
        dlg_cust.close()

    # ConfidenceThresholdDialog: slider 50-99, chk_auto_assign, "Update threshold", live count
    dlg_conf = ConfidenceThresholdDialog(
        current_threshold=0.85,
        auto_assign_enabled=False,
        unassigned_max_probs={"u1": 0.92, "u2": 0.75, "u3": 0.60},
    )
    try:
        assert dlg_conf.slider_threshold.minimum() == 50
        assert dlg_conf.slider_threshold.maximum() == 99
        assert dlg_conf.chk_auto_assign is not None
        assert dlg_conf.btn_update.text() == "Update threshold"
        assert "1" in dlg_conf.lbl_above_count.text()

        dlg_conf.slider_threshold.setValue(70)
        qapp.processEvents()
        assert "2" in dlg_conf.lbl_above_count.text()
        assert dlg_conf.lbl_threshold_value.text() == "70%"
    finally:
        dlg_conf.close()

    # ActiveLearningSettingsDialog
    dlg_al = ActiveLearningSettingsDialog()
    try:
        assert dlg_al.btn_apply.text() == "Apply and retrain"
    finally:
        dlg_al.close()

    # VisualizationSettingsDialog
    dlg_vis = VisualizationSettingsDialog()
    try:
        assert dlg_vis.btn_apply.text() == "Apply"
    finally:
        dlg_vis.close()

    # ResetClusteringDialog & RevertAutoAssignmentsDialog
    dlg_reset = ResetClusteringDialog()
    try:
        assert dlg_reset.btn_reset.text() == "Reset"
    finally:
        dlg_reset.close()

    dlg_revert = RevertAutoAssignmentsDialog(auto_count=3)
    try:
        assert dlg_revert.btn_revert.text() == "Revert"
    finally:
        dlg_revert.close()


# ============================================================================
# 4. § 14 WP8 MULTI-EXPERIMENT PROJECT ROUNDTRIP & REMOVAL
# ============================================================================


def test_wp8_multi_experiment_project_roundtrip_and_removal(qapp, tmp_path: Path):
    """Save, close, reopen multi-experiment project preserving names, colors, keys, statuses, and remove."""
    exp_a = _build_phase1_export(tmp_path, "exp_alpha", n_units=4, n_tagged=3)
    exp_b = _build_phase1_export(tmp_path, "exp_beta", n_units=4, n_tagged=3)
    proj_dir = tmp_path / "wp8_roundtrip_project"

    view = ClusteringView(load_mock=False)
    try:
        assert view.add_experiment_folder(exp_a) is True
        assert view.add_experiment_folder(exp_b) is True
        assert set(view.loaded_experiments.keys()) == {"exp_alpha", "exp_beta"}
        assert view.table_pooled_units.rowCount() == 6

        # Curate units to establish all 4 statuses: Curated, Auto-assigned, Seeded, Unassigned
        view.assign_candidate_to_cluster(0, unit_id="exp_alpha_u0001")
        view.assign_candidate_to_cluster(1, unit_id="exp_alpha_u0002")

        # Auto-assigned unit
        view._pooled_units_map["exp_alpha_u0003"]["cluster_id"] = 1
        view._unit_status_map["exp_alpha_u0003"] = "Auto-assigned"
        view._auto_confidences["exp_alpha_u0003"] = 0.91

        # Seeded unit
        view._pooled_units_map["exp_beta_u0001"]["cluster_id"] = 0
        view._unit_status_map["exp_beta_u0001"] = "Seeded"

        # Remaining exp_beta_u0002, exp_beta_u0003 stay Unassigned
        assert view._unit_status_map.get("exp_beta_u0002", "Unassigned") == "Unassigned"

        # Customize cluster names, colours, and hotkeys
        view.apply_cluster_customizations(
            new_order=[1, 0],
            new_names={0: "Fast Excitatory", 1: "Sustained Inhibitory"},
            new_colors={0: "#10B981", 1: "#F59E0B"},
            new_hotkeys={0: "Shift+A", 1: "Shift+B"},
        )

        saved_yaml = view.save_project(proj_dir)
        assert saved_yaml is not None and saved_yaml.is_file()

        # Close project and verify clean reset
        view.close_clustering_project()
        assert len(view.loaded_experiments) == 0
        assert len(view._pooled_units_map) == 0
    finally:
        view.close()

    # Reopen from disk in a fresh ClusteringView
    reopened = ClusteringView(load_mock=False)
    try:
        assert reopened.load_project(proj_dir) is True
        assert set(reopened.loaded_experiments.keys()) == {"exp_alpha", "exp_beta"}
        assert reopened.table_pooled_units.rowCount() == 6

        # Cluster metadata preserved
        assert reopened._cluster_card_order == [1, 0]
        assert reopened._cluster_names[0] == "Fast Excitatory"
        assert reopened._cluster_names[1] == "Sustained Inhibitory"
        assert reopened._cluster_colors[0].upper() == "#10B981"
        assert reopened._cluster_colors[1].upper() == "#F59E0B"
        assert reopened._get_cluster_hotkey(0) == "Shift+A"
        assert reopened._get_cluster_hotkey(1) == "Shift+B"

        # Unit cluster_ids and statuses preserved
        assert reopened._pooled_units_map["exp_alpha_u0001"]["cluster_id"] == 0
        assert reopened._unit_status_map["exp_alpha_u0001"] == "Curated"

        assert reopened._pooled_units_map["exp_alpha_u0002"]["cluster_id"] == 1
        assert reopened._unit_status_map["exp_alpha_u0002"] == "Curated"

        assert reopened._pooled_units_map["exp_alpha_u0003"]["cluster_id"] == 1
        assert reopened._unit_status_map["exp_alpha_u0003"] == "Auto-assigned"

        assert reopened._pooled_units_map["exp_beta_u0001"]["cluster_id"] == 0
        assert reopened._unit_status_map["exp_beta_u0001"] == "Seeded"

        assert reopened._pooled_units_map["exp_beta_u0002"]["cluster_id"] == -1
        assert reopened._unit_status_map["exp_beta_u0002"] == "Unassigned"

        # Verify remove_clustering_experiment(session_id=..., confirm=False)
        removed = reopened.remove_clustering_experiment(session_id="exp_beta", confirm=False)
        assert removed is True
        assert "exp_beta" not in reopened.loaded_experiments
        assert set(reopened.loaded_experiments.keys()) == {"exp_alpha"}
        assert reopened.table_pooled_units.rowCount() == 3
        assert all(
            rec.get("session_id") == "exp_alpha"
            for rec in reopened._pooled_units_map.values()
        )
    finally:
        reopened.close()


def test_mainwindow_close_and_recent_open_prompt_dirty_changes(qapp, tmp_path: Path, monkeypatch):
    """MainWindow.closeEvent, recent open, and close-session cancel-save must protect unsaved changes (Bug UI4)."""
    from PyQt6.QtGui import QCloseEvent
    from PyQt6.QtWidgets import QMessageBox
    from optocurator.ui.app import MainWindow

    win = MainWindow()
    try:
        win.detection_view._set_dirty(True)
        win.clustering_view._set_dirty(True)

        # 1. Cancelling QMessageBox blocks closeEvent
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.Cancel)
        ev_cancel = QCloseEvent()
        win.closeEvent(ev_cancel)
        assert ev_cancel.isAccepted() is False

        # 2. Cancelling QMessageBox aborts _open_recent_detection_folder and _open_recent_clustering_project before loading
        called_loads = []
        monkeypatch.setattr(win.detection_view, "load_experiment_folder", lambda *a, **kw: called_loads.append("det") or True)
        monkeypatch.setattr(win.clustering_view, "load_project", lambda *a, **kw: called_loads.append("clust") or True)
        win._open_recent_detection_folder(str(tmp_path / "some_det"))
        win._open_recent_clustering_project(str(tmp_path / "some_proj"))
        assert called_loads == []

        # 3. Clicking Save in _on_close_detection_session then cancelling file dialog (_is_dirty remains True) must NOT close session
        closed_det = []
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.Save)
        monkeypatch.setattr(win.detection_view, "export_curated_session", lambda *a, **kw: None)
        monkeypatch.setattr(win.detection_view, "close_session", lambda: closed_det.append(True))
        win._on_close_detection_session()
        assert closed_det == []

        # 4. Choosing Discard allows closeEvent to proceed
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.Discard)
        ev_discard = QCloseEvent()
        win.closeEvent(ev_discard)
        assert ev_discard.isAccepted() is True
    finally:
        win.detection_view._set_dirty(False)
        win.clustering_view._set_dirty(False)
        win.close()


def test_mainwindow_hierarchical_clustering_respects_replace_all(qapp, qtbot, monkeypatch):
    """MainWindow._on_hierarchical_clustering must delegate to prompt_hierarchical_clustering and honor Replace all (Bug UI6)."""
    from PyQt6.QtWidgets import QDialog
    from optocurator.ui.app import MainWindow
    from optocurator.ui.dialog_clustering import HierarchicalClusteringDialog

    win = MainWindow()
    try:
        win.switch_to_clustering()
        win.clustering_view._load_mock_data()
        qtbot.waitUntil(
            lambda: win.clustering_view._active_learning_worker is None
            or not win.clustering_view._active_learning_worker.isRunning(),
            timeout=10000,
        )
        curated_gid = next(iter(win.clustering_view._pooled_units_map))
        win.clustering_view.assign_candidate_to_cluster(0, unit_id=curated_gid)
        assert win.clustering_view._unit_status_map[curated_gid] == "Curated"

        def _fake_exec(dlg):
            dlg.radio_replace_all.setChecked(True)
            return QDialog.DialogCode.Accepted

        monkeypatch.setattr(HierarchicalClusteringDialog, "exec", _fake_exec)
        win._on_hierarchical_clustering()
        assert win.clustering_view._unit_status_map[curated_gid] != "Curated"
    finally:
        qtbot.waitUntil(
            lambda: win.clustering_view._active_learning_worker is None
            or not win.clustering_view._active_learning_worker.isRunning(),
            timeout=10000,
        )
        win.detection_view._set_dirty(False)
        win.clustering_view._set_dirty(False)
        win.close()
