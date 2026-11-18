"""Unit tests for Multi-Cell Selection (Shift/Ctrl-click) and Batch Cluster Assignment in Phase 2."""

from pathlib import Path
from typing import Sequence
import numpy as np
import pytest
from PyQt6.QtCore import QItemSelectionModel, QPoint, Qt
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem

from optocurator.ui.view_clusters import ClusteringView
from tests.test_detection_workflow import create_synthetic_experiment_folder


def select_rows(table: QTableWidget, rows: Sequence[int]) -> None:
    """Multi-select table rows concurrently using QItemSelectionModel flags."""
    table.clearSelection()
    selection_model = table.selectionModel()
    model = table.model()
    flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    for r in rows:
        idx = model.index(r, 0)
        selection_model.select(idx, flags)


def test_table_pooled_units_selection_mode(qapp):
    """Verify table_pooled_units uses ExtendedSelection and CustomContextMenu."""
    view = ClusteringView(start_with_clusters=False)
    view.show()

    assert view.table_pooled_units.selectionMode() == QTableWidget.SelectionMode.ExtendedSelection
    assert view.table_pooled_units.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
    assert hasattr(view, "btn_assign_candidate")
    assert view.btn_assign_candidate is view.btn_comp_assign

    view.close()


def test_batch_assignment_explicit_list(qapp, tmp_path):
    """Verify assign_candidate_to_cluster accepts a list of unit IDs and batches updates."""
    exp_dir = create_synthetic_experiment_folder(tmp_path / "exp_batch")
    view = ClusteringView(start_with_clusters=True)
    view.show()
    view.add_experiment_folder(exp_dir)

    all_gids = [str(r["global_unit_id"]) for r in view._pooled_rows]
    assert len(all_gids) >= 3

    # Pick 3 units to assign to Cluster 1
    batch_gids = all_gids[:3]
    success = view.assign_candidate_to_cluster(1, unit_id=batch_gids)
    assert success is True

    for gid in batch_gids:
        assert view._pooled_units_map[gid]["cluster_id"] == 1
        assert view._pooled_units_map[gid]["status"] == "Curated"
        assert view._unit_status_map[gid] == "Curated"
        assert gid in view._skipped_gids

    view.close()


def test_batch_assignment_from_table_selection(qapp, tmp_path):
    """Verify assign_candidate_to_cluster with unit_id=None extracts all selected rows."""
    exp_dir = create_synthetic_experiment_folder(tmp_path / "exp_table_sel")
    view = ClusteringView(start_with_clusters=True)
    view.show()
    view.add_experiment_folder(exp_dir)

    assert view.table_pooled_units.rowCount() >= 3

    # Multi-select first 3 rows concurrently
    select_rows(view.table_pooled_units, [0, 1, 2])

    selected_rows = view.table_pooled_units.selectionModel().selectedRows()
    assert len(selected_rows) == 3

    gids = [view._get_gid_for_row(r.row()) for r in selected_rows]
    assert len(gids) == 3

    # Assign all 3 to Cluster 2
    success = view.assign_candidate_to_cluster(2)
    assert success is True

    for gid in gids:
        assert view._pooled_units_map[gid]["cluster_id"] == 2
        assert view._unit_status_map[gid] == "Curated"

    view.close()


def test_keyboard_hotkey_multi_cell_assignment(qapp, tmp_path):
    """Verify _on_cluster_hotkey assigns all selected rows when multiple cells are selected."""
    exp_dir = create_synthetic_experiment_folder(tmp_path / "exp_hotkey")
    view = ClusteringView(start_with_clusters=True)
    view.show()
    view.add_experiment_folder(exp_dir)

    # Select rows 0 and 1 concurrently
    select_rows(view.table_pooled_units, [0, 1])

    gids = [view._get_gid_for_row(0), view._get_gid_for_row(1)]

    # Trigger hotkey 0
    view._on_cluster_hotkey(0)

    for gid in gids:
        assert view._pooled_units_map[gid]["cluster_id"] == 0
        assert view._unit_status_map[gid] == "Curated"

    view.close()


def test_create_new_cluster_multi_cell(qapp, tmp_path):
    """Verify create_new_cluster_for_candidate assigns all selected cells to next new cluster."""
    exp_dir = create_synthetic_experiment_folder(tmp_path / "exp_new_clust")
    view = ClusteringView(start_with_clusters=True)
    view.show()
    view.add_experiment_folder(exp_dir)

    existing_cids = set(view._cluster_card_order)

    # Select rows 1 and 2 concurrently
    select_rows(view.table_pooled_units, [1, 2])

    gids = [view._get_gid_for_row(1), view._get_gid_for_row(2)]

    new_cid = view.create_new_cluster_for_candidate()
    assert new_cid >= 0
    assert new_cid not in existing_cids
    assert new_cid in view._cluster_card_order

    for gid in gids:
        assert view._pooled_units_map[gid]["cluster_id"] == new_cid
        assert view._unit_status_map[gid] == "Curated"

    view.close()


def test_selection_ui_button_and_label_text(qapp, tmp_path):
    """Verify btn_assign_candidate and lbl_candidate show multi-cell selection text."""
    exp_dir = create_synthetic_experiment_folder(tmp_path / "exp_ui_text")
    view = ClusteringView(start_with_clusters=True)
    view.show()
    view.add_experiment_folder(exp_dir)

    # Single selection
    select_rows(view.table_pooled_units, [0])
    assert view.btn_assign_candidate.text() == "✓ Assign Focused"
    assert "Cells Selected" not in view.lbl_candidate.text()

    # Multi selection: 3 rows
    select_rows(view.table_pooled_units, [0, 1, 2])
    assert view.btn_assign_candidate.text() == "✓ Assign Selected (3)"
    assert "3 Cells Selected" in view.lbl_candidate.text()

    view.close()


def test_comparison_pane_assign_selected(qapp, tmp_path):
    """Verify _on_assign_to_comparison_cluster assigns all selected cells to comparison cluster."""
    exp_dir = create_synthetic_experiment_folder(tmp_path / "exp_comp_assign")
    view = ClusteringView(start_with_clusters=True)
    view.show()
    view.add_experiment_folder(exp_dir)

    # Set comparison cluster to 1
    view.set_cluster_comparison(1)
    assert view._comparison_cluster_id == 1

    # Select rows 0 and 2 concurrently
    select_rows(view.table_pooled_units, [0, 2])
    gids = [view._get_gid_for_row(0), view._get_gid_for_row(2)]

    view._on_assign_to_comparison_cluster()

    for gid in gids:
        assert view._pooled_units_map[gid]["cluster_id"] == 1
        assert view._unit_status_map[gid] == "Curated"

    view.close()


def test_context_menu_generation(qapp, tmp_path, monkeypatch):
    """Verify _show_pooled_units_context_menu generates cluster options and unassign action."""
    from PyQt6.QtWidgets import QMenu
    exp_dir = create_synthetic_experiment_folder(tmp_path / "exp_ctx_menu")
    view = ClusteringView(start_with_clusters=True)
    view.show()
    view.add_experiment_folder(exp_dir)

    # Multi-select rows 0 and 1 concurrently
    select_rows(view.table_pooled_units, [0, 1])

    captured_menu_actions = []

    def mock_exec(menu_self, *args, **kwargs):
        for act in menu_self.actions():
            captured_menu_actions.append(act.text())
        return None

    monkeypatch.setattr(QMenu, "exec", mock_exec)

    # Right-click inside row 0 (which is within selected rows [0, 1])
    click_pos = view.table_pooled_units.visualRect(view.table_pooled_units.model().index(0, 0)).center()
    view._show_pooled_units_context_menu(click_pos)

    assert any("Assign 2 Selected Cells To:" in a for a in captured_menu_actions)
    assert any("➕ New Cluster" in a for a in captured_menu_actions)
    assert any("Unassign (-1)" in a for a in captured_menu_actions)

    view.close()
