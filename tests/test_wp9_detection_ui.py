"""WP9 Detection Mode UI tests covering docs/ui_overhaul/UI_OVERHAUL_SPEC.md § 13 & § 14 WP9."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QPushButton

from optocurator.ui.app import MainWindow
from optocurator.ui.theme import (
    COLOR_DANGER,
    DETECTION_PLOT_LEFT_MARGIN,
    DETECTION_PSTH_SIZE,
    DETECTION_RASTER_SIZE,
)
from optocurator.ui.view_detection import DetectionView
from tests.test_detection_workflow import create_synthetic_experiment_folder


def _get_qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _find_row_for_unit(view: DetectionView, unit_id: int) -> int:
    for r in range(view.table_units.rowCount()):
        if view._row_unit_id(r) == unit_id:
            return r
    raise AssertionError(f"Unit {unit_id} not found in table_units")


def test_wp9_detection_header_menus_and_statusbar(qapp, tmp_path: Path) -> None:
    """Verify Detection header mode switch, session/protocol controls, menus, and status bar."""
    exp_folder = tmp_path / "wp9_header_session"
    create_synthetic_experiment_folder(exp_folder, n_units=4, n_pulses=20)

    # Add a second stimulus protocol to verify switchable protocol dropdown
    sr = 30000.0
    onsets_2 = np.linspace(12.0, 28.0, 15)
    offsets_2 = onsets_2 + 0.015
    np.save(
        exp_folder / "opto_idxs_OPTS_10.npy",
        np.column_stack([(onsets_2 * sr).astype(np.uint32), (offsets_2 * sr).astype(np.uint32)]),
    )

    window = MainWindow()
    try:
        window.switch_to_detection()
        view = window.detection_view

        # Header mode switch & visibility
        assert window.stack.currentIndex() == MainWindow.VIEW_DETECTION
        assert window.btn_mode_detection.isChecked() is True
        assert window.btn_mode_clustering.isChecked() is False
        assert not window.btn_detection_session.isHidden()
        assert not window.combo_detection_protocol.isHidden()
        assert window.btn_experiments.isHidden()
        assert window.stats_menu.menuAction().isVisible() is True
        assert window.clustering_menu.menuAction().isVisible() is False

        # Status bar in Detection mode
        assert window.status_clustering_left.isHidden() is True
        assert not window.lbl_status_version.isHidden()
        assert not window.lbl_status_save_state.isHidden()

        # Load session and verify header session button & protocol sync
        assert view.load_experiment_folder(exp_folder) is True
        assert "wp9_header_session" in window.btn_detection_session.text()
        assert "▾" in window.btn_detection_session.text()

        assert window.combo_detection_protocol.count() == 2
        assert window.combo_detection_protocol.isEnabled() is True
        assert window.combo_detection_protocol.currentIndex() == view.combo_proto.currentIndex()

        # Switching header protocol updates detection_view.combo_proto & current_protocol
        window.combo_detection_protocol.setCurrentIndex(1)
        assert view.combo_proto.currentIndex() == 1
        assert view.current_protocol is not None
        assert view.current_protocol.name in window.combo_detection_protocol.currentText()

        # Dirty state & save indicator transitions
        assert window.lbl_status_save_state.text() == "● Saved"
        view.row_latency.chk_enabled.setChecked(False)
        assert view._is_dirty is True
        assert window.lbl_status_save_state.text() == "● Unsaved changes"

        csv_path = view.export_curated_session(exp_folder, show_dialog=False, precompute_pngs=False)
        assert csv_path is not None and csv_path.exists()
        assert view._is_dirty is False
        assert window.lbl_status_save_state.text() == "● Saved"

        # Mode switch toggles between Detection and Clustering
        window.btn_mode_clustering.click()
        assert window.stack.currentIndex() == MainWindow.VIEW_CLUSTERING
        assert window.btn_mode_clustering.isChecked() is True
        window.btn_mode_detection.click()
        assert window.stack.currentIndex() == MainWindow.VIEW_DETECTION
        assert window.btn_mode_detection.isChecked() is True
    finally:
        window.close()


def test_wp9_units_column_table_and_probe(qapp, tmp_path: Path) -> None:
    """Verify 8-column units table, failing red cells, SALT floor, WF placeholder, Show filters, Next likely, and Probe."""
    exp_folder = tmp_path / "wp9_units_session"
    create_synthetic_experiment_folder(exp_folder, n_units=4, n_pulses=25)

    view = DetectionView()
    try:
        assert view.load_experiment_folder(exp_folder) is True

        # 8 columns with expected headers
        assert view.table_units.columnCount() == 8
        headers = [view.table_units.horizontalHeaderItem(i).text() for i in range(8)]
        assert headers == ["Unit", "Lat.", "Jit.", "Rel.", "SALT p", "WF r", "Likelihood", "Status"]

        row_u0 = _find_row_for_unit(view, 0)
        row_u2 = _find_row_for_unit(view, 2)

        # Unit 2 fails latency (18 ms > 8 ms) -> red COLOR_DANGER foreground
        assert view.table_units.item(row_u2, 1).foreground().color().name().lower() == COLOR_DANGER.lower()

        # SALT <= 0.01 display with exact sort_key
        salt_item_u0 = view.table_units.item(row_u0, 4)
        assert salt_item_u0.text() == "≤ 0.01"
        assert salt_item_u0.sort_key <= 0.01

        # WF r shows "—" when waveforms are unavailable
        assert view.units_by_id[0].waveforms is None
        assert view.table_units.item(row_u0, 5).text() == "—"

        # Show dropdown options
        show_items = [view.combo_filter_status.itemText(i) for i in range(view.combo_filter_status.count())]
        assert show_items == [
            "All units",
            "Tagged",
            "Untagged",
            "Passing current criteria",
            "Changed manually",
            "Likely but untagged",
        ]

        # SiliconProbeWidget updated with all units and candidate marker visible
        assert len(view.probe_widget.units_scatter.data) == len(view.units)
        assert view.probe_widget.cand_marker.isVisible() is True
    finally:
        view.close()


def test_wp9_focused_unit_plots_waveform_visibility_and_pairplot(qapp, tmp_path: Path) -> None:
    """Verify focused unit header, display-only controls, fixed plot sizes, no baseline lines, waveform visibility, and pairplot click."""
    exp_folder = tmp_path / "wp9_focused_session"
    create_synthetic_experiment_folder(exp_folder, n_units=4, n_pulses=20)

    view = DetectionView()
    try:
        assert view.load_experiment_folder(exp_folder) is True
        row_u0 = _find_row_for_unit(view, 0)
        view.table_units.selectRow(row_u0)
        view._on_unit_selected()

        # Focused unit header & manual override toggle
        assert view.lbl_focused_unit_title.text() == "Unit u0000"
        assert view.lbl_focused_status.text().startswith("● Tagged")
        assert view.btn_toggle_tag.text() == "Mark as untagged"

        view.btn_toggle_tag.click()
        assert view.lbl_focused_status.text() == "○ Untagged (manual)"
        assert view.btn_toggle_tag.text() == "Mark as tagged"
        view.btn_toggle_tag.click()
        assert view.lbl_focused_status.text() == "● Tagged (manual)"

        # Display-only Window and Pulse order never alter view.metrics
        metrics_snapshot = {
            uid: (m.latency_ms, m.jitter_ms, m.reliability, m.salt_p_value, m.waveform_correlation)
            for uid, m in view.metrics.items()
        }
        view.edit_win_start.setText("-10")
        view.edit_win_end.setText("30")
        view._on_window_text_edited()
        assert view.current_window_ms == (-10.0, 30.0)

        view.combo_order.setCurrentIndex(1)
        assert view.raster_order_by == "latency"

        for uid, snap in metrics_snapshot.items():
            m = view.metrics[uid]
            assert (m.latency_ms, m.jitter_ms, m.reliability, m.salt_p_value, m.waveform_correlation) == snap

        # Invalid window (end <= start + 10) is rejected
        view.edit_win_start.setText("10")
        view.edit_win_end.setText("15")
        view._on_window_text_edited()
        assert view.current_window_ms == (-10.0, 30.0)

        # Fixed 520x240 raster & 520x100 PSTH canvas, no baseline mean/3σ lines
        assert DETECTION_RASTER_SIZE == (520, 240)
        assert DETECTION_PSTH_SIZE == (520, 100)
        assert view.plot_canvas.width() == 520 + DETECTION_PLOT_LEFT_MARGIN
        assert view.plot_canvas.height() == 240 + 100 + 28

        psth_cache = getattr(view.psth_plot, "_psth_cache", {})
        assert psth_cache["baseline_line"].isVisible() is False
        visible_horiz = [
            it for it in view.psth_plot.items
            if isinstance(it, pg.InfiniteLine) and it.angle == 0 and it.isVisible()
        ]
        assert visible_horiz == []

        # Waveform panel hidden when waveforms is None, visible when waveforms are provided
        assert view.waveform_panel.isVisibleTo(view) is False
        view.units_by_id[0].waveforms = np.random.default_rng(42).normal(0.0, 10.0, size=(20, 45))
        view._on_unit_selected()
        assert view.waveform_panel.isHidden() is False

        # Pairplot unit_clicked signal focuses that unit
        view.pairplot_widget.unit_clicked.emit(2)
        assert view._focused_unit_id == 2
    finally:
        view.close()


def test_wp9_live_tagging_undo_reset_and_manual_override_persistence(qapp, tmp_path: Path) -> None:
    """Verify no Apply button, live tagging, Undo, Reset to defaults, manual override persistence, and Save."""
    exp_folder = tmp_path / "wp9_live_session"
    create_synthetic_experiment_folder(exp_folder, n_units=4, n_pulses=20)

    view = DetectionView()
    try:
        assert view.load_experiment_folder(exp_folder) is True
        assert view._is_dirty is False

        # No Apply button
        apply_buttons = [
            b for b in view.findChildren(QPushButton)
            if "apply" in b.text().lower()
        ]
        assert apply_buttons == []

        row_u0 = _find_row_for_unit(view, 0)
        row_u1 = _find_row_for_unit(view, 1)
        row_u2 = _find_row_for_unit(view, 2)

        # Manual override on Unit 2 -> Tagged (manual)
        view.table_units.selectRow(row_u2)
        view.btn_toggle_tag.click()
        assert view.table_units.item(row_u2, 7).text() == "✓ M"
        assert "(manual)" in view.lbl_focused_status.text()

        # Live criteria edit: set latency threshold to 5.0 ms -> Unit 1 (~7.5 ms) immediately untagged
        assert view.table_units.item(row_u1, 7).text() == "✓"
        view.row_latency.edit_threshold.setText("5.0")
        view.row_latency.edit_threshold.editingFinished.emit()

        assert view.table_units.item(row_u1, 7).text() == "○"
        assert view.table_units.item(row_u0, 7).text() == "✓"
        assert view.table_units.item(row_u2, 7).text() == "✓ M"  # manual override survived!
        assert "<b>1</b> of 4 pass" in view.row_latency.lbl_pass_count.text()
        assert "Tagged: <b>2</b>" in view.lbl_summary_counts.text()
        assert view._is_dirty is True
        assert view.btn_criteria_undo.isEnabled() is True

        # Undo restores previous threshold (8.0 ms) and retags Unit 1
        view.btn_criteria_undo.click()
        assert view.row_latency.threshold() == 8.0
        assert view.table_units.item(row_u1, 7).text() == "✓"
        assert view.table_units.item(row_u2, 7).text() == "✓ M"

        # Toggle checkbox off, then Reset to defaults
        view.row_jitter.chk_enabled.setChecked(False)
        assert view.row_jitter.is_enabled() is False
        view.btn_criteria_reset.click()
        assert view.row_jitter.is_enabled() is True
        assert view.row_latency.threshold() == 8.0
        assert view.row_jitter.threshold() == 2.0
        assert view.row_rel.threshold() == 0.60
        assert view.row_salt.threshold() == 0.05
        assert view.row_wf.threshold() == 0.90

        # Save writes CSV + _spikes.npz without PNGs and clears dirty state
        view.btn_save.click()
        csv_path = exp_folder / "wp9_live_session_opto_curated.csv"
        npz_path = exp_folder / "wp9_live_session_spikes.npz"
        plots_dir = exp_folder / "wp9_live_session_plots"
        assert csv_path.exists()
        assert npz_path.exists()
        assert list(plots_dir.glob("*.png")) == []
        assert view._is_dirty is False
    finally:
        view.close()


def test_wp9_re_enabling_latency_while_sorted_by_status_retags_all_units(qapp, tmp_path: Path) -> None:
    """Regression: _apply_thresholds must disable table sorting mid-loop so no units are skipped."""
    from PyQt6.QtCore import Qt
    from optocurator.ui.view_detection import _COL_STATUS

    exp_folder = tmp_path / "wp9_sort_retag_session"
    create_synthetic_experiment_folder(exp_folder, n_units=8, n_pulses=20)

    view = DetectionView()
    try:
        assert view.load_experiment_folder(exp_folder) is True
        for uid in range(4):
            view.metrics[uid].latency_ms = 12.0 + uid
            view.metrics[uid].salt_p_value = 0.005
        view._populate_table()
        view.sort_by_column(_COL_STATUS, Qt.SortOrder.AscendingOrder)

        # Untick all except SALT, change SALT p, focus a unit with latency > 8 ms, then re-enable latency
        view.row_latency.chk_enabled.setChecked(False)
        view.row_jitter.chk_enabled.setChecked(False)
        view.row_rel.chk_enabled.setChecked(False)
        view.row_wf.chk_enabled.setChecked(False)
        view.row_salt.edit_threshold.setText("0.01")
        view.row_salt.edit_threshold.editingFinished.emit()
        view.table_units.selectRow(0)
        view._on_unit_selected()

        view.row_latency.chk_enabled.setChecked(True)

        for r in range(view.table_units.rowCount()):
            uid = view._row_unit_id(r)
            m = view.metrics[uid]
            if m.latency_ms > 8.0:
                assert m.is_tagged is False
                assert "○" in view.table_units.item(r, _COL_STATUS).text()
    finally:
        view.close()


def test_detection_manual_override_undo_and_stack_cleared_on_session_load(qapp, tmp_path: Path) -> None:
    """Manual tag overrides must be undoable and the undo stack must clear on loading a new session (Bug UI5)."""
    from optocurator.ui.view_detection import _COL_STATUS

    exp1 = tmp_path / "wp9_undo_sess1"
    exp2 = tmp_path / "wp9_undo_sess2"
    create_synthetic_experiment_folder(exp1, n_units=4, n_pulses=20)
    create_synthetic_experiment_folder(exp2, n_units=4, n_pulses=20)

    view = DetectionView()
    try:
        assert view.load_experiment_folder(exp1) is True
        row_u2 = _find_row_for_unit(view, 2)
        view.table_units.selectRow(row_u2)
        view._on_unit_selected()
        assert view.table_units.item(row_u2, _COL_STATUS).text() == "○"

        # Manual override must be recorded in undo stack and undone by undo_criteria_change()
        view.btn_toggle_tag.click()
        assert 2 in view.manual_overrides
        assert view.btn_criteria_undo.isEnabled() is True

        view.undo_criteria_change()
        assert 2 not in view.manual_overrides
        assert view.table_units.item(row_u2, _COL_STATUS).text() == "○"

        # Change criteria to populate undo stack, then load a second session -> stack must be cleared
        view.row_latency.edit_threshold.setText("4.5")
        view.row_latency.edit_threshold.editingFinished.emit()
        assert len(view._criteria_undo_stack) >= 1

        assert view.load_experiment_folder(exp2) is True
        assert len(view._criteria_undo_stack) == 0
        assert view.btn_criteria_undo.isEnabled() is False
    finally:
        view.close()
