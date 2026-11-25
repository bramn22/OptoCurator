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
    """Verify 7-column units table, failing red cells, SALT floor, Show filters, and Probe."""
    exp_folder = tmp_path / "wp9_units_session"
    create_synthetic_experiment_folder(exp_folder, n_units=4, n_pulses=25)

    view = DetectionView()
    try:
        assert view.load_experiment_folder(exp_folder) is True

        # 7 columns with expected headers
        assert view.table_units.columnCount() == 7
        headers = [view.table_units.horizontalHeaderItem(i).text() for i in range(7)]
        assert headers == ["Unit", "Lat.", "Jit.", "Rel.", "SALT p", "Likelihood", "Status"]

        row_u0 = _find_row_for_unit(view, 0)
        row_u2 = _find_row_for_unit(view, 2)

        # Unit 2 fails latency (18 ms > 8 ms) -> red COLOR_DANGER foreground
        assert view.table_units.item(row_u2, 1).foreground().color().name().lower() == COLOR_DANGER.lower()

        # SALT <= 0.01 display with exact sort_key
        salt_item_u0 = view.table_units.item(row_u0, 4)
        assert salt_item_u0.text() == "≤ 0.01"
        assert salt_item_u0.sort_key <= 0.01

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
            uid: (m.latency_ms, m.jitter_ms, m.reliability, m.salt_p_value)
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
            assert (m.latency_ms, m.jitter_ms, m.reliability, m.salt_p_value) == snap

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
        assert view.table_units.item(row_u2, 6).text() == "✓ M"
        assert "(manual)" in view.lbl_focused_status.text()

        # Live criteria edit: set latency threshold to 5.0 ms -> Unit 1 (~7.5 ms) immediately untagged
        assert view.table_units.item(row_u1, 6).text() == "✓"
        view.row_latency.edit_threshold.setText("5.0")
        view.row_latency.edit_threshold.editingFinished.emit()

        assert view.table_units.item(row_u1, 6).text() == "○"
        assert view.table_units.item(row_u0, 6).text() == "✓"
        assert view.table_units.item(row_u2, 6).text() == "✓ M"  # manual override survived!
        assert "<b>1</b> of 4 pass" in view.row_latency.lbl_pass_count.text()
        assert "Tagged: <b>2</b>" in view.lbl_summary_counts.text()
        assert view._is_dirty is True
        assert view.btn_criteria_undo.isEnabled() is True

        # Undo restores previous threshold (8.0 ms) and retags Unit 1
        view.btn_criteria_undo.click()
        assert view.row_latency.threshold() == 8.0
        assert view.table_units.item(row_u1, 6).text() == "✓"
        assert view.table_units.item(row_u2, 6).text() == "✓ M"

        # Toggle checkbox off, then Reset to defaults
        view.row_jitter.chk_enabled.setChecked(False)
        assert view.row_jitter.is_enabled() is False
        view.btn_criteria_reset.click()
        assert view.row_jitter.is_enabled() is True
        assert view.row_latency.threshold() == 8.0
        assert view.row_jitter.threshold() == 2.0
        assert view.row_rel.threshold() == 0.60
        assert view.row_salt.threshold() == 0.05

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


def test_wp9_pairplot_3x3_salt_layout_and_vertical_stacking(qapp, tmp_path: Path) -> None:
    """Verify all 6 lower-triangle subplots exist and pairplot_widget is stacked vertically below plot_canvas."""
    from PyQt6.QtCore import QPoint

    exp_folder = tmp_path / "wp9_pairplot_layout"
    create_synthetic_experiment_folder(exp_folder, n_units=4, n_pulses=20)

    view = DetectionView()
    view.resize(1340, 900)
    try:
        assert view.load_experiment_folder(exp_folder) is True
        view.show()
        qapp.processEvents()

        pp = view.pairplot_widget
        for attr in (
            "plot_jit_lat",
            "plot_rel_lat",
            "plot_rel_jit",
            "plot_salt_lat",
            "plot_salt_jit",
            "plot_salt_rel",
        ):
            assert hasattr(pp, attr), f"Missing pairplot subplot {attr}"
            assert isinstance(getattr(pp, attr), pg.PlotWidget)

        canvas_pos = view.plot_canvas.mapTo(view, QPoint(0, 0))
        pairplot_pos = pp.mapTo(view, QPoint(0, 0))
        assert pairplot_pos.y() >= canvas_pos.y() + view.plot_canvas.height()

        # Horizontal alignment: Col 0 left edge and Col 2 right edge align with plot_canvas within <= 2 px
        canvas_left = canvas_pos.x()
        canvas_right = canvas_left + view.plot_canvas.width()
        col0_left = pp.plot_salt_lat.mapTo(view, QPoint(0, 0)).x()
        col2_right = pp.plot_salt_rel.mapTo(view, QPoint(pp.plot_salt_rel.width(), 0)).x()
        assert abs(col0_left - canvas_left) <= 2
        assert abs(col2_right - canvas_right) <= 2
    finally:
        view.close()


def test_wp9_pairplot_salt_log10_mapping_clipping_nan_compat_and_click(qapp) -> None:
    """Verify log10 SALT p mapping, 0.0 clipping to -3.0, NaN omission, backward compat, and click-to-focus."""
    import pytest
    from optocurator.ui.detection_widgets import DetectionPairplotWidget

    pp = DetectionPairplotWidget()
    try:
        pp.set_data(
            unit_ids=[10, 11, 12, 13, 14, 15],
            latencies=[2.0, 3.5, 6.0, 12.0, 4.0, 5.0],
            jitters=[0.4, 0.8, 1.5, 3.0, 0.6, 1.0],
            reliabilities=[0.95, 0.85, 0.65, 0.25, 0.90, 0.80],
            is_tagged=[True, True, False, False, True, False],
            focused_unit_id=10,
            salt_p_values=[0.001, 0.01, 0.1, 1.0, 0.0, float("nan")],
            salt_max=0.05,
            salt_enabled=True,
        )
        qapp.processEvents()

        expected_y = {
            10: -3.0,  # 0.001 -> -3.0
            11: -2.0,  # 0.01  -> -2.0
            12: -1.0,  # 0.1   -> -1.0
            13: 0.0,   # 1.0   ->  0.0
            14: -3.0,  # 0.0   -> clipped to -3.0
        }

        for key in ("salt_lat", "salt_jit", "salt_rel"):
            cell = pp._cells[key]
            plotted: dict[int, float] = {}
            for scatter_key in ("tagged", "untagged"):
                for pt in cell[scatter_key].points():
                    plotted[int(pt.data())] = float(pt.pos().y())
            assert set(plotted.keys()) == {10, 11, 12, 13, 14}
            assert 15 not in plotted  # NaN SALT p omitted cleanly
            for uid, exp_y in expected_y.items():
                assert plotted[uid] == pytest.approx(exp_y, abs=1e-6)

        # Click-to-focus on a SALT subplot emits unit_clicked
        clicked_uids: list[int] = []
        pp.unit_clicked.connect(clicked_uids.append)
        salt_lat_tagged = pp._cells["salt_lat"]["tagged"]
        target_pts = [pt for pt in salt_lat_tagged.points() if int(pt.data()) == 11]
        assert len(target_pts) == 1
        salt_lat_tagged.sigClicked.emit(salt_lat_tagged, target_pts, None)
        assert clicked_uids == [11]

        # Backward compatibility: set_data without salt_p_values works cleanly
        pp.set_data(
            unit_ids=[10, 11, 12],
            latencies=[2.0, 3.5, 6.0],
            jitters=[0.4, 0.8, 1.5],
            reliabilities=[0.95, 0.85, 0.65],
            is_tagged=[True, True, False],
            focused_unit_id=10,
        )
        qapp.processEvents()
        assert len(pp._cells["jit_lat"]["tagged"].points()) == 2
        for key in ("salt_lat", "salt_jit", "salt_rel"):
            assert len(pp._cells[key]["tagged"].points()) == 0
            assert len(pp._cells[key]["untagged"].points()) == 0
    finally:
        pp.close()


def test_wp9_pairplot_salt_criterion_updates_hline_and_pass_rect(qapp, tmp_path: Path) -> None:
    """Toggling row_salt enabled/threshold updates hline and pass_rect on all 3 SALT subplots."""
    import pytest

    exp_folder = tmp_path / "wp9_pairplot_salt_criteria"
    create_synthetic_experiment_folder(exp_folder, n_units=4, n_pulses=20)

    view = DetectionView()
    try:
        assert view.load_experiment_folder(exp_folder) is True
        pp = view.pairplot_widget
        salt_keys = ("salt_lat", "salt_jit", "salt_rel")

        # Default salt_max = 0.05, enabled = True
        default_log = float(np.log10(0.05))
        for key in salt_keys:
            cell = pp._cells[key]
            assert cell["hline"].isVisible() is True
            assert cell["hline"].value() == pytest.approx(default_log, abs=1e-6)
            assert cell["pass_rect"].isVisible() is True
            rect = cell["pass_rect"].rect()
            assert rect.top() == pytest.approx(-3.0, abs=1e-6)
            assert rect.bottom() == pytest.approx(default_log, abs=1e-6)

        # Updating row_salt threshold to 0.01 moves hline and pass_rect top/bottom to -2.0
        view.row_salt.edit_threshold.setText("0.01")
        view.row_salt.edit_threshold.editingFinished.emit()
        qapp.processEvents()

        for key in salt_keys:
            cell = pp._cells[key]
            assert cell["hline"].isVisible() is True
            assert cell["hline"].value() == pytest.approx(-2.0, abs=1e-6)
            rect = cell["pass_rect"].rect()
            assert rect.top() == pytest.approx(-3.0, abs=1e-6)
            assert rect.bottom() == pytest.approx(-2.0, abs=1e-6)

        # Disabling row_salt hides hline and expands pass_rect across full SALT y-range [-3.0, 0.0]
        view.row_salt.chk_enabled.setChecked(False)
        qapp.processEvents()

        for key in salt_keys:
            cell = pp._cells[key]
            assert cell["hline"].isVisible() is False
            assert cell["pass_rect"].isVisible() is True
            rect = cell["pass_rect"].rect()
            assert rect.top() == pytest.approx(-3.0, abs=1e-6)
            assert rect.bottom() == pytest.approx(0.0, abs=1e-6)
    finally:
        view.close()
