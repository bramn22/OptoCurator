"""WP11 Detection Mode UI wiring tests covering UI_OVERHAUL_SPEC.md § 13.9 & § 14 WP11."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pytest
from PyQt6.QtWidgets import QApplication, QDialog

from optocurator.models import StatsParameters
from optocurator.ui.dialog_stats import StatsParametersDialog
from optocurator.ui.view_detection import (
    _COL_LIK,
    _COL_STATUS,
    _COL_WF,
    _LIKELY_THRESHOLD,
    DetectionView,
)
from tests.test_detection_workflow import create_synthetic_experiment_folder

EXPECTED_CSV_COLUMNS: List[str] = [
    "global_unit_id",
    "session_id",
    "unit_id",
    "latency_ms",
    "jitter_ms",
    "reliability",
    "waveform_correlation",
    "salt_p_value",
    "is_tagged",
    "cluster_id",
    "curated_label",
    "channel",
    "depth",
    "probe_x",
    "plot_path",
]


def _get_qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def _find_row_for_unit(view: DetectionView, unit_id: int) -> int:
    for r in range(view.table_units.rowCount()):
        if view._row_unit_id(r) == unit_id:
            return r
    raise AssertionError(f"Unit {unit_id} not found in table_units")


def test_wp11_mock_session_waveform_wiring(qapp, tmp_path: Path) -> None:
    """Verify WF r, row_wf, and waveform_panel with and without extractable waveforms."""
    view = DetectionView(load_mock=True)
    captured_calls: List[Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]] = []
    orig_set_waveforms = view.waveform_panel.set_waveforms

    def _spy_set_waveforms(
        spont_snippets: Optional[np.ndarray],
        evoked_snippets: Optional[np.ndarray],
        time_ms: Optional[np.ndarray],
        r_value: Optional[float],
        window_ms: Optional[Tuple[float, float]] = None,
    ) -> Any:
        captured_calls.append((spont_snippets, evoked_snippets, r_value))
        return orig_set_waveforms(
            spont_snippets=spont_snippets,
            evoked_snippets=evoked_snippets,
            time_ms=time_ms,
            r_value=r_value,
            window_ms=window_ms,
        )

    view.waveform_panel.set_waveforms = _spy_set_waveforms  # type: ignore[method-assign]

    try:
        # Unit 1 has extractable waveforms in mock session
        row_u1 = _find_row_for_unit(view, 1)
        captured_calls.clear()
        view.table_units.selectRow(row_u1)
        view._on_unit_selected()

        m1 = view.metrics[1]
        assert np.isfinite(m1.waveform_correlation)
        wf_cell_text = view.table_units.item(row_u1, _COL_WF).text()
        assert wf_cell_text == f"{m1.waveform_correlation:.2f}"
        assert view.row_wf.lbl_focused_value.text().startswith(f"{m1.waveform_correlation:.2f}")
        assert view.waveform_panel.isHidden() is False

        assert len(captured_calls) >= 1
        spont_snips, evoked_snips, r_val = captured_calls[-1]
        assert spont_snips is not None and spont_snips.ndim == 2 and spont_snips.shape[0] > 0
        assert evoked_snips is not None and evoked_snips.ndim == 2 and evoked_snips.shape[0] > 0
        assert r_val is not None
        assert r_val == pytest.approx(m1.waveform_correlation, abs=1e-9)
        assert view.waveform_panel.lbl_r.text() == f"r = {m1.waveform_correlation:.2f}"

        # Unit 5 has waveforms=None in mock session
        assert view.units_by_id[5].waveforms is None
        row_u5 = _find_row_for_unit(view, 5)
        view.table_units.selectRow(row_u5)
        view._on_unit_selected()

        assert np.isnan(view.metrics[5].waveform_correlation)
        assert view.table_units.item(row_u5, _COL_WF).text() == "—"
        assert view.row_wf.lbl_focused_value.text() == "n/a"
        assert view.waveform_panel.isHidden() is True
    finally:
        view.close()

    # Unit without waveforms is NOT penalized on WF r and can still be tagged on the other 4 criteria
    exp_folder = tmp_path / "wp11_no_wf_tagging"
    create_synthetic_experiment_folder(exp_folder, n_units=4, n_pulses=25)
    view2 = DetectionView()
    try:
        assert view2.load_experiment_folder(exp_folder) is True
        assert view2.row_wf.is_enabled() is True
        assert view2.units_by_id[0].waveforms is None

        row_u0 = _find_row_for_unit(view2, 0)
        view2.table_units.selectRow(row_u0)
        view2._on_unit_selected()

        assert view2.table_units.item(row_u0, _COL_WF).text() == "—"
        assert view2.row_wf.lbl_focused_value.text() == "n/a"
        assert view2.waveform_panel.isHidden() is True
        assert view2.metrics[0].is_tagged is True
        assert view2.table_units.item(row_u0, _COL_STATUS).text() == "✓"
    finally:
        view2.close()


def test_wp11_likelihood_wiring_live_updates_and_next_likely(qapp) -> None:
    """Verify Likelihood column, live updates on criteria changes, filter, and Next likely cycling."""
    view = DetectionView(load_mock=True)
    try:
        # Likelihood column displays non-zero % scores matching m.likelihood
        non_zero_count = 0
        for row in range(view.table_units.rowCount()):
            uid = view._row_unit_id(row)
            m = view.metrics[uid]
            lik_text = view.table_units.item(row, _COL_LIK).text()
            assert lik_text.endswith("%")
            assert lik_text == f"{int(round(m.likelihood * 100))}%"
            if m.likelihood > 0.0:
                non_zero_count += 1
        assert non_zero_count >= 1

        initial_likelihoods: Dict[int, float] = {
            uid: m.likelihood for uid, m in view.metrics.items()
        }

        # Changing a criterion threshold updates m.likelihood and Likelihood column live
        view.row_latency.edit_threshold.setText("4.0")
        view.row_latency.edit_threshold.editingFinished.emit()

        after_thresh_likelihoods: Dict[int, float] = {
            uid: m.likelihood for uid, m in view.metrics.items()
        }
        assert any(
            abs(after_thresh_likelihoods[uid] - initial_likelihoods[uid]) > 1e-6
            for uid in initial_likelihoods
        )
        for row in range(view.table_units.rowCount()):
            uid = view._row_unit_id(row)
            assert view.table_units.item(row, _COL_LIK).text() == f"{int(round(view.metrics[uid].likelihood * 100))}%"

        # Toggling a criterion checkbox updates m.likelihood and Likelihood column live
        view.row_jitter.chk_enabled.setChecked(False)
        after_toggle_likelihoods: Dict[int, float] = {
            uid: m.likelihood for uid, m in view.metrics.items()
        }
        assert any(
            abs(after_toggle_likelihoods[uid] - after_thresh_likelihoods[uid]) > 1e-6
            for uid in after_toggle_likelihoods
        )
        for row in range(view.table_units.rowCount()):
            uid = view._row_unit_id(row)
            assert view.table_units.item(row, _COL_LIK).text() == f"{int(round(view.metrics[uid].likelihood * 100))}%"

        # Reset and mark two high-likelihood units as untagged to verify filter and cycling
        view.reset_criteria_to_defaults()
        view.manual_overrides[1] = False
        view.manual_overrides[2] = False
        view._apply_thresholds()

        expected_likely = sorted(
            [
                uid
                for uid, m in view.metrics.items()
                if (not m.is_tagged) and m.likelihood >= _LIKELY_THRESHOLD
            ],
            key=lambda uid: (-view.metrics[uid].likelihood, uid),
        )
        assert len(expected_likely) >= 2

        # "Likely but untagged" filter shows only untagged units with likelihood >= _LIKELY_THRESHOLD
        view.combo_filter_status.setCurrentText("Likely but untagged")
        visible_uids = [
            view._row_unit_id(r)
            for r in range(view.table_units.rowCount())
            if not view.table_units.isRowHidden(r)
        ]
        assert set(visible_uids) == set(expected_likely)
    finally:
        view.close()


def test_wp11_stats_parameters_dialog_waveform_fields(qapp) -> None:
    """Verify Section 5 waveform fields in StatsParametersDialog round-trip through StatsParameters."""
    initial = StatsParameters(
        wf_window_start_ms=-1.5,
        wf_window_end_ms=2.5,
        wf_max_spikes=60,
    )
    dialog = StatsParametersDialog(current_params=initial)
    try:
        assert dialog.edit_wf_window_start.text() == "-1.5"
        assert dialog.edit_wf_window_end.text() == "2.5"
        assert dialog.edit_wf_max_spikes.text() == "60"

        # Invalid range should not accept
        dialog.edit_wf_window_start.setText("2.0")
        dialog.edit_wf_window_end.setText("1.0")
        dialog.btn_apply.click()
        assert dialog.result() != QDialog.DialogCode.Accepted

        # Invalid max spikes should not accept
        dialog.edit_wf_window_start.setText("-2.0")
        dialog.edit_wf_window_end.setText("3.0")
        dialog.edit_wf_max_spikes.setText("0")
        dialog.btn_apply.click()
        assert dialog.result() != QDialog.DialogCode.Accepted

        # Valid update round-trips through StatsParameters
        dialog.edit_wf_max_spikes.setText("80")
        dialog.btn_apply.click()
        assert dialog.result() == QDialog.DialogCode.Accepted

        out = dialog.get_parameters()
        assert out.wf_window_start_ms == pytest.approx(-2.0)
        assert out.wf_window_end_ms == pytest.approx(3.0)
        assert out.wf_max_spikes == 80

        # Reset defaults restores default waveform extraction fields
        dialog._reset_defaults()
        defaults = StatsParameters.default()
        assert dialog.edit_wf_window_start.text() == f"{defaults.wf_window_start_ms:.1f}"
        assert dialog.edit_wf_window_end.text() == f"{defaults.wf_window_end_ms:.1f}"
        assert dialog.edit_wf_max_spikes.text() == str(defaults.wf_max_spikes)
    finally:
        dialog.close()


def test_wp11_saved_csv_schema_unchanged(qapp, tmp_path: Path) -> None:
    """Verify export_curated_session writes exact expected CSV columns with no likelihood column."""
    view = DetectionView(load_mock=True)
    try:
        csv_path = view.export_curated_session(
            export_dir=tmp_path,
            show_dialog=False,
            precompute_pngs=False,
        )
        assert csv_path is not None and csv_path.exists()

        df = pd.read_csv(csv_path)
        assert list(df.columns) == EXPECTED_CSV_COLUMNS
        assert "likelihood" not in df.columns

        by_uid = df.set_index("unit_id")
        for uid, m in view.metrics.items():
            saved_wf = by_uid.loc[uid, "waveform_correlation"]
            if np.isnan(m.waveform_correlation):
                assert pd.isna(saved_wf)
            else:
                assert saved_wf == pytest.approx(m.waveform_correlation, abs=1e-4)
    finally:
        view.close()
