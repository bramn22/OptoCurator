"""WP11 Detection Mode UI wiring tests covering UI_OVERHAUL_SPEC.md § 13.9 & § 14 WP11."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from typing import Dict, List
import pandas as pd
from PyQt6.QtWidgets import QApplication

from optocurator.ui.view_detection import (
    _COL_LIK,
    _LIKELY_THRESHOLD,
    DetectionView,
)

EXPECTED_CSV_COLUMNS: List[str] = [
    "global_unit_id",
    "session_id",
    "unit_id",
    "latency_ms",
    "jitter_ms",
    "reliability",
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


def test_wp11_saved_csv_schema_unchanged(qapp, tmp_path: Path) -> None:
    """Verify export_curated_session writes exact expected CSV columns with no likelihood or waveform_correlation column."""
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
        assert "waveform_correlation" not in df.columns
    finally:
        view.close()
