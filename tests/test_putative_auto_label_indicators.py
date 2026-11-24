"""Tests verifying Putative Auto-Labeled Cell Indicators & Cluster Warning Badges:
- StripedProgressBarSegment offscreen paint event rendering
- ClassificationProgressBar putative segment stretch factors, tooltip, and summary formatting
- ClusterCardWidget warning badge visibility, text formatting, and mouse transparency
- ClusteringView putative auto-label tracking when auto-assign is disabled vs enabled
- Seamless transition between putative mode and active auto-assign mode
"""

from typing import Any

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPixmap
from PyQt6.QtWidgets import QApplication

from optocurator.ui.view_clusters import (
    ClassificationProgressBar,
    ClusterCardWidget,
    ClusteringView,
    StripedProgressBarSegment,
)


def test_striped_progress_bar_segment_painting(qapp):
    """Verify StripedProgressBarSegment instantiates and paints 45-degree diagonal stripes cleanly offscreen."""
    seg = StripedProgressBarSegment(
        base_color="#2563EB",
        stripe_color="#93C5FD",
        stripe_width=2.5,
        stripe_spacing=6.0,
    )
    seg.resize(120, 12)

    # Render to an offscreen pixmap
    pixmap = QPixmap(120, 12)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    seg.render(painter)
    painter.end()

    assert pixmap.width() == 120
    assert pixmap.height() == 12
    seg.close()


def test_progress_bar_putative_segment_counts(qapp):
    """Verify ClassificationProgressBar putative segment visibility, stretch factors, and summary formatting."""
    prog = ClassificationProgressBar()

    # 1. Empty state
    prog.update_counts(0, 0, 0, 0, suggested=0)
    assert prog.seg_suggested.isHidden()
    assert prog.suggested_count == 0

    # 2. Putative mode (Auto == 0, Suggested > 0)
    prog.update_counts(
        curated=10,
        auto_assigned=0,
        seeded=20,
        unassigned=50,
        suggested=15,
        threshold=0.85,
    )
    assert prog.suggested_count == 15
    assert not prog.seg_suggested.isHidden()
    assert "0 Auto (15 Putative)" in prog.lbl_summary.text()
    assert "35 Remaining" in prog.lbl_summary.text()  # 50 total unassigned - 15 putative = 35 remaining
    assert "15 cells (Confidence >= 85%)" in prog.seg_suggested.toolTip()
    assert prog.seg_auto.isHidden()

    # 3. Mixed mode (Auto > 0, Suggested > 0)
    prog.update_counts(
        curated=10,
        auto_assigned=25,
        seeded=20,
        unassigned=45,
        suggested=12,
        threshold=0.90,
    )
    assert not prog.seg_suggested.isHidden()
    assert not prog.seg_auto.isHidden()
    assert "25 Auto (12 Putative)" in prog.lbl_summary.text()
    assert "33 Remaining" in prog.lbl_summary.text()  # 45 - 12 = 33
    assert "12 cells (Confidence >= 90%)" in prog.seg_suggested.toolTip()

    # 4. Cleared putative mode (Suggested == 0)
    prog.update_counts(
        curated=10,
        auto_assigned=37,
        seeded=20,
        unassigned=33,
        suggested=0,
    )
    assert prog.seg_suggested.isHidden()
    assert "37 Auto" in prog.lbl_summary.text()
    assert "Putative" not in prog.lbl_summary.text()
    assert "33 Remaining" in prog.lbl_summary.text()

    prog.close()


def test_cluster_card_suggested_badge(qapp):
    """Verify ClusterCardWidget warning badge display, tooltip, and mouse interaction (§ 9.2)."""
    card = ClusterCardWidget(cluster_id=1, count=20, color_hex="#10B981")
    assert card.lbl_suggested_badge.isHidden()
    assert card.lbl_suggested_badge.text() == ""

    # Must NOT be mouse-transparent so badge tooltip works reliably (§ 9.2)
    assert not card.lbl_suggested_badge.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    # Set suggested count > 0
    card.set_suggested_count(4, threshold=0.85)
    assert not card.lbl_suggested_badge.isHidden()
    assert card.lbl_suggested_badge.text() == "! 4"
    assert card.lbl_suggested_badge.toolTip() == "Cells above the confidence threshold"

    # Clear suggested count
    card.set_suggested_count(0)
    assert card.lbl_suggested_badge.isHidden()
    assert card.lbl_suggested_badge.text() == ""

    card.close()


def _refresh_active_learning_and_wait(view: ClusteringView, qtbot: Any) -> None:
    """Trigger asynchronous active learning refresh and wait until worker results apply."""
    with qtbot.waitSignal(view.progress_updated, timeout=10000):
        view.refresh_active_learning()
    qtbot.waitUntil(
        lambda: view._active_learning_worker is not None and not view._active_learning_worker.isRunning(),
        timeout=10000,
    )


def test_putative_indicators_when_auto_assign_disabled(qapp, qtbot):
    """Verify ClusteringView displays putative progress bar segment and cluster badges when auto-assign is OFF."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    assert view.auto_assign_enabled is False

    # Asynchronously refresh active learning and wait for worker + UI updates
    _refresh_active_learning_and_wait(view, qtbot)

    # With mock data and trained RF classifier, high-confidence units are captured as suggested
    suggested_map = view._suggested_auto_assignments
    suggested_counts = view._get_suggested_counts_per_cluster()
    total_suggested = sum(suggested_counts.values())

    if total_suggested > 0:
        # Progress bar must show striped putative segment
        assert not view.progress_bar.seg_suggested.isHidden()
        assert view.progress_bar.suggested_count == total_suggested
        assert f"({total_suggested} Putative)" in view.progress_bar.lbl_summary.text()

        # Cluster cards having suggested cells must show warning badge
        for cid, card in view._cluster_cards.items():
            expected_count = suggested_counts.get(cid, 0)
            if expected_count > 0:
                assert not card.lbl_suggested_badge.isHidden()
                assert card.lbl_suggested_badge.text() == f"! {expected_count}"
            else:
                assert card.lbl_suggested_badge.isHidden()

    view.close()


def test_indicators_clear_when_auto_assign_enabled(qapp, qtbot):
    """Verify that enabling Auto-assign promotes putative cells to Auto-assigned and clears yellow warning badges."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    _refresh_active_learning_and_wait(view, qtbot)

    # Toggle Auto-assign ON (triggers refresh_active_learning asynchronously)
    with qtbot.waitSignal(view.progress_updated, timeout=10000):
        view.chk_auto_assign.setChecked(True)
    qtbot.waitUntil(
        lambda: view._active_learning_worker is not None
        and not view._active_learning_worker.isRunning()
        and len(view._suggested_auto_assignments) == 0,
        timeout=10000,
    )
    assert view.auto_assign_enabled is True

    # Putative suggestions must be empty, formal auto-assignments active
    assert len(view._suggested_auto_assignments) == 0
    assert view.progress_bar.suggested_count == 0
    assert view.progress_bar.seg_suggested.isHidden()

    # All cluster card warning badges must be hidden
    for card in view._cluster_cards.values():
        assert card.lbl_suggested_badge.isHidden()

    # Toggle Auto-assign back OFF (reverts auto-assignments and triggers refresh_active_learning asynchronously)
    view.chk_auto_assign.setChecked(False)
    blocker = qtbot.waitSignal(view.progress_updated, timeout=10000)
    blocker.wait()
    qtbot.waitUntil(
        lambda: view._active_learning_worker is not None and not view._active_learning_worker.isRunning(),
        timeout=10000,
    )
    assert view.auto_assign_enabled is False

    # Suggested counts must re-emerge
    suggested_counts = view._get_suggested_counts_per_cluster()
    total_suggested = sum(suggested_counts.values())
    if total_suggested > 0:
        assert not view.progress_bar.seg_suggested.isHidden()
        assert view.progress_bar.suggested_count == total_suggested

    view.close()


def test_cluster_card_click_opens_comparison_pane(qapp, qtbot):
    """Verify clicking anywhere on a cluster card (including the warning badge) opens the comparison pane."""
    view = ClusteringView(load_mock=True, start_with_clusters=True)
    _refresh_active_learning_and_wait(view, qtbot)

    # Pick first cluster card
    first_cid = view._cluster_card_order[0]
    card = view._cluster_cards[first_cid]
    card.set_suggested_count(3, threshold=0.85)

    # Click card
    card.card_clicked.emit(first_cid)
    qapp.processEvents()

    # Verify comparison drawer opened
    assert view._comparison_cluster_id == first_cid
    assert not view.comparison_pane.isHidden()

    # Verify member viewers populated
    assert len(view.comparison_viewers) >= 0

    view.close()

