"""WP5 verification tests: Focused unit column (header, probability bar chart, KDE grid, actions & Undo/Redo/Unassign)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QTextDocument
from PyQt6.QtTest import QTest

from optocurator.ui.theme import (
    COLOR_DIVIDER,
    COLOR_TEXT,
    FOCUSED_UNIT_HEADER_HEIGHT,
)
from optocurator.ui.view_clusters import (
    ClusterProbabilityBarWidget,
    ClusteringView,
    MarginalDistributionsWidget,
    _BoldTopClusterAxisItem,
)


def _make_view() -> ClusteringView:
    return ClusteringView(load_mock=True, start_with_clusters=True)


def _plain_html(html: str) -> str:
    doc = QTextDocument()
    doc.setHtml(html)
    return doc.toPlainText().strip()


def test_wp5_focused_unit_header_and_return_to_suggested_button(qapp):
    """Header has 64px height, title/session labels, and '↩ Suggested unit' button + G shortcut."""
    view = _make_view()
    view.show()
    qapp.processEvents()

    assert view.focused_unit_header.height() == FOCUSED_UNIT_HEADER_HEIGHT == 64

    # Ensure active learning is on and we have an active candidate + another unit
    view.set_active_learning_mode(True)
    qapp.processEvents()
    assert view.active_candidate is not None
    cand_gid = view.active_candidate.global_unit_id

    # Find a different unit in the pool
    all_gids = list(view._pooled_units_map.keys())
    other_gid = next(g for g in all_gids if g != cand_gid)

    # Focus the active candidate: button should be hidden
    view.select_pooled_unit(cand_gid)
    qapp.processEvents()
    rec_cand = view._pooled_units_map[cand_gid]
    assert view.lbl_focused_unit_title.text() == f"Unit {rec_cand['unit_id']}"
    assert str(rec_cand["session_id"]).split("_")[0] in view.lbl_focused_unit_session.text()
    assert view.btn_return_to_candidate.isHidden()

    # Focus a different unit while AL is ON: button becomes visible
    view.select_pooled_unit(other_gid)
    qapp.processEvents()
    rec_other = view._pooled_units_map[other_gid]
    assert view.lbl_focused_unit_title.text() == f"Unit {rec_other['unit_id']}"
    assert not view.btn_return_to_candidate.isHidden()
    assert view.btn_return_to_candidate.isEnabled()

    # Turning AL OFF hides the button even when inspecting another unit
    view.set_active_learning_mode(False)
    qapp.processEvents()
    assert view.btn_return_to_candidate.isHidden()

    # Turning AL back ON restores button visibility while focused on other_gid
    view.set_active_learning_mode(True)
    qapp.processEvents()
    assert not view.btn_return_to_candidate.isHidden()

    # Clicking the button returns to active suggestion and hides button
    view.btn_return_to_candidate.click()
    qapp.processEvents()
    assert view._current_selected_gid == cand_gid
    assert view.btn_return_to_candidate.isHidden()

    # Inspect other_gid again and press 'G' to return to active suggestion
    view.select_pooled_unit(other_gid)
    qapp.processEvents()
    assert not view.btn_return_to_candidate.isHidden()
    QTest.keyClick(view, Qt.Key.Key_G)
    qapp.processEvents()
    assert view._current_selected_gid == cand_gid
    assert view.btn_return_to_candidate.isHidden()

    view.close()


def test_wp5_cluster_probability_bar_chart_widget(qapp):
    """Probability bar chart renders all clusters in order, bolds top C<id>, formats 0-100%, tooltip, click & outline."""
    bar_widget = ClusterProbabilityBarWidget()
    bar_widget.show()
    qapp.processEvents()

    assert bar_widget.lbl_title.text() == "Cluster probability"
    assert bar_widget.lbl_threshold_legend.text() == "- - Auto-assign 85%"
    assert isinstance(bar_widget._bottom_axis, _BoldTopClusterAxisItem)

    left_ticks = bar_widget.plot_widget.getAxis("left")._tickLevels
    assert left_ticks is not None
    assert left_ticks[0] == [(0.0, "0%"), (0.5, "50%"), (1.0, "100%")]

    cluster_order = [2, 4, 0]
    probs = {0: 0.22, 2: 0.37, 4: 0.41}
    colors = {0: "#1f77b4", 2: "#ff7f0e", 4: "#2ca02c"}
    names = {0: "Alpha", 2: "Beta", 4: "Name"}

    bar_widget.set_probabilities(
        probabilities=probs,
        cluster_colors=colors,
        cluster_names=names,
        threshold=0.85,
        cluster_order=cluster_order,
    )
    qapp.processEvents()

    assert len(bar_widget.bar_items) == 3
    assert bar_widget._bar_cids == [2, 4, 0]
    assert bar_widget._bottom_axis._bold_label == "C4"

    bottom_ticks = bar_widget.plot_widget.getAxis("bottom")._tickLevels
    assert bottom_ticks is not None
    assert bottom_ticks[0] == [(0, "C2"), (1, "C4"), (2, "C0")]

    # Hover over bar index 1 (C4) -> tooltip "Name (C4): 41%"
    vb = bar_widget.plot_widget.getViewBox()
    scene_pos_c4 = vb.mapViewToScene(QPointF(1.0, 0.2))
    bar_widget._on_mouse_moved(scene_pos_c4)
    assert bar_widget.plot_widget.toolTip() == "Name (C4): 41%"

    # Hover outside bars clears tooltip
    scene_pos_empty = vb.mapViewToScene(QPointF(1.45, 0.2))
    bar_widget._on_mouse_moved(scene_pos_empty)
    assert bar_widget.plot_widget.toolTip() == ""

    # Clicking bar index 1 emits cluster_clicked(4)
    emitted = []
    bar_widget.cluster_clicked.connect(emitted.append)
    fake_click = SimpleNamespace(
        button=lambda: Qt.MouseButton.LeftButton,
        scenePos=lambda: scene_pos_c4,
    )
    bar_widget._on_mouse_clicked(fake_click)
    assert emitted == [4]

    # set_inspected_cluster applies dark COLOR_TEXT outline to that bar and COLOR_DIVIDER to others
    bar_widget.set_inspected_cluster(4)
    for cid, bar in zip(bar_widget._bar_cids, bar_widget.bar_items):
        pen_col = bar.opts["pen"].color().name().upper()
        if cid == 4:
            assert pen_col == COLOR_TEXT.upper()
        else:
            assert pen_col == COLOR_DIVIDER.upper()

    bar_widget.close()


def test_wp5_cluster_probability_bar_click_opens_inspector_in_view(qapp):
    """Clicking a probability bar in ClusteringView opens that cluster in the inspector and outlines the bar."""
    view = _make_view()
    view.show()
    qapp.processEvents()

    assert len(view.cluster_prob_bar._bar_cids) > 0
    target_idx = 0
    target_cid = view.cluster_prob_bar._bar_cids[target_idx]

    vb = view.cluster_prob_bar.plot_widget.getViewBox()
    scene_pos = vb.mapViewToScene(QPointF(float(target_idx), 0.1))
    fake_click = SimpleNamespace(
        button=lambda: Qt.MouseButton.LeftButton,
        scenePos=lambda: scene_pos,
    )
    view.cluster_prob_bar._on_mouse_clicked(fake_click)
    qapp.processEvents()

    assert view._comparison_cluster_id == target_cid
    assert view.comparison_pane.isVisible()
    pen_col = view.cluster_prob_bar.bar_items[target_idx].opts["pen"].color().name().upper()
    assert pen_col == COLOR_TEXT.upper()

    view.close()


def test_wp5_response_metrics_kde_grid_legend_lines_overlays_and_values(qapp):
    """Response metrics KDE grid shows 4 mini-plots, dynamic legend, dark unit lines, cluster overlays, and formatted values."""
    widget = MarginalDistributionsWidget()
    widget.show()
    qapp.processEvents()

    assert widget.lbl_title.text() == "Response metrics"
    assert _plain_html(widget.lbl_legend.text()) == "■ All · | Unit"
    assert list(widget.plots.keys()) == ["latency_ms", "jitter_ms", "reliability", "salt_p_value"]

    pop_df = pd.DataFrame(
        {
            "latency_ms": [4.0, 6.0, 8.0, 12.4, 15.0],
            "jitter_ms": [0.5, 0.8, 1.2, 1.8, 2.5],
            "reliability": [0.40, 0.65, 0.85, 0.90, 0.95],
            "salt_p_value": [0.002, 0.01, 0.032, 0.15, 0.50],
        }
    )
    widget.update_distributions(pop_df)

    # Set finite unit values including SALT <= 0.01 floor
    widget.set_candidate_values(
        {
            "latency_ms": 12.4,
            "jitter_ms": 1.2,
            "reliability": 0.85,
            "salt_p_value": 0.004,
        }
    )
    qapp.processEvents()

    assert widget.value_items["latency_ms"].toPlainText() == "12.4"
    assert widget.value_items["jitter_ms"].toPlainText() == "1.2"
    assert widget.value_items["reliability"].toPlainText() == "0.85"
    assert widget.value_items["salt_p_value"].toPlainText() == "≤ 0.01"

    for key, line in widget.lines.items():
        assert line.isVisible()
        pen = line.pen
        assert pen.color().name().upper() == COLOR_TEXT.upper()
        assert pen.style() == Qt.PenStyle.SolidLine

    # Test SALT > 0.01 formatting ("0.03") and NaN/None ("—")
    widget.set_candidate_values(
        {
            "latency_ms": float("nan"),
            "jitter_ms": None,
            "reliability": 0.50,
            "salt_p_value": 0.032,
        }
    )
    assert widget.value_items["latency_ms"].toPlainText() == "—"
    assert widget.value_items["jitter_ms"].toPlainText() == "—"
    assert not widget.lines["latency_ms"].isVisible()
    assert not widget.lines["jitter_ms"].isVisible()
    assert widget.value_items["salt_p_value"].toPlainText() == "0.03"

    # Cluster overlay updates legend and curves, and clear resets them
    cluster_df = pop_df.iloc[:3].copy()
    widget.update_cluster_distribution(cluster_df, color_hex="#E74C3C", cluster_name="Fast PV")
    qapp.processEvents()
    assert _plain_html(widget.lbl_legend.text()) == "■ All · ■ Fast PV · | Unit"
    for key, curve in widget.cluster_curves.items():
        x_data, y_data = curve.getData()
        assert x_data is not None and len(x_data) > 0
        assert y_data is not None and len(y_data) > 0

    widget.clear_cluster_distribution()
    qapp.processEvents()
    assert _plain_html(widget.lbl_legend.text()) == "■ All · | Unit"
    for curve in widget.cluster_curves.values():
        x_data, _ = curve.getData()
        assert x_data is None or len(x_data) == 0

    widget.close()


def test_wp5_action_row_undo_redo_and_unassign(qapp):
    """Action row has Undo/Skip/Unassign (Save hidden), Unassign visibility tracks cluster_id>=0, Undo/Redo & Backspace work."""
    view = _make_view()
    view.show()
    qapp.processEvents()

    assert hasattr(view, "btn_undo") and not view.btn_undo.isHidden()
    assert hasattr(view, "btn_skip_candidate") and not view.btn_skip_candidate.isHidden()
    assert hasattr(view, "btn_unassign")
    assert view.btn_save_pool.isHidden()

    # Pick an initially assigned unit and focus it -> btn_unassign is visible
    test_gid = next(
        gid for gid, rec in view._pooled_units_map.items() if int(rec.get("cluster_id", -1)) >= 0
    )
    view.select_pooled_unit(test_gid)
    qapp.processEvents()
    assert not view.btn_unassign.isHidden()
    assert view.btn_unassign.isEnabled()

    # Unassign it via unassign_focused_unit() -> cluster_id becomes -1 and btn_unassign hides
    assert view.unassign_focused_unit()
    qapp.processEvents()
    assert view._pooled_units_map[test_gid]["cluster_id"] == -1
    assert view._unit_status_map[test_gid] == "Unassigned"
    assert view.btn_unassign.isHidden()

    # Assign to an existing cluster
    target_cid = next(iter(view._cluster_cards.keys()))
    assert view.assign_candidate_to_cluster(target_cid, unit_id=test_gid)
    qapp.processEvents()

    assert view._pooled_units_map[test_gid]["cluster_id"] == target_cid
    assert view._unit_status_map[test_gid] == "Curated"
    assert view.btn_undo.isEnabled()

    # Focus the assigned unit -> Unassign button becomes visible and enabled
    view.select_pooled_unit(test_gid)
    qapp.processEvents()
    assert not view.btn_unassign.isHidden()
    assert view.btn_unassign.isEnabled()

    # Undo restores unassigned state AND previous focused unit
    assert view.undo()
    qapp.processEvents()
    assert view._pooled_units_map[test_gid]["cluster_id"] == -1
    assert view._unit_status_map[test_gid] == "Unassigned"
    assert view._current_selected_gid == test_gid
    assert view.btn_unassign.isHidden()

    # Redo restores assignment
    assert view.redo()
    qapp.processEvents()
    assert view._pooled_units_map[test_gid]["cluster_id"] == target_cid
    assert view._unit_status_map[test_gid] == "Curated"

    # Focus assigned unit and test Backspace unassigns and hides btn_unassign
    view.select_pooled_unit(test_gid)
    qapp.processEvents()
    assert not view.btn_unassign.isHidden()

    QTest.keyClick(view, Qt.Key.Key_Backspace)
    qapp.processEvents()
    assert view._pooled_units_map[test_gid]["cluster_id"] == -1
    assert view._unit_status_map[test_gid] == "Unassigned"
    assert view.btn_unassign.isHidden()

    view.close()
