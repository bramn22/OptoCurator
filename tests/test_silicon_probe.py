"""Unit and integration tests for the multi-shank SiliconProbeWidget (horizontal unit bars + vertical depth KDE)."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QContextMenuEvent
from PyQt6.QtWidgets import QApplication, QMenu

from optocurator.ui.figure_export import export_widget_to_svg
from optocurator.ui.plots import SiliconProbeWidget, detect_probe_shanks
from optocurator.ui.theme import (
    COLOR_PROBE_SHANK_FILL,
    COLOR_SUCCESS,
    COLOR_TEXT_MUTED,
    DETECTION_PROBE_LEGEND_HTML,
    PLOT_PROBE_HEIGHT,
    PLOT_PROBE_KDE_WIDTH,
    PLOT_PROBE_LEFT_MARGIN,
    PLOT_PROBE_SHANK_WIDTH,
    PLOT_PROBE_SIZE,
)
from optocurator.ui.view_clusters import ClusteringView
from optocurator.ui.view_detection import DetectionView
from tests.test_detection_workflow import create_synthetic_experiment_folder


def _make_shank_positions(n_shanks: int, shank_pitch_um: float = 250.0) -> np.ndarray:
    """Build synthetic multi-shank channel_positions (N, 2) with 2 columns per shank."""
    rows = []
    for s in range(n_shanks):
        base_x = s * shank_pitch_um
        for depth in np.linspace(0.0, 800.0, 17):
            rows.append((base_x, depth))
            rows.append((base_x + 32.0, depth + 20.0))
    return np.asarray(rows, dtype=np.float64)


def test_detect_probe_shanks_1_2_4_and_fallback() -> None:
    """detect_probe_shanks handles 1-shank, 2-shank, 4-shank, unit fallback, and empty fallback."""
    # 1-shank
    pos_1 = _make_shank_positions(1)
    shanks_1 = detect_probe_shanks(pos_1)
    assert len(shanks_1) == 1
    assert shanks_1[0]["shank_index"] == 0
    assert shanks_1[0]["x_min"] == pytest.approx(0.0)
    assert shanks_1[0]["x_max"] == pytest.approx(32.0)
    assert shanks_1[0]["x_median"] == pytest.approx(16.0)

    # 2-shank
    pos_2 = _make_shank_positions(2, shank_pitch_um=250.0)
    shanks_2 = detect_probe_shanks(pos_2)
    assert len(shanks_2) == 2
    assert [s["shank_index"] for s in shanks_2] == [0, 1]
    assert shanks_2[1]["x_min"] == pytest.approx(250.0)

    # 4-shank
    pos_4 = _make_shank_positions(4, shank_pitch_um=250.0)
    shanks_4 = detect_probe_shanks(pos_4)
    assert len(shanks_4) == 4
    assert [s["shank_index"] for s in shanks_4] == [0, 1, 2, 3]
    assert shanks_4[3]["x_median"] == pytest.approx(766.0)

    # Fallback: no channel_positions, infer from unit_x / unit_y
    shanks_u = detect_probe_shanks(
        None,
        unit_x=[0.0, 16.0, 300.0, 316.0],
        unit_y=[100.0, 400.0, 200.0, 500.0],
    )
    assert len(shanks_u) == 2
    assert shanks_u[0]["y_min"] == pytest.approx(100.0)
    assert shanks_u[0]["y_max"] == pytest.approx(400.0)
    assert shanks_u[1]["x_median"] == pytest.approx(308.0)

    # Fallback: neither channel_positions nor units -> 1 default shank [0, 1000]
    shanks_empty = detect_probe_shanks(None)
    assert len(shanks_empty) == 1
    assert shanks_empty[0]["y_min"] == pytest.approx(0.0)
    assert shanks_empty[0]["y_max"] == pytest.approx(1000.0)


def test_silicon_probe_dynamic_sizing_and_clear_reset(qapp: QApplication) -> None:
    """Widget width scales dynamically as 44 + 32 * n_shanks + 46 (height 268) and resets on clear_probe()."""
    probe = SiliconProbeWidget()
    try:
        assert (probe.width(), probe.height()) == (122, 268)
        assert (probe.width(), probe.height()) == PLOT_PROBE_SIZE

        for n_shanks, expected_w in ((1, 122), (2, 154), (4, 218)):
            assert expected_w == PLOT_PROBE_LEFT_MARGIN + PLOT_PROBE_SHANK_WIDTH * n_shanks + PLOT_PROBE_KDE_WIDTH
            pos = _make_shank_positions(n_shanks)
            units = [
                {"id": i, "probe_x": float(i * 250.0), "depth": 200.0 + i * 50.0, "color": COLOR_SUCCESS}
                for i in range(n_shanks)
            ]
            probe.set_probe_data(pos, units, focused_id=0, bg_kde_depths=[200.0, 300.0], fg_kde_depths=[200.0])
            qapp.processEvents()
            assert (probe.width(), probe.height()) == (expected_w, PLOT_PROBE_HEIGHT)
            assert len(probe._shank_rect_items) == n_shanks

        probe.clear_probe()
        qapp.processEvents()
        assert (probe.width(), probe.height()) == (122, 268)
        assert len(probe.units_scatter.points()) == 0
        assert probe.cand_marker.isVisible() is False
        assert probe._kde_fg_curve.isVisible() is False
    finally:
        probe.close()


def test_silicon_probe_subcolumns_priority_focused_box_and_sites_compat(qapp: QApplication) -> None:
    """Verify dual sub-column placement, foreground priority ordering, focused box, and hidden sites_scatter compat."""
    probe = SiliconProbeWidget()
    try:
        pos = _make_shank_positions(2, shank_pitch_um=250.0)
        units = [
            # Foreground unit on shank 0 left sub-column (px=0 <= x_median=16)
            {"id": "u_fg", "probe_x": 0.0, "depth": 300.0, "color": COLOR_SUCCESS, "alpha": 255, "priority": 1},
            # Background unit on shank 0 right sub-column (px=32 > x_median=16)
            {"id": "u_bg0", "probe_x": 32.0, "depth": 300.0, "color": COLOR_TEXT_MUTED, "alpha": 140, "priority": 0},
            # Background unit on shank 1 left sub-column (px=250 <= x_median=266)
            {"id": "u_bg1", "probe_x": 250.0, "depth": 500.0, "color": COLOR_TEXT_MUTED, "alpha": 140, "priority": 0},
        ]
        probe.set_probe_data(
            channel_positions=pos,
            units=units,
            focused_id="u_fg",
            bg_kde_depths=[300.0, 300.0, 500.0],
            fg_kde_depths=[300.0],
        )
        qapp.processEvents()

        # Priority ordering: background spots (priority=0) precede foreground spots (priority=1)
        pts = probe.units_scatter.points()
        ordered_ids = [pt.data() for pt in pts]
        assert ordered_ids == ["u_bg0", "u_bg1", "u_fg"]

        pos_by_id = {pt.data(): (float(pt.pos().x()), float(pt.pos().y())) for pt in pts}
        assert pos_by_id["u_fg"] == (pytest.approx(-0.17), pytest.approx(300.0))
        assert pos_by_id["u_bg0"] == (pytest.approx(0.17), pytest.approx(300.0))
        assert pos_by_id["u_bg1"] == (pytest.approx(1.0 - 0.17), pytest.approx(500.0))

        # Focused box positioned at focused unit's exact (plot_x, depth)
        assert probe.cand_marker.isVisible() is True
        cand_pts = probe.cand_marker.points()
        assert len(cand_pts) == 1
        assert float(cand_pts[0].pos().x()) == pytest.approx(-0.17)
        assert float(cand_pts[0].pos().y()) == pytest.approx(300.0)

        # Backward-compat sites_scatter populated with channel_positions and kept hidden
        sx, sy = probe.sites_scatter.getData()
        assert len(sx) == len(pos)
        assert len(sy) == len(pos)
        assert probe.sites_scatter.isVisible() is False
    finally:
        probe.close()


def test_detection_mode_probe_tagged_untagged_bars_and_depth_kdes(qapp: QApplication, tmp_path: Path) -> None:
    """Detection mode renders tagged (green) vs untagged (grey) bars and both background & tagged depth KDEs."""
    exp_folder = tmp_path / "probe_detection_sess"
    create_synthetic_experiment_folder(exp_folder, n_units=4, n_pulses=25)

    view = DetectionView()
    try:
        assert view.load_experiment_folder(exp_folder) is True
        qapp.processEvents()

        probe = view.probe_widget
        pts = probe.units_scatter.points()
        assert len(pts) == len(view.units)

        tagged_uids = {uid for uid, m in view.metrics.items() if m.is_tagged}
        untagged_uids = {uid for uid, m in view.metrics.items() if not m.is_tagged}
        assert len(tagged_uids) >= 1 and len(untagged_uids) >= 1

        # All horizontal unit bars are opaque (alpha == 255) to avoid alpha-stacking blur
        assert all(pt.brush().color().alpha() == 255 for pt in pts)

        # Probe legend lists Tagged & Untagged without Focused unit
        assert "Tagged" in DETECTION_PROBE_LEGEND_HTML
        assert "Untagged" in DETECTION_PROBE_LEGEND_HTML
        assert "Focused unit" not in DETECTION_PROBE_LEGEND_HTML

        # Tagged units drawn after untagged units with COLOR_SUCCESS brush
        colors_by_uid = {pt.data(): pt.brush().color().name().upper() for pt in pts}
        for uid in tagged_uids:
            assert colors_by_uid[uid] == COLOR_SUCCESS.upper()
        for uid in untagged_uids:
            assert colors_by_uid[uid] != COLOR_SUCCESS.upper()

        ordered_uids = [pt.data() for pt in pts]
        last_untagged_idx = max(ordered_uids.index(u) for u in untagged_uids)
        first_tagged_idx = min(ordered_uids.index(u) for u in tagged_uids)
        assert first_tagged_idx > last_untagged_idx

        # Both background (all units) and foreground (tagged) vertical depth KDEs are visible & populated
        assert probe._kde_bg_curve.isVisible() is True
        assert probe._kde_fg_curve.isVisible() is True
        bg_x, _ = probe._kde_bg_curve.getData()
        fg_x, _ = probe._kde_fg_curve.getData()
        assert bg_x is not None and len(bg_x) > 0 and float(np.max(bg_x)) == pytest.approx(1.0)
        assert fg_x is not None and len(fg_x) > 0 and float(np.max(fg_x)) == pytest.approx(1.0)
        assert probe._kde_fg_curve.opts["pen"].color().name().upper() == COLOR_SUCCESS.upper()
    finally:
        view.close()


def test_clustering_mode_probe_open_inspector_highlights_and_dims_others(qapp: QApplication) -> None:
    """Opening cluster inspector highlights open cluster's bars & depth KDE and dims other clusters; closing restores."""
    cview = ClusteringView()
    try:
        sess_id = "probe_cluster_sess"
        records = [
            {"global_unit_id": f"{sess_id}_u0000", "session_id": sess_id, "unit_id": 0, "cluster_id": 0, "probe_x": 0.0, "depth": 150.0, "latency_ms": 3.0, "jitter_ms": 0.5, "reliability": 0.9, "salt_p_value": 0.001, "status": "Manual"},
            {"global_unit_id": f"{sess_id}_u0001", "session_id": sess_id, "unit_id": 1, "cluster_id": 0, "probe_x": 16.0, "depth": 220.0, "latency_ms": 3.5, "jitter_ms": 0.6, "reliability": 0.85, "salt_p_value": 0.002, "status": "Manual"},
            {"global_unit_id": f"{sess_id}_u0002", "session_id": sess_id, "unit_id": 2, "cluster_id": 1, "probe_x": 0.0, "depth": 550.0, "latency_ms": 12.0, "jitter_ms": 2.0, "reliability": 0.5, "salt_p_value": 0.02, "status": "Manual"},
            {"global_unit_id": f"{sess_id}_u0003", "session_id": sess_id, "unit_id": 3, "cluster_id": 1, "probe_x": 16.0, "depth": 620.0, "latency_ms": 13.0, "jitter_ms": 2.2, "reliability": 0.45, "salt_p_value": 0.03, "status": "Manual"},
        ]
        cview._pooled_units_map = {r["global_unit_id"]: r for r in records}
        cview._current_selected_gid = f"{sess_id}_u0000"

        c0_color = cview._get_cluster_color(0).upper()
        c1_color = cview._get_cluster_color(1).upper()

        # 1) Normal state (no open cluster inspector): all bars opaque (alpha == 255) with full cluster colors
        cview._comparison_cluster_id = None
        cview._update_probe_view_for_unit(f"{sess_id}_u0000")
        qapp.processEvents()

        pts_normal = cview.probe_widget.units_scatter.points()
        assert all(pt.brush().color().alpha() == 255 for pt in pts_normal)
        colors_normal = {pt.data(): pt.brush().color().name().upper() for pt in pts_normal}
        assert colors_normal[f"{sess_id}_u0000"] == c0_color
        assert colors_normal[f"{sess_id}_u0002"] == c1_color

        # 2) Open cluster 1 inspector: all bars stay opaque (alpha == 255); cluster 1 keeps full color,
        #    cluster 0 uses a muted RGB pre-blended against COLOR_PROBE_SHANK_FILL
        cview._comparison_cluster_id = 1
        cview._update_probe_view_for_unit(f"{sess_id}_u0000")
        qapp.processEvents()

        pts_open = cview.probe_widget.units_scatter.points()
        assert all(pt.brush().color().alpha() == 255 for pt in pts_open)
        colors_open = {pt.data(): pt.brush().color().name().upper() for pt in pts_open}
        assert colors_open[f"{sess_id}_u0002"] == c1_color
        assert colors_open[f"{sess_id}_u0003"] == c1_color
        assert colors_open[f"{sess_id}_u0000"] != c0_color
        assert colors_open[f"{sess_id}_u0001"] != c0_color
        assert colors_open[f"{sess_id}_u0000"] != COLOR_PROBE_SHANK_FILL.upper()

        # Foreground depth KDE colored with Cluster 1's color
        assert cview.probe_widget._kde_fg_curve.isVisible() is True
        assert cview.probe_widget._kde_fg_curve.opts["pen"].color().name().upper() == c1_color

        # 3) Closing inspector restores full-saturation coloring (still opaque alpha == 255)
        cview._comparison_cluster_id = None
        cview._update_probe_view_for_unit(f"{sess_id}_u0000")
        qapp.processEvents()

        pts_restored = cview.probe_widget.units_scatter.points()
        assert all(pt.brush().color().alpha() == 255 for pt in pts_restored)
        colors_restored = {pt.data(): pt.brush().color().name().upper() for pt in pts_restored}
        assert colors_restored[f"{sess_id}_u0000"] == c0_color
        assert colors_restored[f"{sess_id}_u0002"] == c1_color
    finally:
        cview.close()


def test_silicon_probe_right_click_svg_export_both_canvases(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Right-clicking both plot_widget and kde_plot triggers probe SVG export producing valid pure-vector SVG."""
    probe = SiliconProbeWidget()
    try:
        pos = _make_shank_positions(2)
        units = [
            {"id": 0, "probe_x": 0.0, "depth": 200.0, "color": COLOR_SUCCESS, "alpha": 255, "priority": 1},
            {"id": 1, "probe_x": 266.0, "depth": 500.0, "color": COLOR_TEXT_MUTED, "alpha": 160, "priority": 0},
        ]
        probe.set_probe_data(pos, units, focused_id=0, bg_kde_depths=[200.0, 500.0], fg_kde_depths=[200.0])
        qapp.processEvents()

        captured_labels: list[list[str]] = []

        def _fake_exec(self: QMenu, *args, **kwargs):
            captured_labels.append([a.text() for a in self.actions() if not a.isSeparator()])
            return None

        monkeypatch.setattr(QMenu, "exec", _fake_exec)

        for vp in (probe.plot_widget.viewport(), probe.kde_plot.viewport()):
            assert vp is not None
            ev = QContextMenuEvent(
                QContextMenuEvent.Reason.Mouse,
                QPoint(10, 10),
                QPoint(120, 120),
                Qt.KeyboardModifier.NoModifier,
            )
            QApplication.sendEvent(vp, ev)
            assert captured_labels[-1] == ["Save probe figure as SVG…"]

        # Export full composite SiliconProbeWidget (shanks + KDE) to pure vector SVG
        svg_path = export_widget_to_svg(probe, tmp_path / "multi_shank_probe.svg")
        assert svg_path.exists()
        root = ET.fromstring(svg_path.read_text(encoding="utf-8"))
        assert root.tag.endswith("svg")
        images = [el for el in root.iter() if el.tag == "image" or el.tag.endswith("}image")]
        paths = [el for el in root.iter() if el.tag == "path" or el.tag.endswith("}path")]
        assert len(images) == 0
        assert len(paths) > 0
    finally:
        probe.close()


def test_clustering_mode_multi_cluster_probe_depth_kdes(qapp: QApplication) -> None:
    """Multi-cluster probe depth KDEs show all clusters when closed, single cluster when open, and clear on clear_probe."""
    from optocurator.ui.view_clusters import ClusterCardWidget

    cview = ClusteringView()
    try:
        sess_id = "probe_multi_kde_sess"
        records = [
            {"global_unit_id": f"{sess_id}_u0000", "session_id": sess_id, "unit_id": 0, "cluster_id": 0, "probe_x": 0.0, "depth": 120.0, "latency_ms": 3.0, "jitter_ms": 0.5, "reliability": 0.9, "salt_p_value": 0.001, "status": "Curated"},
            {"global_unit_id": f"{sess_id}_u0001", "session_id": sess_id, "unit_id": 1, "cluster_id": 0, "probe_x": 16.0, "depth": 180.0, "latency_ms": 3.2, "jitter_ms": 0.6, "reliability": 0.88, "salt_p_value": 0.002, "status": "Curated"},
            {"global_unit_id": f"{sess_id}_u0002", "session_id": sess_id, "unit_id": 2, "cluster_id": 1, "probe_x": 0.0, "depth": 420.0, "latency_ms": 9.0, "jitter_ms": 1.4, "reliability": 0.65, "salt_p_value": 0.01, "status": "Curated"},
            {"global_unit_id": f"{sess_id}_u0003", "session_id": sess_id, "unit_id": 3, "cluster_id": 1, "probe_x": 16.0, "depth": 480.0, "latency_ms": 9.5, "jitter_ms": 1.6, "reliability": 0.62, "salt_p_value": 0.015, "status": "Curated"},
            {"global_unit_id": f"{sess_id}_u0004", "session_id": sess_id, "unit_id": 4, "cluster_id": 2, "probe_x": 0.0, "depth": 720.0, "latency_ms": 15.0, "jitter_ms": 2.5, "reliability": 0.40, "salt_p_value": 0.04, "status": "Curated"},
            {"global_unit_id": f"{sess_id}_u0005", "session_id": sess_id, "unit_id": 5, "cluster_id": 2, "probe_x": 16.0, "depth": 780.0, "latency_ms": 16.0, "jitter_ms": 2.8, "reliability": 0.35, "salt_p_value": 0.05, "status": "Curated"},
        ]
        cview._pooled_units_map = {r["global_unit_id"]: r for r in records}
        cview._current_selected_gid = f"{sess_id}_u0000"
        cview._cluster_card_order = [0, 1, 2]
        cview._cluster_cards = {
            cid: ClusterCardWidget(cluster_id=cid, count=2, color_hex=cview._get_cluster_color(cid), parent=cview)
            for cid in (0, 1, 2)
        }

        c0_color = cview._get_cluster_color(0).upper()
        c1_color = cview._get_cluster_color(1).upper()
        c2_color = cview._get_cluster_color(2).upper()

        probe = cview.probe_widget

        # 1) No cluster card open -> all 3 clusters render their own foreground KDE curves in cluster colors
        cview._comparison_cluster_id = None
        cview._update_probe_view_for_unit(f"{sess_id}_u0000")
        qapp.processEvents()

        assert probe._kde_fg_curve.isVisible() is True
        assert len(probe._extra_kde_items) == 2
        kde_colors_all = {
            probe._kde_fg_curve.opts["pen"].color().name().upper(),
            *(curve.opts["pen"].color().name().upper() for _fill, curve in probe._extra_kde_items),
        }
        assert kde_colors_all == {c0_color, c1_color, c2_color}

        # 2) Open cluster 1 card via set_cluster_comparison -> only Cluster 1 KDE is rendered
        cview.set_cluster_comparison(1)
        qapp.processEvents()

        assert probe._kde_fg_curve.isVisible() is True
        assert len(probe._extra_kde_items) == 0
        assert probe._kde_fg_curve.opts["pen"].color().name().upper() == c1_color

        # 3) Close cluster card via clear_cluster_comparison -> all 3 clusters' KDEs reappear cleanly
        cview.clear_cluster_comparison(return_to_suggestion=False)
        qapp.processEvents()

        assert probe._kde_fg_curve.isVisible() is True
        assert len(probe._extra_kde_items) == 2
        kde_colors_restored = {
            probe._kde_fg_curve.opts["pen"].color().name().upper(),
            *(curve.opts["pen"].color().name().upper() for _fill, curve in probe._extra_kde_items),
        }
        assert kde_colors_restored == {c0_color, c1_color, c2_color}

        # 4) clear_probe() clears all extra KDE items and hides foreground KDE
        probe.clear_probe()
        qapp.processEvents()
        assert len(probe._extra_kde_items) == 0
        assert probe._kde_fg_curve.isVisible() is False
    finally:
        cview.close()
