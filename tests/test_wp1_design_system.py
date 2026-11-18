"""Tests for Work Package 1 (WP1 — Design system) from UI_OVERHAUL_SPEC.md (§ 4, § 8.1, § 8.2, § 14)."""

import ast
from pathlib import Path
import re
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont

from optocurator.analysis.clustering import CLUSTER_PALETTE as ANALYSIS_CLUSTER_PALETTE
from optocurator.ui import plots, theme
from optocurator.ui.plots import (
    apply_plot_style,
    compute_standard_plot_ticks,
    render_psth,
    render_raster,
)
from optocurator.ui.theme import (
    CLUSTER_CARD_SIZE,
    CLUSTER_PALETTE,
    COLOR_ACCENT,
    COLOR_ACCENT_BG,
    COLOR_AXIS,
    COLOR_BORDER_STRONG,
    COLOR_CANVAS,
    COLOR_DANGER,
    COLOR_DIVIDER,
    COLOR_LIGHT_PULSE_BAND,
    COLOR_PANEL,
    COLOR_PASSING_FILL,
    COLOR_SEEDED,
    COLOR_SUCCESS,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_SECONDARY,
    COLOR_UNASSIGNED,
    COLOR_WARNING,
    COLOR_WARNING_BG,
    ClusterColorService,
    FONT_FALLBACK,
    FONT_FAMILY,
    FONT_FAMILY_MONO,
    FONT_NAME,
    FONT_SIZE_APP_TITLE,
    FONT_SIZE_BODY,
    FONT_SIZE_CAPTION,
    FONT_SIZE_HEADER_TITLE,
    FONT_SIZE_PANEL_TITLE,
    FONT_SIZE_TICK,
    FONT_SIZE_UNIT_TITLE,
    PLOT_CARD_PSTH_LEFT_MARGIN,
    PLOT_CARD_PSTH_SIZE,
    PLOT_CLUSTER_PROB_SIZE,
    PLOT_CRITERION_KDE_SIZE,
    PLOT_DETECTION_LEFT_MARGIN,
    PLOT_DETECTION_PSTH_SIZE,
    PLOT_DETECTION_RASTER_SIZE,
    PLOT_EMBEDDING_SIZE,
    PLOT_METRIC_KDE_SIZE,
    PLOT_PROBE_SIZE,
    PLOT_UNIT_RESPONSE_GAP,
    PLOT_UNIT_RESPONSE_LEFT_MARGIN,
    PLOT_UNIT_RESPONSE_PSTH_SIZE,
    PLOT_UNIT_RESPONSE_RASTER_SIZE,
    PLOT_WAVEFORM_SIZE,
    PSTH_LINE_WIDTH,
    color_for_cluster,
    make_font,
    perceived_luminance,
    text_color_for_cluster,
)
from optocurator.ui.view_clusters import ClusteringView


# ---------------------------------------------------------------------------
# 1. Theme Tokens: § 4.1 Typography, § 4.2 Colours, § 8.1 Fixed Plot Sizes
# ---------------------------------------------------------------------------


def test_wp1_typography_tokens_and_make_font(qapp):
    """Verify § 4.1 typography scale constants and make_font helper."""
    assert FONT_NAME == "Inter"
    assert FONT_FALLBACK == "Segoe UI"
    assert "Inter" in FONT_FAMILY
    assert "Segoe UI" in FONT_FAMILY
    assert "Consolas" in FONT_FAMILY_MONO or "monospace" in FONT_FAMILY_MONO

    # Exact pixel sizes from § 4.1
    assert FONT_SIZE_TICK == 10
    assert FONT_SIZE_CAPTION == 11
    assert FONT_SIZE_BODY == 12
    assert FONT_SIZE_PANEL_TITLE == 13
    assert FONT_SIZE_HEADER_TITLE == 16
    assert FONT_SIZE_UNIT_TITLE == 16
    assert FONT_SIZE_APP_TITLE == 32

    # Verify make_font returns QFont with pixelSize, weight, and italic
    f_body = make_font(FONT_SIZE_BODY)
    assert isinstance(f_body, QFont)
    assert f_body.pixelSize() == 12
    assert not f_body.italic()

    f_title = make_font(FONT_SIZE_PANEL_TITLE, weight=QFont.Weight.DemiBold, italic=True)
    assert f_title.pixelSize() == 13
    assert f_title.weight() == QFont.Weight.DemiBold
    assert f_title.italic() is True


def test_wp1_color_tokens():
    """Verify § 4.2 neutral UI palette, semantic accents, and plot data tokens."""
    # Surfaces & Borders
    assert COLOR_CANVAS == "#F6F7F9"
    assert COLOR_PANEL == "#FFFFFF"
    assert COLOR_DIVIDER == "#E3E5E8"
    assert COLOR_BORDER_STRONG == "#D0D5DB"

    # Text & Axes
    assert COLOR_TEXT == "#1F2328"
    assert COLOR_TEXT_SECONDARY == "#59636E"
    assert COLOR_TEXT_MUTED == "#6E7781"
    assert COLOR_AXIS == "#8C959F"

    # Semantic Accents
    assert COLOR_ACCENT == "#2563EB"
    assert COLOR_ACCENT_BG == "#EEF3FE"
    assert COLOR_SUCCESS == "#1A7F37"
    assert COLOR_WARNING == "#9A6700"
    assert COLOR_WARNING_BG == "#FFF1C2"
    assert COLOR_DANGER == "#CF222E"

    # Plot Data Tokens
    assert COLOR_LIGHT_PULSE_BAND == "#EAF2FD"
    assert COLOR_PASSING_FILL == "#BFD3F7"
    assert COLOR_SEEDED == "#8E7CC3"
    assert COLOR_UNASSIGNED == "#A0A4AA"


def test_wp1_fixed_plot_sizes():
    """Verify § 8.1 fixed plot dimensions across Clustering and Detection views."""
    # Clustering view plot sizes
    assert PLOT_UNIT_RESPONSE_RASTER_SIZE == (232, 110)
    assert PLOT_UNIT_RESPONSE_PSTH_SIZE == (232, 56)
    assert PLOT_UNIT_RESPONSE_GAP == 6
    assert PLOT_UNIT_RESPONSE_LEFT_MARGIN == 44

    assert PLOT_CARD_PSTH_SIZE == (150, 84)
    assert PLOT_CARD_PSTH_LEFT_MARGIN == 34

    assert PLOT_CLUSTER_PROB_SIZE == (232, 90)
    assert PLOT_METRIC_KDE_SIZE == (128, 44)
    assert PLOT_EMBEDDING_SIZE == (268, 268)
    assert PLOT_PROBE_SIZE == (76, 268)
    assert CLUSTER_CARD_SIZE == (217, 148)

    # Detection view plot sizes
    assert PLOT_DETECTION_RASTER_SIZE == (520, 240)
    assert PLOT_DETECTION_PSTH_SIZE == (520, 100)
    assert PLOT_DETECTION_LEFT_MARGIN == 52
    assert PLOT_WAVEFORM_SIZE == (240, 134)
    assert PLOT_CRITERION_KDE_SIZE == (240, 24)

    # PSTH line width (§ 8.2)
    assert PSTH_LINE_WIDTH == 1.5


# ---------------------------------------------------------------------------
# 2. Cluster Colour Palette & ClusterColorService (§ 4.3)
# ---------------------------------------------------------------------------


def test_wp1_cluster_palette_and_cycling():
    """Verify 15-colour Okabe-Ito + Paul Tol palette, cycling by cluster_id, and unassigned colour."""
    expected_palette = [
        "#E69F00",  # 0  Orange
        "#56B4E9",  # 1  Sky blue
        "#009E73",  # 2  Bluish green
        "#0072B2",  # 3  Blue
        "#D55E00",  # 4  Vermilion
        "#CC79A7",  # 5  Reddish purple
        "#332288",  # 6  Indigo
        "#88CCEE",  # 7  Cyan
        "#44AA99",  # 8  Teal
        "#117733",  # 9  Green
        "#999933",  # 10 Olive
        "#DDCC77",  # 11 Sand
        "#CC6677",  # 12 Rose
        "#882255",  # 13 Wine
        "#AA4499",  # 14 Purple
    ]
    assert CLUSTER_PALETTE == expected_palette
    assert ANALYSIS_CLUSTER_PALETTE == expected_palette
    assert len(CLUSTER_PALETTE) == 15

    # Unassigned (< 0) always returns #A0A4AA
    assert color_for_cluster(-1) == "#A0A4AA"
    assert color_for_cluster(-2) == "#A0A4AA"
    assert color_for_cluster(-99, {0: "#FF0000"}) == "#A0A4AA"

    # Palette cycling for >= 0
    for i in range(30):
        assert color_for_cluster(i) == expected_palette[i % 15]

    # Custom override takes precedence
    custom = {2: "#123456", 15: "#ABCDEF"}
    assert color_for_cluster(2, custom) == "#123456"
    assert color_for_cluster(15, custom) == "#ABCDEF"
    assert color_for_cluster(0, custom) == expected_palette[0]


def test_wp1_perceived_luminance_and_text_color_darkening():
    """Verify § 4.3 luminance check darkens light colours (> 0.6) to ~55% for text on white."""
    # Dark colour (e.g. #332288 Indigo, #0072B2 Blue): luminance <= 0.6 -> unchanged
    assert perceived_luminance("#332288") < 0.6
    assert text_color_for_cluster("#332288") == "#332288"
    assert perceived_luminance("#0072B2") < 0.6
    assert text_color_for_cluster("#0072B2") == "#0072B2"

    # Light colours in CLUSTER_PALETTE: #DDCC77 (Sand), #88CCEE (Cyan), #E69F00 (Orange), #56B4E9 (Sky blue)
    for light_hex in ("#DDCC77", "#88CCEE", "#E69F00", "#56B4E9"):
        lum = perceived_luminance(light_hex)
        assert lum > 0.6, f"Expected {light_hex} to have luminance > 0.6, got {lum}"
        darkened = text_color_for_cluster(light_hex)
        assert darkened != light_hex
        # Verify each RGB channel is scaled to ~55% of original
        orig_c = QColor(light_hex)
        dark_c = QColor(darkened)
        assert dark_c.red() == int(round(orig_c.red() * 0.55))
        assert dark_c.green() == int(round(orig_c.green() * 0.55))
        assert dark_c.blue() == int(round(orig_c.blue() * 0.55))
        assert perceived_luminance(darkened) < 0.6


def test_wp1_cluster_color_service():
    """Verify ClusterColorService single source of truth for cluster colours and text colours."""
    svc = ClusterColorService()

    # Default palette lookup & cycling
    assert svc.get_color(-1) == COLOR_UNASSIGNED
    assert svc.get_color(0) == CLUSTER_PALETTE[0]
    assert svc.get_color(11) == "#DDCC77"
    assert svc.get_color(15) == CLUSTER_PALETTE[0]

    # Swatch/plot colour vs text colour for light cluster (#DDCC77)
    assert svc.get_color(11) == "#DDCC77"
    assert svc.get_text_color(11) == text_color_for_cluster("#DDCC77")
    assert svc.get_text_color(11) != svc.get_color(11)

    # Unassigned text colour uses text_color_for_cluster(COLOR_UNASSIGNED)
    assert svc.get_text_color(-1) == text_color_for_cluster(COLOR_UNASSIGNED)

    # Custom override via set_color & set_colors
    svc.set_color(3, "#FFEE88")  # very bright yellow
    assert svc.get_color(3) == "#FFEE88"
    assert svc.get_text_color(3) == text_color_for_cluster("#FFEE88")
    assert svc.to_dict() == {3: "#FFEE88"}

    # Invalid hex or negative cluster_id ignored
    svc.set_color(-1, "#FF0000")
    svc.set_color(3, "not-a-color")
    assert svc.get_color(-1) == COLOR_UNASSIGNED
    assert svc.get_color(3) == "#FFEE88"

    # Reset clears overrides
    svc.reset()
    assert svc.to_dict() == {}
    assert svc.get_color(3) == CLUSTER_PALETTE[3]


# ---------------------------------------------------------------------------
# 3. Cross-View Cluster Colour Consistency in ClusteringView (§ 4.3 & § 14)
# ---------------------------------------------------------------------------


def test_wp1_cluster_color_consistency_across_all_clustering_views(qapp):
    """Verify one cluster has the exact same colour across every view in ClusteringView:
    1. Cluster card (swatch/header border & mean PSTH curve)
    2. Comparison pane (border & header title text colour)
    3. Embedding scatter point
    4. Silicon probe dot
    5. Cluster probability bar
    6. Marginal KDE overlay
    7. Pooled units table (using text_color_for_cluster on table text)
    """
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    assert view.current_clustering_result is not None
    active_cids = sorted(view.current_clustering_result.cluster_summaries.keys())
    assert len(active_cids) >= 2

    def _assert_cluster_color_everywhere(target_cid: int, expected_hex: str) -> None:
        expected_hex_upper = expected_hex.upper()
        expected_text_upper = text_color_for_cluster(expected_hex).upper()

        # 1. Cluster card (border uses swatch hex; title header uses text_color_for_cluster per § 4.3)
        card = view._cluster_cards[target_cid]
        assert card.color_hex.upper() == expected_hex_upper
        assert card.mean_curve.opts["pen"].color().name().upper() == expected_hex_upper
        assert expected_hex_upper in card.styleSheet().upper()
        assert expected_text_upper in card.header.styleSheet().upper()

        # 2. Comparison pane & 6. Marginal KDE overlay (activated when comparing target_cid)
        view.set_cluster_comparison(target_cid)
        assert expected_hex_upper in view.lbl_inspector_swatch.styleSheet().upper()
        assert expected_text_upper in view.lbl_comparison_title.styleSheet().upper()
        for kde_curve in view.marginal_widget.cluster_curves.values():
            assert kde_curve.opts["pen"].color().name().upper() == expected_hex_upper

        # 3. Embedding scatter point
        scatter_spots = view.scatter_item.points()
        matching_scatter = [
            s for s in scatter_spots
            if s.data() in view._pooled_units_map
            and view._pooled_units_map[s.data()].get("cluster_id") == target_cid
        ]
        assert len(matching_scatter) > 0
        for spot in matching_scatter:
            assert spot.brush().color().name().upper() == expected_hex_upper

        # 4. Silicon probe dot (select a unit in target_cid so its session is active on the probe)
        target_gid = matching_scatter[0].data()
        view.select_pooled_unit(target_gid)
        qapp.processEvents()
        probe_pts = view.probe_units_scatter.points()
        matching_probe = [
            pt for pt in probe_pts
            if pt.data() in view._pooled_units_map
            and view._pooled_units_map[pt.data()].get("cluster_id") == target_cid
        ]
        assert len(matching_probe) > 0
        for pt in matching_probe:
            assert pt.brush().color().name().upper() == expected_hex_upper

        # 5. Cluster probability bar
        probs = {cid: 1.0 / len(active_cids) for cid in active_cids}
        view.cluster_prob_bar.set_probabilities(
            probabilities=probs,
            cluster_colors=view._cluster_colors,
            cluster_names=view._cluster_names,
            threshold=view.auto_assign_threshold,
        )
        cid_idx = active_cids.index(target_cid)
        bar_item = view.cluster_prob_bar.bar_items[cid_idx]
        assert bar_item.opts["brush"].color().name().upper() == expected_hex_upper

        # 7. Pooled units table (uses text_color_for_cluster)
        table_matches = 0
        for r in range(view.table_pooled_units.rowCount()):
            gid = view._get_gid_for_row(r)
            if view._pooled_units_map[gid].get("cluster_id") == target_cid:
                item = view.table_pooled_units.item(r, 2)
                assert item.foreground().color().name().upper() == expected_text_upper
                table_matches += 1
        assert table_matches > 0

        view.clear_cluster_comparison()

    # Check default palette colours (Cluster 0 is #E69F00 which has luminance > 0.6; Cluster 2 is #009E73)
    for cid in active_cids:
        _assert_cluster_color_everywhere(cid, CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)])

    # Check custom colour overrides (both a dark colour and a high-luminance colour)
    custom_colors = {
        active_cids[0]: "#1E40AF",  # Dark blue (luminance < 0.6)
        active_cids[1]: "#FDE047",  # Bright yellow (luminance > 0.6 -> darkened in table text)
    }
    view.apply_cluster_customizations(
        new_order=list(view._cluster_card_order),
        new_names=dict(view._cluster_names),
        new_colors=custom_colors,
    )
    _assert_cluster_color_everywhere(active_cids[0], "#1E40AF")
    _assert_cluster_color_everywhere(active_cids[1], "#FDE047")

    view.close()


# ---------------------------------------------------------------------------
# 4. Plot Styling, Ticks, Raster, and PSTH Rules (§ 8.2)
# ---------------------------------------------------------------------------


def test_wp1_compute_standard_plot_ticks():
    """Verify § 8.2 default window ticks at [-20, -10, 0, 5, 10, 20, 30, 40] and compact mode."""
    # Standard mode: all 8 ticks present and labeled
    ticks_std = compute_standard_plot_ticks(-30.0, 50.0, compact=False)
    assert len(ticks_std) == 1
    major_std = ticks_std[0]
    positions = [pos for pos, _ in major_std]
    labels = {pos: lbl for pos, lbl in major_std}
    assert positions == [-20.0, -10.0, 0.0, 5.0, 10.0, 20.0, 30.0, 40.0]
    for pos in positions:
        assert labels[pos] == f"{int(pos)}"

    # Compact mode (for 150x84 cluster card PSTHs): labels only -20, 0, 10, 20, 40
    ticks_compact = compute_standard_plot_ticks(-30.0, 50.0, compact=True)
    major_compact = ticks_compact[0]
    assert [pos for pos, _ in major_compact] == [-20.0, -10.0, 0.0, 5.0, 10.0, 20.0, 30.0, 40.0]
    compact_labels = {pos: lbl for pos, lbl in major_compact}
    for labeled_pos in (-20.0, 0.0, 10.0, 20.0, 40.0):
        assert compact_labels[labeled_pos] == f"{int(labeled_pos)}"
    for unlabeled_pos in (-10.0, 5.0, 30.0):
        assert compact_labels[unlabeled_pos] == ""


def test_wp1_apply_plot_style_and_render_raster_and_psth(qapp):
    """Verify § 8.2 plot rules:
    - No top/right spines, no plot title, autoSIPrefix disabled
    - #EAF2FD light-pulse band (0-10 ms) and dashed t=0 line in both raster and PSTH
    - 1.5 px PSTH line width and no visible baseline/3sigma line in PSTH
    """
    raster_widget = pg.PlotWidget()
    psth_widget = pg.PlotWidget()

    raster_x = np.array([-5.0, 2.0, 4.5, 8.0, 15.0])
    raster_y = np.array([0, 1, 2, 3, 4])
    psth_bins = np.linspace(-30.0, 50.0, 41)
    psth_rates = np.abs(np.sin(psth_bins[:-1] * 0.1)) * 25.0

    render_raster(
        raster_widget,
        raster_x=raster_x,
        raster_y=raster_y,
        n_trials=5,
        title="Should Be Ignored",
        x_range=(-30.0, 50.0),
        show_x_axis=True,
    )
    render_psth(
        psth_widget,
        psth_bins=psth_bins,
        psth_rates=psth_rates,
        baseline_mean=5.0,
        baseline_3sigma=15.0,
        title="Should Also Be Ignored",
        x_range=(-30.0, 50.0),
        show_x_axis=True,
        is_smooth=True,
    )

    for w in (raster_widget, psth_widget):
        pi = w.getPlotItem()
        # No top or right spines
        assert not pi.getAxis("top").isVisible()
        assert not pi.getAxis("right").isVisible()
        # No title inside plot
        assert pi.titleLabel.text in (None, "")
        # autoSIPrefix disabled on left and bottom axes
        assert pi.getAxis("left").autoSIPrefix is False
        assert pi.getAxis("bottom").autoSIPrefix is False
        # Bottom ticks match standard [-20, -10, 0, 5, 10, 20, 30, 40]
        tick_levels = pi.getAxis("bottom")._tickLevels
        assert tick_levels is not None and len(tick_levels) > 0
        tick_vals = [pos for pos, _ in tick_levels[0]]
        assert tick_vals == [-20.0, -10.0, 0.0, 5.0, 10.0, 20.0, 30.0, 40.0]

    # Verify raster cache items: pulse_band (#EAF2FD, 0-10 ms), dashed onset_line at t=0, dark ink scatter
    r_cache = raster_widget.getPlotItem()._raster_cache
    assert "pulse_band" in r_cache
    assert tuple(r_cache["pulse_band"].getRegion()) == (0.0, 10.0)
    assert r_cache["pulse_band"].brush.color().name().upper() == COLOR_LIGHT_PULSE_BAND.upper()
    assert r_cache["onset_line"].value() == 0.0
    assert r_cache["onset_line"].pen.style() == Qt.PenStyle.DashLine
    assert r_cache["scatter"].opts["pen"].color().name().upper() == COLOR_TEXT.upper()

    # Verify PSTH cache items: pulse_band (#EAF2FD, 0-10 ms), dashed onset_line at t=0, 1.5 px curve, hidden baseline_line
    p_cache = psth_widget.getPlotItem()._psth_cache
    assert "pulse_band" in p_cache
    assert tuple(p_cache["pulse_band"].getRegion()) == (0.0, 10.0)
    assert p_cache["pulse_band"].brush.color().name().upper() == COLOR_LIGHT_PULSE_BAND.upper()
    assert p_cache["onset_line"].value() == 0.0
    assert p_cache["onset_line"].pen.style() == Qt.PenStyle.DashLine
    assert p_cache["smooth_curve"].opts["pen"].widthF() == pytest.approx(PSTH_LINE_WIDTH)
    assert not p_cache["baseline_line"].isVisible()

    raster_widget.close()
    psth_widget.close()


# ---------------------------------------------------------------------------
# 5. AST / Source Scan of UI Modules Outside theme.py (§ 14)
# ---------------------------------------------------------------------------


def test_wp1_zero_hardcoded_colors_or_inline_css_outside_theme():
    """Confirm zero hard-coded hex colours, rgb(...), Qt.GlobalColor.*, or inline CSS rules remain outside theme.py."""
    ui_dir = Path(theme.__file__).resolve().parent
    ui_files = sorted(p for p in ui_dir.glob("*.py") if p.name != "theme.py")
    assert len(ui_files) > 0

    hex_re = re.compile(r"#[0-9A-Fa-f]{3,8}\b")
    rgb_re = re.compile(r"\brgba?\s*\(", re.IGNORECASE)
    css_prop_re = re.compile(
        r"(background-color|border-radius|font-weight|font-size|padding|margin)\s*:",
        re.IGNORECASE,
    )

    violations = []

    for py_file in ui_files:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))

        for node in ast.walk(tree):
            # Check string literals for hex colours, rgb(...), or inline CSS declarations
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                s = node.value
                lineno = getattr(node, "lineno", 0)
                if hex_re.search(s):
                    violations.append(f"{py_file.name}:{lineno} hard-coded hex colour: {s!r}")
                if rgb_re.search(s):
                    violations.append(f"{py_file.name}:{lineno} hard-coded rgb/rgba: {s!r}")
                if css_prop_re.search(s):
                    violations.append(f"{py_file.name}:{lineno} inline CSS rule: {s!r}")

            # Check attribute access for Qt.GlobalColor.*
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Attribute) and node.value.attr == "GlobalColor":
                    lineno = getattr(node, "lineno", 0)
                    violations.append(
                        f"{py_file.name}:{lineno} Qt.GlobalColor.{node.attr} usage"
                    )

    assert violations == [], "Found styling violations outside theme.py:\n" + "\n".join(violations)
