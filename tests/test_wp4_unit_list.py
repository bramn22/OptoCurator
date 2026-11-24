"""WP4 verification tests: Unit list column (Show dropdown, 7-column table, Embedding & Probe)."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont

from optocurator.ui.plots import SiliconProbeWidget
from optocurator.ui.theme import (
    COLOR_ACCENT_BG,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    EMBEDDING_DIM_ALPHA,
    PROBE_DIM_ALPHA,
    text_color_for_cluster,
)
from optocurator.ui.view_clusters import ClusteringView


def _make_view() -> ClusteringView:
    return ClusteringView(load_mock=True, start_with_clusters=True)


def test_wp4_show_dropdown_order_swatches_counts_scrolling_and_filtering(qapp):
    """Show dropdown has exact WP4 order, swatches, counts, scrolling, and status/cluster filtering."""
    view = _make_view()
    qapp.processEvents()

    combo = view.combo_filter_cluster
    assert combo.maxVisibleItems() <= 10
    assert combo.view().verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    assert combo.view().maximumHeight() <= 270

    # Verify exact order of top entries + separator + clusters
    assert combo.itemText(0).startswith("All units")
    assert combo.itemData(0, Qt.ItemDataRole.UserRole) == -2

    expected_statuses = [
        (1, "Unassigned", "status:Unassigned"),
        (2, "Curated", "status:Curated"),
        (3, "Seeded", "status:Seeded"),
        (4, "Auto-assigned", "status:Auto-assigned"),
    ]
    for idx, label, key in expected_statuses:
        assert combo.itemText(idx).startswith(label)
        assert combo.itemData(idx, Qt.ItemDataRole.UserRole) == key
        assert not combo.itemIcon(idx).isNull()

    # Separator at index 5
    assert combo.itemData(5, Qt.ItemDataRole.UserRole) is None

    # Active clusters follow separator with swatches and readable cluster text colour
    assert combo.count() > 6
    for idx in range(6, combo.count()):
        cid = combo.itemData(idx, Qt.ItemDataRole.UserRole)
        assert isinstance(cid, int) and cid >= 0
        assert not combo.itemIcon(idx).isNull()
        fg = combo.itemData(idx, Qt.ItemDataRole.ForegroundRole)
        expected_col = text_color_for_cluster(view._get_cluster_color(cid)).upper()
        assert isinstance(fg, QColor)
        assert fg.name().upper() == expected_col

    # Assign one unit as Curated and one as Seeded, then verify filtering by status and cluster
    total_rows = view.table_pooled_units.rowCount()
    assert total_rows >= 3
    gid0 = view._get_gid_for_row(0)
    gid1 = view._get_gid_for_row(1)
    first_cid = combo.itemData(6, Qt.ItemDataRole.UserRole)

    view._pooled_units_map[gid0]["cluster_id"] = first_cid
    view._unit_status_map[gid0] = "Curated"
    view._pooled_units_map[gid1]["cluster_id"] = first_cid
    view._unit_status_map[gid1] = "Seeded"
    view._update_all_pooled_cluster_items({gid0, gid1})
    view._populate_cluster_filter_options()
    qapp.processEvents()

    # Filter to Curated
    curated_idx = combo.findData("status:Curated")
    combo.setCurrentIndex(curated_idx)
    qapp.processEvents()
    visible_gids = [
        view._get_gid_for_row(r)
        for r in range(total_rows)
        if not view.table_pooled_units.isRowHidden(r)
    ]
    assert gid0 in visible_gids
    assert gid1 not in visible_gids

    # Filter to Seeded
    seeded_idx = combo.findData("status:Seeded")
    combo.setCurrentIndex(seeded_idx)
    qapp.processEvents()
    visible_gids = [
        view._get_gid_for_row(r)
        for r in range(total_rows)
        if not view.table_pooled_units.isRowHidden(r)
    ]
    assert gid1 in visible_gids
    assert gid0 not in visible_gids

    # Filter to cluster
    cluster_idx = combo.findData(first_cid)
    combo.setCurrentIndex(cluster_idx)
    qapp.processEvents()
    visible_gids = [
        view._get_gid_for_row(r)
        for r in range(total_rows)
        if not view.table_pooled_units.isRowHidden(r)
    ]
    assert gid0 in visible_gids and gid1 in visible_gids
    assert "units" in view.lbl_pool_count.text().lower()

    view.close()


def test_wp4_table_columns_formatting_salt_floor_status_and_focus_highlight(qapp, qtbot):
    """Table has 7 exact WP4 columns, short session + tooltip, SALT floor, icon-only status, and focused highlight."""
    view = _make_view()
    qapp.processEvents()
    qtbot.waitUntil(
        lambda: (view._active_learning_worker is None or not view._active_learning_worker.isRunning())
        and view.active_candidate is not None,
        timeout=10000,
    )

    tbl = view.table_pooled_units
    headers = [tbl.horizontalHeaderItem(c).text() for c in range(tbl.columnCount())]
    assert headers == ["Unit", "Session", "Cluster", "Lat", "Rel.", "SALT p", "Status"]

    # Inject controlled test values on row 0 and row 1
    gid0 = view._get_gid_for_row(0)
    rec0 = view._pooled_units_map[gid0]
    rec0["session_id"] = "01942_20230307"
    rec0["salt_p_value"] = 0.0040
    rec0["cluster_id"] = -1
    view._unit_status_map[gid0] = "Unassigned"
    view._populate_pooled_table_row(0, rec0)

    gid1 = view._get_gid_for_row(1)
    rec1 = view._pooled_units_map[gid1]
    cid1 = 0
    view._cluster_names[cid1] = "Very Long Cluster Name For Elision Check"
    rec1["cluster_id"] = cid1
    rec1["salt_p_value"] = 0.0320
    view._unit_status_map[gid1] = "Curated"
    view._populate_pooled_table_row(1, rec1)

    # Row 0: unassigned, short session, SALT <= 0.01 floor
    assert tbl.item(0, 1).text() == "01942"
    assert tbl.item(0, 1).toolTip() == "01942_20230307"
    assert tbl.item(0, 2).text() == "—"
    assert tbl.item(0, 2).foreground().color().name().upper() == COLOR_TEXT_MUTED.upper()
    assert tbl.item(0, 5).text() == "≤ 0.01"
    assert tbl.item(0, 5).toolTip() == "SALT p = 0.0040"
    assert tbl.item(0, 6).text() == ""
    assert not tbl.item(0, 6).icon().isNull()
    assert tbl.item(0, 6).toolTip() == "Unassigned"

    # Row 1: assigned, full cluster name in text() with ElideRight + full tooltip + colour + swatch, Curated icon-only status
    assert tbl.item(1, 2).text() == "Very Long Cluster Name For Elision Check"
    assert tbl.textElideMode() == Qt.TextElideMode.ElideRight
    assert tbl.item(1, 2).toolTip() == "Very Long Cluster Name For Elision Check"
    assert tbl.item(1, 2).icon().isNull()
    expected_text_col = text_color_for_cluster(view._get_cluster_color(cid1)).upper()
    assert tbl.item(1, 2).foreground().color().name().upper() == expected_text_col
    assert tbl.item(1, 5).text() == "0.03"
    assert tbl.item(1, 5).toolTip() == "SALT p = 0.0320"
    assert tbl.item(1, 6).text() == ""
    assert not tbl.item(1, 6).icon().isNull()
    assert tbl.item(1, 6).toolTip() == "Curated"

    # Focused row highlight
    assert view.select_pooled_unit(gid1)
    qapp.processEvents()
    assert tbl.item(1, 0).background().color().name().upper() == COLOR_ACCENT_BG.upper()
    assert tbl.item(1, 0).font().weight() >= QFont.Weight.DemiBold
    assert tbl.item(0, 0).font().weight() == QFont.Weight.Normal

    view.close()


def test_wp4_embedding_and_probe_methods_ring_and_cluster_emphasis(qapp):
    """Embedding supports PCA/UMAP only + gear button, dark ring, and cluster-open emphasis/dimming on Embedding & Probe."""
    view = _make_view()
    qapp.processEvents()

    # Methods: PCA and UMAP only (no t-SNE)
    methods = [view.combo_embedding_method.itemData(i) for i in range(view.combo_embedding_method.count())]
    assert methods == ["PCA", "UMAP"]
    assert hasattr(view, "btn_configure_embedding")
    assert view.btn_configure_embedding is view.btn_embedding_settings

    # PCA axis labels and dark focused ring
    bottom_lbl = view.pca_plot.getAxis("bottom").labelText
    left_lbl = view.pca_plot.getAxis("left").labelText
    assert "PC1" in bottom_lbl and "% var." in bottom_lbl
    assert "PC2" in left_lbl and "% var." in left_lbl
    assert view.cand_scatter.opts["pen"].color().name().upper() == COLOR_TEXT.upper()
    assert isinstance(view.probe_widget, SiliconProbeWidget)

    # Open a cluster and verify emphasis/dimming on both Embedding and Probe
    open_cid = next(iter(view._cluster_cards.keys()))
    view.set_cluster_comparison(open_cid)
    qapp.processEvents()

    embed_spots = view.scatter_item.points()
    assert len(embed_spots) > 0
    member_alphas = []
    non_member_alphas = []
    for pt in embed_spots:
        gid = str(pt.data())
        cid = int(view._pooled_units_map.get(gid, {}).get("cluster_id", -1))
        alpha = pt.brush().color().alpha()
        if cid == open_cid:
            member_alphas.append(alpha)
        else:
            non_member_alphas.append(alpha)
    if member_alphas and non_member_alphas:
        assert all(a == 255 for a in member_alphas)
        assert all(a == EMBEDDING_DIM_ALPHA for a in non_member_alphas)

    # Probe points emphasis/dimming (non-focused, non-member units dimmed)
    focused_gid = view._current_selected_gid
    probe_spots = view.probe_units_scatter.points()
    assert len(probe_spots) > 0
    for pt in probe_spots:
        gid = str(pt.data())
        if gid == focused_gid:
            assert pt.pen().color().name().upper() == COLOR_TEXT.upper()
            continue
        cid = int(view._pooled_units_map.get(gid, {}).get("cluster_id", -1))
        if cid == open_cid:
            assert pt.brush().color().alpha() == 255
        else:
            assert pt.brush().color().alpha() == PROBE_DIM_ALPHA

    # Closing cluster restores full opacity
    view.clear_cluster_comparison()
    qapp.processEvents()
    assert all(pt.brush().color().alpha() == 255 for pt in view.scatter_item.points())

    view.close()
