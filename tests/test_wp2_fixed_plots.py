"""Tests for Work Package 2 (WP2 — Fixed-size plot widgets) from UI_OVERHAUL_SPEC.md (§ 7.3, § 8, § 14)."""

import numpy as np
import pytest
from PyQt6.QtCore import Qt

from optocurator.ui.plots import SiliconProbeWidget, UnitResponsePlot
from optocurator.ui.theme import (
    COLOR_LIGHT_PULSE_BAND,
    COLOR_TEXT,
    PLOT_CARD_PSTH_LEFT_MARGIN,
    PLOT_CARD_PSTH_SIZE,
    PLOT_CLUSTER_PROB_SIZE,
    PLOT_EMBEDDING_SIZE,
    PLOT_METRIC_KDE_SIZE,
    PLOT_PROBE_SIZE,
    PLOT_UNIT_RESPONSE_GAP,
    PLOT_UNIT_RESPONSE_LEFT_MARGIN,
    PLOT_UNIT_RESPONSE_PSTH_SIZE,
    PLOT_UNIT_RESPONSE_RASTER_SIZE,
)
from optocurator.ui.view_clusters import ClusteringView


def test_wp2_unit_response_plot_dimensions_and_live_rendering(qapp):
    """Verify UnitResponsePlot fixed dimensions, labels, band, and graceful empty handling."""
    w = UnitResponsePlot()
    expected_w = PLOT_UNIT_RESPONSE_RASTER_SIZE[0] + PLOT_UNIT_RESPONSE_LEFT_MARGIN
    assert w.width() == expected_w
    assert w.raster_plot.width() == expected_w
    assert w.raster_plot.height() == PLOT_UNIT_RESPONSE_RASTER_SIZE[1]
    assert w.psth_plot.width() == expected_w
    assert w.psth_plot.height() == PLOT_UNIT_RESPONSE_PSTH_SIZE[1] + UnitResponsePlot.PSTH_AXIS_HEIGHT
    assert w.layout().spacing() == PLOT_UNIT_RESPONSE_GAP
    assert w.raster_plot.getAxis("left").fixedWidth == PLOT_UNIT_RESPONSE_LEFT_MARGIN
    assert w.raster_plot.getAxis("left").preferredWidth() == PLOT_UNIT_RESPONSE_LEFT_MARGIN
    assert w.psth_plot.getAxis("left").fixedWidth == PLOT_UNIT_RESPONSE_LEFT_MARGIN
    assert w.psth_plot.getAxis("left").preferredWidth() == PLOT_UNIT_RESPONSE_LEFT_MARGIN

    # Live render from spike arrays
    spike_times = np.array([-10.0, 2.5, 4.0, 6.0, 12.0])
    trials = np.array([0, 0, 1, 2, 3])
    w.set_data(spike_times, trials, window_ms=(-30.0, 50.0))

    r_cache = w.raster_plot.getPlotItem()._raster_cache
    p_cache = w.psth_plot.getPlotItem()._psth_cache
    assert tuple(r_cache["pulse_band"].getRegion()) == (0.0, 10.0)
    assert r_cache["pulse_band"].brush.color().name().upper() == COLOR_LIGHT_PULSE_BAND.upper()
    assert r_cache["onset_line"].value() == 0.0
    assert r_cache["onset_line"].pen.style() == Qt.PenStyle.DashLine
    assert not p_cache["baseline_line"].isVisible()

    # Labels
    assert "Trial" in w.raster_plot.getAxis("left").labelText
    assert "Rate (Hz)" in w.psth_plot.getAxis("left").labelText
    assert "Time from pulse onset (ms)" in w.psth_plot.getAxis("bottom").labelText

    # Graceful empty / None handling
    w.set_psth_data(None)
    assert not r_cache["scatter"].isVisible()

    w.close()


def test_wp2_focused_and_member_unit_response_plots_pixel_identical(qapp):
    """Verify focused unit and inspector member UnitResponsePlots are pixel-identical."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    active_cids = sorted(view.current_clustering_result.cluster_summaries.keys())
    assert len(active_cids) >= 1
    target_cid = active_cids[0]

    view.set_cluster_comparison(target_cid)
    assert len(view._comparison_member_viewers_map) >= 1

    focused = view.unit_response_plot
    member_card = next(iter(view._comparison_member_viewers_map.values()))
    member_plot = member_card.unit_response_plot

    assert type(focused) is type(member_plot)
    assert focused.size() == member_plot.size()
    assert focused.raster_plot.size() == member_plot.raster_plot.size()
    assert focused.psth_plot.size() == member_plot.psth_plot.size()
    assert focused.raster_plot.getAxis("left").width() == member_plot.raster_plot.getAxis("left").width()
    assert focused.psth_plot.getAxis("left").width() == member_plot.psth_plot.getAxis("left").width()

    view.close()


def test_wp2_all_clustering_plots_fixed_size_and_do_not_resize_with_window(qapp):
    """Verify all ClusteringView plot widgets lock fixed dimensions across window resizes."""
    view = ClusteringView(start_with_clusters=True, load_mock=True)
    view.show()
    qapp.processEvents()

    def _assert_sizes() -> None:
        # 1. UnitResponsePlot
        expected_ur_w = PLOT_UNIT_RESPONSE_RASTER_SIZE[0] + PLOT_UNIT_RESPONSE_LEFT_MARGIN
        assert view.unit_response_plot.width() == expected_ur_w
        assert view.unit_response_plot.raster_plot.size().width() == expected_ur_w
        assert view.unit_response_plot.raster_plot.size().height() == PLOT_UNIT_RESPONSE_RASTER_SIZE[1]

        # 2. Cluster card PSTH
        assert len(view._cluster_cards) >= 1
        card = next(iter(view._cluster_cards.values()))
        assert card.plot.width() == PLOT_CARD_PSTH_SIZE[0] + PLOT_CARD_PSTH_LEFT_MARGIN
        assert card.plot.height() == PLOT_CARD_PSTH_SIZE[1]
        assert card.plot.getAxis("left").width() == PLOT_CARD_PSTH_LEFT_MARGIN
        assert card.plot.getAxis("left").fixedWidth == PLOT_CARD_PSTH_LEFT_MARGIN
        ticks = card.plot.getAxis("bottom")._tickLevels[0]
        assert [pos for pos, _ in ticks] == [-20.0, -10.0, 0.0, 5.0, 10.0, 20.0, 30.0, 40.0]
        lbl_map = {pos: lbl for pos, lbl in ticks}
        for labeled in (-20.0, 0.0, 10.0, 20.0, 40.0):
            assert lbl_map[labeled] == str(int(labeled))
        for unlabeled in (-10.0, 30.0):
            assert lbl_map[unlabeled] == ""

        # 3. Cluster probability bar
        assert view.cluster_prob_bar.plot_widget.width() == (
            PLOT_CLUSTER_PROB_SIZE[0] + PLOT_UNIT_RESPONSE_LEFT_MARGIN
        )
        assert view.cluster_prob_bar.plot_widget.height() == PLOT_CLUSTER_PROB_SIZE[1]

        # 4. Marginal metric KDEs (2x2 of 128x44)
        assert len(view.marginal_widget.plots) == 4
        for p in view.marginal_widget.plots.values():
            assert (p.width(), p.height()) == PLOT_METRIC_KDE_SIZE

        # 5. Embedding plot (268x268, 1:1 aspect, dark focused ring)
        assert (view.pca_plot.width(), view.pca_plot.height()) == PLOT_EMBEDDING_SIZE
        assert view.cand_scatter.opts["pen"].color().name().upper() == COLOR_TEXT.upper()

        # 6. Probe plot (76x268, matches embedding height)
        assert isinstance(view.probe_widget, SiliconProbeWidget)
        assert (view.probe_plot.width(), view.probe_plot.height()) == PLOT_PROBE_SIZE
        assert view.probe_plot.height() == view.pca_plot.height()

    _assert_sizes()

    view.resize(1920, 1080)
    qapp.processEvents()
    _assert_sizes()

    view.resize(1100, 680)
    qapp.processEvents()
    _assert_sizes()

    view.close()


def test_inferred_pulse_duration_shading_detection_and_clustering(qapp, tmp_path):
    """Verify real stimulation duration (offsets - onsets) is inferred and shaded in Detection and Clustering."""
    import pandas as pd
    from optocurator.models import StimulusProtocol
    from optocurator.ui.plots import DEFAULT_PULSE_DURATION_MS, infer_pulse_duration_ms
    from optocurator.ui.view_detection import DetectionView

    onsets = np.arange(1.0, 11.0, 0.5)
    offsets = onsets + 0.028  # 28 ms real stimulation window
    assert infer_pulse_duration_ms(onsets, offsets) == 28.0
    assert infer_pulse_duration_ms(onsets, None) == DEFAULT_PULSE_DURATION_MS

    # 1. DetectionView uses inferred protocol duration for raster & PSTH shading
    from optocurator.models import UnitData
    det = DetectionView()
    u0 = UnitData(
        unit_id=1,
        session_id="sess_28ms",
        spike_times=np.sort(onsets + 0.005),
        sampling_rate=30000.0,
        waveforms=None,
        channel=0,
        depth=100.0,
        probe_x=0.0,
    )
    det.units = [u0]
    det.current_protocol = StimulusProtocol(onsets=onsets, offsets=offsets, name="OPTS_28ms")
    det._populate_table()
    det.table_units.selectRow(0)
    det._on_unit_selected()
    assert tuple(det.raster_plot.getPlotItem()._raster_cache["pulse_band"].getRegion()) == (0.0, 28.0)
    assert tuple(det.psth_plot.getPlotItem()._psth_cache["pulse_band"].getRegion()) == (0.0, 28.0)
    det.close()

    # 2. ClusteringView infers stimulation duration from exported pulse_offsets in spikes.npz
    sess = "sess_28ms"
    exp_dir = tmp_path / sess
    arr_dir = exp_dir / f"{sess}_arrays"
    arr_dir.mkdir(parents=True)
    pd.DataFrame([
        {
            "global_unit_id": f"{sess}_u1",
            "session_id": sess,
            "unit_id": 1,
            "latency_ms": 5.0,
            "jitter_ms": 1.2,
            "reliability": 0.85,
            "salt_p_value": 0.001,
            "waveform_correlation": 0.96,
            "is_tagged": True,
            "channel": 0,
            "depth": 100.0,
            "probe_x": 0.0,
            "plot_path": "",
        },
        {
            "global_unit_id": f"{sess}_u2",
            "session_id": sess,
            "unit_id": 2,
            "latency_ms": 6.0,
            "jitter_ms": 1.5,
            "reliability": 0.80,
            "salt_p_value": 0.002,
            "waveform_correlation": 0.94,
            "is_tagged": True,
            "channel": 1,
            "depth": 120.0,
            "probe_x": 0.0,
            "plot_path": "",
        },
    ]).to_csv(exp_dir / f"{sess}_opto_curated.csv", index=False)
    np.savez_compressed(
        arr_dir / f"{sess}_spikes.npz",
        pulse_onsets=onsets,
        pulse_offsets=offsets,
        unit_1=np.sort(onsets + 0.005),
        unit_2=np.sort(onsets + 0.006),
    )

    cv = ClusteringView(start_with_clusters=True)
    assert cv.add_experiment_folder(exp_dir) is True
    assert cv._get_pulse_duration_ms(sess) == 28.0
    assert tuple(cv.unit_response_plot.raster_plot.getPlotItem()._raster_cache["pulse_band"].getRegion()) == (0.0, 28.0)
    assert tuple(cv.unit_response_plot.psth_plot.getPlotItem()._psth_cache["pulse_band"].getRegion()) == (0.0, 28.0)
    if cv._cluster_cards:
        first_card = next(iter(cv._cluster_cards.values()))
        assert tuple(first_card.stim_region.getRegion()) == (0.0, 28.0)
    cv.close()
