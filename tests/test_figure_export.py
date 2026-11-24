"""Unit and integration tests for right-click SVG vector figure export."""

from pathlib import Path
from typing import Any, List, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QContextMenuEvent, QMouseEvent
from PyQt6.QtWidgets import QApplication, QFileDialog, QGraphicsView, QMenu

from optocurator.analysis.clustering import ClusterSummary
from optocurator.ui import figure_export
from optocurator.ui.detection_widgets import DetectionPairplotWidget
from optocurator.ui.figure_export import (
    export_widget_to_svg,
    install_figure_export_menu,
    populate_figure_export_menu,
    prompt_save_figure_svg,
    sanitize_filename,
)
from optocurator.ui.plots import (
    SiliconProbeWidget,
    UnitResponsePlot,
    render_psth,
    render_raster,
)
from optocurator.ui.view_clusters import (
    ClusterCardWidget,
    ClusterProbabilityBarWidget,
    ClusteringView,
    MarginalDistributionsWidget,
)
from optocurator.ui.view_detection import DetectionView


def _local_tag(tag: str) -> str:
    """Strip XML namespace prefix from an ElementTree tag name."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _assert_valid_vector_svg(svg_path: Path, *, require_paths: bool = True) -> ET.Element:
    """Parse an SVG file and verify it is valid, pure vector XML."""
    assert svg_path.exists(), f"Expected SVG file to exist: {svg_path}"
    content = svg_path.read_text(encoding="utf-8")
    assert len(content) > 0, "SVG file is empty"

    root = ET.fromstring(content)
    assert _local_tag(root.tag) == "svg", f"Expected <svg> root, got <{root.tag}>"

    image_elems = [el for el in root.iter() if _local_tag(el.tag) == "image"]
    assert len(image_elems) == 0, f"Expected pure vector SVG with 0 <image> elements, found {len(image_elems)}"

    if require_paths:
        path_elems = [el for el in root.iter() if _local_tag(el.tag) == "path"]
        assert len(path_elems) > 0, "Expected > 0 <path> elements in exported vector SVG"

    return root


def _make_context_menu_event(local_pos: QPoint, global_pos: QPoint) -> QContextMenuEvent:
    """Construct a QContextMenuEvent for mouse-triggered context menus."""
    return QContextMenuEvent(
        QContextMenuEvent.Reason.Mouse,
        local_pos,
        global_pos,
        Qt.KeyboardModifier.NoModifier,
    )


def _make_mouse_event(
    event_type: QMouseEvent.Type,
    local_pos: QPoint,
    global_pos: QPoint,
    button: Qt.MouseButton,
    buttons: Qt.MouseButton,
) -> QMouseEvent:
    """Construct a PyQt6 QMouseEvent with explicit local and global coordinates."""
    return QMouseEvent(
        event_type,
        QPointF(local_pos),
        QPointF(global_pos),
        button,
        buttons,
        Qt.KeyboardModifier.NoModifier,
    )


def _capture_menu_exec(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[QMenu, List[str]]]:
    """Monkeypatch QMenu.exec so menus never block and record their action labels."""
    captured: List[Tuple[QMenu, List[str]]] = []

    def _fake_exec(self: QMenu, *args: Any, **kwargs: Any) -> Any:
        labels = [act.text() for act in self.actions() if not act.isSeparator()]
        captured.append((self, labels))
        return None

    monkeypatch.setattr(QMenu, "exec", _fake_exec)
    return captured


# =============================================================================
# 1. sanitize_filename
# =============================================================================


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("my_plot", "my_plot"),
        ("  spaced   name  ", "spaced_name"),
        ("sess/u001\\raster:psth", "sess_u001_raster_psth"),
        ("unit@1#2$3%4&5*6?<>|", "unit_1_2_3_4_5_6"),
        ("already_has_ext.svg", "already_has_ext"),
        ("UPPER_EXT.SVG", "UPPER_EXT"),
        ("...___---clean_stem---___...", "clean_stem"),
        ("a___b   c///d", "a_b_c_d"),
        ("", "figure"),
        ("     ", "figure"),
        ("///\\\\\\***???", "figure"),
        ("...", "figure"),
        (".svg", "figure"),
    ],
)
def test_sanitize_filename_edge_cases(raw: str, expected: str) -> None:
    """Verify sanitize_filename strips unsafe characters, whitespace, slashes, and empty inputs."""
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_none_input() -> None:
    """Verify None input falls back safely to 'figure'."""
    assert sanitize_filename(None) == "figure"  # type: ignore[arg-type]


# =============================================================================
# 2. export_widget_to_svg across all target widgets
# =============================================================================


def test_export_unit_response_plot_to_svg(qapp: QApplication, tmp_path: Path) -> None:
    """Exporting UnitResponsePlot with spikes and PSTH yields valid vector SVG."""
    widget = UnitResponsePlot()
    spike_times = np.array([-15.0, -5.0, 2.5, 3.0, 3.5, 4.2, 12.0, 25.0], dtype=np.float64)
    trial_indices = np.array([0, 1, 0, 1, 2, 3, 4, 5], dtype=np.float64)
    bins = np.linspace(-30.0, 50.0, 41)
    rates = np.exp(-0.5 * ((bins - 3.5) / 2.0) ** 2) * 80.0
    widget.set_data(
        spike_times_ms=spike_times,
        trial_indices=trial_indices,
        psth_bins=bins,
        psth_rates=rates,
        n_trials=10,
        window_ms=(-30.0, 50.0),
    )
    qapp.processEvents()

    out = export_widget_to_svg(widget, tmp_path / "unit_response.svg")
    _assert_valid_vector_svg(out, require_paths=True)


def test_export_silicon_probe_widget_to_svg(qapp: QApplication, tmp_path: Path) -> None:
    """Exporting SiliconProbeWidget with unit markers yields valid vector SVG."""
    probe = SiliconProbeWidget()
    probe.units_scatter.setData(
        x=np.array([0.0, 16.0, 32.0, 0.0]),
        y=np.array([100.0, 200.0, 300.0, 400.0]),
    )
    probe.cand_marker.setData(x=[16.0], y=[200.0])
    probe.plot_widget.setXRange(-20.0, 60.0)
    probe.plot_widget.setYRange(0.0, 500.0)
    qapp.processEvents()

    out = export_widget_to_svg(probe.plot_widget, tmp_path / "silicon_probe.svg")
    _assert_valid_vector_svg(out, require_paths=True)


def test_export_detection_pairplot_widget_to_svg(qapp: QApplication, tmp_path: Path) -> None:
    """Exporting DetectionPairplotWidget yields valid vector SVG with scatter paths."""
    pairplot = DetectionPairplotWidget()
    pairplot.set_data(
        unit_ids=[0, 1, 2, 3, 4],
        latencies=[2.5, 4.0, 6.5, 12.0, 18.0],
        jitters=[0.4, 0.8, 1.5, 3.2, 4.1],
        reliabilities=[0.95, 0.85, 0.70, 0.30, 0.10],
        is_tagged=[True, True, False, False, False],
        focused_unit_id=0,
    )
    qapp.processEvents()

    out = export_widget_to_svg(pairplot, tmp_path / "detection_pairplot.svg")
    _assert_valid_vector_svg(out, require_paths=True)


def test_export_detection_view_canvas_and_criteria_to_svg(qapp: QApplication, tmp_path: Path) -> None:
    """Exporting DetectionView.plot_canvas and DetectionView.criteria_container yields valid vector SVGs."""
    view = DetectionView()
    view.resize(1280, 820)

    # Populate plot_canvas (raster + PSTH)
    rx = np.array([-10.0, 2.0, 2.5, 3.0, 4.0, 15.0], dtype=np.float64)
    ry = np.array([0, 0, 1, 2, 3, 4], dtype=np.float64)
    bins = np.linspace(-30.0, 50.0, 41)
    rates = np.exp(-0.5 * ((bins - 3.0) / 1.5) ** 2) * 60.0
    render_raster(view.raster_plot, rx, ry, n_trials=6, x_range=(-30.0, 50.0))
    render_psth(view.psth_plot, bins, rates, x_range=(-30.0, 50.0), is_smooth=True)

    # Populate criteria distributions
    view.row_latency.set_population_data(
        np.array([2.0, 3.5, 5.0, 11.0, 16.0]),
        threshold=8.0,
        enabled=True,
        focused_value=3.5,
        n_pass=3,
        n_total=5,
    )
    view.row_jitter.set_population_data(
        np.array([0.4, 0.8, 1.2, 2.5, 3.8]),
        threshold=2.0,
        enabled=True,
        focused_value=0.8,
        n_pass=3,
        n_total=5,
    )
    view.row_rel.set_population_data(
        np.array([0.9, 0.8, 0.6, 0.2, 0.1]),
        threshold=0.6,
        enabled=True,
        focused_value=0.8,
        n_pass=3,
        n_total=5,
    )
    qapp.processEvents()

    canvas_svg = export_widget_to_svg(view.plot_canvas, tmp_path / "detection_canvas.svg")
    _assert_valid_vector_svg(canvas_svg, require_paths=True)

    criteria_svg = export_widget_to_svg(view.criteria_container, tmp_path / "detection_criteria.svg")
    _assert_valid_vector_svg(criteria_svg, require_paths=True)


def test_export_clustering_view_pca_prob_marginals_and_card_to_svg(
    qapp: QApplication, tmp_path: Path
) -> None:
    """Exporting ClusteringView.pca_plot, ClusterProbabilityBarWidget, MarginalDistributionsWidget, and ClusterCardWidget.plot."""
    # 1) ClusteringView.pca_plot
    cview = ClusteringView()
    cview.scatter_item.setData(
        x=np.array([-2.0, -1.2, 0.5, 1.8, 2.4]),
        y=np.array([1.1, -0.4, 0.8, -1.5, 0.3]),
    )
    cview.cand_scatter.setData(x=[0.5], y=[0.8])
    cview.pca_plot.setXRange(-3.0, 3.0)
    cview.pca_plot.setYRange(-2.5, 2.5)
    qapp.processEvents()

    pca_svg = export_widget_to_svg(cview.pca_plot, tmp_path / "clustering_pca.svg")
    _assert_valid_vector_svg(pca_svg, require_paths=True)

    # 2) ClusterProbabilityBarWidget
    prob_w = ClusterProbabilityBarWidget()
    prob_w.set_probabilities(
        probabilities={0: 0.72, 1: 0.21, 2: 0.07},
        cluster_colors={0: "#2563EB", 1: "#10B981", 2: "#F59E0B"},
        cluster_names={0: "Fast", 1: "Delayed", 2: "Inhibited"},
        threshold=0.80,
        cluster_order=[0, 1, 2],
    )
    qapp.processEvents()

    prob_svg = export_widget_to_svg(prob_w.plot_widget, tmp_path / "cluster_prob.svg")
    _assert_valid_vector_svg(prob_svg, require_paths=True)

    # 3) MarginalDistributionsWidget
    marg_w = MarginalDistributionsWidget()
    df = pd.DataFrame(
        {
            "latency_ms": [2.1, 3.0, 4.2, 8.5, 12.0, 14.3],
            "jitter_ms": [0.3, 0.5, 0.8, 1.9, 2.5, 3.1],
            "reliability": [0.95, 0.88, 0.80, 0.55, 0.40, 0.30],
            "salt_p_value": [0.001, 0.002, 0.005, 0.02, 0.04, 0.08],
            "waveform_corr": [0.98, 0.96, 0.95, 0.91, 0.87, 0.82],
            "evoked_rate_hz": [65.0, 52.0, 48.0, 25.0, 18.0, 12.0],
        }
    )
    marg_w.update_distributions(df)
    qapp.processEvents()

    marg_svg = export_widget_to_svg(marg_w, tmp_path / "marginal_dists.svg")
    _assert_valid_vector_svg(marg_svg, require_paths=True)

    # 4) ClusterCardWidget.plot
    bins = np.linspace(-30.0, 50.0, 41)
    mean_psth = np.exp(-0.5 * ((bins - 4.0) / 2.5) ** 2)
    sem_psth = np.full_like(mean_psth, 0.08)
    summary = ClusterSummary(
        cluster_id=0,
        count=12,
        mean_psth=mean_psth,
        sem_psth=sem_psth,
        time_bins_ms=bins,
        color_hex="#2563EB",
        name="Fast Direct",
    )
    card = ClusterCardWidget(cluster_id=summary)
    qapp.processEvents()

    card_svg = export_widget_to_svg(card.plot, tmp_path / "cluster_card.svg")
    _assert_valid_vector_svg(card_svg, require_paths=True)


def test_export_auto_suffix_and_parent_dir_creation(qapp: QApplication, tmp_path: Path) -> None:
    """export_widget_to_svg appends .svg when absent and creates missing parent directories."""
    plot = pg.PlotWidget()
    plot.resize(240, 180)
    plot.plot([0.0, 1.0, 2.0], [1.0, 3.0, 2.0])
    qapp.processEvents()

    nested_no_ext = tmp_path / "nested" / "deep" / "my_figure"
    out_path = export_widget_to_svg(plot, nested_no_ext)
    assert out_path == tmp_path / "nested" / "deep" / "my_figure.svg"
    _assert_valid_vector_svg(out_path, require_paths=True)

    # Uppercase .SVG suffix is preserved without double extension
    upper_target = tmp_path / "upper_case.SVG"
    out_upper = export_widget_to_svg(plot, upper_target)
    assert out_upper == upper_target
    _assert_valid_vector_svg(out_upper, require_paths=True)


def test_export_restores_scatter_and_view_cache_settings(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ScatterPlotItem.opts['useCache'] and QGraphicsView.cacheMode() are restored after export (even on error)."""
    plot = pg.PlotWidget()
    plot.resize(260, 200)
    plot.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)

    scatter_cached = pg.ScatterPlotItem(x=[0.0, 1.0, 2.0], y=[1.0, 2.0, 3.0], size=8, useCache=True)
    scatter_uncached = pg.ScatterPlotItem(x=[0.5, 1.5], y=[2.5, 1.5], size=8, useCache=False)
    plot.addItem(scatter_cached)
    plot.addItem(scatter_uncached)
    qapp.processEvents()

    assert plot.cacheMode() == QGraphicsView.CacheModeFlag.CacheBackground
    assert scatter_cached.opts["useCache"] is True
    assert scatter_uncached.opts["useCache"] is False

    # Capture state during widget.render() to verify caching is disabled while painting
    observed_during_render: dict[str, Any] = {}
    orig_render = plot.render

    def _spy_render(*args: Any, **kwargs: Any) -> Any:
        observed_during_render["view_cache"] = plot.cacheMode()
        observed_during_render["scatter_cached"] = scatter_cached.opts["useCache"]
        observed_during_render["scatter_uncached"] = scatter_uncached.opts["useCache"]
        return orig_render(*args, **kwargs)

    monkeypatch.setattr(plot, "render", _spy_render)

    out = export_widget_to_svg(plot, tmp_path / "cache_check.svg")
    _assert_valid_vector_svg(out, require_paths=True)

    # Disabled during render
    assert observed_during_render["view_cache"] == QGraphicsView.CacheModeFlag.CacheNone
    assert observed_during_render["scatter_cached"] is False
    assert observed_during_render["scatter_uncached"] is False

    # Restored after render
    assert plot.cacheMode() == QGraphicsView.CacheModeFlag.CacheBackground
    assert scatter_cached.opts["useCache"] is True
    assert scatter_uncached.opts["useCache"] is False

    # Also verify restoration when render() raises an exception
    def _failing_render(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated render error")

    monkeypatch.setattr(plot, "render", _failing_render)
    with pytest.raises(RuntimeError, match="simulated render error"):
        export_widget_to_svg(plot, tmp_path / "cache_check_fail.svg")

    assert plot.cacheMode() == QGraphicsView.CacheModeFlag.CacheBackground
    assert scatter_cached.opts["useCache"] is True
    assert scatter_uncached.opts["useCache"] is False


# =============================================================================
# 3. prompt_save_figure_svg
# =============================================================================


def test_prompt_save_figure_svg_writes_file_and_persists_last_dir(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """prompt_save_figure_svg uses SVG-only filter, sanitizes default name, writes SVG, and persists last dir."""
    monkeypatch.setattr(figure_export, "_LAST_EXPORT_DIR", None)

    plot = pg.PlotWidget()
    plot.resize(240, 180)
    plot.plot([0.0, 1.0, 2.0], [0.5, 1.5, 1.0])
    qapp.processEvents()

    dialog_calls: List[Tuple[str, str, str]] = []
    export_dir = tmp_path / "exports_dir"
    export_dir.mkdir()
    chosen_file = export_dir / "saved_plot.svg"

    def _fake_get_save(parent: Any, caption: str, directory: str, file_filter: str) -> Tuple[str, str]:
        dialog_calls.append((caption, directory, file_filter))
        return str(chosen_file), file_filter

    monkeypatch.setattr(QFileDialog, "getSaveFileName", _fake_get_save)

    result = prompt_save_figure_svg(
        parent=plot,
        export_target=plot,
        default_name=lambda: "  sess01/unit:05 * raster?  ",
    )
    assert result == chosen_file
    _assert_valid_vector_svg(chosen_file, require_paths=True)

    assert len(dialog_calls) == 1
    caption, initial_path, file_filter = dialog_calls[0]
    assert caption == "Save Figure as SVG"
    assert file_filter == "SVG Vector Graphics (*.svg)"
    assert Path(initial_path).name == "sess01_unit_05_raster.svg"
    assert figure_export._LAST_EXPORT_DIR == export_dir

    # Second call should open in the persisted last export directory
    second_file = export_dir / "second_plot.svg"

    def _fake_get_save_second(parent: Any, caption: str, directory: str, file_filter: str) -> Tuple[str, str]:
        dialog_calls.append((caption, directory, file_filter))
        return str(second_file), file_filter

    monkeypatch.setattr(QFileDialog, "getSaveFileName", _fake_get_save_second)
    result2 = prompt_save_figure_svg(parent=plot, export_target=plot, default_name="next_fig")
    assert result2 == second_file
    assert Path(dialog_calls[1][1]).parent == export_dir
    assert Path(dialog_calls[1][1]).name == "next_fig.svg"


def test_prompt_save_figure_svg_cancel_returns_none(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelling the save dialog returns None and does not overwrite _LAST_EXPORT_DIR."""
    prev_dir = tmp_path / "prev_dir"
    prev_dir.mkdir()
    monkeypatch.setattr(figure_export, "_LAST_EXPORT_DIR", prev_dir)

    plot = pg.PlotWidget()
    plot.resize(200, 150)

    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: ("", ""))

    res = prompt_save_figure_svg(parent=plot, export_target=plot, default_name="cancelled_plot")
    assert res is None
    assert figure_export._LAST_EXPORT_DIR == prev_dir


# =============================================================================
# 4. Event filter & context menu behavior
# =============================================================================


def test_stationary_right_click_builds_expected_menus_on_composite_and_single_widgets(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stationary right-click shows full-panel + subplot actions on composite widgets and single action on simple plots."""
    captured = _capture_menu_exec(monkeypatch)

    # UnitResponsePlot: raster_plot and psth_plot
    urp = UnitResponsePlot()
    QApplication.sendEvent(urp.raster_plot.viewport(), _make_context_menu_event(QPoint(10, 10), QPoint(100, 100)))
    assert len(captured) == 1
    assert captured[-1][1] == ["Save Raster + PSTH as SVG…", "Save raster as SVG…"]

    QApplication.sendEvent(urp.psth_plot.viewport(), _make_context_menu_event(QPoint(10, 10), QPoint(100, 100)))
    assert len(captured) == 2
    assert captured[-1][1] == ["Save Raster + PSTH as SVG…", "Save PSTH as SVG…"]

    # DetectionPairplotWidget
    pairplot = DetectionPairplotWidget()
    QApplication.sendEvent(pairplot.plot_jit_lat.viewport(), _make_context_menu_event(QPoint(10, 10), QPoint(120, 120)))
    assert captured[-1][1] == ["Save pairplot as SVG…", "Save this subplot as SVG…"]

    # MarginalDistributionsWidget
    marg = MarginalDistributionsWidget()
    QApplication.sendEvent(
        marg.plots["latency_ms"].viewport(), _make_context_menu_event(QPoint(10, 10), QPoint(140, 140))
    )
    assert captured[-1][1] == ["Save marginal distributions as SVG…", "Save this subplot as SVG…"]

    # SiliconProbeWidget (single export action)
    probe = SiliconProbeWidget()
    QApplication.sendEvent(probe.plot_widget.viewport(), _make_context_menu_event(QPoint(10, 10), QPoint(160, 160)))
    assert captured[-1][1] == ["Save figure as SVG…"]

    # DetectionView: plot_canvas and criteria row
    dview = DetectionView()
    QApplication.sendEvent(dview.plot_canvas.viewport(), _make_context_menu_event(QPoint(10, 10), QPoint(180, 180)))
    assert captured[-1][1] == ["Save Raster + PSTH as SVG…"]

    QApplication.sendEvent(
        dview.row_latency.plot_dist.viewport(), _make_context_menu_event(QPoint(10, 10), QPoint(190, 190))
    )
    assert captured[-1][1] == ["Save tagging criteria as SVG…"]


def test_populate_figure_export_menu_triggers_expected_targets(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Triggering main and subplot menu actions calls prompt_save_figure_svg with the corresponding targets."""
    urp = UnitResponsePlot()
    urp.export_name_provider = lambda: "sessA_u0007"

    calls: List[Tuple[Any, Any]] = []

    def _spy_prompt(_parent: Any, export_target: Any, default_name: Any) -> None:
        resolved = default_name() if callable(default_name) else default_name
        calls.append((export_target, resolved))

    monkeypatch.setattr(figure_export, "prompt_save_figure_svg", _spy_prompt)

    menu = QMenu(urp)
    populate_figure_export_menu(
        menu,
        export_target=urp,
        default_name=urp._export_base_name,
        label="Save Raster + PSTH as SVG…",
        subplot_target=urp.raster_plot,
        subplot_default_name=lambda: f"{urp._export_base_name()}_raster",
        subplot_label="Save raster as SVG…",
        add_separator=True,
    )
    actions = [a for a in menu.actions() if not a.isSeparator()]
    assert [a.text() for a in actions] == ["Save Raster + PSTH as SVG…", "Save raster as SVG…"]

    actions[0].trigger()
    actions[1].trigger()
    assert calls == [
        (urp, "sessA_u0007"),
        (urp.raster_plot, "sessA_u0007_raster"),
    ]


@pytest.mark.parametrize("widget_kind", ["pca_plot", "detection_plot_canvas"])
def test_right_drag_suppresses_context_menu_on_interactive_plots(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch, widget_kind: str
) -> None:
    """Right-button drag (> 4 px) suppresses the SVG context menu so right-drag zoom is preserved."""
    captured = _capture_menu_exec(monkeypatch)

    if widget_kind == "pca_plot":
        owner = ClusteringView()
        vp = owner.pca_plot.viewport()
    else:
        owner = DetectionView()
        vp = owner.plot_canvas.viewport()

    assert vp is not None

    start_local = QPoint(20, 20)
    start_global = QPoint(200, 200)
    end_local = QPoint(45, 20)
    end_global = QPoint(225, 200)  # manhattanLength = 25 > 4 px

    press = _make_mouse_event(
        QMouseEvent.Type.MouseButtonPress,
        start_local,
        start_global,
        Qt.MouseButton.RightButton,
        Qt.MouseButton.RightButton,
    )
    move = _make_mouse_event(
        QMouseEvent.Type.MouseMove,
        end_local,
        end_global,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.RightButton,
    )
    release = _make_mouse_event(
        QMouseEvent.Type.MouseButtonRelease,
        end_local,
        end_global,
        Qt.MouseButton.RightButton,
        Qt.MouseButton.NoButton,
    )
    ctx_after_drag = _make_context_menu_event(end_local, end_global)

    QApplication.sendEvent(vp, press)
    QApplication.sendEvent(vp, move)
    QApplication.sendEvent(vp, release)
    QApplication.sendEvent(vp, ctx_after_drag)

    assert len(captured) == 0, "Context menu must be suppressed after right-button drag > 4 px"
    assert ctx_after_drag.isAccepted()

    # A subsequent stationary right-click (<= 4 px jitter) must still open the menu
    jitter_local = QPoint(22, 21)
    jitter_global = QPoint(202, 201)  # manhattanLength = 3 <= 4 px
    QApplication.sendEvent(
        vp,
        _make_mouse_event(
            QMouseEvent.Type.MouseButtonPress,
            start_local,
            start_global,
            Qt.MouseButton.RightButton,
            Qt.MouseButton.RightButton,
        ),
    )
    QApplication.sendEvent(
        vp,
        _make_mouse_event(
            QMouseEvent.Type.MouseButtonRelease,
            jitter_local,
            jitter_global,
            Qt.MouseButton.RightButton,
            Qt.MouseButton.NoButton,
        ),
    )
    ctx_stationary = _make_context_menu_event(jitter_local, jitter_global)
    QApplication.sendEvent(vp, ctx_stationary)

    assert len(captured) == 1, "Stationary right-click after a drag must open the SVG export menu"


def test_cluster_card_unified_context_menu_and_viewport_forwarding(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ClusterCardWidget preserves Rename/Change colour/Merge into/Dissolve, adds SVG export, and forwards plot viewport right-clicks."""
    captured = _capture_menu_exec(monkeypatch)

    card = ClusterCardWidget(cluster_id=1, count=5, custom_name="Fast Units")
    card._other_clusters_provider = lambda cid: [(2, "Delayed Units")]

    expected_labels = [
        "Rename…",
        "Change colour…",
        "Merge into…",
        "Save figure as SVG…",
        "Dissolve cluster…",
    ]

    # 1) Direct right-click on the card
    card.contextMenuEvent(_make_context_menu_event(QPoint(10, 10), QPoint(300, 300)))
    assert len(captured) == 1
    assert captured[-1][1] == expected_labels

    # 2) Right-click on card.plot.viewport() forwards to the unified card context menu
    vp = card.plot.viewport()
    assert vp is not None
    ctx_ev = _make_context_menu_event(QPoint(15, 15), QPoint(315, 315))
    QApplication.sendEvent(vp, ctx_ev)
    assert len(captured) == 2
    assert captured[-1][1] == expected_labels
    assert ctx_ev.isAccepted()
