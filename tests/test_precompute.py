"""Unit tests for headless compound raster and PSTH plot pre-rendering."""

from pathlib import Path
import numpy as np
import pytest

from optocurator.analysis.precompute import (
    batch_precompute_plots,
    render_compound_psth_raster,
)
from optocurator.models import StimulusProtocol, UnitData, UnitMetrics


def test_render_compound_psth_raster(tmp_path: Path):
    """Test generating a single compound raster + PSTH image."""
    pulse_onsets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pulse_offsets = pulse_onsets + 0.02
    stim = StimulusProtocol(onsets=pulse_onsets, offsets=pulse_offsets, name="OPTS_03")

    # Unit with direct spikes
    spikes = np.sort(np.concatenate([
        pulse_onsets + 0.004,
        pulse_onsets - 0.020,
    ]))
    unit = UnitData(
        session_id="test_sess",
        unit_id=1,
        spike_times=spikes,
        sampling_rate=30000.0,
    )

    out_file = tmp_path / "test_u0001.png"
    result_path = render_compound_psth_raster(unit, stim, out_file, dpi=100)

    assert result_path == out_file
    assert out_file.exists()
    assert out_file.stat().st_size > 1000  # Valid non-empty PNG file


def test_render_compound_psth_raster_marker_styling(tmp_path: Path, monkeypatch):
    """Verify compound raster uses vertical line marker '|' and black color."""
    from matplotlib.axes import Axes

    scatter_kwargs = {}
    orig_scatter = Axes.scatter

    def spy_scatter(self, *args, **kwargs):
        scatter_kwargs.update(kwargs)
        return orig_scatter(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", spy_scatter)

    pulse_onsets = np.array([1.0, 2.0, 3.0])
    stim = StimulusProtocol(onsets=pulse_onsets, offsets=pulse_onsets + 0.02, name="OPTS_03")
    unit = UnitData(
        session_id="test_sess",
        unit_id=1,
        spike_times=np.array([1.004, 2.004, 3.004]),
        sampling_rate=30000.0,
    )
    out_file = tmp_path / "test_marker.png"
    render_compound_psth_raster(unit, stim, out_file, dpi=80)

    assert scatter_kwargs.get("marker") == "|"
    assert scatter_kwargs.get("c") == "#000000"


def test_batch_precompute_plots_only_tagged(tmp_path: Path):
    """Test batch pre-rendering filters for only tagged units when requested."""
    pulse_onsets = np.arange(1.0, 10.0, 1.0)
    stim = StimulusProtocol(onsets=pulse_onsets, offsets=pulse_onsets + 0.02, name="OPTS")

    # 3 units: u0 (tagged), u1 (untagged), u2 (tagged)
    units = [
        UnitData("sessA", 0, pulse_onsets + 0.004, 30000.0),
        UnitData("sessA", 1, np.array([0.5, 12.0]), 30000.0),
        UnitData("sessA", 2, pulse_onsets + 0.005, 30000.0),
    ]

    metrics_map = {
        0: UnitMetrics(units[0].global_id, "sessA", 0, "", 4.0, 0.5, 1.0, 0.001, is_tagged=True),
        1: UnitMetrics(units[1].global_id, "sessA", 1, "", 25.0, 5.0, 0.1, 0.800, is_tagged=False),
        2: UnitMetrics(units[2].global_id, "sessA", 2, "", 5.0, 0.8, 0.9, 0.002, is_tagged=True),
    }

    out_dir = tmp_path / "plots_tagged_only"
    callbacks_log = []

    def cb(cur, tot):
        callbacks_log.append((cur, tot))

    generated = batch_precompute_plots(
        units=units,
        stimulus=stim,
        output_dir=out_dir,
        only_tagged=True,
        metrics_map=metrics_map,
        progress_callback=cb,
        dpi=100,
    )

    # Should only generate for units 0 and 2
    assert len(generated) == 2
    filenames = {p.name for p in generated}
    assert "sessA_u0000_psth_raster.png" in filenames
    assert "sessA_u0002_psth_raster.png" in filenames
    assert "sessA_u0001_psth_raster.png" not in filenames
    assert len(callbacks_log) == 2


def test_batch_precompute_plots_marker_styling(tmp_path: Path, monkeypatch):
    """Verify batch precompute plots configure vertical line marker '|' and black color."""
    from matplotlib.axes import Axes

    scatter_calls = []
    orig_scatter = Axes.scatter

    def spy_scatter(self, *args, **kwargs):
        scatter_calls.append(kwargs)
        return orig_scatter(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", spy_scatter)

    pulse_onsets = np.array([1.0, 2.0, 3.0])
    stim = StimulusProtocol(onsets=pulse_onsets, offsets=pulse_onsets + 0.02, name="OPTS")
    units = [UnitData("sessA", 0, np.array([1.004, 2.004, 3.004]), 30000.0)]

    out_dir = tmp_path / "batch_marker_test"
    batch_precompute_plots(units=units, stimulus=stim, output_dir=out_dir, dpi=80)

    assert len(scatter_calls) > 0
    assert scatter_calls[0].get("marker") == "|"
    assert scatter_calls[0].get("c") == "#000000"
