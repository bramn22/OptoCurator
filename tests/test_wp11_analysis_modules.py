"""Unit tests for WP11 isolated analysis modules (likelihood & metrics)."""

import numpy as np
import pytest

import optocurator.analysis.metrics as metrics_mod
from optocurator.analysis.likelihood import (
    CriterionSpec,
    compute_population_spreads,
    compute_session_likelihoods,
    compute_unit_likelihood,
)
from optocurator.models import UnitMetrics


def test_old_waveform_correlation_removed_from_metrics():
    """Old compute_waveform_correlation must be removed from metrics.py."""
    assert not hasattr(metrics_mod, "compute_waveform_correlation")


def _make_metrics(
    gid: str,
    lat: float,
    jit: float,
    rel: float,
    salt_p: float,
) -> UnitMetrics:
    return UnitMetrics(
        global_id=gid,
        session_id="s1",
        unit_id=int(gid.split("_u")[-1]),
        plot_path="",
        latency_ms=lat,
        jitter_ms=jit,
        reliability=rel,
        salt_p_value=salt_p,
    )


def test_likelihood_boundary_monotonicity_and_missing_metric():
    """Verify threshold ~0.5 boundary, monotone scoring, log SALT p, and skipped NaN metric."""
    criteria = [
        CriterionSpec("latency_ms", threshold=8.0, direction="le"),
        CriterionSpec("jitter_ms", threshold=2.0, direction="le"),
        CriterionSpec("reliability", threshold=0.6, direction="ge"),
        CriterionSpec("salt_p_value", threshold=0.05, direction="le", is_log=True),
    ]

    # Single-criterion exact threshold check -> ~0.5
    single_crit = [CriterionSpec("latency_ms", threshold=8.0, direction="le")]
    u_thresh = _make_metrics("s1_u0001", lat=8.0, jit=1.0, rel=0.8, salt_p=0.01)
    spreads_single = compute_population_spreads([u_thresh], single_crit)
    assert pytest.approx(compute_unit_likelihood(u_thresh, single_crit, spreads_single), abs=1e-6) == 0.5

    # Population with strong, borderline, and weak units
    u_strong = _make_metrics("s1_u0001", lat=3.0, jit=0.5, rel=0.95, salt_p=0.001)
    u_mid = _make_metrics("s1_u0002", lat=7.5, jit=1.8, rel=0.65, salt_p=0.04)
    u_weak = _make_metrics("s1_u0003", lat=18.0, jit=5.0, rel=0.10, salt_p=0.80)
    u_nan_jit = _make_metrics("s1_u0004", lat=3.0, jit=float("nan"), rel=0.95, salt_p=0.001)

    scores = compute_session_likelihoods([u_strong, u_mid, u_weak, u_nan_jit], criteria)
    assert 0.0 <= scores[u_weak.global_id] < scores[u_mid.global_id] < scores[u_strong.global_id] <= 1.0
    # Missing (NaN) enabled metric receives 0.0 likelihood
    assert scores[u_nan_jit.global_id] == 0.0
    assert u_strong.likelihood == scores[u_strong.global_id]

    # Disabled criteria -> 0.0
    disabled = [CriterionSpec("latency_ms", threshold=8.0, direction="le", enabled=False)]
    assert compute_unit_likelihood(u_strong, disabled, {"latency_ms": 1.0}) == 0.0


def test_likelihood_excludes_999_sentinel_units():
    """Verify silent 999.0 ms sentinel units do not inflate MAD/std spread or receive non-zero likelihood (Bug M1)."""
    criteria = [CriterionSpec("latency_ms", threshold=8.0, direction="le")]
    units = [
        _make_metrics("s1_u0001", lat=4.0, jit=0.8, rel=0.9, salt_p=0.001),
        _make_metrics("s1_u0002", lat=6.0, jit=1.0, rel=0.8, salt_p=0.005),
        _make_metrics("s1_u0003", lat=20.0, jit=2.0, rel=0.4, salt_p=0.10),
    ] + [
        _make_metrics(f"s1_u{i:04d}", lat=999.0, jit=999.0, rel=0.0, salt_p=1.0)
        for i in range(4, 11)
    ]

    spreads = compute_population_spreads(units, criteria)
    assert spreads["latency_ms"] < 10.0

    scores = compute_session_likelihoods(units, criteria)
    assert scores["s1_u0001"] > 0.95
    assert scores["s1_u0003"] < 1e-4
    for i in range(4, 11):
        assert scores[f"s1_u{i:04d}"] == 0.0
