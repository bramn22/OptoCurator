"""Soft-criteria optotagging likelihood scoring across population metrics."""

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Optional, Sequence
import numpy as np

from optocurator.models import UnitMetrics, is_valid_metric_value

_LOG_EPS = 1e-6
_METRIC_FIELDS = frozenset(
    {
        "latency_ms",
        "jitter_ms",
        "reliability",
        "salt_p_value",
        "waveform_correlation",
    }
)


@dataclass(frozen=True)
class CriterionSpec:
    """Configuration for a single soft tagging criterion."""

    name: str
    threshold: float
    direction: str  # "le" or "ge"
    enabled: bool = True
    is_log: bool = False
    min_spread: float = 1e-3


def _get_metric_value(metrics: UnitMetrics, name: str) -> Optional[float]:
    if name not in _METRIC_FIELDS:
        raise ValueError(f"Unsupported criterion metric name: {name!r}")
    val = getattr(metrics, name, None)
    if val is None:
        return None
    f = float(val)
    if not math.isfinite(f):
        return None
    if name in ("latency_ms", "jitter_ms") and not is_valid_metric_value(f):
        return None
    return f


def _transform_value(val: float, is_log: bool) -> float:
    if is_log:
        return float(np.log10(max(val, _LOG_EPS)))
    return float(val)


def compute_population_spreads(
    metrics_list: Sequence[UnitMetrics],
    criteria: Sequence[CriterionSpec],
) -> Dict[str, float]:
    """Compute robust population spreads (1.4826 * MAD) for each criterion.

    Falls back to standard deviation or ``max(min_spread, 0.1 * abs(threshold))``
    when MAD is zero or fewer than two finite values are available.
    """
    spreads: Dict[str, float] = {}
    for spec in criteria:
        thresh_ref = _transform_value(spec.threshold, spec.is_log)
        fallback = max(spec.min_spread, 0.1 * abs(thresh_ref))

        finite_vals = []
        for m in metrics_list:
            v = _get_metric_value(m, spec.name)
            if v is not None and np.isfinite(v):
                finite_vals.append(_transform_value(v, spec.is_log))

        if len(finite_vals) >= 2:
            arr = np.asarray(finite_vals, dtype=np.float64)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            spread = 1.4826 * mad
            if spread <= spec.min_spread:
                std = float(np.std(arr))
                spread = std if std > spec.min_spread else fallback
        else:
            spread = fallback

        spreads[spec.name] = float(max(spread, spec.min_spread))
    return spreads


def compute_unit_likelihood(
    metrics: UnitMetrics,
    criteria: Sequence[CriterionSpec],
    spreads: Mapping[str, float],
) -> float:
    """Compute 0-1 soft-criteria likelihood score for a single unit."""
    product = 1.0
    n_evaluated = 0

    for spec in criteria:
        if not spec.enabled:
            continue

        val = _get_metric_value(metrics, spec.name)
        if spec.name == "waveform_correlation" and (val is None or not np.isfinite(val)):
            continue

        if val is None or not np.isfinite(val):
            product = 0.0
            n_evaluated += 1
            continue

        v = _transform_value(val, spec.is_log)
        t = _transform_value(spec.threshold, spec.is_log)
        fallback_spread = max(spec.min_spread, 0.1 * abs(t))
        spread = max(float(spreads.get(spec.name, fallback_spread)), spec.min_spread)

        if spec.direction == "le":
            z = 3.0 * (t - v) / spread
        elif spec.direction == "ge":
            z = 3.0 * (v - t) / spread
        else:
            raise ValueError(f"Unsupported criterion direction: {spec.direction!r}")

        z_clamped = float(np.clip(z, -30.0, 30.0))
        score = 1.0 / (1.0 + math.exp(-z_clamped))
        product *= score
        n_evaluated += 1

    if n_evaluated == 0:
        return 0.0
    return float(np.clip(product, 0.0, 1.0))


def compute_session_likelihoods(
    metrics_list: Sequence[UnitMetrics],
    criteria: Sequence[CriterionSpec],
) -> Dict[str, float]:
    """Compute and store likelihood scores for all units in a session."""
    spreads = compute_population_spreads(metrics_list, criteria)
    scores: Dict[str, float] = {}
    for m in metrics_list:
        score = compute_unit_likelihood(m, criteria, spreads)
        m.likelihood = score
        scores[m.global_id] = score
    return scores
