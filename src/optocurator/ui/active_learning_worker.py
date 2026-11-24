"""Asynchronous Background Worker for Random Forest Retraining and Active Learning."""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal

from optocurator.analysis.clustering import (
    FeatureConfiguration,
    run_active_learning_cycle,
)


class ActiveLearningWorker(QThread):
    """Background thread that retrains the Random Forest classifier and computes active learning candidates.

    Prevents UI stutter and maintains 60 FPS interaction during intensive model fitting.
    """

    results_ready = pyqtSignal(list, dict, dict, dict, dict)  # (candidates, auto_assignments, confidences, all_probabilities, suggested_assignments)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        df: pd.DataFrame,
        cluster_labels: np.ndarray,
        unit_status_map: Dict[str, str],
        threshold: float = 0.85,
        max_distance_sigma: float = 3.5,
        feature_config: Optional[FeatureConfiguration] = None,
        feature_matrix: Optional[np.ndarray] = None,
        skipped_gids: Optional[Set[str]] = None,
        random_state: int = 42,
        spike_data_map: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        psth_cache: Optional[Dict[Any, np.ndarray]] = None,
        auto_assign_enabled: bool = False,
        n_estimators: int = 100,
        pulse_duration_ms: Optional[float] = None,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.df = df.copy()
        self.cluster_labels = np.array(cluster_labels, copy=True)
        self.unit_status_map = dict(unit_status_map)
        self.threshold = float(threshold)
        self.max_distance_sigma = float(max_distance_sigma)
        self.feature_config = feature_config or FeatureConfiguration()
        self.feature_matrix = np.array(feature_matrix, copy=True) if feature_matrix is not None else None
        self.skipped_gids = set(skipped_gids) if skipped_gids else set()
        self.random_state = random_state
        self.spike_data_map = dict(spike_data_map) if spike_data_map is not None else None
        self.psth_cache = dict(psth_cache) if psth_cache is not None else None
        self.auto_assign_enabled = bool(auto_assign_enabled)
        self.n_estimators = max(10, int(n_estimators))
        self.pulse_duration_ms = pulse_duration_ms

    def run(self) -> None:
        """Execute background Random Forest fitting, auto-assignment, and entropy ranking."""
        try:
            candidates, auto_assignments, confidences, all_probabilities, suggested_assignments = (
                run_active_learning_cycle(
                    df=self.df,
                    cluster_labels=self.cluster_labels,
                    unit_status_map=self.unit_status_map,
                    threshold=self.threshold,
                    max_distance_sigma=self.max_distance_sigma,
                    feature_config=self.feature_config,
                    feature_matrix=self.feature_matrix,
                    skipped_gids=self.skipped_gids,
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    spike_data_map=self.spike_data_map,
                    psth_cache=self.psth_cache,
                    auto_assign_enabled=self.auto_assign_enabled,
                    pulse_duration_ms=self.pulse_duration_ms,
                )
            )
            self.results_ready.emit(candidates, auto_assignments, confidences, all_probabilities, suggested_assignments)
        except Exception as exc:
            self.error_occurred.emit(str(exc))

