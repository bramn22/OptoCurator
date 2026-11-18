"""Asynchronous Background Worker for Random Forest Retraining and Active Learning."""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal

from optocurator.analysis.clustering import (
    ActiveLearningCandidate,
    FeatureConfiguration,
    compute_auto_assignments,
    compute_entropy_ranking,
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
        self.feature_matrix = feature_matrix
        self.skipped_gids = set(skipped_gids) if skipped_gids else set()
        self.random_state = random_state
        self.spike_data_map = spike_data_map
        self.psth_cache = psth_cache
        self.auto_assign_enabled = bool(auto_assign_enabled)
        self.n_estimators = max(10, int(n_estimators))
        self.pulse_duration_ms = pulse_duration_ms

    def run(self) -> None:
        """Execute background Random Forest fitting, auto-assignment, and entropy ranking."""
        try:
            # 1. Compute auto-assignments based on confidence threshold and all unit probabilities
            raw_auto_assignments, confidences, all_probabilities = compute_auto_assignments(
                df=self.df,
                cluster_labels=self.cluster_labels,
                unit_status_map=self.unit_status_map,
                threshold=self.threshold,
                max_distance_sigma=self.max_distance_sigma,
                feature_config=self.feature_config,
                feature_matrix=self.feature_matrix,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                spike_data_map=self.spike_data_map,
                psth_cache=self.psth_cache,
                return_all_probabilities=True,
                pulse_duration_ms=self.pulse_duration_ms,
            )

            # If auto-assignment is enabled, assign units passing threshold.
            # If auto-assignment is disabled, capture suggested putative assignments while suppressing formal assignments.
            if self.auto_assign_enabled:
                auto_assignments = {gid: cid for gid, cid in raw_auto_assignments.items() if cid >= 0}
                suggested_assignments: Dict[str, int] = {}
            else:
                auto_assignments = {}
                suggested_assignments = {
                    gid: cid
                    for gid, cid in raw_auto_assignments.items()
                    if cid >= 0 and self.unit_status_map.get(gid, "Unassigned") not in ("Curated", "Seeded")
                }

            # 2. Derive updated labels incorporating auto-assignments for candidate ranking
            temp_labels = np.array(self.cluster_labels, copy=True)
            unit_ids = (
                self.df["global_unit_id"].astype(str).tolist()
                if "global_unit_id" in self.df.columns
                else [f"u_{i}" for i in range(len(self.df))]
            )

            for i, gid in enumerate(unit_ids):
                status = self.unit_status_map.get(gid, "Unassigned")
                if status == "Curated":
                    continue
                if gid in auto_assignments:
                    pred_cid = auto_assignments[gid]
                    if pred_cid >= 0:
                        temp_labels[i] = pred_cid
                    elif status not in ("Seeded", "Curated"):
                        temp_labels[i] = -1

            # 3. Compute pure Maximum Shannon Entropy ranking with skipped units cycled to back
            candidates = compute_entropy_ranking(
                df=self.df,
                cluster_labels=temp_labels,
                unit_status_map=self.unit_status_map,
                feature_matrix=self.feature_matrix,
                skipped_unit_ids=self.skipped_gids,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                cycle_skipped=True,
                feature_config=self.feature_config,
                spike_data_map=self.spike_data_map,
                psth_cache=self.psth_cache,
                pulse_duration_ms=self.pulse_duration_ms,
            )

            self.results_ready.emit(candidates, auto_assignments, confidences, all_probabilities, suggested_assignments)
        except Exception as exc:
            self.error_occurred.emit(str(exc))

