"""Unsupervised Dimensionality Reduction & Hierarchical Clustering Analysis."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from optocurator.analysis.metrics import compute_psth_and_raster

COLOR_UNASSIGNED: str = "#A0A4AA"

# Colour-blind-safe 15-colour categorical palette (Okabe–Ito + Paul Tol Muted)
CLUSTER_PALETTE: List[str] = [
    "#E69F00",  # 0 — orange
    "#56B4E9",  # 1 — sky blue
    "#009E73",  # 2 — bluish green
    "#0072B2",  # 3 — blue
    "#D55E00",  # 4 — vermilion
    "#CC79A7",  # 5 — reddish purple
    "#332288",  # 6 — indigo
    "#88CCEE",  # 7 — cyan
    "#44AA99",  # 8 — teal
    "#117733",  # 9 — green
    "#999933",  # 10 — olive
    "#DDCC77",  # 11 — sand
    "#CC6677",  # 12 — rose
    "#882255",  # 13 — wine
    "#AA4499",  # 14 — purple
]


def color_for_cluster(
    cluster_id: int,
    custom_colors: Optional[Mapping[int, str]] = None,
) -> str:
    """Resolve the display colour for a cluster ID.

    Colours are bound to cluster identity (never display order).
    Returns ``COLOR_UNASSIGNED`` when ``cluster_id < 0``.
    """
    cid = int(cluster_id)
    if cid < 0:
        return COLOR_UNASSIGNED
    if custom_colors is not None:
        custom = custom_colors.get(cid)
        if custom and str(custom).strip():
            return str(custom).strip()
    return CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]


@dataclass
class FeatureConfiguration:
    """Configuration for features and weights used in 2D embedding, seeding, clustering, and active learning."""

    # Analytical Metrics
    use_latency: bool = True
    weight_latency: float = 5.0
    use_jitter: bool = True
    weight_jitter: float = 1.0
    use_reliability: bool = True
    weight_reliability: float = 1.0
    use_salt: bool = False
    salt_log10_transform: bool = True
    weight_salt: float = 1.0

    # PSTH Summary Scalars
    use_baseline_rate: bool = False
    weight_baseline_rate: float = 1.0
    use_peak_rate: bool = False
    weight_peak_rate: float = 1.0
    use_peak_response_rate: bool = True
    weight_peak_response_rate: float = 1.0
    use_time_to_response: bool = True
    weight_time_to_response: float = 1.0
    response_std_threshold: float = 2.0

    # Temporal PSTH Profile Vector
    use_psth_profile: bool = True
    weight_psth_profile: float = 1.0
    psth_normalization: str = "area_normalized"  # "area_normalized", "peak_normalized", "raw"
    time_window_ms: Tuple[float, float] = (0.0, 25.0)
    bin_size_ms: float = 2.0
    psth_smooth: bool = True
    psth_gaussian_sigma_ms: float = 2.0
    psth_use_pca: bool = True
    psth_pca_variance_explained: float = 0.80

    def __post_init__(self) -> None:
        if str(self.psth_normalization).lower() not in ("area_normalized", "peak_normalized", "raw"):
            self.psth_normalization = "area_normalized"


@dataclass
class EmbeddingParameters:
    """Parameters for 2D dimensionality reduction (PCA / UMAP)."""

    method: str = "PCA"  # "PCA" or "UMAP"
    pca_component_x: int = 1
    pca_component_y: int = 2
    pca_whiten: bool = False
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"


@dataclass
class AffinityPropagationParameters:
    """Parameters for Affinity Propagation cluster seeding."""

    damping: float = 0.80
    similarity_percentile: float = 50.0
    min_cluster_size: int = 2
    preference: Optional[float] = None
    preserve_curated: bool = True


@dataclass
class ClusteringParameters:
    """Hyperparameters and configuration for unsupervised response clustering."""

    metric: str = "euclidean"                  # "euclidean", "cosine", "correlation", "cityblock"
    linkage: str = "ward"                      # "ward", "complete", "average", "single"
    cutoff_mode: str = "k"                     # "k" (count) or "distance" (d_cut)
    n_clusters: int = 4
    distance_threshold: float = 1.5
    feature_config: Optional[FeatureConfiguration] = None
    presented_window_ms: Tuple[float, float] = (-30.0, 50.0)
    presented_bin_size_ms: float = 2.0
    psth_smooth: bool = True
    psth_gaussian_sigma_ms: float = 2.0
    cluster_psth_norm: str = "probability"     # "probability", "peak", "raw"
    # Multi-algorithm extensions
    algorithm: str = "hierarchical"            # "hierarchical", "kmeans", "gmm"
    kmeans_init: str = "k-means++"
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    gmm_covariance_type: str = "full"
    gmm_reg_covar: float = 1e-6
    gmm_max_iter: int = 100
    random_state: int = 42


@dataclass
class ClusterSummary:
    """Statistical summary and grand-average PSTH waveform for a single response cluster."""

    cluster_id: int
    count: int
    color_hex: str
    unit_ids: List[str] = field(default_factory=list)
    time_bins_ms: np.ndarray = field(default_factory=lambda: np.empty(0))
    mean_psth: np.ndarray = field(default_factory=lambda: np.empty(0))
    sem_psth: np.ndarray = field(default_factory=lambda: np.empty(0))
    norm_mode: str = "probability"
    name: Optional[str] = None


@dataclass
class ActiveLearningCandidate:
    """A candidate unit evaluated by Maximum Shannon Entropy active learning."""

    global_unit_id: str
    entropy: float
    probabilities: Dict[int, float]
    feature_values: Dict[str, float] = field(default_factory=dict)
    cluster_id: int = -1
    session_id: str = ""
    unit_id: int = 0
    depth: float = 0.0


@dataclass
class ActiveLearningState:
    """State of the active learning queue for fast unit curation."""

    active_candidate: Optional[ActiveLearningCandidate] = None
    ranked_candidates: List[ActiveLearningCandidate] = field(default_factory=list)
    skipped_unit_ids: Set[str] = field(default_factory=set)


@dataclass
class ClusteringResult:
    """Complete results of dimensionality reduction, clustering, and cluster profiling."""

    cluster_labels: np.ndarray                  # shape (N,), int
    pca_coordinates: np.ndarray                 # shape (N, 2), float
    explained_variance_ratio: List[float]       # [var_pc1, var_pc2]
    cluster_summaries: Dict[int, ClusterSummary]
    feature_matrix: np.ndarray                  # shape (N, D)
    unit_ids: List[str]
    parameters: ClusteringParameters


def generate_psth_matrix(
    df: pd.DataFrame,
    time_window_ms: Tuple[float, float] = (-30.0, 50.0),
    bin_size_ms: float = 2.0,
    spike_data_map: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    psth_cache: Optional[Dict[Any, np.ndarray]] = None,
    smooth: bool = True,
    gaussian_sigma_ms: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate PSTH profile matrix [N, T] and time bins [T] for pooled units across specified time window."""
    n_units = len(df)
    start_ms, end_ms = time_window_ms
    bins = np.arange(start_ms, end_ms + bin_size_ms * 0.5, bin_size_ms)
    t_bins = (bins[:-1] + bins[1:]) / 2.0
    n_bins = len(t_bins)
    if n_units == 0 or n_bins == 0:
        return np.empty((n_units, n_bins), dtype=np.float64), t_bins

    psth_matrix = np.zeros((n_units, n_bins), dtype=np.float64)
    for idx, (_, row) in enumerate(df.iterrows()):
        gid = str(row.get("global_unit_id", ""))
        sess = str(row.get("session_id", ""))
        uid = str(row.get("unit_id", ""))
        cache_key = (
            gid,
            (float(start_ms), float(end_ms)),
            float(bin_size_ms),
            bool(smooth),
            float(gaussian_sigma_ms),
        )

        if psth_cache is not None and cache_key in psth_cache:
            cached_arr = psth_cache[cache_key]
            if len(cached_arr) == n_bins:
                psth_matrix[idx, :] = cached_arr
                continue

        spikes_tuple = None
        if spike_data_map is not None:
            if gid in spike_data_map:
                spikes_tuple = spike_data_map[gid]
            elif f"{sess}_{uid}" in spike_data_map:
                spikes_tuple = spike_data_map[f"{sess}_{uid}"]
            elif sess and uid:
                try:
                    alt_key = f"{sess}_u{int(uid):04d}"
                    if alt_key in spike_data_map:
                        spikes_tuple = spike_data_map[alt_key]
                except (ValueError, TypeError):
                    pass

        if spikes_tuple is not None:
            spk_times, p_onsets = spikes_tuple
            if spk_times is not None and p_onsets is not None and len(spk_times) > 0 and len(p_onsets) > 0:
                res = compute_psth_and_raster(
                    spk_times,
                    p_onsets,
                    window_ms=time_window_ms,
                    bin_size_ms=bin_size_ms,
                    smooth=smooth,
                    gaussian_sigma_ms=gaussian_sigma_ms,
                )
                rates = res["psth_rates"]
                if len(rates) == n_bins:
                    psth_matrix[idx, :] = rates
                    if psth_cache is not None:
                        psth_cache[cache_key] = rates
                    continue
                elif len(rates) > 0:
                    min_len = min(len(rates), n_bins)
                    psth_matrix[idx, :min_len] = rates[:min_len]
                    if psth_cache is not None:
                        psth_cache[cache_key] = psth_matrix[idx, :]
                    continue

        # Fallback to analytical Gaussian formula
        lat = float(row.get("latency_ms", 5.0)) if pd.notna(row.get("latency_ms")) else 10.0
        jit = float(row.get("jitter_ms", 1.5)) if pd.notna(row.get("jitter_ms")) else 2.0
        rel = float(row.get("reliability", 0.8)) if pd.notna(row.get("reliability")) else 0.5
        jit = max(jit, 0.5)

        rng = np.random.RandomState(idx * 37 + 11)
        base_rate = float(rng.uniform(3.0, 10.0))
        baseline = np.ones(n_bins) * base_rate

        evoked_peak = (rel * 80.0) * np.exp(-0.5 * ((t_bins - lat) / jit) ** 2)
        evoked_peak[t_bins < 0] = 0.0

        psth_matrix[idx, :] = baseline + evoked_peak
        if psth_cache is not None:
            psth_cache[cache_key] = psth_matrix[idx, :]

    return psth_matrix, t_bins


def extract_feature_matrix(
    df: pd.DataFrame,
    feature_config: Optional[FeatureConfiguration] = None,
    spike_data_map: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    psth_cache: Optional[Dict[Any, np.ndarray]] = None,
    pulse_duration_ms: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and standardize feature representations from pooled unit records.

    Args:
        df: DataFrame containing pooled unit records.
        feature_config: Optional granular feature toggle and weighting configuration.
        spike_data_map: Optional map from unit identifier to (spike_times, pulse_onsets).
        psth_cache: Optional cache mapping (gid, time_window_ms, bin_size_ms, smooth, sigma) to 1D PSTH array.
        pulse_duration_ms: Optional optogenetic pulse duration in ms (defaults to 20.0 if None or <= 0).

    Returns:
        Tuple of (feature_matrix [N, D], psth_matrix [N, T], time_bins_ms [T]).
    """
    n_units = len(df)
    if n_units == 0:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty(0)

    # 1. Always resolve FeatureConfiguration
    cfg = feature_config if feature_config is not None else FeatureConfiguration()

    # 2. Generate PSTH Profiles for all units over configured window
    psth_matrix, t_bins = generate_psth_matrix(
        df,
        time_window_ms=cfg.time_window_ms,
        bin_size_ms=cfg.bin_size_ms,
        spike_data_map=spike_data_map,
        psth_cache=psth_cache,
        smooth=cfg.psth_smooth,
        gaussian_sigma_ms=cfg.psth_gaussian_sigma_ms,
    )
    n_bins = len(t_bins)

    # 3. Compute baseline stats (base_rates, base_stds)
    if n_bins > 0 and np.any(t_bins < 0):
        baseline_mask = t_bins < 0
        base_rates = np.mean(psth_matrix[:, baseline_mask], axis=1)
        base_stds = np.std(psth_matrix[:, baseline_mask], axis=1)
    else:
        base_psth, _ = generate_psth_matrix(
            df,
            time_window_ms=(-50.0, 0.0),
            bin_size_ms=cfg.bin_size_ms,
            spike_data_map=spike_data_map,
            psth_cache=psth_cache,
            smooth=cfg.psth_smooth,
            gaussian_sigma_ms=cfg.psth_gaussian_sigma_ms,
        )
        if base_psth.shape[1] > 0:
            base_rates = np.mean(base_psth, axis=1)
            base_stds = np.std(base_psth, axis=1)
        else:
            base_rates = np.zeros(n_units, dtype=np.float64)
            base_stds = np.zeros(n_units, dtype=np.float64)

    # 4. Pulse window mask for peak_rate and signed peak_response_rate
    pulse_dur = float(pulse_duration_ms) if (pulse_duration_ms is not None and pulse_duration_ms > 0) else 20.0
    if n_bins > 0:
        pulse_mask = (t_bins >= 0.0) & (t_bins <= pulse_dur)
        if not np.any(pulse_mask):
            pulse_mask = t_bins >= 0.0
            if not np.any(pulse_mask):
                pulse_mask = np.ones(n_bins, dtype=bool)
        pulse_sub = psth_matrix[:, pulse_mask]
        peak_rates = np.max(pulse_sub, axis=1)
        diff_pulse = pulse_sub - base_rates[:, None]
        max_dev_idx = np.argmax(np.abs(diff_pulse), axis=1)
        peak_response_rates = diff_pulse[np.arange(n_units), max_dev_idx]

        # 5. Time to response over t_bins >= 0.0
        evoked_mask = t_bins >= 0.0
        if not np.any(evoked_mask):
            evoked_mask = np.ones(n_bins, dtype=bool)
        evoked_t_bins = t_bins[evoked_mask]
        evoked_sub = psth_matrix[:, evoked_mask]
        k_sd = max(0.0, float(cfg.response_std_threshold))
        fallback_t = float(evoked_t_bins[-1])
        time_to_response = np.full(n_units, fallback_t, dtype=np.float64)
        for i in range(n_units):
            thresh = k_sd * max(float(base_stds[i]), 1e-6)
            hits = np.where(np.abs(evoked_sub[i] - base_rates[i]) >= thresh)[0]
            if len(hits) > 0:
                time_to_response[i] = float(evoked_t_bins[hits[0]])
    else:
        peak_rates = np.zeros(n_units, dtype=np.float64)
        peak_response_rates = np.zeros(n_units, dtype=np.float64)
        time_to_response = np.zeros(n_units, dtype=np.float64)

    # 6. Extract and scale selected Analytical & Summary Metrics
    scalar_columns: List[Tuple[str, np.ndarray, float]] = []

    if cfg.use_latency:
        vals = df["latency_ms"].to_numpy(dtype=np.float64, copy=True) if "latency_ms" in df.columns else np.full(n_units, 20.0)
        vals = np.where(np.isnan(vals), 20.0, vals)
        scalar_columns.append(("latency_ms", vals, float(cfg.weight_latency)))

    if cfg.use_jitter:
        vals = df["jitter_ms"].to_numpy(dtype=np.float64, copy=True) if "jitter_ms" in df.columns else np.full(n_units, 5.0)
        vals = np.where(np.isnan(vals), 5.0, vals)
        scalar_columns.append(("jitter_ms", vals, float(cfg.weight_jitter)))

    if cfg.use_reliability:
        vals = df["reliability"].to_numpy(dtype=np.float64, copy=True) if "reliability" in df.columns else np.zeros(n_units)
        vals = np.where(np.isnan(vals), 0.0, vals)
        scalar_columns.append(("reliability", vals, float(cfg.weight_reliability)))

    if cfg.use_salt:
        vals = df["salt_p_value"].to_numpy(dtype=np.float64, copy=True) if "salt_p_value" in df.columns else np.ones(n_units)
        vals = np.where(np.isnan(vals), 1.0, vals)
        if cfg.salt_log10_transform:
            vals = -np.log10(np.clip(vals, 1e-6, 1.0))
        scalar_columns.append(("salt_p_value", vals, float(cfg.weight_salt)))

    if cfg.use_baseline_rate:
        scalar_columns.append(("baseline_rate", base_rates, float(cfg.weight_baseline_rate)))

    if cfg.use_peak_rate:
        scalar_columns.append(("peak_rate", peak_rates, float(cfg.weight_peak_rate)))

    if cfg.use_peak_response_rate:
        scalar_columns.append(("peak_response_rate", peak_response_rates, float(cfg.weight_peak_response_rate)))

    if cfg.use_time_to_response:
        scalar_columns.append(("time_to_response", time_to_response, float(cfg.weight_time_to_response)))

    if scalar_columns:
        scaled_scalars = np.zeros((n_units, len(scalar_columns)), dtype=np.float64)
        for i, (_, arr, weight) in enumerate(scalar_columns):
            std_val = float(np.std(arr))
            if n_units > 1 and std_val > 1e-8:
                scaled_col = (arr - np.mean(arr)) / std_val
            else:
                scaled_col = np.zeros_like(arr) if n_units > 1 else arr
            scaled_scalars[:, i] = scaled_col * weight
    else:
        scaled_scalars = np.empty((n_units, 0), dtype=np.float64)

    # 7. Normalize, standardize, and optionally PCA-compress temporal PSTH profile
    if cfg.use_psth_profile and n_bins > 0:
        norm_mode = str(cfg.psth_normalization).lower()
        if norm_mode == "area_normalized":
            pos_psth = np.maximum(psth_matrix, 0.0)
            l1 = np.maximum(np.sum(pos_psth, axis=1, keepdims=True), 1e-6)
            norm_psth = pos_psth / l1
        elif norm_mode == "peak_normalized":
            max_per_row = np.maximum(np.max(psth_matrix, axis=1, keepdims=True), 1e-6)
            norm_psth = psth_matrix / max_per_row
        else:
            # "raw" (or unrecognized fallback)
            norm_psth = np.array(psth_matrix, copy=True)

        if n_units > 1:
            scaled_psth = StandardScaler().fit_transform(norm_psth)
        else:
            scaled_psth = norm_psth

        if cfg.psth_use_pca and n_units > 1 and scaled_psth.shape[1] > 1:
            if float(np.max(np.abs(scaled_psth))) <= 1e-12:
                pca_out = np.zeros((n_units, 1), dtype=np.float64)
            else:
                var_exp = float(np.clip(cfg.psth_pca_variance_explained, 0.05, 0.9999))
                try:
                    pca = PCA(n_components=var_exp, svd_solver="full")
                    pca_out = pca.fit_transform(scaled_psth)
                    if pca_out.shape[1] == 0:
                        pca_out = PCA(n_components=1, random_state=42).fit_transform(scaled_psth)
                except Exception:
                    pca_out = PCA(n_components=1, random_state=42).fit_transform(scaled_psth)
            n_pcs = pca_out.shape[1]
            for j in range(n_pcs):
                col_std = float(np.std(pca_out[:, j]))
                if col_std > 1e-8:
                    pca_out[:, j] = (pca_out[:, j] - np.mean(pca_out[:, j])) / col_std
                else:
                    pca_out[:, j] = 0.0
            scaled_psth = pca_out * (float(cfg.weight_psth_profile) / np.sqrt(max(1, n_pcs)))
        else:
            scaled_psth = scaled_psth * (float(cfg.weight_psth_profile) / np.sqrt(max(1, n_bins)))
    else:
        scaled_psth = np.empty((n_units, 0), dtype=np.float64)

    # 8. Combine scalar and PSTH profile features
    feature_parts: List[np.ndarray] = []
    if scaled_scalars.shape[1] > 0:
        feature_parts.append(scaled_scalars)
    if cfg.use_psth_profile and scaled_psth.shape[1] > 0:
        feature_parts.append(scaled_psth)

    if not feature_parts:
        feature_parts.append(np.zeros((n_units, 1), dtype=np.float64))

    feat_matrix = np.hstack(feature_parts) if len(feature_parts) > 1 else feature_parts[0]
    return feat_matrix, psth_matrix, t_bins


def compute_pca_embedding(
    feature_matrix: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[float]]:
    """Project feature matrix into 2D PCA space for scatter visualization.

    Args:
        feature_matrix: Matrix of shape [N, D].
        n_components: Dimensions to retain (default 2).
        random_state: Random state for reproducibility.

    Returns:
        Tuple of (coordinates_2d [N, 2], explained_variance_ratio [var_pc1, var_pc2]).
    """
    n_samples, n_features = feature_matrix.shape
    if n_samples == 0:
        return np.empty((0, 2), dtype=np.float64), [0.0, 0.0]

    if n_samples < 2 or n_features < 2:
        # Degenerate case with 1 sample or 1 feature
        coords = np.zeros((n_samples, 2), dtype=np.float64)
        if n_features == 1:
            coords[:, 0] = feature_matrix[:, 0]
        return coords, [1.0, 0.0]

    n_comp = min(n_components, n_samples, n_features)
    pca = PCA(n_components=n_comp, random_state=random_state)
    transformed = pca.fit_transform(feature_matrix)

    coords_2d = np.zeros((n_samples, 2), dtype=np.float64)
    coords_2d[:, :n_comp] = transformed[:, :n_comp]

    exp_var = [float(v) for v in pca.explained_variance_ratio_]
    while len(exp_var) < 2:
        exp_var.append(0.0)

    return coords_2d, exp_var[:2]


def _find_umap_ab_params(spread: float = 1.0, min_dist: float = 0.1) -> Tuple[float, float]:
    """Fit UMAP smooth approximation parameters (a, b) from spread and min_dist."""
    from scipy.optimize import curve_fit

    spread = max(0.1, float(spread))
    min_dist = max(0.0, min(float(min_dist), spread * 0.99))

    def curve(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return 1.0 / (1.0 + a * np.power(x, 2.0 * b))

    xv = np.linspace(0.0, spread * 3.0, 300)
    yv = np.zeros_like(xv)
    yv[xv <= min_dist] = 1.0
    yv[xv > min_dist] = np.exp(-(xv[xv > min_dist] - min_dist) / spread)
    try:
        params, _ = curve_fit(curve, xv, yv, p0=[1.0, 1.0], bounds=(0.001, 50.0), maxfev=2000)
        return float(params[0]), float(params[1])
    except Exception:
        return 1.5769, 0.8951


def _compute_native_umap(
    feature_matrix: np.ndarray,
    params: EmbeddingParameters,
    seed: int = 42,
) -> np.ndarray:
    """Vectorized UMAP (fuzzy simplicial set + cross-entropy force layout) responsive to all hyperparameters."""
    from scipy.spatial.distance import cdist

    n_samples, n_features = feature_matrix.shape
    if n_samples <= 2:
        coords, _ = compute_pca_embedding(feature_matrix, n_components=2, random_state=seed)
        return coords

    k = max(2, min(int(params.umap_n_neighbors), n_samples - 1))
    min_dist = max(0.0, float(params.umap_min_dist))
    spread = 1.0
    metric_map = {
        "euclidean": "euclidean",
        "cosine": "cosine",
        "correlation": "correlation",
        "manhattan": "cityblock",
        "cityblock": "cityblock",
    }
    scipy_metric = metric_map.get(str(params.umap_metric).lower(), "euclidean")

    X = np.nan_to_num(np.asarray(feature_matrix, dtype=np.float64), nan=0.0)
    try:
        dist_matrix = cdist(X, X, metric=scipy_metric)
        dist_matrix = np.nan_to_num(dist_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        dist_matrix = cdist(X, X, metric="euclidean")

    np.fill_diagonal(dist_matrix, np.inf)
    knn_indices = np.argsort(dist_matrix, axis=1)[:, :k]
    knn_dists = np.take_along_axis(dist_matrix, knn_indices, axis=1)

    # Smooth kNN distances: rho_i = distance to nearest neighbor, sigma_i via binary search for log2(k)
    rho = np.maximum(0.0, knn_dists[:, 0])
    target = np.log2(float(k))
    lo = np.full(n_samples, 1e-5, dtype=np.float64)
    hi = np.full(n_samples, 1e4, dtype=np.float64)
    sigma = np.ones(n_samples, dtype=np.float64)

    shifted = np.maximum(0.0, knn_dists - rho[:, None])
    for _ in range(32):
        psum = np.sum(np.exp(-shifted / sigma[:, None]), axis=1)
        too_high = psum > target
        hi = np.where(too_high, sigma, hi)
        lo = np.where(too_high, lo, sigma)
        sigma = 0.5 * (lo + hi)

    # Directed membership graph A and fuzzy simplicial set union W = A + A^T - A * A^T
    A = np.zeros((n_samples, n_samples), dtype=np.float64)
    row_idx = np.arange(n_samples)[:, None]
    A[row_idx, knn_indices] = np.exp(-shifted / np.maximum(sigma[:, None], 1e-8))
    W = A + A.T - (A * A.T)
    np.fill_diagonal(W, 0.0)

    # Fit exact UMAP curve parameters (a, b) for (spread, min_dist)
    a, b = _find_umap_ab_params(spread=spread, min_dist=min_dist)

    # Spectral / Laplacian initialization blended with RNG seed perturbation
    rng = np.random.RandomState(int(seed))
    deg = np.sum(W, axis=1)
    inv_sqrt_deg = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
    norm_adj = (W * inv_sqrt_deg[:, None]) * inv_sqrt_deg[None, :]
    try:
        eigvals, eigvecs = np.linalg.eigh(norm_adj)
        if n_samples >= 3:
            init_coords = eigvecs[:, -3:-1].copy()
        else:
            init_coords, _ = compute_pca_embedding(X, n_components=2, random_state=seed)
    except Exception:
        init_coords, _ = compute_pca_embedding(X, n_components=2, random_state=seed)

    std_init = np.std(init_coords, axis=0, keepdims=True)
    std_init = np.where(std_init < 1e-8, 1.0, std_init)
    Y = (init_coords / std_init) * (2.0 * spread) + rng.normal(0.0, 0.15 * spread, size=(n_samples, 2))

    # Vectorized UMAP cross-entropy SGD optimization
    n_epochs = 80
    repulsion_weight = 0.15 / max(1.0, float(n_samples - k))
    for epoch in range(n_epochs):
        lr = 1.0 * (1.0 - float(epoch) / float(n_epochs))
        diff = Y[:, None, :] - Y[None, :, :]  # [N, N, 2]
        dist_sq = np.sum(diff * diff, axis=2) + 1e-6  # [N, N]
        np.fill_diagonal(dist_sq, 1.0)
        dist_2b = np.power(dist_sq, b)

        # Attractive gradient coefficient: -2 * a * b * d^{2(b-1)} / (1 + a * d^{2b})
        attr_coeff = (2.0 * a * b * np.power(dist_sq, b - 1.0)) / (1.0 + a * dist_2b)
        np.fill_diagonal(attr_coeff, 0.0)

        # Repulsive gradient coefficient: 2 * b / ((0.01 + d^2) * (1 + a * d^{2b}))
        rep_coeff = (2.0 * b) / ((0.05 + dist_sq) * (1.0 + a * dist_2b))
        np.fill_diagonal(rep_coeff, 0.0)

        net_coeff = (W * attr_coeff) - ((1.0 - W) * repulsion_weight * rep_coeff)
        grad = np.sum(net_coeff[:, :, None] * diff, axis=1)
        grad = np.clip(grad, -4.0, 4.0)
        Y = Y - lr * grad

    Y = Y - np.mean(Y, axis=0, keepdims=True)
    return Y.astype(np.float64)


def compute_2d_embedding(
    feature_matrix: np.ndarray,
    embedding_params: Optional[EmbeddingParameters] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[float]]:
    """Project feature matrix into 2D space using PCA or UMAP.

    Args:
        feature_matrix: Matrix of shape [N, D].
        embedding_params: Configuration for 2D embedding (PCA or UMAP).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (coordinates_2d [N, 2], explained_variance_ratio [var_dim1, var_dim2]).
    """
    n_samples, n_features = feature_matrix.shape
    if n_samples == 0:
        return np.empty((0, 2), dtype=np.float64), [0.0, 0.0]

    params = embedding_params or EmbeddingParameters()
    seed = random_state
    method = params.method.upper()

    if method == "UMAP":
        try:
            import umap
            n_nbrs = max(2, min(params.umap_n_neighbors, n_samples - 1)) if n_samples > 2 else 2
            reducer = umap.UMAP(
                n_neighbors=n_nbrs,
                min_dist=params.umap_min_dist,
                metric=params.umap_metric,
                spread=1.0,
                random_state=seed,
            )
            coords = reducer.fit_transform(feature_matrix)
            return coords.astype(np.float64), [0.0, 0.0]
        except Exception:
            coords = _compute_native_umap(feature_matrix, params, seed=seed)
            return coords, [0.0, 0.0]

    # PCA path
    pc_x = max(1, params.pca_component_x)
    pc_y = max(1, params.pca_component_y)
    n_comp = min(max(pc_x, pc_y), n_samples, n_features)

    if n_comp < 1 or n_samples < 2 or n_features < 2:
        return compute_pca_embedding(feature_matrix, n_components=2, random_state=seed)

    pca = PCA(n_components=n_comp, whiten=params.pca_whiten, random_state=seed)
    transformed = pca.fit_transform(feature_matrix)

    coords_2d = np.zeros((n_samples, 2), dtype=np.float64)
    idx_x = pc_x - 1
    idx_y = pc_y - 1
    if idx_x < transformed.shape[1]:
        coords_2d[:, 0] = transformed[:, idx_x]
    if idx_y < transformed.shape[1]:
        coords_2d[:, 1] = transformed[:, idx_y]

    exp_var = [float(v) for v in pca.explained_variance_ratio_]
    var_x = exp_var[idx_x] if idx_x < len(exp_var) else 0.0
    var_y = exp_var[idx_y] if idx_y < len(exp_var) else 0.0
    return coords_2d, [var_x, var_y]


def run_hierarchical_clustering(
    feature_matrix: np.ndarray,
    params: ClusteringParameters,
) -> np.ndarray:
    """Run Agglomerative Hierarchical Clustering on feature matrix.

    Args:
        feature_matrix: Standardized feature matrix of shape [N, D].
        params: ClusteringParameters configuration.

    Returns:
        1D integer array of cluster assignments (0 to K-1).
    """
    n_samples = feature_matrix.shape[0]
    if n_samples == 0:
        return np.empty(0, dtype=np.int32)
    if n_samples == 1:
        return np.zeros(1, dtype=np.int32)

    # Normalize linkage and metric combinations (Ward requires Euclidean)
    linkage_mode = params.linkage.lower()
    metric_mode = params.metric.lower()
    if linkage_mode == "ward" and metric_mode != "euclidean":
        metric_mode = "euclidean"

    if params.cutoff_mode == "k":
        k = max(2, min(int(params.n_clusters), n_samples))
        # Use AgglomerativeClustering with n_clusters
        model = AgglomerativeClustering(
            n_clusters=k,
            metric=metric_mode,
            linkage=linkage_mode,
        )
        labels = model.fit_predict(feature_matrix)
    else:
        # Distance threshold mode: use scipy hierarchy linkage + fcluster
        # Scipy linkage accepts metric
        try:
            z = linkage(feature_matrix, method=linkage_mode, metric=metric_mode)
            # fcluster criterion='distance' returns 1-based cluster IDs
            raw_labels = fcluster(z, t=params.distance_threshold, criterion="distance")
            # Convert to 0-based cluster IDs
            unique_ids = np.unique(raw_labels)
            id_map = {old: new for new, old in enumerate(unique_ids)}
            labels = np.array([id_map[v] for v in raw_labels], dtype=np.int32)
        except Exception:
            # Fallback to k=4 if distance threshold produces error
            model = AgglomerativeClustering(n_clusters=min(4, n_samples), metric="euclidean", linkage="ward")
            labels = model.fit_predict(feature_matrix)

    return labels.astype(np.int32)


def run_clustering_algorithm(
    feature_matrix: np.ndarray,
    params: ClusteringParameters,
) -> np.ndarray:
    """Run specified clustering algorithm (hierarchical, kmeans, gmm) on feature matrix.

    Args:
        feature_matrix: Extracted feature matrix [N, D].
        params: Clustering parameters including algorithm and hyperparameters.

    Returns:
        1D integer array of cluster assignments (0 to K-1).
    """
    n_samples = feature_matrix.shape[0]
    if n_samples == 0:
        return np.empty(0, dtype=np.int32)
    if n_samples == 1:
        return np.zeros(1, dtype=np.int32)

    X = feature_matrix

    # Dispatch algorithm
    algo = getattr(params, "algorithm", "hierarchical").lower()
    if algo == "kmeans":
        from sklearn.cluster import KMeans
        k = max(2, min(int(params.n_clusters), n_samples))
        km = KMeans(
            n_clusters=k,
            init=params.kmeans_init,
            n_init=params.kmeans_n_init,
            max_iter=params.kmeans_max_iter,
            random_state=params.random_state,
        )
        labels = km.fit_predict(X)
        return labels.astype(np.int32)

    elif algo == "gmm":
        from sklearn.mixture import GaussianMixture
        k = max(2, min(int(params.n_clusters), n_samples))
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=params.gmm_covariance_type,
            reg_covar=params.gmm_reg_covar,
            max_iter=params.gmm_max_iter,
            random_state=params.random_state,
        )
        labels = gmm.fit_predict(X)
        return labels.astype(np.int32)

    else:
        # Default: hierarchical
        return run_hierarchical_clustering(X, params)


def normalize_psth_waveform(psth: np.ndarray, norm_mode: str = "peak") -> np.ndarray:
    """Normalize a 1D or 2D PSTH waveform array by peak, probability, or keep raw.

    Args:
        psth: 1D [T] or 2D [N, T] numpy array of firing rates.
        norm_mode: Normalization mode:
            - "peak": Scale waveform to max = 1.0 (psth / max(psth, 1e-6)).
            - "probability": Scale waveform so total area / sum = 1.0 (psth / sum(psth, 1e-6)).
            - "raw": Return unchanged firing rate values in Hz.

    Returns:
        Normalized numpy array matching input shape.
    """
    arr = np.asarray(psth, dtype=np.float64)
    if arr.size == 0:
        return arr.copy()

    mode = str(norm_mode or "peak").strip().lower()
    if mode == "peak":
        if arr.ndim == 1:
            peak = float(np.max(arr))
            denom = max(peak, 1e-6)
            return arr / denom
        else:
            peak = np.maximum(np.max(arr, axis=-1, keepdims=True), 1e-6)
            return arr / peak
    elif mode == "probability":
        if arr.ndim == 1:
            total = float(np.sum(arr))
            denom = max(total, 1e-6)
            return arr / denom
        else:
            total = np.maximum(np.sum(arr, axis=-1, keepdims=True), 1e-6)
            return arr / total
    elif mode == "raw":
        return arr.copy()
    else:
        # Fallback to peak if unrecognized mode
        if arr.ndim == 1:
            return arr / max(float(np.max(arr)), 1e-6)
        else:
            return arr / np.maximum(np.max(arr, axis=-1, keepdims=True), 1e-6)


def compute_cluster_summaries(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    psth_matrix: np.ndarray,
    time_bins_ms: np.ndarray,
    active_cluster_ids: Optional[Any] = None,
    norm_mode: str = "probability",
) -> Dict[int, ClusterSummary]:
    """Compute grand-average waveforms and confidence envelopes for each response cluster.

    Args:
        df: Pooled units dataframe.
        cluster_labels: 1D integer cluster IDs.
        psth_matrix: Raw or normalized PSTH firing matrix [N, T].
        time_bins_ms: Time vector in ms [T].
        active_cluster_ids: Optional collection of cluster IDs to preserve even if count == 0.
        norm_mode: Normalization mode for member PSTHs ("peak", "probability", or "raw").

    Returns:
        Dictionary mapping cluster_id -> ClusterSummary.
    """
    summaries: Dict[int, ClusterSummary] = {}
    if active_cluster_ids is not None:
        cids_set = set(int(c) for c in active_cluster_ids if c >= 0) | set(int(c) for c in np.unique(cluster_labels) if c >= 0)
        unique_clusters = np.array(sorted(cids_set), dtype=int)
    else:
        unique_clusters = np.sort(np.unique(cluster_labels))

    unit_ids = df["global_unit_id"].astype(str).tolist() if "global_unit_id" in df.columns else [f"u_{i}" for i in range(len(df))]

    for cid in unique_clusters:
        if cid < 0:
            continue
        mask = (cluster_labels == cid)
        count = int(np.sum(mask))
        if count == 0:
            continue
        c_units = [unit_ids[i] for i in range(len(unit_ids)) if mask[i]]
        color = color_for_cluster(int(cid))

        if count > 0 and psth_matrix.shape[0] > 0:
            cluster_psths = psth_matrix[mask, :]
            cluster_psths_norm = normalize_psth_waveform(cluster_psths, norm_mode=norm_mode)
            mean_trace = np.mean(cluster_psths_norm, axis=0)
            if count > 1:
                sem_trace = np.std(cluster_psths_norm, axis=0) / np.sqrt(count)
            else:
                sem_trace = np.zeros_like(mean_trace)
        else:
            mean_trace = np.zeros_like(time_bins_ms)
            sem_trace = np.zeros_like(time_bins_ms)

        summaries[int(cid)] = ClusterSummary(
            cluster_id=int(cid),
            count=count,
            color_hex=color,
            unit_ids=c_units,
            time_bins_ms=time_bins_ms,
            mean_psth=mean_trace,
            sem_psth=sem_trace,
            norm_mode=norm_mode,
        )

    return summaries


def perform_clustering(
    df: pd.DataFrame,
    params: Optional[ClusteringParameters] = None,
    embedding_params: Optional[EmbeddingParameters] = None,
    spike_data_map: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    psth_cache: Optional[Dict[Any, np.ndarray]] = None,
    pulse_duration_ms: Optional[float] = None,
) -> ClusteringResult:
    """Execute end-to-end 2D projection, clustering, and grand-average profile extraction.

    Args:
        df: DataFrame of pooled optotagged units.
        params: Optional ClusteringParameters.
        embedding_params: Optional EmbeddingParameters for 2D projection (PCA or UMAP).
        spike_data_map: Optional map from unit identifier to (spike_times, pulse_onsets).
        psth_cache: Optional cache mapping (gid, time_window_ms, bin_size_ms, smooth, sigma) to 1D PSTH array.
        pulse_duration_ms: Optional optogenetic pulse duration in ms.

    Returns:
        ClusteringResult with cluster labels, coordinates, explained variance, and cluster summaries.
    """
    if params is None:
        params = ClusteringParameters()

    n_units = len(df)
    unit_ids = df["global_unit_id"].astype(str).tolist() if "global_unit_id" in df.columns else [f"u_{i}" for i in range(n_units)]

    if n_units == 0:
        return ClusteringResult(
            cluster_labels=np.empty(0, dtype=np.int32),
            pca_coordinates=np.empty((0, 2), dtype=np.float64),
            explained_variance_ratio=[0.0, 0.0],
            cluster_summaries={},
            feature_matrix=np.empty((0, 0), dtype=np.float64),
            unit_ids=[],
            parameters=params,
        )

    # 1. Extract feature matrix and PSTHs
    feat_matrix, psth_matrix, t_bins = extract_feature_matrix(
        df,
        feature_config=params.feature_config,
        spike_data_map=spike_data_map,
        psth_cache=psth_cache,
        pulse_duration_ms=pulse_duration_ms,
    )

    # 2. Compute 2D projection for visualization (PCA or UMAP)
    coords_2d, exp_var = compute_2d_embedding(
        feat_matrix,
        embedding_params=embedding_params,
        random_state=params.random_state,
    )

    # 3. Run clustering algorithm
    labels = run_clustering_algorithm(feat_matrix, params)

    # 4. Generate grand-average cluster profiles using presented time window
    psth_matrix, t_bins = generate_psth_matrix(
        df,
        time_window_ms=params.presented_window_ms,
        bin_size_ms=params.presented_bin_size_ms,
        spike_data_map=spike_data_map,
        psth_cache=psth_cache,
        smooth=params.psth_smooth,
        gaussian_sigma_ms=params.psth_gaussian_sigma_ms,
    )
    summaries = compute_cluster_summaries(
        df,
        labels,
        psth_matrix,
        t_bins,
        norm_mode=getattr(params, "cluster_psth_norm", "probability"),
    )

    return ClusteringResult(
        cluster_labels=labels,
        pca_coordinates=coords_2d,
        explained_variance_ratio=exp_var,
        cluster_summaries=summaries,
        feature_matrix=feat_matrix,
        unit_ids=unit_ids,
        parameters=params,
    )


def seed_clusters_affinity_propagation(
    df: pd.DataFrame,
    feature_config: Optional[FeatureConfiguration] = None,
    damping: float = 0.8,
    similarity_percentile: float = 50.0,
    random_state: int = 42,
    presented_window_ms: Tuple[float, float] = (-30.0, 50.0),
    presented_bin_size_ms: float = 2.0,
    spike_data_map: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    psth_cache: Optional[Dict[Any, np.ndarray]] = None,
    ap_params: Optional[AffinityPropagationParameters] = None,
    embedding_params: Optional[EmbeddingParameters] = None,
    pulse_duration_ms: Optional[float] = None,
) -> ClusteringResult:
    """Seed initial response clusters using Affinity Propagation with exemplar distance filtering.

    Units with distances to their cluster exemplar within `similarity_percentile` are assigned
    to the cluster; distant/uncertain units remain unassigned (-1).

    Args:
        df: Pooled DataFrame containing unit records.
        feature_config: Optional FeatureConfiguration specifying enabled metrics and PSTH parameters.
        damping: Damping factor for Affinity Propagation (0.5 to 1.0).
        similarity_percentile: Percentile distance threshold within each cluster (e.g. 50.0 = median).
        random_state: Random state for reproducibility.
        presented_window_ms: Time window for visualized cluster PSTH profiles.
        presented_bin_size_ms: Bin width in ms for visualized cluster PSTH profiles.
        spike_data_map: Optional map from unit identifier to (spike_times, pulse_onsets).
        psth_cache: Optional cache mapping (gid, time_window_ms, bin_size_ms, smooth, sigma) to 1D PSTH array.
        ap_params: Optional granular AffinityPropagationParameters.
        embedding_params: Optional EmbeddingParameters for 2D scatter coordinates.
        pulse_duration_ms: Optional optogenetic pulse duration in ms.

    Returns:
        ClusteringResult with initial cluster labels, 2D coordinates, explained variance, and summaries.
    """
    n_units = len(df)
    params = ClusteringParameters(
        feature_config=feature_config,
        presented_window_ms=presented_window_ms,
        presented_bin_size_ms=presented_bin_size_ms,
    )
    if n_units == 0:
        return ClusteringResult(
            cluster_labels=np.empty(0, dtype=np.int32),
            pca_coordinates=np.empty((0, 2), dtype=np.float64),
            explained_variance_ratio=[0.0, 0.0],
            cluster_summaries={},
            feature_matrix=np.empty((0, 0), dtype=np.float64),
            unit_ids=[],
            parameters=params,
        )

    unit_ids = (
        df["global_unit_id"].astype(str).tolist()
        if "global_unit_id" in df.columns
        else [f"u_{i}" for i in range(n_units)]
    )

    # 1. Extract feature matrix and PSTHs
    feat_matrix, psth_matrix, t_bins = extract_feature_matrix(
        df,
        feature_config=feature_config,
        spike_data_map=spike_data_map,
        psth_cache=psth_cache,
        pulse_duration_ms=pulse_duration_ms,
    )

    # Determine hyperparameters from ap_params if provided; internal defaults for AP convergence
    d_factor = ap_params.damping if ap_params else damping
    pct_cutoff = ap_params.similarity_percentile if ap_params else similarity_percentile
    min_size = ap_params.min_cluster_size if ap_params else 2
    seed = int(random_state)
    max_iter = 200
    convergence_iter = 15
    preference: Optional[float] = (
        float(ap_params.preference)
        if (ap_params is not None and ap_params.preference is not None)
        else None
    )

    # 2. Compute 2D projection
    coords_2d, exp_var = compute_2d_embedding(
        feat_matrix,
        embedding_params=embedding_params,
        random_state=seed,
    )

    if n_units < 2:
        labels = np.full(n_units, -1, dtype=np.int32)
        return ClusteringResult(
            cluster_labels=labels,
            pca_coordinates=coords_2d,
            explained_variance_ratio=exp_var,
            cluster_summaries={},
            feature_matrix=feat_matrix,
            unit_ids=unit_ids,
            parameters=params,
        )

    # 3. Fit Affinity Propagation
    try:
        ap = AffinityPropagation(
            damping=d_factor,
            max_iter=max_iter,
            convergence_iter=convergence_iter,
            preference=preference,
            random_state=seed,
        )
        raw_labels = ap.fit_predict(feat_matrix)
        exemplar_indices = getattr(ap, "cluster_centers_indices_", None)
    except Exception:
        raw_labels = np.full(n_units, -1, dtype=np.int32)
        exemplar_indices = None

    if exemplar_indices is None or len(exemplar_indices) == 0:
        final_labels = np.full(n_units, -1, dtype=np.int32)
    else:
        # 4. Exemplar distance thresholding
        final_labels = np.full(n_units, -1, dtype=np.int32)
        unique_cids = [c for c in np.unique(raw_labels) if c >= 0]

        for k in unique_cids:
            c_members = [i for i in range(n_units) if raw_labels[i] == k]
            if len(c_members) < min_size:
                # Discard clusters smaller than min_cluster_size
                continue

            if k < len(exemplar_indices):
                ex_idx = exemplar_indices[k]
                ex_vec = feat_matrix[ex_idx]
                dists = np.linalg.norm(feat_matrix[c_members] - ex_vec, axis=1)
                cutoff = float(np.percentile(dists, pct_cutoff))

                for idx_in_c, unit_idx in enumerate(c_members):
                    if dists[idx_in_c] <= cutoff:
                        final_labels[unit_idx] = k
            else:
                for unit_idx in c_members:
                    final_labels[unit_idx] = k

    # 5. Re-index active clusters to 0, 1, ... K-1 sequentially
    active_raw_cids = sorted(list(set([c for c in final_labels if c >= 0])))
    remap = {old_c: new_c for new_c, old_c in enumerate(active_raw_cids)}
    reindexed_labels = np.array([remap[c] if c in remap else -1 for c in final_labels], dtype=np.int32)

    # 6. Generate grand-average cluster profiles using presented time window
    psth_smooth_flag = params.psth_smooth if params is not None else True
    psth_sigma_val = params.psth_gaussian_sigma_ms if params is not None else 2.0
    psth_matrix, t_bins = generate_psth_matrix(
        df,
        time_window_ms=presented_window_ms,
        bin_size_ms=presented_bin_size_ms,
        spike_data_map=spike_data_map,
        psth_cache=psth_cache,
        smooth=psth_smooth_flag,
        gaussian_sigma_ms=psth_sigma_val,
    )
    norm_mode_val = getattr(params, "cluster_psth_norm", "probability") if params is not None else "probability"
    summaries = compute_cluster_summaries(df, reindexed_labels, psth_matrix, t_bins, norm_mode=norm_mode_val)

    return ClusteringResult(
        cluster_labels=reindexed_labels,
        pca_coordinates=coords_2d,
        explained_variance_ratio=exp_var,
        cluster_summaries=summaries,
        feature_matrix=feat_matrix,
        unit_ids=unit_ids,
        parameters=params,
    )


def compute_auto_assignments(
    df: pd.DataFrame,
    cluster_labels: Sequence[int],
    unit_status_map: Optional[Dict[str, str]] = None,
    threshold: float = 0.85,
    max_distance_sigma: float = 3.5,
    feature_config: Optional[FeatureConfiguration] = None,
    feature_matrix: Optional[np.ndarray] = None,
    n_estimators: int = 100,
    random_state: int = 42,
    spike_data_map: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    psth_cache: Optional[Dict[Any, np.ndarray]] = None,
    return_all_probabilities: bool = False,
    pulse_duration_ms: Optional[float] = None,
) -> Union[
    Tuple[Dict[str, int], Dict[str, float]],
    Tuple[Dict[str, int], Dict[str, float], Dict[str, Dict[int, float]]],
]:
    """Train Random Forest on assigned units and auto-assign non-curated units meeting confidence and distance thresholds.

    Ground-truth isolation: The classifier is strictly trained on units with 'Curated' or 'Seeded' status,
    permanently excluding 'Auto-assigned' units to prevent runaway majority cluster feedback loops.
    Dual-criterion gating: Outlier units with standardized distance to cluster centroid > max_distance_sigma
    are rejected from auto-assignment even if the Random Forest probability exceeds the threshold.

    Args:
        df: Pooled DataFrame containing unit records.
        cluster_labels: Array of current cluster labels (cluster_id >= 0 is assigned).
        unit_status_map: Dictionary mapping global_unit_id to status ('Curated', 'Seeded', 'Auto-assigned', 'Unassigned').
        threshold: Confidence probability threshold (0.50 to 0.99, default 0.85).
        max_distance_sigma: Maximum distance in cluster dispersion sigmas to permit auto-assignment (default 3.5).
        feature_config: Optional FeatureConfiguration.
        feature_matrix: Optional precomputed feature matrix.
        n_estimators: Number of trees in the Random Forest.
        random_state: Random state for reproducibility.
        spike_data_map: Optional map from unit identifier to (spike_times, pulse_onsets).
        psth_cache: Optional cache mapping (gid, time_window_ms, bin_size_ms, smooth, sigma) to 1D PSTH array.
        return_all_probabilities: If True, returns (auto_assignments, confidences, all_unit_probabilities).
        pulse_duration_ms: Optional optogenetic pulse duration in ms.

    Returns:
        Tuple of (auto_assignments {gid: cluster_id}, confidences {gid: max_prob}),
        plus all_unit_probabilities {gid: {cluster_id: prob}} if return_all_probabilities is True.
    """
    status_map = unit_status_map if unit_status_map is not None else {}
    n_units = len(df)
    if n_units == 0:
        if return_all_probabilities:
            return {}, {}, {}
        return {}, {}

    unit_ids = (
        df["global_unit_id"].astype(str).tolist()
        if "global_unit_id" in df.columns
        else [f"u_{i}" for i in range(n_units)]
    )

    if feature_matrix is None or feature_matrix.shape[0] != n_units:
        feat_matrix, _, _ = extract_feature_matrix(
            df,
            feature_config=feature_config,
            spike_data_map=spike_data_map,
            psth_cache=psth_cache,
            pulse_duration_ms=pulse_duration_ms,
        )
    else:
        feat_matrix = feature_matrix

    labels = np.asarray(cluster_labels, dtype=np.int32)

    # Ground-truth isolation: Only train on human-curated or AP-seeded units.
    # Auto-assigned pseudo-labels are strictly excluded from training.
    ground_truth_indices = [
        i for i in range(n_units)
        if labels[i] >= 0 and status_map.get(unit_ids[i]) in ("Curated", "Seeded")
    ]
    if len(ground_truth_indices) >= 2 and len(np.unique(labels[ground_truth_indices])) >= 2:
        train_indices = ground_truth_indices
    else:
        # Fallback if no status annotations exist yet
        train_indices = [i for i in range(n_units) if labels[i] >= 0]

    unique_train_classes = np.unique(labels[train_indices]) if train_indices else []

    # Need at least 2 samples and 2 distinct classes to train classifier
    if len(train_indices) < 2 or len(unique_train_classes) < 2:
        if return_all_probabilities:
            return {}, {}, {}
        return {}, {}

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(feat_matrix[train_indices], labels[train_indices])

    # Compute centroids for each trained class
    cluster_centroids: Dict[int, np.ndarray] = {}
    for c in unique_train_classes:
        c_int = int(c)
        c_members = [i for i in train_indices if labels[i] == c_int]
        if c_members:
            cluster_centroids[c_int] = np.mean(feat_matrix[c_members], axis=0)

    all_unit_probabilities: Dict[str, Dict[int, float]] = {}
    if return_all_probabilities:
        all_probs = clf.predict_proba(feat_matrix)
        classes_all = clf.classes_
        for i, gid in enumerate(unit_ids):
            all_unit_probabilities[gid] = {
                int(c): round(float(all_probs[i, k]), 4)
                for k, c in enumerate(classes_all)
            }

    # Evaluate all units that are NOT curated
    eval_indices = [i for i in range(n_units) if status_map.get(unit_ids[i]) != "Curated"]
    if not eval_indices:
        if return_all_probabilities:
            return {}, {}, all_unit_probabilities
        return {}, {}

    probs = clf.predict_proba(feat_matrix[eval_indices])
    classes = clf.classes_

    auto_assignments: Dict[str, int] = {}
    confidences: Dict[str, float] = {}

    for j, idx in enumerate(eval_indices):
        gid = unit_ids[idx]
        p_vec = probs[j]
        max_idx = int(np.argmax(p_vec))
        best_class = int(classes[max_idx])
        max_prob = float(p_vec[max_idx])

        confidences[gid] = round(max_prob, 4)

        # Dual-criterion distance gate: candidate Euclidean distance in standardized feature space <= max_distance_sigma
        passes_distance_gate = True
        if best_class in cluster_centroids:
            centroid = cluster_centroids[best_class]
            cand_feat = feat_matrix[idx]
            dist = float(np.linalg.norm(cand_feat - centroid))
            if dist > max_distance_sigma:
                passes_distance_gate = False

        if max_prob >= threshold and passes_distance_gate:
            auto_assignments[gid] = best_class
        else:
            auto_assignments[gid] = -1

    if return_all_probabilities:
        return auto_assignments, confidences, all_unit_probabilities
    return auto_assignments, confidences


def compute_cluster_candidate_suggestions(
    df: pd.DataFrame,
    target_cluster_id: int,
    cluster_labels: Sequence[int],
    unit_status_map: Optional[Dict[str, str]] = None,
    threshold: float = 0.50,
    max_distance_sigma: float = 3.5,
    feature_config: Optional[FeatureConfiguration] = None,
    feature_matrix: Optional[np.ndarray] = None,
    n_estimators: int = 100,
    random_state: int = 42,
    spike_data_map: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    psth_cache: Optional[Dict[Any, np.ndarray]] = None,
    precomputed_probabilities: Optional[Dict[str, Dict[int, float]]] = None,
    dismissed_gids: Optional[Set[str]] = None,
    pulse_duration_ms: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Compute and rank candidate units for a specific cluster based on Random Forest posteriors and centroid distance.

    Args:
        df: Pooled DataFrame containing unit records.
        target_cluster_id: The cluster ID to find suggestions for.
        cluster_labels: Sequence of integer cluster assignments.
        unit_status_map: Dict mapping global_unit_id to curation status ('Curated', 'Seeded', 'Auto-assigned', 'Unassigned').
        threshold: Minimum posterior probability for suggestion (default: 0.50).
        max_distance_sigma: Maximum distance in units of cluster dispersion sigma to be suggested (default: 3.5).
        feature_config: Optional FeatureConfiguration.
        feature_matrix: Optional precomputed feature matrix.
        n_estimators: Number of trees in Random Forest.
        random_state: Random state for reproducibility.
        spike_data_map: Optional spike data map.
        psth_cache: Optional PSTH cache.
        precomputed_probabilities: Optional dictionary of {gid: {cluster_id: prob}}.
        dismissed_gids: Optional set of global_unit_ids dismissed by the user for this cluster.
        pulse_duration_ms: Optional optogenetic pulse duration in ms.

    Returns:
        List of suggestion dictionaries containing unit info, posterior probability, centroid distance sigma, and analytical metrics,
        sorted by probability descending and distance ascending.
    """
    status_map = unit_status_map if unit_status_map is not None else {}
    dismissed = set(dismissed_gids) if dismissed_gids else set()
    n_units = len(df)
    if n_units == 0:
        return []

    unit_ids = (
        df["global_unit_id"].astype(str).tolist()
        if "global_unit_id" in df.columns
        else [f"u_{i}" for i in range(n_units)]
    )
    labels = np.asarray(cluster_labels, dtype=np.int32)

    if feature_matrix is None or feature_matrix.shape[0] != n_units:
        feat_matrix, _, _ = extract_feature_matrix(
            df,
            feature_config=feature_config,
            spike_data_map=spike_data_map,
            psth_cache=psth_cache,
            pulse_duration_ms=pulse_duration_ms,
        )
    else:
        feat_matrix = feature_matrix

    # Identify confirmed/reference members of target_cluster_id
    target_members = [
        i for i in range(n_units)
        if (labels[i] == target_cluster_id or df.iloc[i].get("cluster_id") == target_cluster_id)
        and status_map.get(unit_ids[i], df.iloc[i].get("status")) in ("Curated", "Seeded")
    ]
    if not target_members:
        target_members = [
            i for i in range(n_units)
            if labels[i] == target_cluster_id or df.iloc[i].get("cluster_id") == target_cluster_id
        ]

    if not target_members:
        return []

    target_feats = feat_matrix[target_members]
    centroid = np.mean(target_feats, axis=0)

    # Determine probabilities for target_cluster_id
    probs_for_target: Dict[str, float] = {}
    if precomputed_probabilities is not None:
        for gid in unit_ids:
            if gid in precomputed_probabilities:
                probs_for_target[gid] = float(precomputed_probabilities[gid].get(target_cluster_id, 0.0))

    # If probabilities not available from precomputation, train RF or compute from distance
    if len(probs_for_target) < n_units:
        ground_truth_indices = [
            i for i in range(n_units)
            if labels[i] >= 0 and status_map.get(unit_ids[i]) in ("Curated", "Seeded")
        ]
        if len(ground_truth_indices) >= 2 and len(np.unique(labels[ground_truth_indices])) >= 2:
            train_indices = ground_truth_indices
        else:
            train_indices = [i for i in range(n_units) if labels[i] >= 0]

        unique_classes = np.unique(labels[train_indices]) if train_indices else []
        if len(train_indices) >= 2 and len(unique_classes) >= 2 and target_cluster_id in unique_classes:
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                class_weight="balanced",
                random_state=random_state,
            )
            clf.fit(feat_matrix[train_indices], labels[train_indices])
            all_probs = clf.predict_proba(feat_matrix)
            target_class_col = int(np.where(clf.classes_ == target_cluster_id)[0][0])
            for i, gid in enumerate(unit_ids):
                if gid not in probs_for_target:
                    probs_for_target[gid] = float(all_probs[i, target_class_col])
        else:
            # Fallback for single-cluster state: Radial basis distance decay
            for i, gid in enumerate(unit_ids):
                if gid not in probs_for_target:
                    cand_feat = feat_matrix[i]
                    diff = cand_feat - centroid
                    rms_z = float(np.sqrt(np.mean(diff ** 2)))
                    sim = float(np.exp(-0.5 * (rms_z ** 2)))
                    probs_for_target[gid] = sim

    candidates: List[Dict[str, Any]] = []
    for i in range(n_units):
        gid = unit_ids[i]
        row = df.iloc[i]
        curr_status = status_map.get(gid, row.get("curated_status", row.get("status", "Unassigned")))
        curr_cid = row.get("cluster_id", labels[i] if i < len(labels) else -1)

        # A cell already in the target cluster must NEVER be in suggested candidates for that cluster
        if curr_cid == target_cluster_id or (i < len(labels) and labels[i] == target_cluster_id):
            continue

        # Curated units are confirmed ground truth and already displayed in Curated Members (or locked in their cluster)
        if curr_status == "Curated":
            continue

        if gid in dismissed:
            continue

        cand_feat = feat_matrix[i]
        dist = float(np.linalg.norm(cand_feat - centroid))

        p = probs_for_target.get(gid, 0.0)

        # Gate on distance and minimum interest
        if dist > max_distance_sigma:
            continue
        if p < threshold and dist > 2.0:
            continue

        row = df.iloc[i]
        candidates.append({
            "global_unit_id": gid,
            "session_id": str(row.get("session_id", "")),
            "unit_id": str(row.get("unit_id", "")),
            "probability": round(float(p), 4),
            "distance_sigma": round(float(dist), 2),
            "latency_ms": round(float(row.get("latency_ms", 0.0)), 2) if pd.notna(row.get("latency_ms")) else 0.0,
            "reliability": round(float(row.get("reliability", 0.0)), 2) if pd.notna(row.get("reliability")) else 0.0,
            "jitter_ms": round(float(row.get("jitter_ms", 0.0)), 2) if pd.notna(row.get("jitter_ms")) else 0.0,
            "salt_p_value": round(float(row.get("salt_p_value", 1.0)), 4) if pd.notna(row.get("salt_p_value")) else 1.0,
            "current_status": curr_status,
            "current_cluster_id": int(labels[i]),
        })

    # Sort descending by probability, tie-breaking by lowest distance_sigma
    candidates.sort(key=lambda c: (-c["probability"], c["distance_sigma"]))
    return candidates


def compute_entropy_ranking(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    unit_status_map: Optional[Dict[str, str]] = None,
    feature_matrix: Optional[np.ndarray] = None,
    skipped_unit_ids: Optional[Set[str]] = None,
    candidate_gids: Optional[List[str]] = None,
    n_estimators: int = 100,
    random_state: int = 42,
    cycle_skipped: bool = False,
    feature_config: Optional[FeatureConfiguration] = None,
    spike_data_map: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    psth_cache: Optional[Dict[Any, np.ndarray]] = None,
    pulse_duration_ms: Optional[float] = None,
) -> List[ActiveLearningCandidate]:
    """Train Random Forest classifier on clustered units and rank candidate units by Shannon entropy.

    Formulation:
        For candidate units x:
            P(y = c | x), for all active clusters c
            H(x) = -sum_c P(y = c | x) * log2(P(y = c | x))
            x* = argmax_x H(x)

    Args:
        df: Pooled DataFrame containing unit records.
        cluster_labels: Integer array of cluster assignments (cluster_id >= 0 is assigned).
        unit_status_map: Optional dictionary of {gid: status}. When provided, classifier trains on Curated/Seeded units.
        feature_matrix: Optional precomputed feature matrix of shape (N, D). If None, extracted from df.
        skipped_unit_ids: Set of unit global IDs to exclude from candidate ranking.
        candidate_gids: Optional explicit list of global IDs to evaluate as candidates.
        n_estimators: Number of trees in the Random Forest (default: 100).
        random_state: Random state for reproducibility.
        cycle_skipped: If True, skipped units are appended after active candidates instead of dropped.
        feature_config: Optional FeatureConfiguration used when extracting feature_matrix.
        spike_data_map: Optional map from unit identifier to (spike_times, pulse_onsets).
        psth_cache: Optional cache mapping (gid, time_window_ms, bin_size_ms, smooth, sigma) to 1D PSTH array.
        pulse_duration_ms: Optional optogenetic pulse duration in ms.

    Returns:
        List of ActiveLearningCandidate instances sorted in descending order of Shannon entropy H(x).
    """
    skipped = set(skipped_unit_ids) if skipped_unit_ids else set()
    n_units = len(df)
    if n_units == 0:
        return []

    unit_ids = (
        df["global_unit_id"].astype(str).tolist()
        if "global_unit_id" in df.columns
        else [f"u_{i}" for i in range(n_units)]
    )

    if feature_matrix is None or feature_matrix.shape[0] != n_units:
        feat_matrix, _, _ = extract_feature_matrix(
            df,
            feature_config=feature_config,
            spike_data_map=spike_data_map,
            psth_cache=psth_cache,
            pulse_duration_ms=pulse_duration_ms,
        )
    else:
        feat_matrix = feature_matrix

    labels = np.asarray(cluster_labels, dtype=np.int32)

    # Units with valid non-negative cluster assignments are available for training
    # Ground-truth isolation: Only train on human-curated or AP-seeded units when status map is available
    if unit_status_map:
        ground_truth_indices = [
            i for i in range(n_units)
            if labels[i] >= 0 and unit_status_map.get(unit_ids[i]) in ("Curated", "Seeded")
        ]
        if len(ground_truth_indices) >= 2 and len(np.unique(labels[ground_truth_indices])) >= 2:
            train_indices = ground_truth_indices
        else:
            train_indices = [i for i in range(n_units) if labels[i] >= 0]
    else:
        train_indices = [i for i in range(n_units) if labels[i] >= 0]

    # Determine candidate indices
    if candidate_gids is not None:
        active_indices = [i for i, gid in enumerate(unit_ids) if gid in candidate_gids and gid not in skipped]
        skipped_indices = [i for i, gid in enumerate(unit_ids) if gid in candidate_gids and gid in skipped] if cycle_skipped else []
    elif any(labels < 0):
        # Unassigned units are primary candidates
        active_indices = [i for i in range(n_units) if labels[i] < 0 and unit_ids[i] not in skipped]
        skipped_indices = [i for i in range(n_units) if labels[i] < 0 and unit_ids[i] in skipped] if cycle_skipped else []
    else:
        # When all units are assigned, evaluate non-skipped units to find borderline boundary units
        active_indices = [i for i in range(n_units) if unit_ids[i] not in skipped]
        skipped_indices = [i for i in range(n_units) if unit_ids[i] in skipped] if cycle_skipped else []

    cand_indices = active_indices + skipped_indices

    if not cand_indices:
        return []

    # If training set has fewer than 2 samples or fewer than 2 unique classes,
    # fallback gracefully: assign uniform probability and 0.0 entropy
    unique_train_classes = np.unique(labels[train_indices]) if train_indices else []
    if len(train_indices) < 2 or len(unique_train_classes) < 2:
        candidates = []
        for idx in cand_indices:
            gid = unit_ids[idx]
            row = df.iloc[idx]
            prob_dict = {int(c): 1.0 / max(1, len(unique_train_classes)) for c in unique_train_classes}
            candidates.append(
                ActiveLearningCandidate(
                    global_unit_id=gid,
                    entropy=0.0,
                    probabilities=prob_dict,
                    feature_values={
                        "latency_ms": float(row.get("latency_ms", 5.0)) if pd.notna(row.get("latency_ms")) else 5.0,
                        "jitter_ms": float(row.get("jitter_ms", 1.5)) if pd.notna(row.get("jitter_ms")) else 1.5,
                        "reliability": float(row.get("reliability", 0.8)) if pd.notna(row.get("reliability")) else 0.8,
                        "waveform_correlation": float(row.get("waveform_correlation", 0.9)) if pd.notna(row.get("waveform_correlation")) else 0.9,
                        "salt_p_value": float(row.get("salt_p_value", 0.01)) if pd.notna(row.get("salt_p_value")) else 0.01,
                    },
                    cluster_id=int(labels[idx]),
                    session_id=str(row.get("session_id", "")),
                    unit_id=int(row.get("unit_id", 0)) if str(row.get("unit_id", "")).isdigit() else 0,
                    depth=float(row.get("depth", (idx % 32) * 100 + 400)),
                )
            )
        active_cands = [c for c in candidates if c.global_unit_id not in skipped]
        skipped_cands = [c for c in candidates if c.global_unit_id in skipped]
        return active_cands + skipped_cands

    # Train Random Forest classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
    )
    X_train = feat_matrix[train_indices]
    y_train = labels[train_indices]
    clf.fit(X_train, y_train)

    # Predict class probabilities on candidate set
    X_cand = feat_matrix[cand_indices]
    probs = clf.predict_proba(X_cand)
    classes = clf.classes_

    eps = 1e-12
    candidates = []
    for j, idx in enumerate(cand_indices):
        p_vec = probs[j]
        # Shannon entropy: H(x) = -sum(P(y=c|x) * log2(P(y=c|x)))
        h_val = -float(np.sum(p_vec * np.log2(p_vec + eps)))
        h_val = max(0.0, round(h_val, 4))

        prob_dict = {int(c): float(p_vec[k]) for k, c in enumerate(classes)}
        row = df.iloc[idx]
        gid = unit_ids[idx]

        candidates.append(
            ActiveLearningCandidate(
                global_unit_id=gid,
                entropy=h_val,
                probabilities=prob_dict,
                feature_values={
                    "latency_ms": float(row.get("latency_ms", 5.0)) if pd.notna(row.get("latency_ms")) else 5.0,
                    "jitter_ms": float(row.get("jitter_ms", 1.5)) if pd.notna(row.get("jitter_ms")) else 1.5,
                    "reliability": float(row.get("reliability", 0.8)) if pd.notna(row.get("reliability")) else 0.8,
                    "waveform_correlation": float(row.get("waveform_correlation", 0.9)) if pd.notna(row.get("waveform_correlation")) else 0.9,
                    "salt_p_value": float(row.get("salt_p_value", 0.01)) if pd.notna(row.get("salt_p_value")) else 0.01,
                },
                cluster_id=int(labels[idx]),
                session_id=str(row.get("session_id", "")),
                unit_id=int(row.get("unit_id", 0)) if str(row.get("unit_id", "")).isdigit() else 0,
                depth=float(row.get("depth", (idx % 32) * 100 + 400)),
            )
        )

    # Rank by Shannon entropy descending: active candidates first, cycled skipped candidates at the end
    active_cands = [c for c in candidates if c.global_unit_id not in skipped]
    skipped_cands = [c for c in candidates if c.global_unit_id in skipped]
    active_cands.sort(key=lambda c: c.entropy, reverse=True)
    skipped_cands.sort(key=lambda c: c.entropy, reverse=True)
    return active_cands + skipped_cands

