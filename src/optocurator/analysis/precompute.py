"""Headless batch plot pre-rendering for raster and PSTH compound images."""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import matplotlib
# Ensure headless backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# pyrefly: ignore[missing-import]
import numpy as np

from optocurator.analysis.metrics import compute_psth_and_raster, compute_psth_time_ticks
from optocurator.models import StimulusProtocol, UnitData, UnitMetrics


def render_compound_psth_raster(
    unit: UnitData,
    stimulus: StimulusProtocol,
    output_path: Union[str, Path],
    window_ms: Tuple[float, float] = (-50.0, 50.0),
    bin_size_ms: float = 0.5,
    dpi: int = 150,
    order_by: str = "presentation",
    smooth: bool = True,
    gaussian_sigma_ms: float = 1.0,
) -> Path:
    """Render a high-resolution two-panel compound raster and PSTH plot for a single unit.

    Args:
        unit: UnitData object.
        stimulus: StimulusProtocol with pulse onsets.
        output_path: Target .png filepath.
        window_ms: (start_ms, end_ms) relative to pulse onset.
        bin_size_ms: Bin width in ms for PSTH histogram.
        dpi: Image resolution in dots per inch.
        order_by: Pulse ordering mode ("presentation" or "latency").
        smooth: Whether to render smooth firing rate PSTH curve using Gaussian kernel.
        gaussian_sigma_ms: Gaussian kernel sigma in ms for smoothing.

    Returns:
        Path to the saved image file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    psth_data = compute_psth_and_raster(
        unit.spike_times,
        stimulus.onsets,
        window_ms=window_ms,
        bin_size_ms=bin_size_ms,
        order_by=order_by,
        smooth=smooth,
        gaussian_sigma_ms=gaussian_sigma_ms,
    )

    fig, (ax_raster, ax_psth) = plt.subplots(
        2, 1, figsize=(6.5, 5.5), dpi=dpi, sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.08}
    )
    fig.patch.set_facecolor("#FFFFFF")

    # --- Top Panel: Spike Raster ---
    ax_raster.set_facecolor("#FFFFFF")
    if len(psth_data["raster_x"]) > 0:
        ax_raster.scatter(
            psth_data["raster_x"],
            psth_data["raster_y"],
            s=12.0,
            c="#000000",
            marker="|",
            linewidths=1.0,
            alpha=0.9,
        )

    ax_raster.axvline(0.0, color="#2563EB", linestyle="--", linewidth=1.2, alpha=0.9)
    y_lbl = "Pulse Trial # (Latency)" if order_by == "latency" else "Pulse Trial #"
    ax_raster.set_ylabel(y_lbl, fontsize=10, color="#334155")
    ax_raster.set_ylim(-0.5, max(1.0, float(psth_data["n_trials"] - 0.5)))
    ax_raster.set_xlim(window_ms[0], window_ms[1])
    ax_raster.tick_params(colors="#475569", labelsize=9, labelbottom=False)
    for spine in ax_raster.spines.values():
        spine.set_color("#CBD5E1")

    # --- Bottom Panel: PSTH Histogram / Curve ---
    ax_psth.set_facecolor("#FFFFFF")
    bins = psth_data["psth_bins"]
    rates = psth_data["psth_rates"]
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    bin_width = bins[1] - bins[0]

    if smooth:
        if len(rates) > 0:
            ax_psth.plot(bin_centers, rates, color="#2563EB", linewidth=1.8)
            ax_psth.fill_between(bin_centers, 0, rates, color="#3B82F6", alpha=0.35)
    else:
        if len(rates) > 0:
            ax_psth.bar(
                bin_centers,
                rates,
                width=bin_width,
                color="#3B82F6",
                edgecolor="#1D4ED8",
                linewidth=0.5,
                alpha=0.9,
            )

    ax_psth.axvline(0.0, color="#2563EB", linestyle="--", linewidth=1.2, alpha=0.9)

    # 3-sigma baseline threshold line
    thresh_3sig = psth_data["baseline_3sigma"]
    base_mean = psth_data["baseline_mean"]
    if thresh_3sig > 0:
        ax_psth.axhline(
            thresh_3sig,
            color="#D97706",
            linestyle="--",
            linewidth=1.2,
            label=f"Baseline 3σ ({thresh_3sig:.1f} Hz, Mean: {base_mean:.1f} Hz)",
        )
        ax_psth.legend(loc="upper right", frameon=True, fontsize=8, facecolor="#F8FAFC", edgecolor="#E2E8F0")

    ax_psth.set_xlabel("Time from Pulse Onset (ms)", fontsize=10, color="#334155")
    ax_psth.set_ylabel("Firing Rate (Hz)", fontsize=10, color="#334155")
    y_max = max(10.0, float(np.max(rates)) * 1.2 if len(rates) > 0 else 10.0, thresh_3sig * 1.2)
    ax_psth.set_ylim(0.0, y_max)
    psth_ticks = compute_psth_time_ticks(window_ms[0], window_ms[1])
    ax_psth.set_xticks(psth_ticks)
    ax_psth.set_xticklabels([str(int(t)) if float(t).is_integer() else f"{t:g}" for t in psth_ticks])
    ax_psth.tick_params(colors="#475569", labelsize=9)
    for spine in ax_psth.spines.values():
        spine.set_color("#CBD5E1")

    # Clean layout with no title above spike raster
    fig.subplots_adjust(top=0.96, bottom=0.10, left=0.12, right=0.95, hspace=0.08)
    fig.savefig(str(output_path), dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)

    return output_path


def batch_precompute_plots(
    units: List[UnitData],
    stimulus: StimulusProtocol,
    output_dir: Union[str, Path],
    only_tagged: bool = False,
    metrics_map: Optional[Dict[int, UnitMetrics]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    dpi: int = 150,
    smooth: bool = True,
    gaussian_sigma_ms: float = 1.0,
) -> List[Path]:
    """Headless batch generator rendering raster+PSTH compound plots into output_dir.

    Args:
        units: List of UnitData objects.
        stimulus: StimulusProtocol object.
        output_dir: Directory where plots will be saved (e.g., <session_id>_plots/).
        only_tagged: If True, only render units where metrics_map[u.unit_id].is_tagged is True.
        metrics_map: Mapping of unit_id -> UnitMetrics.
        progress_callback: Optional callback receiving (completed_count, total_count).
        dpi: Image resolution in DPI.
        smooth: Whether to render smooth firing rate PSTH curve using Gaussian kernel.
        gaussian_sigma_ms: Gaussian kernel sigma in ms for smoothing.

    Returns:
        List of Paths to generated PNG files.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter units if only_tagged is requested
    if only_tagged and metrics_map is not None:
        target_units = [u for u in units if metrics_map.get(u.unit_id) and metrics_map[u.unit_id].is_tagged]
        if not target_units:
            # Fallback: if no units tagged, render all to avoid empty gallery
            target_units = units
    else:
        target_units = units

    total = len(target_units)
    generated_paths: List[Path] = []

    # Optimized rendering reusing figure across iterations
    fig, (ax_raster, ax_psth) = plt.subplots(
        2, 1, figsize=(6.5, 5.5), dpi=dpi, sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0]}
    )
    fig.patch.set_facecolor("#FFFFFF")

    try:
        for idx, unit in enumerate(target_units):
            filename = f"{unit.global_id}_psth_raster.png"
            file_path = out_dir / filename

            psth_data = compute_psth_and_raster(
                unit.spike_times,
                stimulus.onsets,
                smooth=smooth,
                gaussian_sigma_ms=gaussian_sigma_ms,
            )

            # Clear previous contents
            ax_raster.clear()
            ax_psth.clear()

            # Render raster
            ax_raster.set_facecolor("#FFFFFF")
            if len(psth_data["raster_x"]) > 0:
                ax_raster.scatter(
                    psth_data["raster_x"],
                    psth_data["raster_y"],
                    s=12.0,
                    c="#000000",
                    marker="|",
                    linewidths=1.0,
                    alpha=0.9,
                )
            ax_raster.axvline(0.0, color="#2563EB", linestyle="--", linewidth=1.2)
            ax_raster.set_ylabel("Pulse Trial #", fontsize=10, color="#334155")
            ax_raster.set_ylim(-0.5, max(1.0, float(psth_data["n_trials"] - 0.5)))
            ax_raster.set_xlim(-50.0, 50.0)
            ax_raster.tick_params(colors="#475569", labelsize=9)
            for spine in ax_raster.spines.values():
                spine.set_color("#CBD5E1")

            # Render PSTH
            ax_psth.set_facecolor("#FFFFFF")
            bins = psth_data["psth_bins"]
            rates = psth_data["psth_rates"]
            bin_centers = (bins[:-1] + bins[1:]) / 2.0
            bin_width = bins[1] - bins[0]

            if smooth:
                if len(rates) > 0:
                    ax_psth.plot(bin_centers, rates, color="#2563EB", linewidth=1.8)
                    ax_psth.fill_between(bin_centers, 0, rates, color="#3B82F6", alpha=0.35)
            else:
                if len(rates) > 0:
                    ax_psth.bar(
                        bin_centers,
                        rates,
                        width=bin_width,
                        color="#3B82F6",
                        edgecolor="#1D4ED8",
                        linewidth=0.5,
                        alpha=0.9,
                    )
            ax_psth.axvline(0.0, color="#2563EB", linestyle="--", linewidth=1.2)

            thresh_3sig = psth_data["baseline_3sigma"]
            base_mean = psth_data["baseline_mean"]
            if thresh_3sig > 0:
                ax_psth.axhline(
                    thresh_3sig,
                    color="#D97706",
                    linestyle="--",
                    linewidth=1.2,
                    label=f"3σ ({thresh_3sig:.1f} Hz)",
                )
                ax_psth.legend(loc="upper right", frameon=True, fontsize=8, facecolor="#F8FAFC", edgecolor="#E2E8F0")

            ax_psth.set_xlabel("Time from Pulse Onset (ms)", fontsize=10, color="#334155")
            ax_psth.set_ylabel("Firing Rate (Hz)", fontsize=10, color="#334155")
            y_max = max(10.0, float(np.max(rates)) * 1.2 if len(rates) > 0 else 10.0, thresh_3sig * 1.2)
            ax_psth.set_ylim(0.0, y_max)
            ax_psth.set_title(
                f"PSTH (Baseline Mean: {base_mean:.1f} Hz)",
                fontsize=10,
                color="#475569",
                pad=6,
            )
            psth_ticks = compute_psth_time_ticks(-50.0, 50.0)
            ax_psth.set_xticks(psth_ticks)
            ax_psth.set_xticklabels([str(int(t)) if float(t).is_integer() else f"{t:g}" for t in psth_ticks])
            ax_psth.tick_params(colors="#475569", labelsize=9)
            for spine in ax_psth.spines.values():
                spine.set_color("#CBD5E1")

            fig.tight_layout()
            fig.savefig(str(file_path), dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
            generated_paths.append(file_path)

            if progress_callback:
                progress_callback(idx + 1, total)

    finally:
        plt.close(fig)

    return generated_paths
