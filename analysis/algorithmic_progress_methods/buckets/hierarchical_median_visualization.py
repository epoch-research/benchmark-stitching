#!/usr/bin/env python3
"""Visualization utilities for hierarchical compute reduction estimates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
os.chdir(PROJECT_ROOT)

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.cli_utils import generate_title_suffix  # noqa: E402
from shared.plotting.base import apply_plot_style, save_figure  # noqa: E402
from shared.plotting.distributions import plot_histogram_with_stats  # noqa: E402


def _infer_observations_path(summary_path: Path) -> Path:
    """Given a summary path, guess the matching observations CSV."""
    stem = summary_path.stem.replace("hierarchical_median_summary", "hierarchical_median_observations")
    candidate = summary_path.with_name(f"{stem}{summary_path.suffix}")
    if candidate.exists():
        return candidate
    # Fallback to CSV extension
    candidate_csv = candidate.with_suffix(".csv")
    if candidate_csv.exists():
        return candidate_csv
    raise FileNotFoundError(
        f"Could not locate observations CSV next to {summary_path}. "
        "Specify --observations-csv explicitly."
    )


def _default_output_base(summary_path: Path) -> Path:
    """Derive default output filename stem based on the summary path."""
    stem = summary_path.stem.replace(
        "hierarchical_median_summary", "hierarchical_median_diagnostics"
    )
    return summary_path.with_name(stem)


def load_summary_and_data(summary_path: Path, observations_path: Optional[Path]) -> tuple[dict, pd.DataFrame]:
    """Load hierarchical summary JSON and observations dataframe."""
    with open(summary_path, "r", encoding="utf-8") as fp:
        summary = json.load(fp)

    obs_path = observations_path or _infer_observations_path(summary_path)
    observations_df = pd.read_csv(obs_path)
    return summary, observations_df


def make_plots(summary: dict, observations_df: pd.DataFrame, output_path: Path,
               title_suffix: str = "") -> Path:
    """Create diagnostic plots for hierarchical median outputs."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    mu_samples = np.asarray(summary.get("posterior_mu_samples", []), dtype=float)
    multiplier_samples = 10 ** mu_samples if mu_samples.size else np.array([])

    # 1. Posterior distribution of compute reduction (OOMs/year)
    ax = axes[0, 0]
    if mu_samples.size > 0:
        plot_histogram_with_stats(
            ax,
            mu_samples,
            bins=40,
            xlabel="Compute Reduction (OOMs/year)",
            title=f"Posterior for Global Compute Reduction{title_suffix}",
            color="royalblue",
        )
    else:
        summary_mu = summary.get("posterior_mu_oom_per_year", {})
        median = summary_mu.get("median")
        ci_lower = summary_mu.get("ci_lower")
        ci_upper = summary_mu.get("ci_upper")
        if median is not None:
            ax.axvline(median, color="red", linestyle="--", linewidth=2, label=f"Median: {median:.2f}")
        if ci_lower is not None and ci_upper is not None:
            ax.axvspan(ci_lower, ci_upper, color="steelblue", alpha=0.2, label="95% CI")
        apply_plot_style(
            ax,
            title=f"Global Compute Reduction Summary{title_suffix}",
            xlabel="Compute Reduction (OOMs/year)",
            ylabel="",
            legend=True,
        )

    # 2. Posterior distribution in multiplier space
    ax = axes[0, 1]
    if multiplier_samples.size > 0:
        bins = np.logspace(np.log10(multiplier_samples.min()), np.log10(multiplier_samples.max()), 40)
        ax.hist(multiplier_samples, bins=bins, color="seagreen", alpha=0.75, edgecolor="black")
        ax.set_xscale("log")
        stats = np.percentile(multiplier_samples, [50, 2.5, 97.5])
        ax.axvline(stats[0], color="red", linestyle="--", linewidth=2, label=f"Median: {stats[0]:.1f}×")
        ax.axvline(stats[1], color="gray", linestyle=":", linewidth=2, alpha=0.7)
        ax.axvline(stats[2], color="gray", linestyle=":", linewidth=2, alpha=0.7,
                   label=f"95% CI: [{stats[1]:.1f}×, {stats[2]:.1f}×]")
        apply_plot_style(
            ax,
            title=f"Posterior Compute Multiplier (per year){title_suffix}",
            xlabel="Multiplier (× per year, log scale)",
            ylabel="Frequency",
            legend=True,
        )
    else:
        apply_plot_style(
            ax,
            title=f"Posterior Compute Multiplier (per year){title_suffix}",
            xlabel="Multiplier (× per year)",
            ylabel="",
        )

    # 3. Bucket-level posterior intervals vs ECI center
    ax = axes[1, 0]
    bucket_posteriors = summary.get("bucket_posteriors", [])
    if bucket_posteriors:
        centers = [bp["bucket_center"] for bp in bucket_posteriors]
        medians = [bp["median"] for bp in bucket_posteriors]
        lower = [bp["ci_lower"] for bp in bucket_posteriors]
        upper = [bp["ci_upper"] for bp in bucket_posteriors]
        sizes = [bp.get("n_models_sota", 1) for bp in bucket_posteriors]

        yerr = np.array(
            [[m - l for m, l in zip(medians, lower)],
             [u - m for m, u in zip(medians, upper)]]
        )
        ax.errorbar(
            centers,
            medians,
            yerr=yerr,
            fmt="o",
            markersize=6,
            ecolor="gray",
            capsize=4,
            alpha=0.8,
        )
        scatter = ax.scatter(
            centers,
            medians,
            s=np.array(sizes) * 20,
            c=[bp["bucket_size"] for bp in bucket_posteriors],
            cmap="viridis",
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
            label="Bucket posterior median",
        )
        fig.colorbar(scatter, ax=ax, label="ECI bucket width")

        summary_mu = summary.get("posterior_mu_oom_per_year", {})
        median = summary_mu.get("median")
        ci_lower = summary_mu.get("ci_lower")
        ci_upper = summary_mu.get("ci_upper")
        if median is not None:
            ax.axhline(median, color="red", linestyle="--", linewidth=2,
                       label=f"Global median: {median:.2f}")
        if ci_lower is not None and ci_upper is not None:
            ax.axhspan(ci_lower, ci_upper, color="red", alpha=0.1, label="Global 95% CI")

    apply_plot_style(
        ax,
        title=f"Bucket Posterior Intervals by Capability Level{title_suffix}",
        xlabel="ECI Bucket Center",
        ylabel="Compute Reduction (OOMs/year)",
        legend=True,
    )

    # 4. Bucket width sensitivity (observed data)
    ax = axes[1, 1]
    if len(observations_df) > 0:
        ax.scatter(
            observations_df["bucket_size"],
            observations_df["compute_reduction_oom_per_year"],
            alpha=0.3,
            s=40,
            color="gray",
            label="Bucket estimates",
        )
        group = observations_df.groupby("bucket_size")["compute_reduction_oom_per_year"]
        medians = group.median()
        lower = group.quantile(0.25)
        upper = group.quantile(0.75)
        ax.errorbar(
            medians.index,
            medians.values,
            yerr=[medians.values - lower.values, upper.values - medians.values],
            fmt="o-",
            color="steelblue",
            linewidth=2,
            markersize=8,
            capsize=4,
            label="Observed median ± IQR",
        )

    apply_plot_style(
        ax,
        title=f"Bucket Size Sensitivity (Observed){title_suffix}",
        xlabel="ECI Bucket Size",
        ylabel="Compute Reduction (OOMs/year)",
        legend=True,
    )

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def generate_plots(summary_path: Path, observations_path: Optional[Path] = None,
                   output_path: Optional[Path] = None, title_suffix: str = "") -> Path:
    """Public API to load data and create plots, returns saved path stem."""
    summary_path = Path(summary_path)
    observations_path = Path(observations_path) if observations_path is not None else None
    output_path = Path(output_path) if output_path is not None else _default_output_base(summary_path)

    summary, observations_df = load_summary_and_data(summary_path, observations_path)
    return make_plots(summary, observations_df, output_path, title_suffix=title_suffix)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize hierarchical compute reduction outputs."
    )
    parser.add_argument("--summary-json", type=str, required=True,
                        help="Path to hierarchical_median_summary*.json file.")
    parser.add_argument("--observations-csv", type=str, default=None,
                        help="Path to hierarchical_median_observations*.csv (optional).")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Output path stem for figures (default: next to summary).")
    parser.add_argument("--title-suffix", type=str, default="",
                        help="Human-readable suffix appended to plot titles.")
    parser.add_argument("--exclude-distilled", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--include-low-confidence", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--use-website-data", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    title_suffix = args.title_suffix
    if not title_suffix and (args.exclude_distilled or args.include_low_confidence or args.use_website_data):
        title_suffix = generate_title_suffix(
            exclude_distilled=args.exclude_distilled,
            include_low_confidence=args.include_low_confidence,
            use_website_data=args.use_website_data,
        )

    saved_path = generate_plots(
        summary_path=Path(args.summary_json),
        observations_path=Path(args.observations_csv) if args.observations_csv else None,
        output_path=Path(args.output_path) if args.output_path else None,
        title_suffix=title_suffix,
    )
    print(f"Hierarchical plots saved to: {saved_path}.[png/svg]")


if __name__ == "__main__":
    main()
