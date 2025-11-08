#!/usr/bin/env python3
"""Collect and summarize algorithmic progress results from all methods."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


def collect_buckets_results(output_dir: Path, suffix: str) -> Optional[Dict]:
    """Extract median compute reduction from buckets method results.

    Args:
        output_dir: Output directory containing results
        suffix: Filename suffix for this configuration

    Returns:
        Dict with median_ooms_per_year and CI, or None if not found
    """
    results_file = output_dir / f"compute_reduction_results{suffix}.csv"
    if not results_file.exists():
        return None

    try:
        df = pd.read_csv(results_file)
        if len(df) == 0:
            return None

        # Calculate median across all buckets (negating slope to get compute reduction)
        median_slope = float(-df["slope_oom_per_year"].median())
        # For CI, take median of the CIs (also negating and swapping)
        ci_lower = float(-df["slope_ci_upper"].median())
        ci_upper = float(-df["slope_ci_lower"].median())

        return {
            "median_ooms_per_year": median_slope,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }
    except Exception as e:
        print(f"Warning: Could not extract buckets results from {results_file}: {e}")
        return None


def collect_hierarchical_results(output_dir: Path, suffix: str) -> Optional[Dict]:
    """Extract hierarchical median estimate from results.

    Args:
        output_dir: Output directory containing results
        suffix: Filename suffix for this configuration

    Returns:
        Dict with median_ooms_per_year and CI, or None if not found
    """
    summary_file = output_dir / f"hierarchical_median_summary{suffix}.json"
    if not summary_file.exists():
        return None

    try:
        with open(summary_file) as f:
            summary = json.load(f)
        posterior_mu = summary.get("posterior_mu_oom_per_year", {})

        return {
            "median_ooms_per_year": posterior_mu.get("median"),
            "ci_lower": posterior_mu.get("ci_lower"),
            "ci_upper": posterior_mu.get("ci_upper")
        }
    except Exception as e:
        print(f"Warning: Could not extract hierarchical results from {summary_file}: {e}")
        return None


def collect_linear_model_results(output_dir: Path, suffix: str) -> Optional[Dict]:
    """Extract compute-year tradeoff from linear model results.

    Note: Linear model doesn't save results to a file, so we need to
    add that capability first. For now, this returns None.

    Args:
        output_dir: Output directory containing results
        suffix: Filename suffix for this configuration

    Returns:
        Dict with median_ooms_per_year and CI, or None if not found
    """
    # TODO: Modify linear model to save summary statistics to JSON
    # For now, we'll return None
    summary_file = output_dir / f"linear_model_summary{suffix}.json"
    if not summary_file.exists():
        return None

    try:
        with open(summary_file) as f:
            summary = json.load(f)
        tradeoff = summary.get("compute_year_tradeoff", {})

        return {
            "median_ooms_per_year": tradeoff.get("median"),
            "ci_lower": tradeoff.get("ci_lower"),
            "ci_upper": tradeoff.get("ci_upper")
        }
    except Exception as e:
        print(f"Warning: Could not extract linear model results from {summary_file}: {e}")
        return None


def format_result(value: Optional[float], ci_lower: Optional[float],
                  ci_upper: Optional[float], decimals: int = 2) -> str:
    """Format a result with confidence interval.

    Args:
        value: Point estimate
        ci_lower: Lower bound of 95% CI
        ci_upper: Upper bound of 95% CI
        decimals: Number of decimal places

    Returns:
        Formatted string like "1.23 [0.89, 1.67]"
    """
    if value is None or np.isnan(value):
        return "N/A"

    result = f"{value:.{decimals}f}"

    if ci_lower is not None and ci_upper is not None and \
       not np.isnan(ci_lower) and not np.isnan(ci_upper):
        result += f" [{ci_lower:.{decimals}f}, {ci_upper:.{decimals}f}]"

    return result


def print_summary_table(results: List[Dict]):
    """Print a formatted summary table of all results.

    Args:
        results: List of result dictionaries with keys:
                 - method: Method name
                 - config: Configuration name
                 - median_ooms_per_year: Point estimate
                 - ci_lower: Lower CI bound
                 - ci_upper: Upper CI bound
    """
    if not results:
        print("\nNo results to display.")
        return

    print("\n" + "=" * 100)
    print("ALGORITHMIC PROGRESS SUMMARY: MEDIAN OOMS/YEAR OF COMPUTE REDUCTION")
    print("=" * 100)
    print("\nInterpretation: How many orders of magnitude less compute is needed per year")
    print("to achieve the same capability level due to algorithmic improvements.")
    print("Example: 1.5 OOMs/year means 10^1.5 ≈ 31× compute reduction per year.\n")

    # Group by configuration
    configs = sorted(set(r["config"] for r in results))
    methods = sorted(set(r["method"] for r in results))

    # Create table header
    header = f"{'Configuration':<50} | " + " | ".join(f"{m:^30}" for m in methods)
    print(header)
    print("-" * len(header))

    # Print each configuration
    for config in configs:
        row = f"{config:<50} |"
        for method in methods:
            # Find result for this config and method
            result = next((r for r in results
                          if r["config"] == config and r["method"] == method), None)
            if result:
                formatted = format_result(
                    result.get("median_ooms_per_year"),
                    result.get("ci_lower"),
                    result.get("ci_upper"),
                    decimals=2
                )
                row += f" {formatted:^30} |"
            else:
                row += f" {'N/A':^30} |"
        print(row)

    print("=" * 100)
    print("\nNotes:")
    print("  - Values shown as: median [95% CI lower, 95% CI upper]")
    print("  - Buckets: Median across ECI buckets of compute reduction slopes")
    print("  - Hierarchical Median: Bayesian meta-analysis pooling across bucket sizes")
    print("  - Linear Model: Compute-year tradeoff from multivariate regression")
    print("=" * 100)


def infer_suffix_from_dir(dir_name: str) -> str:
    """Infer the file suffix from directory name.

    Args:
        dir_name: Directory name like 'website_no_distilled_all_models'

    Returns:
        File suffix like '_no_distilled_website' or empty string
    """
    # The suffix order is: filtering + data_source + frontier + date
    # E.g.: _no_distilled_website, _no_med_high_distilled_frontier_only, _frontier_only
    parts = []

    # Model filtering (comes first in suffix)
    # Note: directory has "_all" suffix but files don't
    if "no_distilled_all" in dir_name or ("no_distilled" in dir_name and "no_med_high" not in dir_name):
        parts.append("no_distilled")
    elif "no_med_high_distilled" in dir_name:
        parts.append("no_med_high_distilled")

    # Frontier only (linear model)
    if "frontier_only" in dir_name:
        parts.append("frontier_only")

    # Data source (comes after filtering)
    if "website" in dir_name:
        parts.append("website")

    # Date filtering
    if "from_" in dir_name:
        # Extract the date part
        import re
        match = re.search(r'from_(\d{4}-\d{2}-\d{2})', dir_name)
        if match:
            parts.append(f"from_{match.group(1)}")

    return ("_" + "_".join(parts)) if parts else ""


def infer_config_name_from_dir(dir_name: str) -> str:
    """Infer a human-readable configuration name from directory name.

    Args:
        dir_name: Directory name like 'internal_with_distilled_all_models'

    Returns:
        Human-readable name like 'Internal data (all models)'
    """
    name_parts = []

    # Data source
    if "website" in dir_name:
        name_parts.append("Website data")
    else:
        name_parts.append("Internal data")

    # Model filtering - check both frontier and distilled flags
    filter_parts = []

    if "frontier_only" in dir_name:
        filter_parts.append("frontier only")

    if "no_distilled_all" in dir_name or ("no_distilled" in dir_name and "no_med_high" not in dir_name):
        filter_parts.append("excluding all distilled")
    elif "no_med_high_distilled" in dir_name:
        filter_parts.append("excluding med/high distilled")

    if not filter_parts:
        filter_parts.append("all models")

    filter_desc = ", ".join(filter_parts)
    name_parts.append(f"({filter_desc})")

    # Date filtering
    if "from_" in dir_name:
        import re
        match = re.search(r'from_(\d{4}-\d{2}-\d{2})', dir_name)
        if match:
            name_parts.append(f"from {match.group(1)}")

    return " ".join(name_parts)


def collect_all_results() -> List[Dict]:
    """Scan output directories and collect all available results.

    Returns:
        List of result dictionaries
    """
    # __file__ is: .../benchmark-stitching-clean/analysis/algorithmic_progress_methods/result_collector.py
    # So parent.parent.parent gets us to benchmark-stitching-clean
    project_root = Path(__file__).parent.parent.parent
    base_output_dir = project_root / "outputs" / "algorithmic_progress_methods"

    results = []

    # Scan buckets directory
    buckets_dir = base_output_dir / "buckets"
    if buckets_dir.exists():
        for subdir in buckets_dir.iterdir():
            if not subdir.is_dir():
                continue

            config_name = infer_config_name_from_dir(subdir.name)
            suffix = infer_suffix_from_dir(subdir.name)

            # Try to collect buckets results
            buckets_result = collect_buckets_results(subdir, suffix)
            if buckets_result:
                results.append({
                    "method": "Buckets",
                    "config": config_name,
                    **buckets_result
                })

            # Try to collect hierarchical results
            hierarchical_result = collect_hierarchical_results(subdir, suffix)
            if hierarchical_result:
                results.append({
                    "method": "Hierarchical Median",
                    "config": config_name,
                    **hierarchical_result
                })

    # Scan linear model directory
    linear_dir = base_output_dir / "linear_model"
    if linear_dir.exists():
        for subdir in linear_dir.iterdir():
            if not subdir.is_dir():
                continue

            config_name = infer_config_name_from_dir(subdir.name)
            suffix = infer_suffix_from_dir(subdir.name)

            # Try to collect linear model results
            linear_result = collect_linear_model_results(subdir, suffix)
            if linear_result:
                results.append({
                    "method": "Linear Model",
                    "config": config_name,
                    **linear_result
                })

    return results


if __name__ == "__main__":
    results = collect_all_results()
    print_summary_table(results)
