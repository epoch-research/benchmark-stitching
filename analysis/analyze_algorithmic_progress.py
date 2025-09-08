#!/usr/bin/env python3
"""
Algorithmic progress analysis - converts algorithmic_progress.ipynb to a Python script

This script analyzes algorithmic progress by examining:
1. Compute requirements reduction over time for achieving the same capability
2. Capability improvements at fixed compute budgets over time
3. Statistical relationships between compute, capability, and time

Usage: python analyze_algorithmic_progress_1.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from data_loader import scores_df
from fit import fit_statistical_model
import statsmodels.api as sm
from scipy.stats import linregress, gmean
from typing import Union, Optional


def setup_output_directory():
    """Setup output directory for results"""
    output_dir = Path("outputs/algorithmic_progress")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_and_prepare_data():
    """Load and prepare the data for analysis"""
    print("Loading and preparing data...")
    
    # Fit the statistical model
    anchor_mode = "benchmark"
    anchor_benchmark = "Winogrande"
    anchor_difficulty = 0
    anchor_slope = 1
    anchor_model1 = "claude-2.0"
    anchor_model1_capability = 1.177630
    anchor_model2 = "claude-3-opus-20240229"
    anchor_model2_capability = 1.311554

    df1, df_cm1, df_db1 = fit_statistical_model(
        scores_df,
        anchor_mode=anchor_mode,
        anchor_benchmark=anchor_benchmark,
        anchor_difficulty=anchor_difficulty,
        anchor_slope=anchor_slope,
        anchor_model1=anchor_model1,
        anchor_model1_capability=anchor_model1_capability,
        anchor_model2=anchor_model2,
        anchor_model2_capability=anchor_model2_capability
    )

    df_cm1['date_obj'] = pd.to_datetime(df_cm1['date'])
    df_cap = df_cm1.copy(deep=True)

    # Load compute data
    try:
        pcd_dataset = pd.read_csv("data/all_ai_models_20250908.csv")[["Model", "Training compute (FLOP)", "Parameters", "Training dataset size (datapoints)"]]
        columns = {"Training compute (FLOP)": "compute", "Parameters": "parameters", "Training dataset size (datapoints)": "data"}
        pcd_dataset = pcd_dataset.rename(columns=columns)
        
        df_cap = df_cap.merge(pcd_dataset, on="Model")
        print(f"Successfully loaded compute data for {len(df_cap)} models")
    except Exception as e:
        print(f"Warning: Could not load compute data: {e}")
        return None, None, None
    
    return df_cap, df_cm1, df_db1


def later_lower_compute(
    row: pd.Series,
    df_level: pd.DataFrame,
    cap_tol_below: Union[int, float] = 0.1,
    cap_tol_above: Union[int, float] = 0.1,
) -> pd.DataFrame:
    """
    Return all models that are:
    • released after row["date"]
    • use ≤ the same compute
    • have capability within ±cap_tol of the current row

    Parameters
    ----------
    cap_tol_below : float
        Tolerance below the anchor capability
    cap_tol_above : float  
        Tolerance above the anchor capability

    Returns
    -------
    pd.DataFrame
        Qualifying later models, sorted by date (may be empty)
    """
    later_mask   = df_level["date_obj"].gt(row["date_obj"])
    compute_mask = df_level["compute"].le(row["compute"])

    lower_bound  = row["estimated_capability"] - cap_tol_below
    upper_bound  = row["estimated_capability"] + cap_tol_above
    cap_mask     = df_level["estimated_capability"].between(lower_bound, upper_bound, inclusive="both")

    later_matches = df_level.loc[later_mask & compute_mask & cap_mask]
    out = pd.concat([row.to_frame().T, later_matches]).sort_values("date_obj")

    return out


def compute_regression_stats(
    df_slice: pd.DataFrame,
    *,
    date_col: str = "date",
    compute_col: str = "compute",
) -> dict:
    """
    Fit log10(compute) ~ time (ordinary least squares).

    Returns
    -------
    dict with keys
        slope_log10              – slope in log10 units per year (should be < 0)
        factor_per_year          – 10**slope (e.g. 0.75 ⇒ 25% drop/yr)
        pct_reduction_per_year   – (1-factor)*100
        intercept_log10, r_value, p_value, stderr
    """
    df = df_slice.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # numeric X axis = fractional years since first point
    years = (
        (df[date_col] - df[date_col].iloc[0])
        .dt.total_seconds()
        / (365.25 * 24 * 3600)
    )

    # log-10 transform of compute
    y = np.log10(df[compute_col].astype(float))

    slope, intercept, r, p, se = linregress(years, y)

    factor_per_year        = 10 ** slope          # < 1 if slope negative
    pct_reduction_per_year = (1 - factor_per_year) * 100

    return {
        "slope_log10": slope,
        "intercept_log10": intercept,
        "r_value": r,
        "p_value": p,
        "stderr": se,
        "factor_per_year": 1/factor_per_year,
        "pct_reduction_per_year": pct_reduction_per_year,
    }


def plot_compute_trend(
    df_slice: pd.DataFrame,
    *,
    date_col: str = "date",
    compute_col: str = "compute",
    title: str | None = None,
    ax: plt.Axes | None = None,
    output_path: Path = None
):
    """
    Scatter of compute vs date (log-y) with best-fit line overlaid.
    """
    stats = compute_regression_stats(df_slice, date_col=date_col, compute_col=compute_col)

    df = df_slice.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # X for plotting
    years = (
        (df[date_col] - df[date_col].iloc[0])
        .dt.total_seconds()
        / (365.25 * 24 * 3600)
    )

    # Fitted line in *linear* compute space
    y_fit_log10 = stats["intercept_log10"] + stats["slope_log10"] * years
    y_fit       = 10 ** y_fit_log10

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(df[date_col], df[compute_col], label="models", alpha=0.7)
    ax.plot(df[date_col], y_fit, label="OLS fit", linewidth=2, color='red')

    ax.set_yscale("log")
    ax.set_xlabel("Date")
    ax.set_ylabel("Training compute (FLOP)")
    ax.set_title(
        title
        or "Compute trend (≈ constant capability)\n"
           f"≈ {stats['pct_reduction_per_year']:.1f}% less compute per year"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        # plt.show(block=False)
    
    return stats


def analyze_compute_efficiency(df: pd.DataFrame, output_dir: Path):
    """Analyze compute requirements reduction over time"""
    print("Analyzing compute efficiency improvements...")
    
    n_min = 10
    cap_tol_below = 0.05
    cap_tol_above = 99
    date_col = "date_obj"
    compute_col = "compute"

    results = []

    for idx, anchor in df.iterrows():
        window = later_lower_compute(anchor, df,
                                      cap_tol_below=cap_tol_below,
                                      cap_tol_above=cap_tol_above)

        if len(window) < n_min:
            continue

        stats = compute_regression_stats(
            window, date_col=date_col, compute_col=compute_col
        )

        results.append(
            {
                "anchor_idx": idx,
                "anchor_date": anchor[date_col],
                "anchor_model": anchor["model"],
                "n_points": len(window),
                **stats,
            }
        )

    compute_reduction_df = pd.DataFrame(results)
    
    # Save detailed results
    compute_reduction_df.to_csv(output_dir / "compute_reduction_analysis.csv", index=False)
    
    if len(compute_reduction_df) > 0:
        # Calculate geometric mean of improvement factors
        geometric_mean_improvement = gmean(compute_reduction_df["factor_per_year"])
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by factor for better visualization
        df_sorted = compute_reduction_df.sort_values('factor_per_year')
        
        bars = ax.barh(range(len(df_sorted)), df_sorted['factor_per_year'])
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['anchor_model'], fontsize=8)
        ax.set_xlabel('Compute Efficiency Improvement Factor (per year)')
        ax.set_title('Annual Compute Efficiency Improvements by Anchor Model')
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='No improvement')
        ax.axvline(x=geometric_mean_improvement, color='green', linestyle='--', alpha=0.7, 
                  label=f'Geometric mean: {geometric_mean_improvement:.2f}×')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_dir / "compute_efficiency_improvements.png", dpi=300, bbox_inches='tight')
        # plt.show(block=False)
        plt.close()
        
        print(f"Geometric mean compute efficiency improvement: {geometric_mean_improvement:.2f}× per year")
        
        return compute_reduction_df, geometric_mean_improvement
    
    return None, None


def later_higher_capability(
    df_level: pd.DataFrame,
    *,
    compute_target: Union[int, float],
    date_from: Union[str, pd.Timestamp],
    compute_tol_below: Union[int, float] = 1.0,
    compute_tol_above: Union[int, float] = 1.0,
    compare_row: Optional[pd.Series] = None,
    cap_floor: Optional[Union[int, float]] = None,
) -> pd.DataFrame:
    """
    Return all models that:
    • are released after date_from
    • use ≈ the compute_target (within tolerance)
    • beat either compare_row capability or cap_floor
    """
    date_from = pd.to_datetime(date_from)

    # Use date_obj column which is already datetime
    later_mask = df_level["date_obj"].gt(date_from)

    lower_compute = compute_target / compute_tol_below
    upper_compute = compute_target * compute_tol_above
    compute_mask = df_level["compute"].between(lower_compute, upper_compute, inclusive="both")

    if compare_row is not None:
        cap_threshold = compare_row["estimated_capability"]
    elif cap_floor is not None:
        cap_threshold = cap_floor
    else:
        raise ValueError("Supply either compare_row or cap_floor")

    cap_mask = df_level["estimated_capability"].gt(cap_threshold)

    later_better = df_level.loc[later_mask & compute_mask & cap_mask]
    return later_better.sort_values("date_obj")


def capability_regression_stats(
    df_slice: pd.DataFrame,
    *,
    date_col: str = "date",
    cap_col: str = "estimated_capability",
) -> dict:
    """
    Fit capability ~ time (ordinary least squares).
    """
    df = df_slice.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    years = (
        (df[date_col] - df[date_col].iloc[0])
        .dt.total_seconds()
        / (365.25 * 24 * 3600)
    )

    y = df[cap_col].astype(float)

    slope, intercept, r, p, se = linregress(years, y)

    return {
        "slope_per_year": slope,
        "intercept": intercept,
        "r_value": r,
        "p_value": p,
        "stderr": se,
    }


def plot_capability_trend(
    df_slice: pd.DataFrame,
    *,
    date_col: str = "date",
    cap_col: str = "estimated_capability",
    title: str | None = None,
    ax: plt.Axes | None = None,
    output_path: Path = None
):
    """
    Scatter of capability vs date (linear y) with best-fit line.
    """
    stats = capability_regression_stats(df_slice, date_col=date_col, cap_col=cap_col)

    df = df_slice.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    years = (
        (df[date_col] - df[date_col].iloc[0])
        .dt.total_seconds()
        / (365.25 * 24 * 3600)
    )
    y_fit = stats["intercept"] + stats["slope_per_year"] * years

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(df[date_col], df[cap_col], label="models", alpha=0.7)
    ax.plot(df[date_col], y_fit, label="OLS fit", linewidth=2, color='red')

    ax.set_xlabel("Date")
    ax.set_ylabel("Estimated capability")
    ax.set_title(
        title
        or "Capability trend (≈ constant compute)\n"
           f"≈ {stats['slope_per_year']:.3f} capability units gained per year"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        # plt.show(block=False)
    
    return stats


def analyze_capability_improvements(df: pd.DataFrame, output_dir: Path):
    """Analyze capability improvements at fixed compute"""
    print("Analyzing capability improvements at fixed compute...")
    
    n_min = 5
    compute_tol_below = 1e5
    compute_tol_above = 1.05

    results = []

    for idx, anchor in df.iterrows():
        window = later_higher_capability(
            df,
            compute_target=anchor["compute"],
            date_from=anchor["date_obj"],
            compare_row=anchor,
            compute_tol_below=compute_tol_below,
            compute_tol_above=compute_tol_above,
        )

        if len(window) < n_min:
            continue

        stats = capability_regression_stats(window)

        results.append(
            {
                "anchor_idx": idx,
                "anchor_model": anchor["model"],
                "anchor_date": anchor["date"],
                "n_points": len(window),
                **stats,
            }
        )

    summary_df = pd.DataFrame(results)
    
    # Save detailed results
    summary_df.to_csv(output_dir / "capability_improvements_analysis.csv", index=False)
    
    if len(summary_df) > 0:
        # Calculate statistics
        positive_slopes = summary_df[summary_df["slope_per_year"] > 0]["slope_per_year"]
        if len(positive_slopes) > 0:
            geo_mean = gmean(positive_slopes)
            mean_slope = np.mean(summary_df["slope_per_year"])
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Sort by slope for better visualization
            df_sorted = summary_df.sort_values('slope_per_year')
            
            bars = ax.barh(range(len(df_sorted)), df_sorted['slope_per_year'])
            ax.set_yticks(range(len(df_sorted)))
            ax.set_yticklabels(df_sorted['anchor_model'], fontsize=8)
            ax.set_xlabel('Annual Capability Improvement (capability units per year)')
            ax.set_title('Annual Capability Improvements at Fixed Compute by Anchor Model')
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='No improvement')
            ax.axvline(x=mean_slope, color='green', linestyle='--', alpha=0.7,
                      label=f'Mean: {mean_slope:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(output_dir / "capability_improvements_fixed_compute.png", dpi=300, bbox_inches='tight')
            # plt.show()
            plt.close()
            
            print(f"Mean capability improvement: {mean_slope:.3f} units per year")
            if len(positive_slopes) > 0:
                print(f"Geometric mean of positive improvements: {geo_mean:.3f} units per year")
            
            return summary_df, mean_slope
    
    return None, None


def create_overview_plots(df: pd.DataFrame, output_dir: Path):
    """Create overview visualization plots"""
    print("Creating overview plots...")
    
    # Capability over time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Capability over time
    ax1.scatter(df["date_obj"], df["estimated_capability"], alpha=0.7)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Estimated capability")
    ax1.set_title("AI Model Capabilities Over Time")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Capability vs compute
    ax2.scatter(df["compute"], df["estimated_capability"], alpha=0.7)
    ax2.set_xscale("log")
    ax2.set_xlabel("Training compute (FLOP)")
    ax2.set_ylabel("Estimated capability")
    ax2.set_title("Model Capability vs Training Compute")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "overview_plots.png", dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


def save_summary_results(compute_results, capability_results, output_dir: Path):
    """Save summary of all results"""
    summary_text = []
    summary_text.append("="*60)
    summary_text.append("ALGORITHMIC PROGRESS ANALYSIS SUMMARY")
    summary_text.append("="*60)
    summary_text.append("")
    
    if compute_results[0] is not None and compute_results[1] is not None:
        summary_text.append("COMPUTE EFFICIENCY IMPROVEMENTS:")
        summary_text.append(f"  Geometric mean improvement factor: {compute_results[1]:.2f}× per year")
        summary_text.append(f"  Number of anchor models analyzed: {len(compute_results[0])}")
        summary_text.append("")
    
    if capability_results[0] is not None and capability_results[1] is not None:
        summary_text.append("CAPABILITY IMPROVEMENTS AT FIXED COMPUTE:")
        summary_text.append(f"  Mean capability improvement: {capability_results[1]:.3f} units per year")
        summary_text.append(f"  Number of anchor models analyzed: {len(capability_results[0])}")
        summary_text.append("")
    
    summary_text.append("Files generated:")
    summary_text.append("  - compute_reduction_analysis.csv")
    summary_text.append("  - capability_improvements_analysis.csv")
    summary_text.append("  - compute_efficiency_improvements.png")
    summary_text.append("  - capability_improvements_fixed_compute.png")
    summary_text.append("  - overview_plots.png")
    summary_text.append("  - algorithmic_progress_summary.txt")
    
    with open(output_dir / "algorithmic_progress_summary.txt", "w") as f:
        f.write("\n".join(summary_text))
    
    print("\n".join(summary_text))


def main():
    """Main analysis function"""
    print("Starting algorithmic progress analysis...")
    
    # Setup
    output_dir = setup_output_directory()
    
    # Load and prepare data
    df_cap, df_cm1, df_db1 = load_and_prepare_data()
    
    if df_cap is None:
        print("Could not load required data. Exiting.")
        return
    
    print(f"Loaded data for {len(df_cap)} models with compute information")
    
    # Create overview plots
    create_overview_plots(df_cap, output_dir)
    
    # Analyze compute efficiency improvements
    compute_results = analyze_compute_efficiency(df_cap, output_dir)
    
    # Analyze capability improvements at fixed compute
    capability_results = analyze_capability_improvements(df_cap, output_dir)
    
    # Specific analysis for index 62 (gpt-4o-2024-05-13) as requested
    try:
        print(f"\nGenerating specific plots for index 62...")
        if len(df_cap) > 62:
            baseline = df_cap.iloc[62]
            print(f"Index 62 model: {baseline['model']}")
            print(f"Date: {baseline['date']}")
            print(f"Capability: {baseline['estimated_capability']:.4f}")
            print(f"Compute: {baseline['compute']:.2e}")
            
            # 1. Compute reduction plot (same capability, less compute over time)
            print("Generating compute reduction plot...")
            window_compute = later_lower_compute(baseline, df_cap, cap_tol_below=0.05, cap_tol_above=99)
            if len(window_compute) >= 10:
                # Calculate stats first
                stats_compute = compute_regression_stats(window_compute, date_col="date_obj", compute_col="compute")
                
                # Create plot with stats in title
                plot_compute_trend(
                    window_compute,
                    date_col="date_obj",
                    title=f"Compute Reduction for {baseline['model']} (Index 62)\nSame/better capability, less compute over time\n{stats_compute['pct_reduction_per_year']:.1f}% annual compute reduction",
                    output_path=output_dir / "compute_reduction_index62.png"
                )
                # plt.show(block=False)
                plt.close()
                print(f"  - Found {len(window_compute)} models")
                print(f"  - Annual compute reduction: {stats_compute['pct_reduction_per_year']:.1f}%")
            else:
                print(f"  - Not enough models for compute trend ({len(window_compute)} < 10)")
            
            # 2. Capability increase plot (same compute, better capability over time)
            print("Generating capability increase plot...")
            window_capability = later_higher_capability(
                df_cap,
                compute_target=baseline["compute"],
                date_from=baseline["date_obj"],
                compare_row=baseline,
                compute_tol_below=1000,
                compute_tol_above=1.05
            )
            if len(window_capability) >= 5:
                # Calculate stats first
                stats_capability = capability_regression_stats(window_capability, date_col="date_obj", cap_col="estimated_capability")
                
                # Create plot with stats in title
                plot_capability_trend(
                    window_capability,
                    date_col="date_obj",
                    title=f"Capability Increase for {baseline['model']} (Index 62)\nSame/less compute, better capability over time\n{stats_capability['slope_per_year']:.3f} capability units gained per year",
                    output_path=output_dir / "capability_increase_index62.png"
                )
                # plt.show(block=False)
                plt.close()
                print(f"  - Found {len(window_capability)} models")
                print(f"  - Annual capability gain: {stats_capability['slope_per_year']:.3f} units")
            else:
                print(f"  - Not enough models for capability trend ({len(window_capability)} < 5)")
        else:
            print(f"Not enough models to access index 62 (only {len(df_cap)} models)")
                
    except Exception as e:
        print(f"Could not create index 62 specific plots: {e}")
    
    # Save summary
    save_summary_results(compute_results, capability_results, output_dir)
    
    # Keep all plot windows open
    # plt.show()
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()