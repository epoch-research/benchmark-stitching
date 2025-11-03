#!/usr/bin/env python3
"""
Buckets method for analyzing algorithmic progress.

This script implements the buckets approach described in BUCKET_NOTES.md:
1. For compute reduction: filter models to narrow ECI buckets, identify SOTA in compute
   efficiency at release, and measure compute reduction over time.
2. For capability gains: filter models to narrow compute buckets, identify SOTA at release,
   and measure capability improvements over time.

Outputs are saved to outputs/algorithmic_progress_methods/buckets/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy import stats
import pickle


def load_model_capabilities_and_compute(use_website_data=False, exclude_distilled=False,
                                       include_low_confidence=False):
    """Load ECI scores and merge with compute data

    Args:
        use_website_data: If True, load from data/website/epoch_capabilities_index.csv
        exclude_distilled: If True, exclude distilled models
        include_low_confidence: If True, also exclude low-confidence distilled models

    Returns:
        DataFrame with columns: model, date, date_obj, estimated_capability, compute
    """
    if use_website_data:
        print("Loading data from data/website/epoch_capabilities_index.csv...")
        eci_df = pd.read_csv("data/website/epoch_capabilities_index.csv")

        # Rename columns to match expected format
        column_mapping = {
            'Model version': 'model',
            'ECI Score': 'estimated_capability',
            'Release date': 'date',
            'Training compute (FLOP)': 'compute'
        }
        eci_df = eci_df.rename(columns=column_mapping)
        eci_df['date_obj'] = pd.to_datetime(eci_df['date'])
        eci_df['Model'] = eci_df['model']

        print(f"Loaded {len(eci_df)} models from website data")
    else:
        print("Loading ECI scores from outputs/model_fit/model_capabilities.csv...")
        eci_df = pd.read_csv("outputs/model_fit/model_capabilities.csv")
        print(f"Loaded ECI scores for {len(eci_df)} models")

        # Ensure date_obj is datetime
        if 'date_obj' not in eci_df.columns:
            eci_df['date_obj'] = pd.to_datetime(eci_df['date'])
        else:
            eci_df['date_obj'] = pd.to_datetime(eci_df['date_obj'])

        # Load compute data and merge
        try:
            pcd_dataset = pd.read_csv("data/all_ai_models.csv")[
                ["Model", "Training compute (FLOP)"]
            ]
            pcd_dataset = pcd_dataset.rename(columns={"Training compute (FLOP)": "compute"})
            eci_df = eci_df.merge(pcd_dataset, on="Model", how="left")
            print(f"Merged compute data: {eci_df['compute'].notna().sum()} models have compute info")
        except Exception as e:
            print(f"Error: Could not load compute data: {e}")
            return None

    # Filter out distilled models if requested
    if exclude_distilled:
        print("\nFiltering out distilled models...")
        distilled_df = pd.read_csv("data/distilled_models.csv")

        if include_low_confidence:
            confidence_levels = ['high', 'medium', 'low']
            print("  Excluding: high, medium, AND low confidence distilled models")
        else:
            confidence_levels = ['high', 'medium']
            print("  Excluding: high and medium confidence distilled models only")

        distilled_models = distilled_df[
            (distilled_df['distilled'] == True) &
            (distilled_df['confidence'].isin(confidence_levels))
        ]['model'].tolist()

        before_count = len(eci_df)
        eci_df = eci_df[~eci_df['model'].isin(distilled_models)]
        after_count = len(eci_df)

        print(f"Excluded {before_count - after_count} distilled models "
              f"({100 * (before_count - after_count) / before_count:.1f}%)")
        print(f"Remaining models: {after_count}")

    # Filter to only models with complete data
    df = eci_df.dropna(subset=['date_obj', 'compute', 'estimated_capability']).copy()
    df['log_compute'] = np.log10(df['compute'])

    # Convert date to numeric (years since 2020)
    df['date_numeric'] = (df['date_obj'] - pd.Timestamp('2020-01-01')).dt.total_seconds() / (365.25 * 24 * 3600)

    print(f"Prepared {len(df)} models with complete data")
    return df


def find_sota_in_compute_efficiency_at_release(df_bucket):
    """
    Identify models that were SOTA in compute efficiency at their release date.

    A model is SOTA in compute efficiency if no earlier model in the bucket achieved
    the same (or higher) capability with less (or equal) compute.

    Args:
        df_bucket: DataFrame filtered to a specific ECI bucket

    Returns:
        DataFrame of SOTA models
    """
    df_sorted = df_bucket.sort_values('date_obj').copy()
    sota_indices = []

    for idx, row in df_sorted.iterrows():
        # Get all models released on or before this date in the bucket
        earlier_or_same = df_sorted[df_sorted['date_obj'] <= row['date_obj']]

        # Check if any earlier model had lower or equal compute
        # (since we're in same ECI bucket, capability is approximately equal)
        min_compute_so_far = earlier_or_same[earlier_or_same['date_obj'] < row['date_obj']]['compute'].min()

        # If this is the first model or has lower compute than all previous, it's SOTA
        if pd.isna(min_compute_so_far) or row['compute'] < min_compute_so_far:
            sota_indices.append(idx)

    return df_sorted.loc[sota_indices]


def find_sota_in_capability_at_release(df_bucket):
    """
    Identify models that were SOTA in capability at their release date.

    A model is SOTA in capability if no earlier model in the bucket achieved
    higher (or equal) capability.

    Args:
        df_bucket: DataFrame filtered to a specific compute bucket

    Returns:
        DataFrame of SOTA models
    """
    df_sorted = df_bucket.sort_values('date_obj').copy()
    sota_indices = []

    for idx, row in df_sorted.iterrows():
        # Get all models released on or before this date in the bucket
        earlier_or_same = df_sorted[df_sorted['date_obj'] <= row['date_obj']]

        # Check if any earlier model had higher or equal capability
        max_capability_so_far = earlier_or_same[earlier_or_same['date_obj'] < row['date_obj']]['estimated_capability'].max()

        # If this is the first model or has higher capability than all previous, it's SOTA
        if pd.isna(max_capability_so_far) or row['estimated_capability'] > max_capability_so_far:
            sota_indices.append(idx)

    return df_sorted.loc[sota_indices]


def bootstrap_bucket_analysis(df_sota, n_bootstrap=1000):
    """
    Perform bootstrap analysis on a single bucket to estimate uncertainty.

    Args:
        df_sota: DataFrame of SOTA models in the bucket
        n_bootstrap: Number of bootstrap iterations

    Returns:
        dict with bootstrap results including slopes and confidence intervals
    """
    X = df_sota['date_numeric'].values
    y = df_sota['log_compute'].values

    bootstrap_slopes = []
    bootstrap_intercepts = []

    np.random.seed(42)
    n_samples = len(df_sota)

    for i in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]

        if len(np.unique(X_boot)) > 1:  # Need at least 2 unique x values
            slope, intercept, _, _, _ = stats.linregress(X_boot, y_boot)
            bootstrap_slopes.append(slope)
            bootstrap_intercepts.append(intercept)

    bootstrap_slopes = np.array(bootstrap_slopes)
    bootstrap_intercepts = np.array(bootstrap_intercepts)

    return {
        'slopes': bootstrap_slopes,
        'intercepts': bootstrap_intercepts,
        'slope_mean': bootstrap_slopes.mean(),
        'slope_std': bootstrap_slopes.std(),
        'slope_ci': np.percentile(bootstrap_slopes, [2.5, 97.5]),
        'intercept_mean': bootstrap_intercepts.mean(),
        'intercept_ci': np.percentile(bootstrap_intercepts, [2.5, 97.5])
    }


def bootstrap_capability_bucket_analysis(df_sota, n_bootstrap=1000):
    """
    Perform bootstrap analysis for capability gains bucket.

    Args:
        df_sota: DataFrame of SOTA models in the bucket
        n_bootstrap: Number of bootstrap iterations

    Returns:
        dict with bootstrap results including slopes and confidence intervals
    """
    X = df_sota['date_numeric'].values
    y = df_sota['estimated_capability'].values

    bootstrap_slopes = []
    bootstrap_intercepts = []

    np.random.seed(42)
    n_samples = len(df_sota)

    for i in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]

        if len(np.unique(X_boot)) > 1:  # Need at least 2 unique x values
            slope, intercept, _, _, _ = stats.linregress(X_boot, y_boot)
            bootstrap_slopes.append(slope)
            bootstrap_intercepts.append(intercept)

    bootstrap_slopes = np.array(bootstrap_slopes)
    bootstrap_intercepts = np.array(bootstrap_intercepts)

    return {
        'slopes': bootstrap_slopes,
        'intercepts': bootstrap_intercepts,
        'slope_mean': bootstrap_slopes.mean(),
        'slope_std': bootstrap_slopes.std(),
        'slope_ci': np.percentile(bootstrap_slopes, [2.5, 97.5]),
        'intercept_mean': bootstrap_intercepts.mean(),
        'intercept_ci': np.percentile(bootstrap_intercepts, [2.5, 97.5])
    }


def analyze_compute_reduction(df, bucket_size_oom=0.3, min_models_per_bucket=3, n_bootstrap=1000):
    """
    Analyze compute reduction over time for fixed capability levels (ECI buckets).

    For each ECI bucket:
    1. Filter to models within the bucket
    2. Identify models that were SOTA in compute efficiency at release
    3. Fit linear model of log(compute) vs date
    4. Extract slope = OOMs of compute reduction per year

    Args:
        df: DataFrame with model data
        bucket_size_oom: Width of ECI buckets (in ECI units, not OOMs)
        min_models_per_bucket: Minimum number of SOTA models required for analysis
        n_bootstrap: Number of bootstrap iterations for uncertainty estimation

    Returns:
        results_df: DataFrame with bucket analysis results
        bucket_data: List of dicts with detailed bucket information
    """
    print("\n" + "="*70)
    print("COMPUTE REDUCTION ANALYSIS")
    print("Measuring how compute requirements decrease over time for fixed capabilities")
    print("="*70)

    # Create ECI buckets
    eci_min = df['estimated_capability'].min()
    eci_max = df['estimated_capability'].max()

    # Create bucket centers
    bucket_centers = np.arange(
        eci_min + bucket_size_oom/2,
        eci_max,
        bucket_size_oom
    )

    results = []
    bucket_data = []

    for bucket_center in bucket_centers:
        bucket_min = bucket_center - bucket_size_oom / 2
        bucket_max = bucket_center + bucket_size_oom / 2

        # Filter to models in this bucket
        df_bucket = df[
            (df['estimated_capability'] >= bucket_min) &
            (df['estimated_capability'] < bucket_max)
        ].copy()

        if len(df_bucket) < 2:
            continue

        # Find SOTA models in compute efficiency
        df_sota = find_sota_in_compute_efficiency_at_release(df_bucket)

        if len(df_sota) < min_models_per_bucket:
            continue

        # Fit linear model: log(compute) ~ date
        X = df_sota['date_numeric'].values.reshape(-1, 1)
        y = df_sota['log_compute'].values

        if len(X) < 2:
            continue

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            X.flatten(), y
        )

        # Bootstrap analysis for uncertainty
        bootstrap_results = bootstrap_bucket_analysis(df_sota, n_bootstrap=n_bootstrap)

        results.append({
            'bucket_center': bucket_center,
            'bucket_min': bucket_min,
            'bucket_max': bucket_max,
            'n_models_total': len(df_bucket),
            'n_models_sota': len(df_sota),
            'slope_oom_per_year': slope,  # Negative means compute is decreasing
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
            'slope_ci_lower': bootstrap_results['slope_ci'][0],
            'slope_ci_upper': bootstrap_results['slope_ci'][1]
        })

        bucket_data.append({
            'bucket_center': bucket_center,
            'df_bucket': df_bucket,
            'df_sota': df_sota,
            'bootstrap_results': bootstrap_results
        })

        print(f"\nECI Bucket [{bucket_min:.2f}, {bucket_max:.2f}], center={bucket_center:.2f}:")
        print(f"  Total models: {len(df_bucket)}, SOTA models: {len(df_sota)}")
        print(f"  Compute reduction: {-slope:.4f} OOMs/year (R²={r_value**2:.3f}, p={p_value:.3f})")
        if slope < 0:
            print(f"  Interpretation: {10**(-slope):.2f}× compute reduction per year")

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        # Summary statistics
        print("\n" + "="*70)
        print("SUMMARY: Compute Reduction Across All Buckets")
        print("="*70)
        mean_reduction = results_df['slope_oom_per_year'].mean()
        median_reduction = results_df['slope_oom_per_year'].median()
        std_reduction = results_df['slope_oom_per_year'].std()

        print(f"Mean slope: {mean_reduction:.4f} OOMs/year")
        print(f"Median slope: {median_reduction:.4f} OOMs/year")
        print(f"Std dev: {std_reduction:.4f} OOMs/year")
        print(f"\nMean compute reduction: {10**(-mean_reduction):.2f}× per year")
        print(f"Median compute reduction: {10**(-median_reduction):.2f}× per year")
        print(f"Range: {10**(-results_df['slope_oom_per_year'].max()):.2f}× to "
              f"{10**(-results_df['slope_oom_per_year'].min()):.2f}× per year")

    return results_df, bucket_data


def analyze_capability_gains(df, bucket_size_oom=0.3, min_models_per_bucket=3, n_bootstrap=1000):
    """
    Analyze capability gains over time for fixed compute levels (compute buckets).

    For each compute bucket:
    1. Filter to models within the bucket
    2. Identify models that were SOTA in capability at release
    3. Fit linear model of ECI vs date
    4. Extract slope = capability units gained per year

    Args:
        df: DataFrame with model data
        bucket_size_oom: Width of compute buckets in log10 scale
        min_models_per_bucket: Minimum number of SOTA models required for analysis
        n_bootstrap: Number of bootstrap iterations for uncertainty estimation

    Returns:
        results_df: DataFrame with bucket analysis results
        bucket_data: List of dicts with detailed bucket information
    """
    print("\n" + "="*70)
    print("CAPABILITY GAINS ANALYSIS")
    print("Measuring how capabilities improve over time for fixed compute budgets")
    print("="*70)

    # Create compute buckets (in log scale)
    log_compute_min = df['log_compute'].min()
    log_compute_max = df['log_compute'].max()

    # Create bucket centers
    bucket_centers = np.arange(
        log_compute_min + bucket_size_oom/2,
        log_compute_max,
        bucket_size_oom
    )

    results = []
    bucket_data = []

    for bucket_center in bucket_centers:
        bucket_min = bucket_center - bucket_size_oom / 2
        bucket_max = bucket_center + bucket_size_oom / 2

        # Filter to models in this bucket
        df_bucket = df[
            (df['log_compute'] >= bucket_min) &
            (df['log_compute'] < bucket_max)
        ].copy()

        if len(df_bucket) < 2:
            continue

        # Find SOTA models in capability
        df_sota = find_sota_in_capability_at_release(df_bucket)

        if len(df_sota) < min_models_per_bucket:
            continue

        # Fit linear model: ECI ~ date
        X = df_sota['date_numeric'].values.reshape(-1, 1)
        y = df_sota['estimated_capability'].values

        if len(X) < 2:
            continue

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            X.flatten(), y
        )

        # Bootstrap analysis for uncertainty
        bootstrap_results = bootstrap_capability_bucket_analysis(df_sota, n_bootstrap=n_bootstrap)

        compute_range = 10**bucket_min, 10**bucket_max

        results.append({
            'bucket_center_log': bucket_center,
            'bucket_center_compute': 10**bucket_center,
            'bucket_min_log': bucket_min,
            'bucket_max_log': bucket_max,
            'bucket_min_compute': compute_range[0],
            'bucket_max_compute': compute_range[1],
            'n_models_total': len(df_bucket),
            'n_models_sota': len(df_sota),
            'slope_eci_per_year': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
            'slope_ci_lower': bootstrap_results['slope_ci'][0],
            'slope_ci_upper': bootstrap_results['slope_ci'][1]
        })

        bucket_data.append({
            'bucket_center': bucket_center,
            'df_bucket': df_bucket,
            'df_sota': df_sota,
            'bootstrap_results': bootstrap_results
        })

        print(f"\nCompute Bucket [{bucket_min:.2f}, {bucket_max:.2f}] log10(FLOP), "
              f"center={bucket_center:.2f} ({10**bucket_center:.2e} FLOP):")
        print(f"  Total models: {len(df_bucket)}, SOTA models: {len(df_sota)}")
        print(f"  Capability gain: {slope:.4f} ECI units/year (R²={r_value**2:.3f}, p={p_value:.3f})")

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        # Summary statistics
        print("\n" + "="*70)
        print("SUMMARY: Capability Gains Across All Buckets")
        print("="*70)
        mean_gain = results_df['slope_eci_per_year'].mean()
        median_gain = results_df['slope_eci_per_year'].median()
        std_gain = results_df['slope_eci_per_year'].std()

        print(f"Mean slope: {mean_gain:.4f} ECI units/year")
        print(f"Median slope: {median_gain:.4f} ECI units/year")
        print(f"Std dev: {std_gain:.4f} ECI units/year")
        print(f"Range: {results_df['slope_eci_per_year'].min():.4f} to "
              f"{results_df['slope_eci_per_year'].max():.4f} ECI units/year")

    return results_df, bucket_data


def plot_compute_reduction_results(results_df, bucket_data, output_dir, suffix=""):
    """Create visualizations for compute reduction analysis"""

    if len(results_df) == 0:
        print("No results to plot for compute reduction")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Slope vs ECI bucket
    ax = axes[0, 0]
    ax.scatter(results_df['bucket_center'], results_df['slope_oom_per_year'],
              s=results_df['n_models_sota']*20, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.axhline(results_df['slope_oom_per_year'].mean(), color='blue',
              linestyle='--', linewidth=2, label=f'Mean: {results_df["slope_oom_per_year"].mean():.3f}')
    ax.set_xlabel('ECI Bucket Center', fontsize=12)
    ax.set_ylabel('Compute Reduction (OOMs/year)', fontsize=12)
    ax.set_title('Compute Reduction Rate vs Capability Level', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()

    # 2. Distribution of slopes
    ax = axes[0, 1]
    ax.hist(results_df['slope_oom_per_year'], bins=15, alpha=0.7,
           edgecolor='black', color='steelblue')
    ax.axvline(results_df['slope_oom_per_year'].mean(), color='red',
              linestyle='--', linewidth=2, label=f'Mean: {results_df["slope_oom_per_year"].mean():.3f}')
    ax.axvline(results_df['slope_oom_per_year'].median(), color='orange',
              linestyle='--', linewidth=2, label=f'Median: {results_df["slope_oom_per_year"].median():.3f}')
    ax.set_xlabel('Compute Reduction (OOMs/year)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Compute Reduction Rates', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Example bucket: show data and fit
    ax = axes[1, 0]
    if len(bucket_data) > 0:
        # Choose middle bucket
        mid_idx = len(bucket_data) // 2
        bucket = bucket_data[mid_idx]
        df_bucket = bucket['df_bucket']
        df_sota = bucket['df_sota']
        bucket_center = bucket['bucket_center']

        # Plot all models in bucket
        ax.scatter(df_bucket['date_obj'], df_bucket['log_compute'],
                  alpha=0.3, s=50, label='All models in bucket', color='gray')

        # Plot SOTA models
        ax.scatter(df_sota['date_obj'], df_sota['log_compute'],
                  alpha=0.8, s=100, label='SOTA models', color='red',
                  edgecolors='black', linewidth=1, zorder=3)

        # Plot linear fit
        if len(df_sota) >= 2:
            result = results_df[results_df['bucket_center'] == bucket_center].iloc[0]
            date_range = pd.date_range(df_sota['date_obj'].min(), df_sota['date_obj'].max(), periods=100)
            date_numeric = (date_range - pd.Timestamp('2020-01-01')).total_seconds() / (365.25 * 24 * 3600)
            log_compute_fit = result['slope_oom_per_year'] * date_numeric + result['intercept']
            ax.plot(date_range, log_compute_fit, 'b--', linewidth=2,
                   label=f'Linear fit: {result["slope_oom_per_year"]:.3f} OOMs/year')

        ax.set_xlabel('Release Date', fontsize=12)
        ax.set_ylabel('log₁₀(Compute)', fontsize=12)
        ax.set_title(f'Example: ECI Bucket {bucket_center:.2f}', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # 4. R² values
    ax = axes[1, 1]
    ax.scatter(results_df['bucket_center'], results_df['r_squared'],
              s=results_df['n_models_sota']*20, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.axhline(results_df['r_squared'].mean(), color='blue',
              linestyle='--', linewidth=2, label=f'Mean R²: {results_df["r_squared"].mean():.3f}')
    ax.set_xlabel('ECI Bucket Center', fontsize=12)
    ax.set_ylabel('R² (Goodness of Fit)', fontsize=12)
    ax.set_title('Model Fit Quality vs Capability Level', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    output_path = output_dir / f"compute_reduction_analysis{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCompute reduction plot saved to: {output_path}")
    plt.close()


def plot_capability_gains_results(results_df, bucket_data, output_dir, suffix=""):
    """Create visualizations for capability gains analysis"""

    if len(results_df) == 0:
        print("No results to plot for capability gains")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Slope vs compute bucket
    ax = axes[0, 0]
    ax.scatter(results_df['bucket_center_log'], results_df['slope_eci_per_year'],
              s=results_df['n_models_sota']*20, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.axhline(results_df['slope_eci_per_year'].mean(), color='blue',
              linestyle='--', linewidth=2, label=f'Mean: {results_df["slope_eci_per_year"].mean():.3f}')
    ax.set_xlabel('log₁₀(Compute) Bucket Center', fontsize=12)
    ax.set_ylabel('Capability Gain (ECI units/year)', fontsize=12)
    ax.set_title('Capability Gain Rate vs Compute Level', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()

    # 2. Distribution of slopes
    ax = axes[0, 1]
    ax.hist(results_df['slope_eci_per_year'], bins=15, alpha=0.7,
           edgecolor='black', color='forestgreen')
    ax.axvline(results_df['slope_eci_per_year'].mean(), color='red',
              linestyle='--', linewidth=2, label=f'Mean: {results_df["slope_eci_per_year"].mean():.3f}')
    ax.axvline(results_df['slope_eci_per_year'].median(), color='orange',
              linestyle='--', linewidth=2, label=f'Median: {results_df["slope_eci_per_year"].median():.3f}')
    ax.set_xlabel('Capability Gain (ECI units/year)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Capability Gain Rates', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Example bucket: show data and fit
    ax = axes[1, 0]
    if len(bucket_data) > 0:
        # Choose middle bucket
        mid_idx = len(bucket_data) // 2
        bucket = bucket_data[mid_idx]
        df_bucket = bucket['df_bucket']
        df_sota = bucket['df_sota']
        bucket_center = bucket['bucket_center']

        # Plot all models in bucket
        ax.scatter(df_bucket['date_obj'], df_bucket['estimated_capability'],
                  alpha=0.3, s=50, label='All models in bucket', color='gray')

        # Plot SOTA models
        ax.scatter(df_sota['date_obj'], df_sota['estimated_capability'],
                  alpha=0.8, s=100, label='SOTA models', color='blue',
                  edgecolors='black', linewidth=1, zorder=3)

        # Plot linear fit
        if len(df_sota) >= 2:
            result = results_df[results_df['bucket_center_log'] == bucket_center].iloc[0]
            date_range = pd.date_range(df_sota['date_obj'].min(), df_sota['date_obj'].max(), periods=100)
            date_numeric = (date_range - pd.Timestamp('2020-01-01')).total_seconds() / (365.25 * 24 * 3600)
            eci_fit = result['slope_eci_per_year'] * date_numeric + result['intercept']
            ax.plot(date_range, eci_fit, 'r--', linewidth=2,
                   label=f'Linear fit: {result["slope_eci_per_year"]:.3f} ECI/year')

        ax.set_xlabel('Release Date', fontsize=12)
        ax.set_ylabel('ECI (Capability)', fontsize=12)
        ax.set_title(f'Example: Compute Bucket {bucket_center:.2f} log₁₀(FLOP)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # 4. R² values
    ax = axes[1, 1]
    ax.scatter(results_df['bucket_center_log'], results_df['r_squared'],
              s=results_df['n_models_sota']*20, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.axhline(results_df['r_squared'].mean(), color='blue',
              linestyle='--', linewidth=2, label=f'Mean R²: {results_df["r_squared"].mean():.3f}')
    ax.set_xlabel('log₁₀(Compute) Bucket Center', fontsize=12)
    ax.set_ylabel('R² (Goodness of Fit)', fontsize=12)
    ax.set_title('Model Fit Quality vs Compute Level', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    output_path = output_dir / f"capability_gains_analysis{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCapability gains plot saved to: {output_path}")
    plt.close()


def plot_all_bucket_regressions(df, results_df, bucket_data, output_dir, suffix="", label_points=False):
    """
    Plot all bucket regression lines on a single plot, labeled with OOMs/year.
    This shows compute reduction trends at different capability levels.

    Args:
        df: DataFrame with model data
        results_df: Results DataFrame
        bucket_data: List of bucket information dicts
        output_dir: Output directory
        suffix: Filename suffix
        label_points: If True, label SOTA points with model name and ECI
    """
    if len(results_df) == 0 or len(bucket_data) == 0:
        print("No results to plot for all bucket regressions")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    # Get overall date range for plotting
    date_min = df['date_obj'].min()
    date_max = df['date_obj'].max()
    date_range = pd.date_range(date_min, date_max, periods=100)
    date_numeric = (date_range - pd.Timestamp('2020-01-01')).total_seconds() / (365.25 * 24 * 3600)

    # Use a colormap for different buckets
    colors = plt.cm.viridis(np.linspace(0, 1, len(bucket_data)))

    for i, (bucket_info, color) in enumerate(zip(bucket_data, colors)):
        bucket_center = bucket_info['bucket_center']
        df_sota = bucket_info['df_sota']
        bootstrap_results = bucket_info['bootstrap_results']

        # Get the corresponding result row
        result_row = results_df[results_df['bucket_center'] == bucket_center].iloc[0]
        slope = result_row['slope_oom_per_year']
        intercept = result_row['intercept']

        # Compute fitted line
        log_compute_fit = slope * date_numeric + intercept

        # Plot regression line
        ax.plot(date_range, log_compute_fit, color=color, linewidth=2.5, alpha=0.8,
               label=f'ECI={bucket_center:.2f}: {-slope:.3f} OOMs/yr ({10**(-slope):.1f}× reduction)')

        # Plot bootstrap uncertainty as shaded region
        slope_ci = bootstrap_results['slope_ci']
        intercept_mean = bootstrap_results['intercept_mean']

        log_compute_ci_lower = slope_ci[0] * date_numeric + intercept_mean
        log_compute_ci_upper = slope_ci[1] * date_numeric + intercept_mean

        ax.fill_between(date_range, log_compute_ci_lower, log_compute_ci_upper,
                       color=color, alpha=0.15)

        # Plot SOTA models for this bucket
        ax.scatter(df_sota['date_obj'], df_sota['log_compute'],
                  color=color, s=80, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=3)

        # Label points with model name and ECI if requested
        if label_points:
            for _, row in df_sota.iterrows():
                label = f"{row['model']}\nECI={row['estimated_capability']:.2f}"
                ax.annotate(label,
                           xy=(row['date_obj'], row['log_compute']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=7, alpha=0.8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3, edgecolor='none'))

    # Format plot
    ax.set_xlabel('Release Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('log₁₀(Training Compute)', fontsize=13, fontweight='bold')
    ax.set_title('Compute Reduction at Fixed Capability Levels\n' +
                'All ECI Buckets with Bootstrap Uncertainty (95% CI)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', fontsize=9, framealpha=0.95, ncol=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    output_path = output_dir / f"all_bucket_regressions_compute_reduction{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nAll bucket regressions plot saved to: {output_path}")
    plt.close()


def plot_all_capability_bucket_regressions(df, results_df, bucket_data, output_dir, suffix="", label_points=False):
    """
    Plot all capability gain bucket regression lines on a single plot.
    This shows capability improvement trends at different compute levels.

    Args:
        df: DataFrame with model data
        results_df: Results DataFrame
        bucket_data: List of bucket information dicts
        output_dir: Output directory
        suffix: Filename suffix
        label_points: If True, label SOTA points with model name and compute
    """
    if len(results_df) == 0 or len(bucket_data) == 0:
        print("No results to plot for all capability bucket regressions")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    # Get overall date range for plotting
    date_min = df['date_obj'].min()
    date_max = df['date_obj'].max()
    date_range = pd.date_range(date_min, date_max, periods=100)
    date_numeric = (date_range - pd.Timestamp('2020-01-01')).total_seconds() / (365.25 * 24 * 3600)

    # Use a colormap for different buckets
    colors = plt.cm.plasma(np.linspace(0, 1, len(bucket_data)))

    for i, (bucket_info, color) in enumerate(zip(bucket_data, colors)):
        bucket_center_log = bucket_info['bucket_center']
        df_sota = bucket_info['df_sota']
        bootstrap_results = bucket_info['bootstrap_results']

        # Get the corresponding result row
        result_row = results_df[results_df['bucket_center_log'] == bucket_center_log].iloc[0]
        slope = result_row['slope_eci_per_year']
        intercept = result_row['intercept']

        # Compute fitted line
        eci_fit = slope * date_numeric + intercept

        # Plot regression line
        ax.plot(date_range, eci_fit, color=color, linewidth=2.5, alpha=0.8,
               label=f'Compute={bucket_center_log:.1f} log₁₀(FLOP): {slope:.3f} ECI/yr')

        # Plot bootstrap uncertainty as shaded region
        slope_ci = bootstrap_results['slope_ci']
        intercept_mean = bootstrap_results['intercept_mean']

        eci_ci_lower = slope_ci[0] * date_numeric + intercept_mean
        eci_ci_upper = slope_ci[1] * date_numeric + intercept_mean

        ax.fill_between(date_range, eci_ci_lower, eci_ci_upper,
                       color=color, alpha=0.15)

        # Plot SOTA models for this bucket
        ax.scatter(df_sota['date_obj'], df_sota['estimated_capability'],
                  color=color, s=80, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=3)

        # Label points with model name and compute if requested
        if label_points:
            for _, row in df_sota.iterrows():
                label = f"{row['model']}\nCompute={row['log_compute']:.1f}"
                ax.annotate(label,
                           xy=(row['date_obj'], row['estimated_capability']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=7, alpha=0.8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3, edgecolor='none'))

    # Format plot
    ax.set_xlabel('Release Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('ECI (Capability)', fontsize=13, fontweight='bold')
    ax.set_title('Capability Gains at Fixed Compute Levels\n' +
                'All Compute Buckets with Bootstrap Uncertainty (95% CI)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', fontsize=9, framealpha=0.95, ncol=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    output_path = output_dir / f"all_bucket_regressions_capability_gains{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nAll capability bucket regressions plot saved to: {output_path}")
    plt.close()


def plot_bootstrap_distributions(results_df, bucket_data, output_dir, suffix=""):
    """
    Create diagnostic plots showing bootstrap distributions for all buckets.
    Similar to the linear model's bootstrap uncertainty diagnostics.
    """
    if len(results_df) == 0 or len(bucket_data) == 0:
        print("No results to plot for bootstrap distributions")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Distribution of slope estimates across buckets
    ax = axes[0, 0]
    slopes = results_df['slope_oom_per_year'].values
    slope_cis_lower = results_df['slope_ci_lower'].values
    slope_cis_upper = results_df['slope_ci_upper'].values

    x_pos = np.arange(len(slopes))
    ax.errorbar(x_pos, -slopes, yerr=[-(slope_cis_lower - slopes), -(slopes - slope_cis_upper)],
               fmt='o', markersize=8, capsize=5, capthick=2, linewidth=2, alpha=0.7)
    ax.axhline(-slopes.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {-slopes.mean():.3f} OOMs/yr')
    ax.set_xlabel('Bucket Index', fontsize=12)
    ax.set_ylabel('Compute Reduction (OOMs/year)', fontsize=12)
    ax.set_title('Compute Reduction with 95% Bootstrap CI\n(All ECI Buckets)',
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Combined bootstrap distribution from all buckets
    ax = axes[0, 1]
    all_bootstrap_slopes = []
    for bucket_info in bucket_data:
        all_bootstrap_slopes.extend(bucket_info['bootstrap_results']['slopes'])

    all_bootstrap_slopes = -np.array(all_bootstrap_slopes)  # Convert to positive (reduction)
    ax.hist(all_bootstrap_slopes, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    ax.axvline(all_bootstrap_slopes.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {all_bootstrap_slopes.mean():.3f} OOMs/yr')
    ax.axvline(np.median(all_bootstrap_slopes), color='orange', linestyle='--', linewidth=2,
              label=f'Median: {np.median(all_bootstrap_slopes):.3f} OOMs/yr')
    ci = np.percentile(all_bootstrap_slopes, [2.5, 97.5])
    ax.axvline(ci[0], color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax.axvline(ci[1], color='gray', linestyle=':', linewidth=2, alpha=0.7,
              label=f'95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]')
    ax.set_xlabel('Compute Reduction (OOMs/year)', fontsize=12)
    ax.set_ylabel('Frequency (All Bootstrap Samples)', fontsize=12)
    ax.set_title('Pooled Bootstrap Distribution\n(All Buckets Combined)',
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Compute reduction as multiplier
    ax = axes[1, 0]
    multipliers = 10 ** all_bootstrap_slopes
    ax.hist(multipliers, bins=50, alpha=0.7, edgecolor='black', color='forestgreen')
    mean_mult = multipliers.mean()
    median_mult = np.median(multipliers)
    ci_mult = 10 ** ci

    ax.axvline(mean_mult, color='red', linestyle='--', linewidth=2,
              label=f'Mean: {mean_mult:.1f}×')
    ax.axvline(median_mult, color='orange', linestyle='--', linewidth=2,
              label=f'Median: {median_mult:.1f}×')
    ax.axvline(ci_mult[0], color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax.axvline(ci_mult[1], color='gray', linestyle=':', linewidth=2, alpha=0.7,
              label=f'95% CI: [{ci_mult[0]:.1f}×, {ci_mult[1]:.1f}×]')
    ax.set_xlabel('Compute Reduction Multiplier (×/year)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Algorithmic Progress as Compute Multiplier\n(1 year = X× more compute)',
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    if ci_mult[1] / ci_mult[0] > 100:
        ax.set_xscale('log')

    # 4. Slope vs bucket center (with uncertainty)
    ax = axes[1, 1]
    bucket_centers = results_df['bucket_center'].values
    ax.scatter(bucket_centers, -slopes, s=100, alpha=0.7, edgecolors='black', linewidth=1)

    for i, (center, slope, ci_lower, ci_upper) in enumerate(zip(bucket_centers, -slopes,
                                                                 -slope_cis_lower, -slope_cis_upper)):
        ax.plot([center, center], [ci_upper, ci_lower], color='gray', alpha=0.5, linewidth=2)

    ax.axhline(-slopes.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {-slopes.mean():.3f} OOMs/yr')
    ax.set_xlabel('ECI Bucket Center', fontsize=12)
    ax.set_ylabel('Compute Reduction (OOMs/year)', fontsize=12)
    ax.set_title('Compute Reduction vs Capability Level\n(with 95% Bootstrap CI)',
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / f"bootstrap_distributions_compute_reduction{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nBootstrap distributions plot saved to: {output_path}")
    plt.close()


def save_results(compute_reduction_df, capability_gains_df, output_dir, suffix=""):
    """Save analysis results to CSV files"""

    if len(compute_reduction_df) > 0:
        output_path = output_dir / f"compute_reduction_results{suffix}.csv"
        compute_reduction_df.to_csv(output_path, index=False)
        print(f"\nCompute reduction results saved to: {output_path}")

    if len(capability_gains_df) > 0:
        output_path = output_dir / f"capability_gains_results{suffix}.csv"
        capability_gains_df.to_csv(output_path, index=False)
        print(f"Capability gains results saved to: {output_path}")


def plot_bucket_size_sensitivity(df, output_dir, suffix="",
                                 eci_bucket_sizes=None,
                                 compute_bucket_sizes=None,
                                 min_models_per_bucket=3,
                                 n_bucket_sizes=5):
    """
    Sweep over different bucket sizes to assess sensitivity.

    Bucket sizes are automatically computed as fractions of the data range if not provided.

    Args:
        df: DataFrame with model data
        output_dir: Output directory
        suffix: Suffix for filenames
        eci_bucket_sizes: List of ECI bucket sizes to try (if None, computed from data)
        compute_bucket_sizes: List of compute bucket sizes to try (if None, computed from data)
        min_models_per_bucket: Minimum SOTA models per bucket
        n_bucket_sizes: Number of bucket sizes to test (default: 5)
    """
    print("\n" + "="*70)
    print("BUCKET SIZE SENSITIVITY ANALYSIS")
    print("="*70)

    # Compute bucket sizes from data if not provided
    if eci_bucket_sizes is None:
        eci_range = df['estimated_capability'].max() - df['estimated_capability'].min()
        # Test bucket sizes from 10% to 50% of the range
        eci_bucket_sizes = np.linspace(0.1 * eci_range, 0.5 * eci_range, n_bucket_sizes)
        print(f"\nECI range: {df['estimated_capability'].min():.2f} to {df['estimated_capability'].max():.2f}")
        print(f"ECI total range: {eci_range:.2f}")
        print(f"Auto-computed ECI bucket sizes: {[f'{x:.2f}' for x in eci_bucket_sizes]}")

    if compute_bucket_sizes is None:
        log_compute_range = df['log_compute'].max() - df['log_compute'].min()
        # Test bucket sizes from 10% to 50% of the range
        compute_bucket_sizes = np.linspace(0.1 * log_compute_range, 0.5 * log_compute_range, n_bucket_sizes)
        print(f"\nlog₁₀(Compute) range: {df['log_compute'].min():.2f} to {df['log_compute'].max():.2f}")
        print(f"log₁₀(Compute) total range: {log_compute_range:.2f}")
        print(f"Auto-computed compute bucket sizes: {[f'{x:.2f}' for x in compute_bucket_sizes]}")

    # Sweep over ECI bucket sizes for compute reduction
    compute_reduction_results = []

    print("\nSweeping ECI bucket sizes for compute reduction analysis...")
    for bucket_size in eci_bucket_sizes:
        print(f"  Testing ECI bucket size: {bucket_size}")
        results_df, _ = analyze_compute_reduction(
            df,
            bucket_size_oom=bucket_size,
            min_models_per_bucket=min_models_per_bucket,
            n_bootstrap=1000
        )

        if len(results_df) > 0:
            compute_reduction_results.append({
                'bucket_size': bucket_size,
                'mean_slope': results_df['slope_oom_per_year'].mean(),
                'median_slope': results_df['slope_oom_per_year'].median(),
                'std_slope': results_df['slope_oom_per_year'].std(),
                'n_buckets': len(results_df),
                'total_models': results_df['n_models_sota'].sum()
            })

    # Sweep over compute bucket sizes for capability gains
    capability_gains_results = []

    print("\nSweeping compute bucket sizes for capability gains analysis...")
    for bucket_size in compute_bucket_sizes:
        print(f"  Testing compute bucket size: {bucket_size}")
        results_df, _ = analyze_capability_gains(
            df,
            bucket_size_oom=bucket_size,
            min_models_per_bucket=min_models_per_bucket,
            n_bootstrap=1000
        )

        if len(results_df) > 0:
            capability_gains_results.append({
                'bucket_size': bucket_size,
                'mean_slope': results_df['slope_eci_per_year'].mean(),
                'median_slope': results_df['slope_eci_per_year'].median(),
                'std_slope': results_df['slope_eci_per_year'].std(),
                'n_buckets': len(results_df),
                'total_models': results_df['n_models_sota'].sum()
            })

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Compute reduction: Mean slope vs bucket size
    ax = axes[0, 0]
    cr_df = pd.DataFrame(compute_reduction_results)
    if len(cr_df) > 0:
        ax.plot(cr_df['bucket_size'], -cr_df['mean_slope'], 'o-',
               linewidth=2, markersize=8, label='Mean')
        ax.plot(cr_df['bucket_size'], -cr_df['median_slope'], 's--',
               linewidth=2, markersize=8, label='Median')
        ax.fill_between(cr_df['bucket_size'],
                       -cr_df['mean_slope'] - cr_df['std_slope'],
                       -cr_df['mean_slope'] + cr_df['std_slope'],
                       alpha=0.2, label='±1 Std Dev')
        ax.set_xlabel('ECI Bucket Size', fontsize=12)
        ax.set_ylabel('Compute Reduction (OOMs/year)', fontsize=12)
        ax.set_title('Sensitivity: Compute Reduction vs ECI Bucket Size',
                    fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    # 2. Compute reduction: Number of buckets vs bucket size
    ax = axes[0, 1]
    if len(cr_df) > 0:
        ax.plot(cr_df['bucket_size'], cr_df['n_buckets'], 'o-',
               linewidth=2, markersize=8, color='steelblue')
        ax.set_xlabel('ECI Bucket Size', fontsize=12)
        ax.set_ylabel('Number of Valid Buckets', fontsize=12)
        ax.set_title('Number of Buckets vs ECI Bucket Size',
                    fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)

    # 3. Capability gains: Mean slope vs bucket size
    ax = axes[1, 0]
    cg_df = pd.DataFrame(capability_gains_results)
    if len(cg_df) > 0:
        ax.plot(cg_df['bucket_size'], cg_df['mean_slope'], 'o-',
               linewidth=2, markersize=8, label='Mean')
        ax.plot(cg_df['bucket_size'], cg_df['median_slope'], 's--',
               linewidth=2, markersize=8, label='Median')
        ax.fill_between(cg_df['bucket_size'],
                       cg_df['mean_slope'] - cg_df['std_slope'],
                       cg_df['mean_slope'] + cg_df['std_slope'],
                       alpha=0.2, label='±1 Std Dev')
        ax.set_xlabel('Compute Bucket Size (log₁₀ FLOP)', fontsize=12)
        ax.set_ylabel('Capability Gain (ECI units/year)', fontsize=12)
        ax.set_title('Sensitivity: Capability Gain vs Compute Bucket Size',
                    fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    # 4. Capability gains: Number of buckets vs bucket size
    ax = axes[1, 1]
    if len(cg_df) > 0:
        ax.plot(cg_df['bucket_size'], cg_df['n_buckets'], 'o-',
               linewidth=2, markersize=8, color='forestgreen')
        ax.set_xlabel('Compute Bucket Size (log₁₀ FLOP)', fontsize=12)
        ax.set_ylabel('Number of Valid Buckets', fontsize=12)
        ax.set_title('Number of Buckets vs Compute Bucket Size',
                    fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / f"bucket_size_sensitivity{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nBucket size sensitivity plot saved to: {output_path}")
    plt.close()

    # Save results to CSV
    if len(cr_df) > 0:
        output_path = output_dir / f"bucket_size_sensitivity_compute_reduction{suffix}.csv"
        cr_df.to_csv(output_path, index=False)
        print(f"Compute reduction sensitivity results saved to: {output_path}")

    if len(cg_df) > 0:
        output_path = output_dir / f"bucket_size_sensitivity_capability_gains{suffix}.csv"
        cg_df.to_csv(output_path, index=False)
        print(f"Capability gains sensitivity results saved to: {output_path}")

    # Print summary
    print("\n" + "="*70)
    print("BUCKET SIZE SENSITIVITY SUMMARY")
    print("="*70)

    if len(cr_df) > 0:
        print("\nCompute Reduction:")
        print(f"  Bucket sizes tested: {eci_bucket_sizes}")
        print(f"  Mean slope range: {-cr_df['mean_slope'].max():.3f} to {-cr_df['mean_slope'].min():.3f} OOMs/yr")
        print(f"  Range of means: {10**(-cr_df['mean_slope'].max()):.1f}× to {10**(-cr_df['mean_slope'].min()):.1f}× reduction/yr")
        print(f"  Variability (std of means): {cr_df['mean_slope'].std():.3f} OOMs/yr")

    if len(cg_df) > 0:
        print("\nCapability Gains:")
        print(f"  Bucket sizes tested: {compute_bucket_sizes}")
        print(f"  Mean slope range: {cg_df['mean_slope'].min():.3f} to {cg_df['mean_slope'].max():.3f} ECI/yr")
        print(f"  Variability (std of means): {cg_df['mean_slope'].std():.3f} ECI/yr")


def main():
    """Main analysis function"""
    import argparse
    parser = argparse.ArgumentParser(
        description='Buckets method for algorithmic progress analysis')
    parser.add_argument('--eci-bucket-size', type=float, default=0.5,
                       help='Width of ECI buckets (default: 0.5 ECI units)')
    parser.add_argument('--compute-bucket-size', type=float, default=0.5,
                       help='Width of compute buckets in log10 scale (default: 0.5 OOMs)')
    parser.add_argument('--min-models', type=int, default=3,
                       help='Minimum number of SOTA models per bucket (default: 3)')
    parser.add_argument('--exclude-distilled', action='store_true',
                       help='Exclude distilled models from analysis')
    parser.add_argument('--include-low-confidence', action='store_true',
                       help='When excluding distilled, also exclude low-confidence ones')
    parser.add_argument('--use-website-data', action='store_true',
                       help='Use data from data/website/epoch_capabilities_index.csv')
    parser.add_argument('--sweep-bucket-sizes', action='store_true',
                       help='Perform sensitivity analysis by sweeping over bucket sizes')
    parser.add_argument('--eci-bucket-sizes', type=str, default=None,
                       help='Comma-separated ECI bucket sizes for sweep (default: auto-compute from data range)')
    parser.add_argument('--compute-bucket-sizes', type=str, default=None,
                       help='Comma-separated compute bucket sizes for sweep (default: auto-compute from data range)')
    parser.add_argument('--n-bucket-sizes', type=int, default=5,
                       help='Number of bucket sizes to test in sweep (default: 5)')
    parser.add_argument('--label-points', action='store_true',
                       help='Label data points with model name and complementary variable (ECI or compute)')
    args = parser.parse_args()

    # Validate arguments
    if args.include_low_confidence and not args.exclude_distilled:
        parser.error('--include-low-confidence requires --exclude-distilled')

    # Load data
    df = load_model_capabilities_and_compute(
        use_website_data=args.use_website_data,
        exclude_distilled=args.exclude_distilled,
        include_low_confidence=args.include_low_confidence
    )

    if df is None or len(df) == 0:
        print("Failed to load data. Exiting.")
        return

    # Create output directory
    output_dir = Path("outputs/algorithmic_progress_methods/buckets")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate suffix for file names
    suffix_parts = []
    if args.exclude_distilled:
        suffix_parts.append("no_distilled_all" if args.include_low_confidence else "no_distilled")
    if args.use_website_data:
        suffix_parts.append("website")
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

    # If sweep mode, run sensitivity analysis
    if args.sweep_bucket_sizes:
        # Parse bucket sizes from command line if provided
        eci_bucket_sizes = None
        compute_bucket_sizes = None

        if args.eci_bucket_sizes is not None:
            eci_bucket_sizes = [float(x) for x in args.eci_bucket_sizes.split(',')]

        if args.compute_bucket_sizes is not None:
            compute_bucket_sizes = [float(x) for x in args.compute_bucket_sizes.split(',')]

        plot_bucket_size_sensitivity(
            df, output_dir, suffix,
            eci_bucket_sizes=eci_bucket_sizes,
            compute_bucket_sizes=compute_bucket_sizes,
            min_models_per_bucket=args.min_models,
            n_bucket_sizes=args.n_bucket_sizes
        )

        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS COMPLETE")
        print("="*70)
        print(f"Results saved to: {output_dir}")
        return

    # Standard analysis with single bucket size
    # Analyze compute reduction (fixed capability)
    compute_reduction_df, compute_reduction_buckets = analyze_compute_reduction(
        df,
        bucket_size_oom=args.eci_bucket_size,
        min_models_per_bucket=args.min_models
    )

    # Analyze capability gains (fixed compute)
    capability_gains_df, capability_gains_buckets = analyze_capability_gains(
        df,
        bucket_size_oom=args.compute_bucket_size,
        min_models_per_bucket=args.min_models
    )

    # Save results
    save_results(compute_reduction_df, capability_gains_df, output_dir, suffix)

    # Create visualizations
    plot_compute_reduction_results(compute_reduction_df, compute_reduction_buckets,
                                   output_dir, suffix)
    plot_capability_gains_results(capability_gains_df, capability_gains_buckets,
                                  output_dir, suffix)

    # Create new combined visualizations
    print("\nCreating combined bucket regression plots...")
    plot_all_bucket_regressions(df, compute_reduction_df, compute_reduction_buckets,
                                output_dir, suffix, label_points=args.label_points)
    plot_all_capability_bucket_regressions(df, capability_gains_df, capability_gains_buckets,
                                          output_dir, suffix, label_points=args.label_points)

    # Create bootstrap distribution plots
    print("\nCreating bootstrap distribution plots...")
    plot_bootstrap_distributions(compute_reduction_df, compute_reduction_buckets,
                                output_dir, suffix)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
