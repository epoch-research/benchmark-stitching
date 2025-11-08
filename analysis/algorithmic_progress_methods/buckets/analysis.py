"""Pure analysis code for the buckets method.

This module contains the core algorithmic progress analysis logic using the buckets approach:
1. Compute reduction: measure how compute requirements decrease over time for fixed capabilities
2. Capability gains: measure how capabilities improve over time for fixed compute budgets
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats

from shared.data_loading import load_model_capabilities_and_compute
from shared.bootstrap import bootstrap_scipy_regression


def generate_bucket_centers(range_min, range_max, bucket_size):
    """Return bucket centers that fully cover the observed range.

    np.arange excludes the stop value, so we extend the stop far enough to ensure
    the final bucket includes range_max and backfill a single center when the
    data are narrower than the bucket width.
    """
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")

    start = range_min + bucket_size / 2
    stop = range_max + bucket_size / 2 + np.finfo(float).eps
    centers = np.arange(start, stop, bucket_size)

    if len(centers) == 0:
        centers = np.array([(range_min + range_max) / 2])

    return centers


def find_sota_in_compute_efficiency_at_release(df_bucket):
    """Identify models that were SOTA in compute efficiency at their release date.

    A model is SOTA in compute efficiency if no earlier model in the bucket achieved
    the same (or higher) capability with less (or equal) compute.

    Args:
        df_bucket: DataFrame filtered to a specific ECI bucket

    Returns:
        DataFrame of SOTA models
    """
    # Sort by date, then compute so the most compute-efficient model for a day comes first
    df_sorted = df_bucket.sort_values(['date_obj', 'compute']).copy()
    sota_indices = []
    best_compute_so_far = np.inf

    for idx, row in df_sorted.iterrows():
        if row['compute'] < best_compute_so_far:
            sota_indices.append(idx)
            best_compute_so_far = row['compute']

    return df_sorted.loc[sota_indices]


def find_sota_in_capability_at_release(df_bucket):
    """Identify models that were SOTA in capability at their release date.

    A model is SOTA in capability if no earlier model in the bucket achieved
    higher (or equal) capability.

    Args:
        df_bucket: DataFrame filtered to a specific compute bucket

    Returns:
        DataFrame of SOTA models
    """
    # Sort by date, then capability so the most capable model for a day comes first
    df_sorted = df_bucket.sort_values(
        ['date_obj', 'estimated_capability'],
        ascending=[True, False]
    ).copy()
    sota_indices = []
    best_capability_so_far = -np.inf

    for idx, row in df_sorted.iterrows():
        if row['estimated_capability'] > best_capability_so_far:
            sota_indices.append(idx)
            best_capability_so_far = row['estimated_capability']

    return df_sorted.loc[sota_indices]


def analyze_compute_reduction(df, bucket_size_oom=0.3, min_models_per_bucket=3, n_bootstrap=1000):
    """Analyze compute reduction over time for fixed capability levels (ECI buckets).

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

    bucket_centers = generate_bucket_centers(eci_min, eci_max, bucket_size_oom)

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
        X = df_sota['date_numeric'].values
        y = df_sota['log_compute'].values

        if len(X) < 2:
            continue

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

        # Bootstrap analysis for uncertainty
        bootstrap_results = bootstrap_scipy_regression(X, y, n_bootstrap=n_bootstrap)

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
    """Analyze capability gains over time for fixed compute levels (compute buckets).

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

    bucket_centers = generate_bucket_centers(log_compute_min, log_compute_max, bucket_size_oom)

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
        X = df_sota['date_numeric'].values
        y = df_sota['estimated_capability'].values

        if len(X) < 2:
            continue

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

        # Bootstrap analysis for uncertainty
        bootstrap_results = bootstrap_scipy_regression(X, y, n_bootstrap=n_bootstrap)

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


def bucket_size_sensitivity_analysis(df, eci_bucket_sizes=None, compute_bucket_sizes=None,
                                     min_models_per_bucket=3, n_bootstrap=1000, n_bucket_sizes=20):
    """Sweep over different bucket sizes to assess sensitivity.

    Args:
        df: DataFrame with model data
        eci_bucket_sizes: List of ECI bucket sizes to try (if None, computed from data)
        compute_bucket_sizes: List of compute bucket sizes to try (if None, computed from data)
        min_models_per_bucket: Minimum SOTA models per bucket
        n_bootstrap: Number of bootstrap iterations
        n_bucket_sizes: Number of bucket sizes to test (default: 20)

    Returns:
        compute_reduction_results: DataFrame with sensitivity results
        capability_gains_results: DataFrame with sensitivity results
    """
    print("\n" + "="*70)
    print("BUCKET SIZE SENSITIVITY ANALYSIS")
    print("="*70)

    # Compute bucket sizes from data if not provided
    if eci_bucket_sizes is None:
        eci_range = df['estimated_capability'].max() - df['estimated_capability'].min()
        # Test bucket sizes from 5% to 25% of the range
        eci_bucket_sizes = np.linspace(0.05 * eci_range, 0.25 * eci_range, n_bucket_sizes)
        print(f"\nECI range: {df['estimated_capability'].min():.2f} to {df['estimated_capability'].max():.2f}")
        print(f"ECI total range: {eci_range:.2f}")
        print(f"Auto-computed ECI bucket sizes: {[f'{x:.2f}' for x in eci_bucket_sizes]}")

    if compute_bucket_sizes is None:
        log_compute_range = df['log_compute'].max() - df['log_compute'].min()
        # Test bucket sizes from 5% to 25% of the range
        compute_bucket_sizes = np.linspace(0.05 * log_compute_range, 0.25 * log_compute_range, n_bucket_sizes)
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
            n_bootstrap=n_bootstrap
        )

        if len(results_df) > 0:
            compute_reduction_results.append({
                'bucket_size': bucket_size,
                'mean_slope': results_df['slope_oom_per_year'].mean(),
                'median_slope': results_df['slope_oom_per_year'].median(),
                'std_slope': results_df['slope_oom_per_year'].std(),
                'n_buckets': len(results_df),
                'total_models': results_df['n_models_total'].sum()
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
            n_bootstrap=n_bootstrap
        )

        if len(results_df) > 0:
            capability_gains_results.append({
                'bucket_size': bucket_size,
                'mean_slope': results_df['slope_eci_per_year'].mean(),
                'median_slope': results_df['slope_eci_per_year'].median(),
                'std_slope': results_df['slope_eci_per_year'].std(),
                'n_buckets': len(results_df),
                'total_models': results_df['n_models_total'].sum()
            })

    return pd.DataFrame(compute_reduction_results), pd.DataFrame(capability_gains_results)
