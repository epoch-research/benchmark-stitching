"""Plotting functions for buckets method."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from shared.plotting import (
    setup_date_axis, apply_plot_style, save_figure,
    plot_histogram_with_stats, plot_bootstrap_distribution,
    plot_multiplier_distribution, plot_compute_vs_date_with_regressions
)


def plot_compute_reduction_results(results_df, bucket_data, output_dir, suffix=""):
    """Create visualizations for compute reduction analysis."""
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
    apply_plot_style(ax, title='Compute Reduction Rate vs Capability Level',
                    xlabel='ECI Bucket Center', ylabel='Compute Reduction (OOMs/year)',
                    legend=True)

    # 2. Distribution of slopes
    ax = axes[0, 1]
    plot_histogram_with_stats(ax, results_df['slope_oom_per_year'], bins=15,
                             xlabel='Compute Reduction (OOMs/year)',
                             title='Distribution of Compute Reduction Rates',
                             color='steelblue')

    # 3. Example bucket: show data and fit
    ax = axes[1, 0]
    if len(bucket_data) > 0:
        mid_idx = len(bucket_data) // 2
        bucket = bucket_data[mid_idx]
        df_bucket = bucket['df_bucket']
        df_sota = bucket['df_sota']
        bucket_center = bucket['bucket_center']

        ax.scatter(df_bucket['date_obj'], df_bucket['log_compute'],
                  alpha=0.3, s=50, label='All models in bucket', color='gray')
        ax.scatter(df_sota['date_obj'], df_sota['log_compute'],
                  alpha=0.8, s=100, label='SOTA models', color='red',
                  edgecolors='black', linewidth=1, zorder=3)

        if len(df_sota) >= 2:
            result = results_df[results_df['bucket_center'] == bucket_center].iloc[0]
            date_range = pd.date_range(df_sota['date_obj'].min(), df_sota['date_obj'].max(), periods=100)
            date_numeric = (date_range - pd.Timestamp('2020-01-01')).total_seconds() / (365.25 * 24 * 3600)
            log_compute_fit = result['slope_oom_per_year'] * date_numeric + result['intercept']
            ax.plot(date_range, log_compute_fit, 'b--', linewidth=2,
                   label=f'Linear fit: {result["slope_oom_per_year"]:.3f} OOMs/year')

        setup_date_axis(ax)
        apply_plot_style(ax, title=f'Example: ECI Bucket {bucket_center:.2f}',
                        xlabel='Release Date', ylabel='log₁₀(Compute)', legend=True)

    # 4. R² values
    ax = axes[1, 1]
    ax.scatter(results_df['bucket_center'], results_df['r_squared'],
              s=results_df['n_models_sota']*20, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.axhline(results_df['r_squared'].mean(), color='blue',
              linestyle='--', linewidth=2, label=f'Mean R²: {results_df["r_squared"].mean():.3f}')
    ax.set_ylim([0, 1])
    apply_plot_style(ax, title='Model Fit Quality vs Capability Level',
                    xlabel='ECI Bucket Center', ylabel='R² (Goodness of Fit)', legend=True)

    plt.tight_layout()
    save_figure(fig, output_dir / f"compute_reduction_analysis{suffix}")
    print(f"\nCompute reduction plot saved to: {output_dir / f'compute_reduction_analysis{suffix}.png'}")
    plt.close()


def plot_capability_gains_results(results_df, bucket_data, output_dir, suffix=""):
    """Create visualizations for capability gains analysis."""
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
    apply_plot_style(ax, title='Capability Gain Rate vs Compute Level',
                    xlabel='log₁₀(Compute) Bucket Center', ylabel='Capability Gain (ECI units/year)',
                    legend=True)

    # 2. Distribution of slopes
    ax = axes[0, 1]
    plot_histogram_with_stats(ax, results_df['slope_eci_per_year'], bins=15,
                             xlabel='Capability Gain (ECI units/year)',
                             title='Distribution of Capability Gain Rates',
                             color='forestgreen')

    # 3. Example bucket
    ax = axes[1, 0]
    if len(bucket_data) > 0:
        mid_idx = len(bucket_data) // 2
        bucket = bucket_data[mid_idx]
        df_bucket = bucket['df_bucket']
        df_sota = bucket['df_sota']
        bucket_center = bucket['bucket_center']

        ax.scatter(df_bucket['date_obj'], df_bucket['estimated_capability'],
                  alpha=0.3, s=50, label='All models in bucket', color='gray')
        ax.scatter(df_sota['date_obj'], df_sota['estimated_capability'],
                  alpha=0.8, s=100, label='SOTA models', color='blue',
                  edgecolors='black', linewidth=1, zorder=3)

        if len(df_sota) >= 2:
            result = results_df[results_df['bucket_center_log'] == bucket_center].iloc[0]
            date_range = pd.date_range(df_sota['date_obj'].min(), df_sota['date_obj'].max(), periods=100)
            date_numeric = (date_range - pd.Timestamp('2020-01-01')).total_seconds() / (365.25 * 24 * 3600)
            eci_fit = result['slope_eci_per_year'] * date_numeric + result['intercept']
            ax.plot(date_range, eci_fit, 'r--', linewidth=2,
                   label=f'Linear fit: {result["slope_eci_per_year"]:.3f} ECI/year')

        setup_date_axis(ax)
        apply_plot_style(ax, title=f'Example: Compute Bucket {bucket_center:.2f} log₁₀(FLOP)',
                        xlabel='Release Date', ylabel='ECI (Capability)', legend=True)

    # 4. R² values
    ax = axes[1, 1]
    ax.scatter(results_df['bucket_center_log'], results_df['r_squared'],
              s=results_df['n_models_sota']*20, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.axhline(results_df['r_squared'].mean(), color='blue',
              linestyle='--', linewidth=2, label=f'Mean R²: {results_df["r_squared"].mean():.3f}')
    ax.set_ylim([0, 1])
    apply_plot_style(ax, title='Model Fit Quality vs Compute Level',
                    xlabel='log₁₀(Compute) Bucket Center', ylabel='R² (Goodness of Fit)', legend=True)

    plt.tight_layout()
    save_figure(fig, output_dir / f"capability_gains_analysis{suffix}")
    print(f"\nCapability gains plot saved to: {output_dir / f'capability_gains_analysis{suffix}.png'}")
    plt.close()


def plot_all_bucket_regressions(df, results_df, bucket_data, output_dir, suffix="", label_points=False):
    """Plot all bucket regression lines on a single plot using unified plotting."""
    if len(results_df) == 0 or len(bucket_data) == 0:
        print("No results to plot for all bucket regressions")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    # Prepare data for unified plotting function
    # Combine all SOTA points with bucket information
    all_df_parts = []
    for bucket_info in bucket_data:
        df_sota = bucket_info['df_sota'].copy()
        df_sota['bucket_center'] = bucket_info['bucket_center']
        all_df_parts.append(df_sota)

    df_combined = pd.concat(all_df_parts, ignore_index=True)

    # Prepare regression line data
    regression_lines = []
    for bucket_info in bucket_data:
        bucket_center = bucket_info['bucket_center']
        df_sota = bucket_info['df_sota']
        bootstrap_results = bucket_info['bootstrap_results']
        result_row = results_df[results_df['bucket_center'] == bucket_center].iloc[0]
        slope = result_row['slope_oom_per_year']
        intercept = result_row['intercept']

        # Create label with reduction info
        reduction_multiplier = 10**(-slope)
        label = (f'ECI={bucket_center:.2f}: {-slope:.3f} OOMs/yr '
                f'({reduction_multiplier:.1f}× reduction)')

        regression_lines.append({
            'slope': slope,
            'intercept': intercept,
            'label': label,
            'df_sota': df_sota,
            'bootstrap_slopes': bootstrap_results['slopes'],
            'bootstrap_intercepts': np.full(len(bootstrap_results['slopes']),
                                           bootstrap_results['intercept_mean'])
        })

    # Use unified plotting function
    plot_compute_vs_date_with_regressions(
        ax, df_combined, regression_lines,
        label_points=label_points,
        label_format='box',
        colormap='viridis',
        title='Compute Reduction at Fixed Capability Levels\nAll ECI Buckets',
        show_uncertainty=False
    )

    plt.tight_layout()
    save_figure(fig, output_dir / f"all_bucket_regressions_compute_reduction{suffix}")
    print("\nAll bucket regressions plot saved")
    plt.close()


def plot_bootstrap_distributions(results_df, bucket_data, output_dir, suffix=""):
    """Create diagnostic plots showing bootstrap distributions for all buckets."""
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
    apply_plot_style(ax, title='Compute Reduction with 95% Bootstrap CI\n(All ECI Buckets)',
                    xlabel='Bucket Index', ylabel='Compute Reduction (OOMs/year)', legend=True)
    ax.set_ylim(0, 10)

    # 2. Combined bootstrap distribution
    ax = axes[0, 1]
    all_bootstrap_slopes = []
    for bucket_info in bucket_data:
        all_bootstrap_slopes.extend(bucket_info['bootstrap_results']['slopes'])
    all_bootstrap_slopes = -np.array(all_bootstrap_slopes)

    plot_bootstrap_distribution(ax, all_bootstrap_slopes,
                               xlabel='Compute Reduction (OOMs/year)',
                               title='Pooled Bootstrap Distribution\n(All Buckets Combined)')
    ax.set_xlim(0, 10)

    # 3. Compute reduction as multiplier
    ax = axes[1, 0]
    plot_multiplier_distribution(ax, all_bootstrap_slopes,
                                xlabel='Compute Reduction Multiplier (×/year)',
                                title='Algorithmic Progress as Compute Multiplier\n(1 year = X× more compute)')
    # Set reasonable x-axis limit (10^0 to 10^10 = 1x to 10 billion x)
    ax.set_xlim(1, 10**10)

    # 4. Slope vs bucket center
    ax = axes[1, 1]
    bucket_centers = results_df['bucket_center'].values
    ax.scatter(bucket_centers, -slopes, s=100, alpha=0.7, edgecolors='black', linewidth=1)

    for i, (center, slope, ci_lower, ci_upper) in enumerate(zip(bucket_centers, -slopes,
                                                                 -slope_cis_lower, -slope_cis_upper)):
        ax.plot([center, center], [ci_upper, ci_lower], color='gray', alpha=0.5, linewidth=2)

    ax.axhline(-slopes.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {-slopes.mean():.3f} OOMs/yr')
    apply_plot_style(ax, title='Compute Reduction vs Capability Level\n(with 95% Bootstrap CI)',
                    xlabel='ECI Bucket Center', ylabel='Compute Reduction (OOMs/year)', legend=True)
    ax.set_ylim(0, 10)

    plt.tight_layout()
    save_figure(fig, output_dir / f"bootstrap_distributions_compute_reduction{suffix}")
    print(f"\nBootstrap distributions plot saved")
    plt.close()


def save_results(compute_reduction_df, capability_gains_df, output_dir, suffix=""):
    """Save analysis results to CSV files."""
    if len(compute_reduction_df) > 0:
        output_path = output_dir / f"compute_reduction_results{suffix}.csv"
        compute_reduction_df.to_csv(output_path, index=False)
        print(f"\nCompute reduction results saved to: {output_path}")

    if len(capability_gains_df) > 0:
        output_path = output_dir / f"capability_gains_results{suffix}.csv"
        capability_gains_df.to_csv(output_path, index=False)
        print(f"Capability gains results saved to: {output_path}")


def plot_bucket_size_sensitivity(compute_reduction_df, capability_gains_df, output_dir, suffix=""):
    """Create visualization showing how results vary with bucket size.

    Args:
        compute_reduction_df: DataFrame with sensitivity results for compute reduction
        capability_gains_df: DataFrame with sensitivity results for capability gains
        output_dir: Directory to save plots
        suffix: Filename suffix
    """
    if len(compute_reduction_df) == 0 and len(capability_gains_df) == 0:
        print("No sensitivity data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Compute reduction: mean, median, std vs bucket size
    if len(compute_reduction_df) > 0:
        ax = axes[0, 0]
        # Convert slopes to compute reduction multipliers
        mean_reductions = 10**(-compute_reduction_df['mean_slope'])
        median_reductions = 10**(-compute_reduction_df['median_slope'])
        std_reductions = compute_reduction_df['std_slope']

        # Plot mean and median
        ax.plot(compute_reduction_df['bucket_size'], mean_reductions,
                'o-', markersize=10, linewidth=2.5, color='steelblue', label='Mean')
        ax.plot(compute_reduction_df['bucket_size'], median_reductions,
                's-', markersize=10, linewidth=2.5, color='forestgreen', label='Median')

        # Add shaded region for ±1 standard deviation around mean
        mean_slopes = compute_reduction_df['mean_slope']
        upper_reductions = 10**(-(mean_slopes - std_reductions))
        lower_reductions = 10**(-(mean_slopes + std_reductions))
        ax.fill_between(compute_reduction_df['bucket_size'],
                        lower_reductions, upper_reductions,
                        alpha=0.2, color='steelblue', label='±1 Std Dev')

        apply_plot_style(ax,
                        title='Compute Reduction vs ECI Bucket Size',
                        xlabel='ECI Bucket Size (capability units)',
                        ylabel='Compute Reduction (× per year)',
                        legend=True)
        ax.set_yscale('log')
        ax.grid(True, which='both', alpha=0.3)

    # 2. Compute reduction: number of buckets vs bucket size
    if len(compute_reduction_df) > 0:
        ax = axes[0, 1]
        ax.plot(compute_reduction_df['bucket_size'], compute_reduction_df['n_buckets'],
                'o-', markersize=10, linewidth=2.5, color='steelblue', label='Number of Buckets')
        ax.plot(compute_reduction_df['bucket_size'], compute_reduction_df['total_models'],
                's-', markersize=10, linewidth=2.5, color='orange',
                label='Total Models Used in Regression\n(SOTA in compute efficiency)', alpha=0.7)
        apply_plot_style(ax,
                        title='Data Coverage vs ECI Bucket Size',
                        xlabel='ECI Bucket Size (capability units)',
                        ylabel='Count',
                        legend=True)
        ax.grid(True, alpha=0.3)

    # 3. Capability gains: mean, median, std vs bucket size
    if len(capability_gains_df) > 0:
        ax = axes[1, 0]
        mean_gains = capability_gains_df['mean_slope']
        median_gains = capability_gains_df['median_slope']
        std_gains = capability_gains_df['std_slope']

        # Plot mean and median
        ax.plot(capability_gains_df['bucket_size'], mean_gains,
                'o-', markersize=10, linewidth=2.5, color='steelblue', label='Mean')
        ax.plot(capability_gains_df['bucket_size'], median_gains,
                's-', markersize=10, linewidth=2.5, color='forestgreen', label='Median')

        # Add shaded region for ±1 standard deviation around mean
        ax.fill_between(capability_gains_df['bucket_size'],
                        mean_gains - std_gains,
                        mean_gains + std_gains,
                        alpha=0.2, color='steelblue', label='±1 Std Dev')

        apply_plot_style(ax,
                        title='Capability Gain vs Compute Bucket Size',
                        xlabel='Compute Bucket Size (log₁₀(FLOP) units)',
                        ylabel='Capability Gain (ECI units per year)',
                        legend=True)
        ax.grid(True, alpha=0.3)

    # 4. Capability gains: number of buckets vs bucket size
    if len(capability_gains_df) > 0:
        ax = axes[1, 1]
        ax.plot(capability_gains_df['bucket_size'], capability_gains_df['n_buckets'],
                'o-', markersize=10, linewidth=2.5, color='forestgreen', label='Number of Buckets')
        ax.plot(capability_gains_df['bucket_size'], capability_gains_df['total_models'],
                's-', markersize=10, linewidth=2.5, color='orange',
                label='Total Models Used in Regression\n(SOTA in capability)', alpha=0.7)
        apply_plot_style(ax,
                        title='Data Coverage vs Compute Bucket Size',
                        xlabel='Compute Bucket Size (log₁₀(FLOP) units)',
                        ylabel='Count',
                        legend=True)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_dir / f"bucket_size_sensitivity{suffix}")
    print(f"\nBucket size sensitivity plot saved to: {output_dir / f'bucket_size_sensitivity{suffix}.png'}")
    plt.close()
