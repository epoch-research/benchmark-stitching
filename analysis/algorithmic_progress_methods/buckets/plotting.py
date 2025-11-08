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
    plot_multiplier_distribution
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
    """Plot all bucket regression lines on a single plot."""
    if len(results_df) == 0 or len(bucket_data) == 0:
        print("No results to plot for all bucket regressions")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    date_min = df['date_obj'].min()
    date_max = df['date_obj'].max()
    date_range = pd.date_range(date_min, date_max, periods=100)
    date_numeric = (date_range - pd.Timestamp('2020-01-01')).total_seconds() / (365.25 * 24 * 3600)

    colors = plt.cm.viridis(np.linspace(0, 1, len(bucket_data)))

    for i, (bucket_info, color) in enumerate(zip(bucket_data, colors)):
        bucket_center = bucket_info['bucket_center']
        df_sota = bucket_info['df_sota']
        bootstrap_results = bucket_info['bootstrap_results']
        result_row = results_df[results_df['bucket_center'] == bucket_center].iloc[0]
        slope = result_row['slope_oom_per_year']
        intercept = result_row['intercept']

        log_compute_fit = slope * date_numeric + intercept
        ax.plot(date_range, log_compute_fit, color=color, linewidth=2.5, alpha=0.8,
               label=f'ECI={bucket_center:.2f}: {-slope:.3f} OOMs/yr ({10**(-slope):.1f}× reduction)')

        # Bootstrap uncertainty
        slope_ci = bootstrap_results['slope_ci']
        intercept_mean = bootstrap_results['intercept_mean']
        log_compute_ci_lower = slope_ci[0] * date_numeric + intercept_mean
        log_compute_ci_upper = slope_ci[1] * date_numeric + intercept_mean
        ax.fill_between(date_range, log_compute_ci_lower, log_compute_ci_upper,
                       color=color, alpha=0.15)

        ax.scatter(df_sota['date_obj'], df_sota['log_compute'],
                  color=color, s=80, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=3)

        if label_points:
            for _, row in df_sota.iterrows():
                ax.annotate(f"{row['model']}\nECI={row['estimated_capability']:.2f}",
                           xy=(row['date_obj'], row['log_compute']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=7, alpha=0.8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3, edgecolor='none'))

    setup_date_axis(ax)
    apply_plot_style(ax,
                    title='Compute Reduction at Fixed Capability Levels\nAll ECI Buckets with Bootstrap Uncertainty (95% CI)',
                    xlabel='Release Date', ylabel='log₁₀(Training Compute)',
                    legend=True, legend_kwargs={'fontsize': 9, 'ncol': 1})

    plt.tight_layout()
    save_figure(fig, output_dir / f"all_bucket_regressions_compute_reduction{suffix}")
    print(f"\nAll bucket regressions plot saved")
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

    # 2. Combined bootstrap distribution
    ax = axes[0, 1]
    all_bootstrap_slopes = []
    for bucket_info in bucket_data:
        all_bootstrap_slopes.extend(bucket_info['bootstrap_results']['slopes'])
    all_bootstrap_slopes = -np.array(all_bootstrap_slopes)

    plot_bootstrap_distribution(ax, all_bootstrap_slopes,
                               xlabel='Compute Reduction (OOMs/year)',
                               title='Pooled Bootstrap Distribution\n(All Buckets Combined)')

    # 3. Compute reduction as multiplier
    ax = axes[1, 0]
    plot_multiplier_distribution(ax, all_bootstrap_slopes,
                                xlabel='Compute Reduction Multiplier (×/year)',
                                title='Algorithmic Progress as Compute Multiplier\n(1 year = X× more compute)')

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
