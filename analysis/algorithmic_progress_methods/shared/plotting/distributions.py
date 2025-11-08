"""Distribution plotting utilities - histograms, bootstrap distributions."""

import numpy as np
import matplotlib.pyplot as plt


def plot_histogram_with_stats(ax, data, bins=50, xlabel=None, ylabel='Frequency',
                              title=None, color='steelblue', show_stats=True):
    """Plot histogram with mean, median, and confidence interval lines.

    Args:
        ax: Matplotlib axis
        data: 1D array of values
        bins: Number of bins
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        color: Histogram color
        show_stats: Whether to show statistical lines

    Returns:
        dict with statistics (mean, median, ci)
    """
    ax.hist(data, bins=bins, alpha=0.7, edgecolor='black', color=color)

    stats_dict = {}

    if show_stats:
        mean_val = data.mean()
        median_val = np.median(data)
        ci_vals = np.percentile(data, [2.5, 97.5])

        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='orange', linestyle='--', linewidth=2,
                  label=f'Median: {median_val:.3f}')
        ax.axvline(ci_vals[0], color='gray', linestyle=':', linewidth=2, alpha=0.7)
        ax.axvline(ci_vals[1], color='gray', linestyle=':', linewidth=2, alpha=0.7,
                  label=f'95% CI: [{ci_vals[0]:.3f}, {ci_vals[1]:.3f}]')

        stats_dict = {
            'mean': mean_val,
            'median': median_val,
            'ci_lower': ci_vals[0],
            'ci_upper': ci_vals[1]
        }

        ax.legend(fontsize=10)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')

    ax.grid(alpha=0.3)

    return stats_dict


def plot_bootstrap_distribution(ax, bootstrap_values, xlabel=None,
                                title=None, color='steelblue'):
    """Plot bootstrap distribution with statistics.

    Args:
        ax: Matplotlib axis
        bootstrap_values: Array of bootstrap estimates
        xlabel: X-axis label
        title: Plot title
        color: Histogram color

    Returns:
        dict with statistics
    """
    return plot_histogram_with_stats(
        ax, bootstrap_values,
        bins=50,
        xlabel=xlabel,
        ylabel='Frequency',
        title=title,
        color=color,
        show_stats=True
    )


def plot_multiplier_distribution(ax, bootstrap_log_values, xlabel=None,
                                 title=None, color='forestgreen'):
    """Plot distribution of multipliers (10^x values) from log-scale bootstrap.

    Args:
        ax: Matplotlib axis
        bootstrap_log_values: Array of log-scale bootstrap estimates
        xlabel: X-axis label (default includes '×')
        title: Plot title
        color: Histogram color

    Returns:
        dict with statistics
    """
    multipliers = 10 ** bootstrap_log_values

    stats = plot_histogram_with_stats(
        ax, multipliers,
        bins=50,
        xlabel=xlabel or 'Multiplier (×)',
        ylabel='Frequency',
        title=title,
        color=color,
        show_stats=False
    )

    # Add custom statistics in multiplier space
    mean_mult = multipliers.mean()
    median_mult = np.median(multipliers)
    ci_mult = np.percentile(multipliers, [2.5, 97.5])

    ax.axvline(mean_mult, color='red', linestyle='--', linewidth=2,
              label=f'Mean: {mean_mult:.1f}×')
    ax.axvline(median_mult, color='orange', linestyle='--', linewidth=2,
              label=f'Median: {median_mult:.1f}×')
    ax.axvline(ci_mult[0], color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax.axvline(ci_mult[1], color='gray', linestyle=':', linewidth=2, alpha=0.7,
              label=f'95% CI: [{ci_mult[0]:.1f}×, {ci_mult[1]:.1f}×]')

    # Use log scale if range is large
    if ci_mult[1] / ci_mult[0] > 100:
        ax.set_xscale('log')
        if xlabel:
            ax.set_xlabel(f'{xlabel} (log scale)', fontsize=12)

    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    return {
        'mean': mean_mult,
        'median': median_mult,
        'ci_lower': ci_mult[0],
        'ci_upper': ci_mult[1]
    }
