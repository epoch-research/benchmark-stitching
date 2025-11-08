"""Unified plotting functions for compute vs date visualizations across methods."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from .base import setup_date_axis, apply_plot_style, save_figure


def plot_compute_vs_date_with_regressions(
    ax, df_data, regression_lines,
    label_points=False, label_format='auto',
    colormap='viridis', title=None,
    show_uncertainty=True
):
    """Unified plotting function for compute vs date with regression lines.

    This function provides a consistent interface for plotting compute vs date
    visualizations across different algorithmic progress methods (buckets, linear, etc.).

    Args:
        ax: Matplotlib axis to plot on
        df_data: DataFrame with columns: 'date_obj', 'log_compute', 'model', 'estimated_capability'
        regression_lines: List of dicts, each containing:
            - 'slope': float (slope in log_compute vs date_numeric space)
            - 'intercept': float
            - 'label': str (legend label)
            - 'color': color (optional, auto-assigned if None)
            - 'df_sota': DataFrame with SOTA points for this regression (optional)
            - 'bootstrap_slopes': array for uncertainty bands (optional)
            - 'bootstrap_intercepts': array for uncertainty bands (optional)
            - 'date_range': tuple (date_min, date_max) for plotting line (optional)
        label_points: bool or str
            - False: no labels
            - True: label with model name and ECI
            - 'eci': label with ECI only
            - 'model': label with model name only
        label_format: str
            - 'auto': automatically choose format based on label_points
            - 'box': use bbox background (better for dense plots)
            - 'simple': plain text (better for sparse plots)
        colormap: str or None - colormap name for auto-coloring regression lines
        title: str - plot title (optional)
        show_uncertainty: bool - whether to show bootstrap uncertainty bands

    Returns:
        dict with 'scatter' and 'lines' matplotlib objects
    """
    # Determine label mode
    if label_points is False or label_points is None:
        show_labels = False
        label_content = None
    elif label_points is True:
        show_labels = True
        label_content = 'both'  # model + ECI
    elif label_points == 'eci':
        show_labels = True
        label_content = 'eci'
    elif label_points == 'model':
        show_labels = True
        label_content = 'model'
    else:
        show_labels = bool(label_points)
        label_content = 'both'

    # Auto-assign colors if needed
    n_lines = len(regression_lines)
    if colormap is not None and n_lines > 0:
        colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, n_lines))
        for i, reg in enumerate(regression_lines):
            if 'color' not in reg or reg['color'] is None:
                reg['color'] = colors[i]

    # Get date range from data if not specified per regression
    date_min = df_data['date_obj'].min()
    date_max = df_data['date_obj'].max()
    date_range = pd.date_range(date_min, date_max, periods=100)
    date_numeric = (date_range - pd.Timestamp('2020-01-01')).total_seconds() / (365.25 * 24 * 3600)

    # Plot regression lines with uncertainty
    lines = []
    for reg in regression_lines:
        slope = reg['slope']
        intercept = reg['intercept']
        color = reg.get('color', 'blue')
        label = reg.get('label', '')

        # Use custom date range if provided
        if 'date_range' in reg:
            custom_date_min, custom_date_max = reg['date_range']
            plot_date_range = pd.date_range(custom_date_min, custom_date_max, periods=100)
            plot_date_numeric = (plot_date_range - pd.Timestamp('2020-01-01')).total_seconds() / (365.25 * 24 * 3600)
        else:
            plot_date_range = date_range
            plot_date_numeric = date_numeric

        # Compute fit line
        log_compute_fit = slope * plot_date_numeric + intercept

        # Plot main line
        line, = ax.plot(plot_date_range, log_compute_fit,
                       color=color, linewidth=2.5, alpha=0.8,
                       label=label, zorder=2)
        lines.append(line)

        # Add uncertainty band if bootstrap results available
        if show_uncertainty and 'bootstrap_slopes' in reg and 'bootstrap_intercepts' in reg:
            bootstrap_slopes = reg['bootstrap_slopes']
            bootstrap_intercepts = reg['bootstrap_intercepts']

            # Compute CI from bootstrap samples
            bootstrap_preds = []
            for bs_slope, bs_intercept in zip(bootstrap_slopes, bootstrap_intercepts):
                bs_pred = bs_slope * plot_date_numeric + bs_intercept
                bootstrap_preds.append(bs_pred)

            bootstrap_preds = np.array(bootstrap_preds)

            # Use bootstrap mean for intercept (more stable)
            intercept_mean = bootstrap_intercepts.mean()
            slope_ci = np.percentile(bootstrap_slopes, [2.5, 97.5])

            log_compute_ci_lower = slope_ci[0] * plot_date_numeric + intercept_mean
            log_compute_ci_upper = slope_ci[1] * plot_date_numeric + intercept_mean

            ax.fill_between(plot_date_range, log_compute_ci_lower, log_compute_ci_upper,
                          color=color, alpha=0.15, zorder=1)

        # Plot SOTA points for this regression if provided
        if 'df_sota' in reg and reg['df_sota'] is not None:
            df_sota = reg['df_sota']
            ax.scatter(df_sota['date_obj'], df_sota['log_compute'],
                      color=color, s=80, alpha=0.7,
                      edgecolors='black', linewidth=0.5, zorder=3)

            # Add labels to SOTA points if requested
            if show_labels:
                for _, row in df_sota.iterrows():
                    _add_point_label(ax, row, label_content, label_format, color)

    # Also plot all background data points if they're not already in df_sota
    # (gray points showing all models in buckets)
    all_sota_indices = set()
    for reg in regression_lines:
        if 'df_sota' in reg and reg['df_sota'] is not None:
            all_sota_indices.update(reg['df_sota'].index)

    # Plot non-SOTA points in gray if there are any
    if len(all_sota_indices) < len(df_data):
        df_background = df_data[~df_data.index.isin(all_sota_indices)]
        if len(df_background) > 0:
            ax.scatter(df_background['date_obj'], df_background['log_compute'],
                      alpha=0.3, s=50, color='gray',
                      label='All models in buckets' if regression_lines else 'All models',
                      zorder=1)

    # Setup axes
    setup_date_axis(ax, major_locator='year', minor_locator=[1, 4, 7, 10])

    ax.set_xlabel('Release Date', fontsize=12)
    ax.set_ylabel('log₁₀(Training Compute)', fontsize=12)

    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)

    # Add legend if there are labeled items
    if any(reg.get('label') for reg in regression_lines) or len(all_sota_indices) < len(df_data):
        ax.legend(fontsize=9, ncol=1, framealpha=0.9, loc='best')

    return {
        'lines': lines
    }


def _add_point_label(ax, row, label_content, label_format, color):
    """Helper function to add label to a data point.

    Args:
        ax: Matplotlib axis
        row: DataFrame row with 'date_obj', 'log_compute', 'model', 'estimated_capability'
        label_content: 'both', 'eci', or 'model'
        label_format: 'auto', 'box', or 'simple'
        color: Color for the label background
    """
    # Construct label text
    if label_content == 'both':
        text = f"{row['model']}\nECI={row['estimated_capability']:.2f}"
    elif label_content == 'eci':
        text = f"{row['estimated_capability']:.2f}"
    elif label_content == 'model':
        text = f"{row['model']}"
    else:
        return

    # Determine format style
    if label_format == 'auto':
        # Use box format for 'both', simple for single items
        use_box = (label_content == 'both')
    else:
        use_box = (label_format == 'box')

    # Create annotation
    if use_box:
        bbox = dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3, edgecolor='none')
        fontsize = 7
        offset = (5, 5)
    else:
        bbox = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
        fontsize = 8
        offset = (3, 3)

    ax.annotate(
        text,
        xy=(row['date_obj'], row['log_compute']),
        xytext=offset,
        textcoords='offset points',
        fontsize=fontsize,
        alpha=0.8,
        bbox=bbox,
        zorder=4
    )


def plot_compute_vs_date_scatter_only(
    ax, df_data, color_by='estimated_capability',
    cmap='viridis', label_points='eci', label_format='simple',
    title=None, colorbar=True, colorbar_label='ECI'
):
    """Plot compute vs date as scatter plot colored by a variable.

    This is useful for the linear model visualization style.

    Args:
        ax: Matplotlib axis
        df_data: DataFrame with 'date_obj', 'log_compute', 'model', and color_by column
        color_by: Column name to use for coloring points
        cmap: Colormap name
        label_points: False, True, 'eci', or 'model' (see plot_compute_vs_date_with_regressions)
        label_format: 'auto', 'box', or 'simple'
        title: Plot title
        colorbar: Whether to add colorbar
        colorbar_label: Label for colorbar

    Returns:
        dict with 'scatter' object
    """
    # Create scatter plot
    scatter = ax.scatter(
        df_data['date_obj'],
        df_data['log_compute'],
        c=df_data[color_by],
        cmap=cmap,
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5,
        zorder=3
    )

    # Add colorbar
    if colorbar:
        cbar = plt.colorbar(scatter, ax=ax, label=colorbar_label)

    # Add labels if requested
    if label_points:
        label_content = 'both' if label_points is True else str(label_points)
        for _, row in df_data.iterrows():
            # For scatter-only plot, use gray color for label backgrounds
            _add_point_label(ax, row, label_content, label_format, color='lightgray')

    # Setup axes
    setup_date_axis(ax, major_locator='year', minor_locator=[1, 4, 7, 10])

    ax.set_xlabel('Release Date', fontsize=12)
    ax.set_ylabel('log₁₀(Training Compute)', fontsize=12)

    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=1)

    return {'scatter': scatter}
