"""Regression plotting utilities - scatter plots with fits and uncertainty."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_scatter_with_fit(ax, x, y, slope, intercept, x_label=None, y_label=None,
                          scatter_kwargs=None, line_kwargs=None, show_fit_label=True):
    """Plot scatter plot with linear fit line.

    Args:
        ax: Matplotlib axis
        x: X values (can be dates, numeric, etc.)
        y: Y values
        slope: Slope of fit line
        intercept: Intercept of fit line
        x_label: X-axis label
        y_label: Y-axis label
        scatter_kwargs: Dict of kwargs for scatter plot
        line_kwargs: Dict of kwargs for fit line
        show_fit_label: Whether to add fit equation to legend

    Returns:
        scatter, line: Matplotlib objects
    """
    # Default scatter kwargs
    default_scatter = {
        'alpha': 0.6,
        's': 80,
        'edgecolors': 'black',
        'linewidth': 0.5,
        'zorder': 3
    }
    if scatter_kwargs:
        default_scatter.update(scatter_kwargs)

    # Default line kwargs
    default_line = {
        'linewidth': 2,
        'linestyle': '--',
        'color': 'red',
        'alpha': 0.8,
        'zorder': 2
    }
    if line_kwargs:
        default_line.update(line_kwargs)

    # Plot scatter
    scatter = ax.scatter(x, y, **default_scatter)

    # Plot fit line
    if isinstance(x, pd.Series):
        x_vals = x.values
    elif hasattr(x, '__iter__'):
        x_vals = np.array(list(x))
    else:
        x_vals = x

    # Handle datetime x-axis
    if hasattr(x_vals[0], 'toordinal'):  # datetime-like
        x_numeric = np.array([xi.toordinal() for xi in x_vals])
        y_fit = slope * x_numeric + intercept
    else:
        y_fit = slope * x_vals + intercept

    if show_fit_label:
        default_line['label'] = f'Fit: slope={slope:.3f}'

    line, = ax.plot(x, y_fit, **default_line)

    if x_label:
        ax.set_xlabel(x_label, fontsize=12)
    if y_label:
        ax.set_ylabel(y_label, fontsize=12)

    return scatter, line


def add_bootstrap_uncertainty_band(ax, x, bootstrap_slopes, bootstrap_intercepts,
                                   x_numeric=None, color='gray', alpha=0.15,
                                   use_ci=True, ci_percentiles=(2.5, 97.5)):
    """Add shaded uncertainty band from bootstrap estimates.

    Args:
        ax: Matplotlib axis
        x: X values for plotting
        bootstrap_slopes: Array of bootstrap slope estimates
        bootstrap_intercepts: Array of bootstrap intercept estimates
        x_numeric: Numeric x values for computation (if x is dates)
        color: Band color
        alpha: Band transparency
        use_ci: If True, use confidence interval; if False, use std dev
        ci_percentiles: Percentiles for CI (default: 2.5, 97.5 for 95% CI)

    Returns:
        fill_between object
    """
    # Get numeric x for computation
    if x_numeric is None:
        if hasattr(x[0], 'toordinal'):  # datetime-like
            x_numeric = np.array([xi.toordinal() for xi in x])
        else:
            x_numeric = np.array(x)

    # Compute predictions for each bootstrap sample
    bootstrap_preds = []
    for slope, intercept in zip(bootstrap_slopes, bootstrap_intercepts):
        y_pred = slope * x_numeric + intercept
        bootstrap_preds.append(y_pred)

    bootstrap_preds = np.array(bootstrap_preds)

    # Get uncertainty bounds
    if use_ci:
        lower = np.percentile(bootstrap_preds, ci_percentiles[0], axis=0)
        upper = np.percentile(bootstrap_preds, ci_percentiles[1], axis=0)
    else:
        mean = bootstrap_preds.mean(axis=0)
        std = bootstrap_preds.std(axis=0)
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std

    return ax.fill_between(x, lower, upper, color=color, alpha=alpha, zorder=1)


def plot_multiple_regressions_with_uncertainty(ax, x, regressions_data, colormap='viridis',
                                               show_scatter=True, show_uncertainty=True):
    """Plot multiple regression lines with bootstrap uncertainty bands.

    Args:
        ax: Matplotlib axis
        x: Common x values (e.g., date range)
        regressions_data: List of dicts, each containing:
            - 'label': str
            - 'slope': float
            - 'intercept': float
            - 'bootstrap_slopes': array (optional)
            - 'bootstrap_intercepts': array (optional)
            - 'scatter_x': array (optional, for data points)
            - 'scatter_y': array (optional, for data points)
        colormap: Colormap name for different regressions
        show_scatter: Whether to show data points
        show_uncertainty: Whether to show bootstrap uncertainty bands

    Returns:
        List of line objects
    """
    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(regressions_data)))
    lines = []

    for i, (reg_data, color) in enumerate(zip(regressions_data, colors)):
        slope = reg_data['slope']
        intercept = reg_data['intercept']
        label = reg_data.get('label', f'Regression {i+1}')

        # Convert x to numeric if needed
        if hasattr(x[0], 'toordinal'):
            x_numeric = np.array([xi.toordinal() for xi in x])
        else:
            x_numeric = np.array(x)

        # Plot regression line
        y_fit = slope * x_numeric + intercept
        line, = ax.plot(x, y_fit, color=color, linewidth=2.5, alpha=0.8,
                       label=label, zorder=2)
        lines.append(line)

        # Add uncertainty band if bootstrap data available
        if show_uncertainty and 'bootstrap_slopes' in reg_data:
            add_bootstrap_uncertainty_band(
                ax, x, reg_data['bootstrap_slopes'],
                reg_data['bootstrap_intercepts'],
                x_numeric=x_numeric,
                color=color,
                alpha=0.15
            )

        # Add scatter points if available
        if show_scatter and 'scatter_x' in reg_data and 'scatter_y' in reg_data:
            ax.scatter(reg_data['scatter_x'], reg_data['scatter_y'],
                      color=color, s=80, alpha=0.7,
                      edgecolors='black', linewidth=0.5, zorder=3)

    return lines
