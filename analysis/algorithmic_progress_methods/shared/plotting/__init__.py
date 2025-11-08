"""Plotting utilities for algorithmic progress analysis."""

from .base import (
    setup_date_axis,
    apply_plot_style,
    save_figure
)

from .distributions import (
    plot_histogram_with_stats,
    plot_bootstrap_distribution,
    plot_multiplier_distribution
)

from .regressions import (
    plot_scatter_with_fit,
    add_bootstrap_uncertainty_band
)

from .diagnostics import (
    plot_coefficient_distributions,
    plot_prediction_uncertainty,
    plot_residuals_with_ci,
    plot_coefficient_correlation
)

__all__ = [
    'setup_date_axis',
    'apply_plot_style',
    'save_figure',
    'plot_histogram_with_stats',
    'plot_bootstrap_distribution',
    'plot_multiplier_distribution',
    'plot_scatter_with_fit',
    'add_bootstrap_uncertainty_band',
    'plot_coefficient_distributions',
    'plot_prediction_uncertainty',
    'plot_residuals_with_ci',
    'plot_coefficient_correlation'
]
