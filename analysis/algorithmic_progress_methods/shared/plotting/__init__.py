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

from .unified_plots import (
    plot_compute_vs_date_with_regressions,
    plot_compute_vs_date_scatter_only
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
    'plot_coefficient_correlation',
    'plot_compute_vs_date_with_regressions',
    'plot_compute_vs_date_scatter_only'
]
