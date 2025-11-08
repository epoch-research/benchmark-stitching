"""Diagnostic plotting utilities - uncertainty, residuals, coefficient analysis."""

import numpy as np
import matplotlib.pyplot as plt


def plot_coefficient_distributions(ax, bootstrap_coefs, coef_index, coef_name,
                                   bins=50, color='steelblue'):
    """Plot bootstrap distribution of a single coefficient.

    Args:
        ax: Matplotlib axis
        bootstrap_coefs: Array of bootstrap coefficients (n_bootstrap, n_coefs)
        coef_index: Index of coefficient to plot
        coef_name: Name of coefficient for labeling
        bins: Number of histogram bins
        color: Histogram color

    Returns:
        dict with statistics
    """
    if bootstrap_coefs.ndim == 1:
        coef_values = bootstrap_coefs
    else:
        coef_values = bootstrap_coefs[:, coef_index]

    ax.hist(coef_values, bins=bins, alpha=0.7, edgecolor='black', color=color)

    mean_val = coef_values.mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
              label=f'Mean: {mean_val:.4f}')

    ax.set_xlabel(f'{coef_name} coefficient', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Bootstrap Distribution: {coef_name} Coefficient',
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    return {'mean': mean_val, 'std': coef_values.std()}


def plot_prediction_uncertainty(ax, actual_values, pred_std, color_by=None,
                                cmap='viridis', xlabel='Actual Values',
                                ylabel='Prediction Std Dev'):
    """Plot prediction uncertainty vs actual values.

    Args:
        ax: Matplotlib axis
        actual_values: Actual Y values
        pred_std: Standard deviation of predictions from bootstrap
        color_by: Optional values to color points by
        cmap: Colormap name
        xlabel: X-axis label
        ylabel: Y-axis label

    Returns:
        scatter object
    """
    if color_by is not None:
        scatter = ax.scatter(actual_values, pred_std, c=color_by,
                           cmap=cmap, alpha=0.6, s=80,
                           edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Color by')
    else:
        scatter = ax.scatter(actual_values, pred_std, alpha=0.6, s=80,
                           edgecolors='black', linewidth=0.5)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title('Prediction Uncertainty vs Actual Values',
                fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    return scatter


def plot_residuals_with_ci(ax, predicted_values, residuals, pred_std,
                           show_ci_band=True, ci_multiplier=1.96):
    """Plot residuals with confidence interval bands.

    Args:
        ax: Matplotlib axis
        predicted_values: Predicted Y values
        residuals: Actual - Predicted
        pred_std: Standard deviation of predictions from bootstrap
        show_ci_band: Whether to show CI band
        ci_multiplier: Multiplier for CI (1.96 for 95%)

    Returns:
        scatter object
    """
    scatter = ax.scatter(predicted_values, residuals, alpha=0.6, s=80,
                        edgecolors='black', linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)

    if show_ci_band:
        # Sort for proper fill_between
        sorted_idx = np.argsort(predicted_values)
        pred_sorted = predicted_values[sorted_idx]
        std_sorted = pred_std[sorted_idx]

        ax.fill_between(pred_sorted,
                       -ci_multiplier * std_sorted,
                       ci_multiplier * std_sorted,
                       alpha=0.2, color='gray',
                       label=f'{int(ci_multiplier*50)}% CI')

    ax.set_xlabel('Predicted Values (mean)', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title('Residuals with Confidence Interval',
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    return scatter


def plot_coefficient_correlation(ax, bootstrap_coefs, coef_names=None,
                                 alpha=0.3, s=20):
    """Plot correlation between two coefficients from bootstrap.

    Args:
        ax: Matplotlib axis
        bootstrap_coefs: Array of bootstrap coefficients (n_bootstrap, 2)
        coef_names: List of two coefficient names
        alpha: Point transparency
        s: Point size

    Returns:
        scatter object, correlation coefficient
    """
    if bootstrap_coefs.shape[1] != 2:
        raise ValueError("This function requires exactly 2 coefficients")

    scatter = ax.scatter(bootstrap_coefs[:, 0], bootstrap_coefs[:, 1],
                        alpha=alpha, s=s)

    # Calculate correlation
    corr = np.corrcoef(bootstrap_coefs[:, 0], bootstrap_coefs[:, 1])[0, 1]

    # Labels
    if coef_names is None:
        coef_names = ['Coefficient 1', 'Coefficient 2']

    ax.set_xlabel(f'{coef_names[0]} coefficient', fontsize=12)
    ax.set_ylabel(f'{coef_names[1]} coefficient', fontsize=12)
    ax.set_title('Bootstrap Coefficient Correlation', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    # Add correlation text box
    ax.text(0.05, 0.95, f'Correlation: {corr:.4f}',
           transform=ax.transAxes, fontsize=12,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return scatter, corr


def plot_uncertainty_diagnostics_grid(fig, axes, bootstrap_results, actual_values,
                                     coef_names=None):
    """Create a 2x2 grid of uncertainty diagnostic plots.

    Args:
        fig: Matplotlib figure
        axes: 2x2 array of axes
        bootstrap_results: Dict from bootstrap function with keys:
            - coefs: bootstrap coefficients
            - pred_mean: mean predictions
            - pred_std: std of predictions
        actual_values: Actual Y values
        coef_names: List of coefficient names

    Returns:
        dict with plot objects
    """
    # 1. Coefficient distribution (first coefficient)
    ax = axes[0, 0]
    coef_name = coef_names[0] if coef_names else 'Coefficient 1'
    plot_coefficient_distributions(ax, bootstrap_results['coefs'], 0, coef_name)

    # 2. Coefficient distribution (second coefficient if available)
    ax = axes[0, 1]
    if bootstrap_results['coefs'].ndim > 1 and bootstrap_results['coefs'].shape[1] > 1:
        coef_name = coef_names[1] if coef_names and len(coef_names) > 1 else 'Coefficient 2'
        plot_coefficient_distributions(ax, bootstrap_results['coefs'], 1, coef_name)
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=20)
        ax.set_title('Second Coefficient (N/A)', fontsize=12, fontweight='bold')

    # 3. Prediction uncertainty
    ax = axes[1, 0]
    if 'log_compute' in bootstrap_results:
        plot_prediction_uncertainty(ax, actual_values, bootstrap_results['pred_std'],
                                   color_by=bootstrap_results['log_compute'])
    else:
        plot_prediction_uncertainty(ax, actual_values, bootstrap_results['pred_std'])

    # 4. Residuals with CI
    ax = axes[1, 1]
    residuals = actual_values - bootstrap_results['pred_mean']
    plot_residuals_with_ci(ax, bootstrap_results['pred_mean'], residuals,
                          bootstrap_results['pred_std'])

    plt.tight_layout()
    return axes
