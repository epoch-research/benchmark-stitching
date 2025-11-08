"""Base plotting utilities - styling, saving, axis setup."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def setup_date_axis(ax, date_format='%Y', major_locator='year', minor_locator=None):
    """Setup date formatting for x-axis.

    Args:
        ax: Matplotlib axis
        date_format: Date format string (default: '%Y' for year)
        major_locator: 'year', 'month', 'quarter' or None
        minor_locator: List of months [1, 4, 7, 10] for quarters, or None
    """
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))

    if major_locator == 'year':
        ax.xaxis.set_major_locator(mdates.YearLocator())
    elif major_locator == 'month':
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    elif major_locator == 'quarter':
        ax.xaxis.set_major_locator(mdates.MonthLocator([1, 4, 7, 10]))

    if minor_locator is not None:
        ax.xaxis.set_minor_locator(mdates.MonthLocator(minor_locator))

    # Rotate labels for readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def apply_plot_style(ax, title=None, xlabel=None, ylabel=None,
                     grid=True, grid_alpha=0.3, legend=False, legend_kwargs=None):
    """Apply standard plot styling.

    Args:
        ax: Matplotlib axis
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        grid: Whether to show grid
        grid_alpha: Grid transparency
        legend: Whether to show legend
        legend_kwargs: Dictionary of legend kwargs
    """
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)

    if grid:
        ax.grid(True, alpha=grid_alpha, linestyle='--', linewidth=0.5)

    if legend:
        legend_kwargs = legend_kwargs or {}
        default_kwargs = {'fontsize': 10, 'framealpha': 0.9}
        default_kwargs.update(legend_kwargs)
        ax.legend(**default_kwargs)


def save_figure(fig, output_path, dpi=300, formats=None, bbox_inches='tight'):
    """Save figure in multiple formats.

    Args:
        fig: Matplotlib figure
        output_path: Output path (string or Path), extension will be replaced
        dpi: DPI for raster formats
        formats: List of formats ['png', 'svg'] or None for default ['png', 'svg']
        bbox_inches: Bounding box setting for tight layout
    """
    if formats is None:
        formats = ['png', 'svg']

    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        save_path = output_path.with_suffix(f'.{fmt}')
        if fmt == 'svg':
            fig.savefig(save_path, format='svg', bbox_inches=bbox_inches)
        else:
            fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved: {save_path}")
