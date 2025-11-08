"""CLI utilities for algorithmic progress analysis scripts."""

import argparse
from pathlib import Path


def add_distilled_filter_args(parser):
    """Add arguments for filtering distilled models.

    Args:
        parser: argparse.ArgumentParser instance
    """
    parser.add_argument('--exclude-med-high-distilled', action='store_true',
                       help='Exclude medium and high confidence distilled models from analysis')
    parser.add_argument('--exclude-distilled', action='store_true',
                       help='Exclude all distilled models (all confidence levels) from analysis')


def add_data_source_args(parser):
    """Add arguments for data source selection.

    Args:
        parser: argparse.ArgumentParser instance
    """
    parser.add_argument('--use-website-data', action='store_true',
                       help='Use data from data/website/epoch_capabilities_index.csv instead of outputs/model_fit/model_capabilities.csv')


def add_date_filter_args(parser):
    """Add arguments for filtering by release date.

    Args:
        parser: argparse.ArgumentParser instance
    """
    parser.add_argument('--min-release-date', type=str, default=None,
                       help='Only include models released on or after this date (format: YYYY-MM-DD)')


def validate_distilled_args(args):
    """Validate that distilled model filtering arguments are consistent.

    Args:
        args: Parsed arguments from argparse

    Raises:
        argparse.ArgumentError if validation fails
    """
    if args.exclude_distilled and args.exclude_med_high_distilled:
        raise argparse.ArgumentTypeError(
            'Cannot use both --exclude-distilled and --exclude-med-high-distilled')


def generate_output_suffix(exclude_distilled=False, exclude_med_high_distilled=False,
                           frontier_only=False, use_website_data=False,
                           min_release_date=None, **kwargs):
    """Generate consistent file suffix based on analysis options.

    Args:
        exclude_distilled: Whether all distilled models were excluded
        exclude_med_high_distilled: Whether med/high confidence distilled models were excluded
        frontier_only: Whether only frontier models were included
        use_website_data: Whether website data was used
        min_release_date: Minimum release date filter (if applied)
        **kwargs: Additional options (ignored)

    Returns:
        str: Suffix string (e.g., "_no_distilled_website" or "")
    """
    suffix_parts = []

    if exclude_distilled:
        suffix_parts.append("no_distilled")
    elif exclude_med_high_distilled:
        suffix_parts.append("no_med_high_distilled")
    if frontier_only:
        suffix_parts.append("frontier_only")
    if use_website_data:
        suffix_parts.append("website")
    if min_release_date:
        suffix_parts.append(f"from_{min_release_date}")

    return "_" + "_".join(suffix_parts) if suffix_parts else ""


def create_output_directory(method_name, base_dir="outputs/algorithmic_progress_methods",
                           exclude_distilled=False, exclude_med_high_distilled=False,
                           frontier_only=False, use_website_data=False,
                           min_release_date=None):
    """Create output directory for a specific method with subdirectory for configuration.

    Args:
        method_name: Name of the method (e.g., "buckets", "linear_model")
        base_dir: Base output directory
        exclude_distilled: Whether all distilled models were excluded
        exclude_med_high_distilled: Whether med/high confidence distilled models were excluded
        frontier_only: Whether only frontier models were included
        use_website_data: Whether website data was used
        min_release_date: Minimum release date filter (if applied)

    Returns:
        Path: Created directory path
    """
    # Build configuration-based subdirectory name
    config_parts = []

    # Data source (most important distinguisher)
    if use_website_data:
        config_parts.append("website")
    else:
        config_parts.append("internal")

    # Distilled model filtering
    if exclude_distilled:
        config_parts.append("no_distilled")
    elif exclude_med_high_distilled:
        config_parts.append("no_med_high_distilled")
    else:
        config_parts.append("with_distilled")

    # Frontier filtering
    if frontier_only:
        config_parts.append("frontier_only")
    else:
        config_parts.append("all_models")

    # Date filtering
    if min_release_date:
        config_parts.append(f"from_{min_release_date}")

    subdir_name = "_".join(config_parts)
    output_dir = Path(base_dir) / method_name / subdir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_title_suffix(exclude_distilled=False, exclude_med_high_distilled=False,
                          frontier_only=False, use_website_data=False,
                          min_release_date=None, **kwargs):
    """Generate human-readable suffix for plot titles.

    Args:
        exclude_distilled: Whether all distilled models were excluded
        exclude_med_high_distilled: Whether med/high confidence distilled models were excluded
        frontier_only: Whether only frontier models were included
        use_website_data: Whether website data was used
        min_release_date: Minimum release date filter (if applied)
        **kwargs: Additional options (ignored)

    Returns:
        str: Title suffix (e.g., " (excluding all distilled models, website data)")
    """
    title_parts = []

    if exclude_distilled:
        title_parts.append('excluding all distilled models')
    elif exclude_med_high_distilled:
        title_parts.append('excluding med/high confidence distilled models')
    if frontier_only:
        title_parts.append('frontier models only')
    if use_website_data:
        title_parts.append('website data')
    if min_release_date:
        title_parts.append(f'models from {min_release_date}+')

    return ' (' + ', '.join(title_parts) + ')' if title_parts else ''
