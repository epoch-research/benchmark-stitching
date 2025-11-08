#!/usr/bin/env python3
"""Main entry point for linear model method analysis."""

import sys
import os
from pathlib import Path

# Change to project root for data access
project_root = Path(__file__).parent.parent.parent.parent
os.chdir(project_root)

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from shared.cli_utils import (
    add_distilled_filter_args,
    add_data_source_args,
    add_date_filter_args,
    validate_distilled_args,
    create_output_directory
)

from analysis import (
    load_and_filter_data,
    fit_linear_predictor
)

from plotting import (
    plot_uncertainty_diagnostics,
    plot_main_figure,
    print_summary_statistics
)


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description='Linear model method for algorithmic progress analysis')

    add_distilled_filter_args(parser)
    add_data_source_args(parser)
    add_date_filter_args(parser)

    parser.add_argument('--show-predicted-frontier', action='store_true',
                       help='Show the Pareto frontier predicted by the linear model for each month')
    parser.add_argument('--frontier-only', action='store_true',
                       help='Only include models that were on the Pareto frontier at their release date')
    parser.add_argument('--label-points', action='store_true',
                       help='Label data points with ECI values')

    args = parser.parse_args()

    # Validate arguments
    try:
        validate_distilled_args(args)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    # Load and filter data
    df_plot = load_and_filter_data(
        exclude_distilled=args.exclude_distilled,
        include_low_confidence=args.include_low_confidence,
        frontier_only=args.frontier_only,
        use_website_data=args.use_website_data,
        min_release_date=args.min_release_date
    )

    if df_plot is None:
        print("Failed to load data. Exiting.")
        return

    # Fit linear predictor with bootstrap
    model, df_plot, bootstrap_results = fit_linear_predictor(df_plot, n_bootstrap=1000)

    # Create output directory with configuration-based subdirectory
    output_dir = create_output_directory(
        "linear_model",
        exclude_distilled=args.exclude_distilled,
        include_low_confidence=args.include_low_confidence,
        frontier_only=args.frontier_only,
        use_website_data=args.use_website_data,
        min_release_date=args.min_release_date
    )

    # Create uncertainty diagnostic plots
    print("\nCreating uncertainty diagnostic plots...")
    plot_uncertainty_diagnostics(
        df_plot, bootstrap_results, output_dir,
        args.exclude_distilled, args.include_low_confidence,
        args.frontier_only, args.use_website_data,
        args.min_release_date
    )

    # Create main plot
    print("\nCreating main plot...")
    plot_main_figure(
        df_plot, model, bootstrap_results, output_dir,
        show_predicted_frontier=args.show_predicted_frontier,
        label_points=args.label_points,
        exclude_distilled=args.exclude_distilled,
        include_low_confidence=args.include_low_confidence,
        frontier_only=args.frontier_only,
        use_website_data=args.use_website_data,
        min_release_date=args.min_release_date
    )

    # Print summary statistics
    print_summary_statistics(df_plot)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
