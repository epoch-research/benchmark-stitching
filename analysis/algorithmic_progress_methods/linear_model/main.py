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
    parser.add_argument('--contour-spacing', type=float, default=None,
                       help='Spacing between ECI contour lines (e.g., 0.5 for every 0.5 ECI points). '
                            'If not specified, spacing is automatically determined based on data range.')
    parser.add_argument('--color-contours', action='store_true',
                       help='Color the ECI contour lines by their ECI value using viridis colormap')
    parser.add_argument('--eci-min', type=float, default=None,
                       help='Minimum ECI value to display on plot (analysis still uses all data)')
    parser.add_argument('--eci-max', type=float, default=None,
                       help='Maximum ECI value to display on plot (analysis still uses all data)')

    args = parser.parse_args()

    # Validate arguments
    try:
        validate_distilled_args(args)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    # Load and filter data
    df_plot = load_and_filter_data(
        exclude_distilled=args.exclude_distilled,
        exclude_med_high_distilled=args.exclude_med_high_distilled,
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
        exclude_med_high_distilled=args.exclude_med_high_distilled,
        frontier_only=args.frontier_only,
        use_website_data=args.use_website_data,
        min_release_date=args.min_release_date
    )

    # Create uncertainty diagnostic plots
    print("\nCreating uncertainty diagnostic plots...")
    plot_uncertainty_diagnostics(
        df_plot, bootstrap_results, output_dir,
        args.exclude_distilled, args.exclude_med_high_distilled,
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
        exclude_med_high_distilled=args.exclude_med_high_distilled,
        frontier_only=args.frontier_only,
        use_website_data=args.use_website_data,
        min_release_date=args.min_release_date,
        contour_spacing=args.contour_spacing,
        color_contours=args.color_contours,
        eci_min=args.eci_min,
        eci_max=args.eci_max
    )

    # Print summary statistics
    print_summary_statistics(df_plot)

    # Save summary to JSON for result collection
    try:
        import json
        from shared.cli_utils import generate_output_suffix

        suffix = generate_output_suffix(
            exclude_distilled=args.exclude_distilled,
            exclude_med_high_distilled=args.exclude_med_high_distilled,
            frontier_only=args.frontier_only,
            use_website_data=args.use_website_data,
            min_release_date=args.min_release_date
        )

        summary_file = output_dir / f"linear_model_summary{suffix}.json"
        tradeoff_summary = bootstrap_results.get('tradeoff_summary', {})

        summary_data = {
            "compute_year_tradeoff": {
                "median": tradeoff_summary.get('tradeoff_median'),
                "mean": tradeoff_summary.get('tradeoff_mean'),
                "ci_lower": tradeoff_summary.get('tradeoff_ci', [None, None])[0],
                "ci_upper": tradeoff_summary.get('tradeoff_ci', [None, None])[1],
            },
            "model_coefficients": {
                "log_compute": float(model.coef_[0]),
                "date": float(model.coef_[1]),
                "intercept": float(model.intercept_)
            },
            "n_models": len(df_plot)
        }

        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)

        print(f"\nSummary saved to: {summary_file}")
    except Exception as e:
        print(f"\nWarning: Could not save summary JSON: {e}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
