#!/usr/bin/env python3
"""Main entry point for buckets method analysis."""

import sys
import os
from pathlib import Path

# Change to project root for data access
project_root = Path(__file__).parent.parent.parent.parent
os.chdir(project_root)

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from shared.data_loading import load_model_capabilities_and_compute
from shared.cli_utils import (
    add_distilled_filter_args,
    add_data_source_args,
    validate_distilled_args,
    generate_output_suffix,
    create_output_directory,
    generate_title_suffix
)

from analysis import (
    analyze_compute_reduction,
    analyze_capability_gains,
    bucket_size_sensitivity_analysis
)

from plotting import (
    plot_compute_reduction_results,
    plot_capability_gains_results,
    plot_all_bucket_regressions,
    plot_bootstrap_distributions,
    plot_bucket_size_sensitivity,
    save_results
)
from hierarchical_median import run as run_hierarchical_median
from hierarchical_median_visualization import generate_plots as generate_hierarchical_plots


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description='Buckets method for algorithmic progress analysis')
    parser.add_argument('--eci-bucket-size', type=float, default=0.5,
                       help='Width of ECI buckets (default: 0.5 ECI units)')
    parser.add_argument('--compute-bucket-size', type=float, default=0.5,
                       help='Width of compute buckets in log10 scale (default: 0.5 OOMs)')
    parser.add_argument('--min-models', type=int, default=3,
                       help='Minimum number of SOTA models per bucket (default: 3)')

    add_distilled_filter_args(parser)
    add_data_source_args(parser)

    parser.add_argument('--sweep-bucket-sizes', action='store_true',
                       help='Perform sensitivity analysis by sweeping over bucket sizes')
    parser.add_argument('--eci-bucket-sizes', type=str, default=None,
                       help='Comma-separated ECI bucket sizes for sweep')
    parser.add_argument('--compute-bucket-sizes', type=str, default=None,
                       help='Comma-separated compute bucket sizes for sweep')
    parser.add_argument('--n-bucket-sizes', type=int, default=5,
                       help='Number of bucket sizes to test in sweep (default: 5)')
    parser.add_argument('--label-points', action='store_true',
                       help='Label data points with model names')
    parser.add_argument('--run-hierarchical-median', action='store_true',
                       help='Estimate pooled median compute reduction via hierarchical model and plot diagnostics')
    parser.add_argument('--hierarchical-bucket-sizes', type=str, default=None,
                       help='Comma-separated ECI bucket sizes for the hierarchical sweep (default: auto 5%-50% of range)')
    parser.add_argument('--hierarchical-n-bucket-sizes', type=int, default=5,
                       help='Number of auto-generated bucket sizes for the hierarchical sweep')
    parser.add_argument('--hierarchical-min-models', type=int, default=None,
                       help='Minimum SOTA models per bucket for the hierarchical analysis (default: --min-models)')
    parser.add_argument('--hierarchical-n-bootstrap', type=int, default=1000,
                       help='Bootstrap iterations per bucket within the hierarchical analysis')
    parser.add_argument('--hierarchical-n-iter', type=int, default=20000,
                       help='Total Gibbs iterations for the hierarchical sampler')
    parser.add_argument('--hierarchical-burn-in', type=int, default=5000,
                       help='Burn-in iterations for the hierarchical sampler')
    parser.add_argument('--hierarchical-thin', type=int, default=10,
                       help='Thinning factor for retained posterior samples')
    parser.add_argument('--hierarchical-seed', type=int, default=0,
                       help='Random seed for the hierarchical sampler')
    parser.add_argument('--hierarchical-mu-prior-mean', type=float, default=1.0,
                       help='Prior mean (OOMs/year) for the global compute reduction rate')
    parser.add_argument('--hierarchical-mu-prior-sd', type=float, default=3.0,
                       help='Prior standard deviation (OOMs/year) for the global compute reduction rate')
    parser.add_argument('--hierarchical-tau-prior-alpha', type=float, default=2.0,
                       help='Inverse-gamma alpha for between-bucket variance prior')
    parser.add_argument('--hierarchical-tau-prior-beta', type=float, default=0.5,
                       help='Inverse-gamma beta for between-bucket variance prior')

    args = parser.parse_args()

    # Validate arguments
    try:
        validate_distilled_args(args)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    # Load data
    df = load_model_capabilities_and_compute(
        use_website_data=args.use_website_data,
        exclude_distilled=args.exclude_distilled,
        include_low_confidence=args.include_low_confidence
    )

    if df is None or len(df) == 0:
        print("Failed to load data. Exiting.")
        return

    # Create output directory with configuration-based subdirectory
    output_dir = create_output_directory(
        "buckets",
        exclude_distilled=args.exclude_distilled,
        include_low_confidence=args.include_low_confidence,
        use_website_data=args.use_website_data
    )

    # Generate suffix for file names
    suffix = generate_output_suffix(
        exclude_distilled=args.exclude_distilled,
        include_low_confidence=args.include_low_confidence,
        use_website_data=args.use_website_data
    )

    # If sweep mode, run sensitivity analysis
    if args.sweep_bucket_sizes:
        eci_bucket_sizes = None
        compute_bucket_sizes = None

        if args.eci_bucket_sizes is not None:
            eci_bucket_sizes = [float(x) for x in args.eci_bucket_sizes.split(',')]
        if args.compute_bucket_sizes is not None:
            compute_bucket_sizes = [float(x) for x in args.compute_bucket_sizes.split(',')]

        cr_df, cg_df = bucket_size_sensitivity_analysis(
            df,
            eci_bucket_sizes=eci_bucket_sizes,
            compute_bucket_sizes=compute_bucket_sizes,
            min_models_per_bucket=args.min_models,
            n_bucket_sizes=args.n_bucket_sizes
        )

        # Save sensitivity results
        if len(cr_df) > 0:
            cr_df.to_csv(output_dir / f"bucket_size_sensitivity_compute_reduction{suffix}.csv", index=False)
        if len(cg_df) > 0:
            cg_df.to_csv(output_dir / f"bucket_size_sensitivity_capability_gains{suffix}.csv", index=False)

        # Create sensitivity plot
        print("\nCreating bucket size sensitivity plot...")
        plot_bucket_size_sensitivity(cr_df, cg_df, output_dir, suffix)

        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS COMPLETE")
        print("="*70)
        print(f"Results saved to: {output_dir}")
        return

    # Standard analysis with single bucket size
    compute_reduction_df, compute_reduction_buckets = analyze_compute_reduction(
        df,
        bucket_size_oom=args.eci_bucket_size,
        min_models_per_bucket=args.min_models
    )

    capability_gains_df, capability_gains_buckets = analyze_capability_gains(
        df,
        bucket_size_oom=args.compute_bucket_size,
        min_models_per_bucket=args.min_models
    )

    # Save results
    save_results(compute_reduction_df, capability_gains_df, output_dir, suffix)

    # Create visualizations
    plot_compute_reduction_results(compute_reduction_df, compute_reduction_buckets,
                                   output_dir, suffix)
    plot_capability_gains_results(capability_gains_df, capability_gains_buckets,
                                  output_dir, suffix)

    print("\nCreating combined bucket regression plots...")
    plot_all_bucket_regressions(df, compute_reduction_df, compute_reduction_buckets,
                                output_dir, suffix, label_points=args.label_points)

    print("\nCreating bootstrap distribution plots...")
    plot_bootstrap_distributions(compute_reduction_df, compute_reduction_buckets,
                                output_dir, suffix)

    if args.run_hierarchical_median:
        print("\nRunning hierarchical median analysis...")
        hierarchical_args = argparse.Namespace(
            bucket_sizes=args.hierarchical_bucket_sizes,
            n_bucket_sizes=args.hierarchical_n_bucket_sizes,
            min_models=args.hierarchical_min_models or args.min_models,
            n_bootstrap=args.hierarchical_n_bootstrap,
            n_iter=args.hierarchical_n_iter,
            burn_in=args.hierarchical_burn_in,
            thin=args.hierarchical_thin,
            seed=args.hierarchical_seed,
            mu_prior_mean=args.hierarchical_mu_prior_mean,
            mu_prior_sd=args.hierarchical_mu_prior_sd,
            tau_prior_alpha=args.hierarchical_tau_prior_alpha,
            tau_prior_beta=args.hierarchical_tau_prior_beta,
            exclude_distilled=args.exclude_distilled,
            include_low_confidence=args.include_low_confidence,
            use_website_data=args.use_website_data
        )
        hier_results = run_hierarchical_median(hierarchical_args)
        if hier_results:
            title_suffix = generate_title_suffix(
                exclude_distilled=args.exclude_distilled,
                include_low_confidence=args.include_low_confidence,
                use_website_data=args.use_website_data
            )
            plot_path = generate_hierarchical_plots(
                summary_path=hier_results["summary_path"],
                observations_path=hier_results["observations_path"],
                title_suffix=title_suffix
            )
            print(f"Hierarchical diagnostic plots saved to: {plot_path}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
